"""Abstractions used in torch_to_nnef internal graph Operations IR.

(decoupled from PyTorch and NNEF)

The goal is that these elements are:
- extracted/parsed from PyTorch graph operations
- translated to NNEF graph operations

"""

import logging
import typing as T
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import torch

from torch_to_nnef.dtypes import (
    dtype_is_whole_number,
    is_quantized_dtype,
    str_to_torch_dtype,
)
from torch_to_nnef.exceptions import (
    T2NErrorIRDataConsistency,
    T2NErrorNotImplemented,
    T2NErrorRuntime,
    T2NErrorTorchCheck,
    T2NErrorTorchOpTranslatedDifferently,
    T2NErrorTorchUnableToTraceData,
)
from torch_to_nnef.torch_graph.ir_data import (
    Data,
    TensorVariable,
    TtupleOrVar,
    TupleTensors,
)
from torch_to_nnef.torch_graph.ir_helpers import (
    _expand_containers_if_exists,
    _expand_node_containers_if_exists,
    _extract_op_infos,
    _reconstruct_view_dims,
    _rerouted_parsing,
    dynamic_tensor_list_parse,
)
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_ALIAS,
    ATEN_ARANGE,
    ATEN_BADDMM,
    ATEN_CLONE,
    ATEN_CUMSUM,
    ATEN_EINSUM,
    ATEN_EMBEDDING,
    ATEN_EMPTY,
    ATEN_EMPTY_LIKE,
    ATEN_FULL,
    ATEN_FULL_LIKE,
    ATEN_GELU,
    ATEN_INT,
    ATEN_LINALG_NORM,
    ATEN_LINALG_VECTOR_NORM,
    ATEN_LINEAR,
    ATEN_MASKED_FILL,
    ATEN_MASKED_FILL_,
    ATEN_MATMUL,
    ATEN_NEW_EMPTY,
    ATEN_NEW_ONES,
    ATEN_NEW_ZEROS,
    ATEN_ONES_LIKE,
    ATEN_PROD,
    ATEN_REPEAT_INTERLEAVE,
    ATEN_SCALAR_TENSOR,
    ATEN_SCALED_DOT_PRODUCT_ATTENTION,
    ATEN_SIZE_KIND,
    ATEN_STARTID,
    ATEN_TO,
    ATEN_TO_COPY,
    ATEN_VIEW_KIND,
    ATEN_WHERE,
    ATEN_ZERO_LIKE,
    ATEN_ZEROS,
    CALL_KIND,
    LISTTYPE_KIND,
    MAP_TO_TENSOR_FN,
    NONETYPE_KIND,
    NUMTOTENSOR_KIND,
    TUPLETYPE_KIND,
    WIRED_CUSTOM_LSTM,
)
from torch_to_nnef.utils import ReactiveNamedItemDict

LOGGER = logging.getLogger(__name__)


class InputsAlignBetweenAtenAndTorch:
    """Mapping inputs between Python `torch.$1` and cpp `aten::$2`.

    Because function arguments are not 1 to 1
    """

    @classmethod
    def align_inputs(cls, kind: str, args, kwargs):
        map_align = {
            ATEN_ARANGE: cls.aten_arange,
            ATEN_BADDMM: cls.aten_baddmm,
            ATEN_CUMSUM: cls.aten_cumsum,
            ATEN_EINSUM: cls.aten_einsum,
            ATEN_EMPTY: cls.aten_zero,
            ATEN_EMPTY_LIKE: cls.aten_empty_like,
            ATEN_NEW_EMPTY: cls.aten_new_empty,
            ATEN_FULL: cls.aten_full,
            ATEN_FULL_LIKE: cls.aten_full_like,
            ATEN_GELU: cls.aten_gelu,
            ATEN_LINALG_NORM: cls.aten_linalg_norm,
            ATEN_LINALG_VECTOR_NORM: cls.aten_linalg_norm,
            ATEN_MASKED_FILL: cls.aten_masked_fill,
            ATEN_MASKED_FILL_: cls.aten_masked_fill,
            ATEN_NEW_ONES: cls.aten_new_ones,
            ATEN_ONES_LIKE: cls.aten_ones_like,
            ATEN_PROD: cls.aten_prod,
            ATEN_REPEAT_INTERLEAVE: cls.aten_repeat_interleave,
            ATEN_SCALAR_TENSOR: cls.aten_scalar_tensor,
            ATEN_SCALED_DOT_PRODUCT_ATTENTION: cls.aten_scaled_dot_product_attention,  # noqa: E501
            ATEN_TO: cls.aten_to,
            ATEN_TO_COPY: cls.aten_to_copy,
            ATEN_WHERE: cls.aten_where,
            ATEN_NEW_ZEROS: cls.aten_new_zero,
            ATEN_ZEROS: cls.aten_zero,
            ATEN_ZERO_LIKE: cls.aten_zero,
            WIRED_CUSTOM_LSTM: cls.wired_custom_lstm,
        }
        to_call = map_align.get(kind)
        if to_call:
            return to_call(args, kwargs)
        return args, kwargs

    @staticmethod
    def wired_custom_lstm(args, kwargs):
        if len(args) > 2:
            args = (args[0], (args[1], args[2]))
        return args, kwargs

    @staticmethod
    def aten_where(args, kwargs):
        args = list(args)
        args[0] = args[0].bool()
        args = tuple(args)
        return args, kwargs

    @staticmethod
    def aten_baddmm(args, kwargs):
        args = list(args[:3])
        return args, kwargs

    @staticmethod
    def aten_scalar_tensor(args, kwargs):
        val, dtype, layout, device, _ = args
        args = (val,)
        kwargs["dtype"] = dtype
        kwargs["layout"] = layout
        kwargs["device"] = device
        return args, kwargs

    @staticmethod
    def aten_zero(args, kwargs):
        args = args[:1]
        return args, kwargs

    @staticmethod
    def aten_new_zero(args, kwargs):
        args = args[:2]
        return args, kwargs

    @staticmethod
    def aten_masked_fill(args, kwargs):
        if len(args) >= 2:
            largs = list(args)
            largs[1] = largs[1].bool()
            args = tuple(largs)
        return args, kwargs

    @staticmethod
    def aten_ones_like(args, kwargs):
        args = args[:1]
        return args, kwargs

    @staticmethod
    def aten_to(args, kwargs):
        if isinstance(args[2], int):
            LOGGER.debug("wrongly `ordered` to parameters")
            args = args[:2]
        return args, kwargs

    @staticmethod
    def aten_to_copy(args, kwargs):
        val, dtype, _, _, _, _, _ = args
        args = (val,)
        kwargs["dtype"] = dtype
        return args, kwargs

    @staticmethod
    def aten_gelu(args, kwargs):
        args = args[:1]  # skip the 'none' param starting torch 1.12.0
        return args, kwargs

    @staticmethod
    def aten_arange(args, kwargs):
        # ensure there is no args as tesnsor
        args = [a.tolist() if isinstance(a, torch.Tensor) else a for a in args]
        if len(args) >= 4:
            kwargs["dtype"] = args.pop(3)
        return args, kwargs

    @staticmethod
    def aten_full(args, kwargs):
        args = list(args[:2])
        args[0] = [
            a.tolist() if isinstance(a, torch.Tensor) else a for a in args[0]
        ]
        if not isinstance(args[-1], (float, int)):
            args[-1] = args[-1].tolist()
        return args, kwargs

    @staticmethod
    def aten_full_like(args, kwargs):
        # orig, value, *_ = args
        args = args[:2]
        return args, kwargs

    @staticmethod
    def aten_cumsum(args, kwargs):
        args = list(args[:2])
        return args, kwargs

    @staticmethod
    def aten_new_ones(args, kwargs):
        args = list(args[:2])
        return args, kwargs

    @staticmethod
    def aten_prod(args, kwargs):
        args = list(args[:1])
        return args, kwargs

    @staticmethod
    def aten_linalg_norm(args, kwargs):
        # last is likely param: 'out'
        args = list(args[:4])
        return args, kwargs

    @staticmethod
    def aten_einsum(args, kwargs):
        args = list(args[:2])
        return args, kwargs

    @staticmethod
    def aten_empty_like(args, kwargs):
        args = list(args[:1])
        return args, kwargs

    @staticmethod
    def aten_new_empty(args, kwargs):
        args = list(args[:2])
        return args, kwargs

    @staticmethod
    def aten_scaled_dot_product_attention(args, kwargs):
        args = list(args[:6])
        if len(args) >= 7 and args[6].data is not None:
            kwargs["scale"] = None
        return args, kwargs

    @staticmethod
    def aten_repeat_interleave(args, kwargs):
        assert len(args) == 4
        args = args[:3]
        return args, kwargs


@dataclass
class TorchOp:
    kind: str
    module_path: str
    inputs: T.List[Data]
    outputs: T.List[TtupleOrVar]
    scope: str
    op_ref: T.Optional[T.Callable]  # multiple ins and outs possible
    call_name: T.Optional[str]

    def __hash__(self):
        # cache for speed-up
        if not hasattr(self, "_cached_hash"):
            hs = hash(f"{self.kind}{self.inputs}{self.outputs}")
            self._cached_hash = hs  # pylint: disable=attribute-defined-outside-init
        return self._cached_hash

    @property
    def is_callmethod(self) -> bool:
        return self.kind == CALL_KIND

    @classmethod
    def _parse_outputs(
        cls, node: torch._C.Node, data_nodes: ReactiveNamedItemDict
    ):
        outputs: T.List[TtupleOrVar] = []
        for out_node in node.outputs():  #: torch._C.Value
            if out_node.type().annotation_str != NONETYPE_KIND:
                if out_node.type().kind() == LISTTYPE_KIND:
                    fixed_tensor_list = dynamic_tensor_list_parse(out_node)
                    data_nodes += fixed_tensor_list.data
                    data_nodes.append(fixed_tensor_list)
                    outputs += fixed_tensor_list.data
                elif out_node.type().kind() == TUPLETYPE_KIND:
                    tuple_out = TupleTensors.parse_from_tuple_type(out_node)
                    for tupitem in tuple_out.data:
                        data_nodes.append(tupitem)
                    # ducktyping/factorize tensor_out & tuple_out
                    # lead to mypy complain hence repeated
                    outputs.append(tuple_out)
                    data_nodes.append(tuple_out)
                else:
                    tensor_out = TensorVariable.parse(out_node)
                    outputs.append(tensor_out)
                    data_nodes.append(tensor_out)
        return outputs

    @classmethod
    def parse(
        cls,
        module,
        node: torch._C.Node,
        scope: str,
        data_nodes: ReactiveNamedItemDict,
        traced_module,
    ) -> "TorchOp":
        op_ref = None
        _rerouted_parsing(node, data_nodes, module)

        if node.kind() in [ATEN_INT]:  # , NUMTOTENSOR_KIND
            raise T2NErrorNotImplemented(f"node: {node} should create an ops")
        (
            kind,
            call_name,
            module_getter_ref,
            op_ref,
            inputs,
        ) = _extract_op_infos(module, data_nodes, node, traced_module)

        outputs = cls._parse_outputs(node, data_nodes)

        if not outputs:
            raise T2NErrorTorchOpTranslatedDifferently(
                "Avoid reccording no return operations"
            )

        return cls(
            kind=kind,
            inputs=inputs,
            outputs=outputs,
            scope=scope,
            module_path=module_getter_ref,
            op_ref=op_ref,
            call_name=call_name,
        )

    def update_call_op_arg_kwargs(self, args):
        """Custom adaptation to call aten fn with torch exposed py fn."""
        kwargs = {}
        if self.kind == "aten::div" and len(args) >= 3:
            kwargs["rounding_mode"] = args[2]
            args = args[:-1]
            self.op_ref = torch.div
        # }
        args, kwargs = InputsAlignBetweenAtenAndTorch.align_inputs(
            self.kind, args, kwargs
        )
        return args, kwargs

    def call_op(self):
        """Produce operation output based on traced inputs with real torch call.

        This operation call is done via self.args arguments (for now).
        Which means that we need to have all args needed in parameters order,
        following at least 1 underling torch operation signature.

        NOTE: we use a different approach than original torch.onnx which pass
        parameter by keyword arguments, this is due to the fact that we are not
        aware of argument name being provided in exported graph (
            from what we understand torch.onnx solve this via explicit
            rerouting of all signatures, which might be a bit bulky
            in most case
        ).

        """
        if self.op_ref is not None:
            if self.kind in MAP_TO_TENSOR_FN:
                args = self.args
                tensor = args[0]
                subargs = args[1:]
                if self.kind == ATEN_VIEW_KIND and None in subargs[0]:
                    # custom reconstruction of missing dimensions infos
                    subargs = list(subargs)
                    subargs[0] = _reconstruct_view_dims(
                        tensor.shape, subargs[0]
                    )
                    self.inputs[1].data.set_data(subargs[0])
                    subargs = tuple(subargs)
                return getattr(tensor, self.kind.replace(ATEN_STARTID, ""))(
                    *subargs
                )

            # hacky/bad way to pass argument that are named argument only {
            args, kwargs = self.update_call_op_arg_kwargs(self.args)

            try:
                return self.op_ref(*args, **kwargs)
            except RuntimeError as exp:
                raise T2NErrorRuntime(
                    f"running {self.op_ref}(args={args}, kwargs={kwargs})"
                ) from exp
        raise T2NErrorNotImplemented(self)

    @property
    def has_constant_inputs(self) -> bool:
        for input_node in self.inputs:
            if input_node.is_constant:
                continue
            return False
        return True

    @property
    def args(self) -> T.Tuple[T.Any, ...]:
        return tuple(_.tracing_data for _ in self.inputs)

    def _infer_trace_result(self, approx: bool = True):
        """Infer empty tensor of correct shape and dtype as output.

        This is a best-effort inference based on input shapes and dtypes,
        and operator semantics.

        This allows to have a trace of the output tensor without
        executing the actual operation, which speed-up significantly
        the tracing process.
        """
        if self.kind in [NUMTOTENSOR_KIND, ATEN_CLONE, ATEN_ALIAS]:
            results = self.args[0]
        elif self.kind == ATEN_EMBEDDING and approx:
            ax = list(self.args[0].shape)
            bx = list(self.args[1].shape)
            shape = bx + ax[1:]
            results = torch.empty(shape, dtype=self.args[0].dtype)
        elif (
            self.kind == ATEN_MATMUL
            and {ar.dtype for ar in self.args}
            and approx
        ):
            a = self.args[0]
            b = self.args[1]
            ax = list(a.shape)
            bx = list(b.shape)
            # Basic rank check (matmul requires at least 2D tensors here)
            assert len(ax) >= 2 and len(bx) >= 2, (
                "Expected tensors with rank >= 2"
            )
            # Inner dimension compatibility: (..., M, K) @ (..., K, N)
            assert ax[-1] == bx[-2], "Incompatible matmul inner dimensions"
            # Explicit batch broadcasting resolution
            batch_shape = torch.broadcast_shapes(ax[:-2], bx[:-2])
            # Output shape: (..., M, N)
            cx = list(batch_shape) + [ax[-2], bx[-1]]
            # Simulated output tensor
            results = torch.empty(cx, dtype=a.dtype)
        elif (
            self.kind == ATEN_LINEAR
            and {ar.dtype for ar in self.args if ar is not None}
            and approx
        ):
            x = self.args[0]  # input
            w = self.args[1]  # weight
            ax = list(x.shape)
            wx = list(w.shape)
            # input: (*batch, in_features)
            # weight: (out_features, in_features)
            assert len(ax) >= 1, "linear expects input rank >= 1"
            assert len(wx) == 2, "linear weight must be 2D"
            assert ax[-1] == wx[-1], "in_features mismatch"
            # output: (*batch, out_features)
            cx = ax[:-1] + [wx[0]]
            results = torch.empty(cx, dtype=x.dtype)

        else:
            results = self.call_op()
        return results

    # pylint: disable-next=too-many-branches
    def realise_output_type_and_size(self, approx: bool = True) -> bool:
        """Trace output and try to find type shape and constant realisation."""
        if self.kind == CALL_KIND:
            return False

        if not all(_.tracable for _ in self.inputs):
            return False

        if isinstance(self.op_ref, TorchModuleTracer):
            self.op_ref.args = self.args

        # generate all data and call ops to infer missing infos
        try:
            results = self._infer_trace_result()
        except T2NErrorTorchUnableToTraceData:
            return False

        if isinstance(results, int):
            results = torch.tensor(results, dtype=torch.int64)

        if isinstance(results, torch.Tensor):
            results = (results,)

        output_nodes = list(
            _expand_node_containers_if_exists(
                self.outputs, filter_container=True
            )
        )
        output_values = list(
            _expand_containers_if_exists(results, filter_container=True)
        )
        if len(output_nodes) != len(output_values):
            raise T2NErrorTorchCheck(
                "Arity Missmatch between extracted from graph "
                f"len({len(output_nodes)}) and the one experienced "
                f"in tracing simulation len({len(output_values)}) "
                f"for {self.op_ref}"
            )
        for data_node, result in zip(output_nodes, output_values):
            if self.has_constant_inputs:
                try:
                    data_node.set_data(result)
                except T2NErrorIRDataConsistency:
                    logging.debug(
                        "Conflicting TensorVariable specification "
                        "data for %s with result %s, "
                        " trusting result from realise_output_type_and_size",
                        data_node,
                        result,
                    )
                    data_node.dtype = result.dtype
                    data_node.shape = list(result.shape)
                    data_node.set_data(result)
            if isinstance(data_node, TensorVariable) and result is not None:
                if self.kind == ATEN_SIZE_KIND:
                    # note this is a special case where we fix variable value
                    data_node.set_data(result)
                if isinstance(result, torch.Tensor):
                    data_node.dtype = result.dtype
                    data_node.shape = list(result.shape)
                    if dtype_is_whole_number(result.dtype):
                        data_node._traced_data = result

                    if is_quantized_dtype(result.dtype):
                        data_node.quant = {
                            "scale": result.q_scale(),
                            "zero_point": result.q_zero_point(),
                        }
                else:
                    if isinstance(result, int):
                        data_node.dtype = str_to_torch_dtype("int")
                        data_node.shape = []
                        data_node._traced_data = result
                    elif isinstance(result, float):
                        data_node.dtype = str_to_torch_dtype("float")
                        data_node.shape = []
                    else:
                        raise T2NErrorNotImplemented(result)
        return True

    def __repr__(self):
        body = f"\tkind={self.kind}\n"
        inputs = "".join(f"\t\t{input_},\n" for input_ in self.inputs)
        body += f"\tinputs=(\n{inputs}\n\t)\n"

        outputs = "".join(f"\t\t{output},\n" for output in self.outputs)
        body += f"\toutputs=(\n{outputs}\t)\n"
        return f"TorchOp(\n{body}\n)".replace("\t", " " * 2)


class CacheDataNodeTarget(str, Enum):
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    ALL = "all"


class CacheDataToOpsNode:
    def __init__(self, target: CacheDataNodeTarget, ops: T.Sequence[TorchOp]):
        self.target = target
        self._map: T.Dict[Data, T.List[TorchOp]] = defaultdict(list)
        self.build_cache(ops)

    def build_cache(self, ops):
        y_inputs = self.target in [
            CacheDataNodeTarget.ALL,
            CacheDataNodeTarget.INPUTS,
        ]
        y_outputs = self.target in [
            CacheDataNodeTarget.ALL,
            CacheDataNodeTarget.OUTPUTS,
        ]
        for op in ops:
            if y_inputs:
                for inp in _expand_node_containers_if_exists(op.inputs):
                    self._map[inp].append(op)
            if y_outputs:
                for out in _expand_node_containers_if_exists(op.outputs):
                    self._map[out].append(op)

    def get(self, data_node: Data):
        return self._map[data_node]
