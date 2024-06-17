"""Abstractions used in torch_to_nnef internal graph Operations IR
(decoupled from PyTorch and NNEF)

The goal is that these elements are:
- extracted/parsed from PyTorch graph operations
- translated to NNEF graph operations

"""

import logging
import typing as T
from dataclasses import dataclass

import torch

from torch_to_nnef.dtypes import is_quantized_dtype
from torch_to_nnef.exceptions import (
    TorchCheckError,
    TorchOpTranslatedDifferently,
    TorchToNNEFNotImplementedError,
)
from torch_to_nnef.torch_graph.ir_data import (
    Data,
    TensorVariable,
    TtupleOrVar,
    TupleTensors,
)
from torch_to_nnef.torch_graph.ir_helpers import (
    _expand_containers_if_exists,
    _extract_op_infos,
    _reconstruct_view_dims,
    _rerouted_parsing,
    dynamic_tensor_list_parse,
)
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_ARANGE,
    ATEN_CUMSUM,
    ATEN_EMPTY,
    ATEN_FULL,
    ATEN_GELU,
    ATEN_INT,
    ATEN_NEW_ONES,
    ATEN_PROD,
    ATEN_REPEAT_INTERLEAVE,
    ATEN_SCALED_DOT_PRODUCT_ATTENTION,
    ATEN_SIZE_KIND,
    ATEN_STARTID,
    ATEN_TO,
    ATEN_VIEW_KIND,
    ATEN_ZERO_LIKE,
    ATEN_ZEROS,
    CALL_KIND,
    LISTTYPE_KIND,
    MAP_TO_TENSOR_FN,
    NONETYPE_KIND,
    TUPLETYPE_KIND,
)

LOGGER = logging.getLogger(__name__)


class InputsAlignBetweenAtenAndTorch:
    """Mapping inputs between Python `torch.$1` and cpp `aten::$2`
    Because function arguments are not 1 to 1
    """

    @classmethod
    def align_inputs(cls, kind: str, args, kwargs):
        map_align = {
            ATEN_ZERO_LIKE: cls.aten_zero,
            ATEN_ZEROS: cls.aten_zero,
            ATEN_EMPTY: cls.aten_zero,
            ATEN_TO: cls.aten_to,
            ATEN_GELU: cls.aten_gelu,
            ATEN_ARANGE: cls.aten_arange,
            ATEN_FULL: cls.aten_full,
            ATEN_CUMSUM: cls.aten_cumsum,
            ATEN_NEW_ONES: cls.aten_new_ones,
            ATEN_PROD: cls.aten_prod,
            ATEN_SCALED_DOT_PRODUCT_ATTENTION: cls.aten_scaled_dot_product_attention,
            ATEN_REPEAT_INTERLEAVE: cls.aten_repeat_interleave,
        }
        to_call = map_align.get(kind)
        if to_call:
            return to_call(args, kwargs)
        return args, kwargs

    @staticmethod
    def aten_zero(args, kwargs):
        args = args[:1]
        return args, kwargs

    @staticmethod
    def aten_to(args, kwargs):
        if isinstance(args[2], int):
            LOGGER.debug("wrongly `ordered` to parameters")
            args = args[:2]
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
        if not isinstance(args[-1], float):
            args[-1] = args[-1].tolist()
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
        args = list(args[:3])
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
        return hash(f"{self.kind}{self.inputs}{self.outputs}")

    @property
    def is_callmethod(self) -> bool:
        return self.kind == CALL_KIND

    @classmethod
    def _parse_outputs(cls, node: torch._C.Node, data_nodes: T.List[Data]):
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
        data_nodes: T.List[Data],
        traced_module,
    ) -> "TorchOp":
        op_ref = None
        _rerouted_parsing(node, data_nodes, module)

        if node.kind() in [ATEN_INT]:  # , NUMTOTENSOR_KIND
            raise NotImplementedError(f"node: {node} should create an ops")
        (
            kind,
            call_name,
            module_getter_ref,
            op_ref,
            inputs,
        ) = _extract_op_infos(module, data_nodes, node, traced_module)

        outputs = cls._parse_outputs(node, data_nodes)

        if not outputs:
            raise TorchOpTranslatedDifferently(
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
        """Custom adaptation to call aten fn with torch exposed py fn"""
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
        """Produce operation output based on traced inputs with real torch call

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
                    self.inputs[1].data = subargs[0]
                    subargs = tuple(subargs)
                return getattr(tensor, self.kind.replace(ATEN_STARTID, ""))(
                    *subargs
                )

            # hacky/bad way to pass argument that are named argument only {
            args, kwargs = self.update_call_op_arg_kwargs(self.args)

            return self.op_ref(*args, **kwargs)
        raise TorchToNNEFNotImplementedError(self)

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

    def realise_output_type_and_size(self) -> bool:
        """Trace output and try to find type shape and constant realisation"""
        if not all(_.tracable for _ in self.inputs):
            return False

        if isinstance(self.op_ref, TorchModuleTracer):
            self.op_ref.args = self.args

        # generate all data and call ops to infer missing infos
        results = self.call_op()

        if isinstance(results, int):
            results = torch.tensor(results, dtype=torch.int32)

        if isinstance(results, torch.Tensor):
            results = (results,)

        output_nodes = list(
            _expand_containers_if_exists(self.outputs, filter_container=True)
        )
        output_values = list(
            _expand_containers_if_exists(results, filter_container=True)
        )
        if len(output_nodes) != len(output_values):
            raise TorchCheckError(
                "Arity Missmatch between extracted from graph "
                f"len({len(output_nodes)}) and the one experienced "
                f"in tracing simulation len({len(output_values)}) "
                f"for {self.op_ref}"
            )
        for data_node, result in zip(output_nodes, output_values):
            if self.has_constant_inputs:
                data_node.data = result
            if isinstance(data_node, TensorVariable):
                if self.kind == ATEN_SIZE_KIND:
                    # note this is a special case where we fix variable value
                    data_node.data = result
                data_node.dtype = result.dtype
                data_node.shape = list(result.shape)
                if is_quantized_dtype(result.dtype):
                    data_node.quant = {
                        "scale": result.q_scale(),
                        "zero_point": result.q_zero_point(),
                    }

        return True

    def __repr__(self):
        body = f"\tkind={self.kind}\n"
        inputs = "".join(f"\t\t{input_},\n" for input_ in self.inputs)
        body += f"\tinputs=(\n{inputs}\n\t)\n"

        outputs = "".join(f"\t\t{output},\n" for output in self.outputs)
        body += f"\toutputs=(\n{outputs}\t)\n"
        return f"TorchOp(\n{body}\n)".replace("\t", " " * 2)
