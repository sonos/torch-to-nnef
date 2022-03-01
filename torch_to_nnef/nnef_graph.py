import logging
import typing as T
from datetime import datetime
from itertools import chain

import numpy as np
import torch
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor
from torch import nn

from torch_to_nnef.torch_graph import (
    InternalPytorchGraphHelper,
    NodeConstant,
    NodeInput,
    NodeTensorSized,
    _access_module,
    clean_dtype_name,
)

LOGGER = logging.getLogger(__name__)


def _torch_to_nnef_typestr(torch_type_str: str):

    if torch_type_str == "QUInt8":
        return np.int8
    if torch_type_str == "Long":
        return np.int64
    if torch_type_str == "Float":
        return np.float32
    if torch_type_str == "int":
        return np.int32

    raise NotImplementedError(torch_type_str)


class _ModuleInfoRegistery(type):

    """Allow extract in NNEF behavior from specific nn.Module"""

    JIT_CLASS_NAME = None

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        if new_cls.JIT_CLASS_NAME is not None:
            cls.REGISTRY[new_cls.JIT_CLASS_NAME] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)


class ModuleInfoExtractor(metaclass=_ModuleInfoRegistery):

    """Allow extract in NNEF behavior from specific nn.Module"""

    JIT_CLASS_NAME = None

    def __init__(self, node: NodeTensorSized, module: nn.Module, g: NGraph):
        assert self.JIT_CLASS_NAME is not None
        self.node = node
        self.mod = module
        self.g = g

    def _quant_from_ref(self, tensor, ref_qtensor):
        return torch.quantize_per_tensor(
            tensor,
            scale=ref_qtensor.q_scale(),
            zero_point=ref_qtensor.q_zero_point(),
            dtype=ref_qtensor.dtype,
        )

    def _torch_qtensor_to_ntensor(self, tensor, name):
        np_int_tensor = tensor.int_repr().numpy()
        return NTensor(
            self.g,
            name=name,
            shape=tuple(tensor.shape),
            dtype=np_int_tensor.dtype.type,
            data=np_int_tensor,
            # https://docs.rs/onnx-pb/0.1.1/onnx_pb/struct.TensorAnnotation.html
            quant={
                "scale": tensor.q_scale(),
                "zero_point": tensor.q_zero_point(),
                "bits": 8,
                "signed": True,
                "symmetric": False,
                "op-name": "zero_point_linear_quantize",
            },
        )

    def _add_idem_type_shape_output_tensor(
        self, name_to_tensor, force_dtype=None
    ):
        out_tensor_name = self.node.export_name
        input_node = name_to_tensor[self.node.export_inputs[1]]
        output_tensor = NTensor(
            graph=self.g,
            name=out_tensor_name,
            dtype=force_dtype or input_node.dtype,
            shape=input_node.shape,
        )
        name_to_tensor[out_tensor_name] = output_tensor
        return output_tensor

    def add_quantized_tensor_to_ngraph(
        self,
        qtensor: torch.Tensor,
        tensor_name: str,
        name_to_tensors: T.Dict[str, NTensor],
    ):
        name = f"{self.node.export_name}_{tensor_name}"
        ntensor = self._torch_qtensor_to_ntensor(qtensor, name)
        name_to_tensors[name] = ntensor
        return ntensor

    def add_tensor_to_ngraph(
        self,
        tensor: torch.Tensor,
        tensor_name: str,
        name_to_tensors: T.Dict[str, NTensor],
    ):
        name = f"{self.node.export_name}_{tensor_name}"
        tensor_np = tensor.numpy()
        ntensor = NTensor(
            self.g,
            name=name,
            shape=tuple(tensor.shape),
            dtype=tensor_np.dtype.type,
            data=tensor_np,
        )
        name_to_tensors[name] = ntensor
        return ntensor

    def extract_extra_tensor_from_module(
        self, name_to_tensor: T.Dict[str, NTensor]
    ):
        return name_to_tensor

    def extract_operations(self, name_to_tensor: T.Dict[str, NTensor]):
        raise NotImplementedError()

    def extract(self, name_to_tensor: T.Dict[str, NTensor]):
        self.extract_extra_tensor_from_module(name_to_tensor)
        self.extract_operations(name_to_tensor)


class ReLUExtractor(ModuleInfoExtractor):
    JIT_CLASS_NAME = "__torch__.torch.nn.modules.activation.ReLU"

    def extract_operations(self, name_to_tensor: T.Dict[str, NTensor]):
        output_tensor = self._add_idem_type_shape_output_tensor(name_to_tensor)
        NOperation(
            graph=self.g,
            type="relu",
            name=f"{self.node.export_name}_relu",
            inputs=name_to_tensor[self.node.export_inputs[1]],
            outputs=output_tensor,
        )


class QuantizeExtractor(ModuleInfoExtractor):

    JIT_CLASS_NAME = "__torch__.torch.nn.quantized.modules.Quantize"

    def extract_operations(self, name_to_tensor: T.Dict[str, NTensor]):
        output_tensor = self._add_idem_type_shape_output_tensor(
            name_to_tensor, force_dtype=np.int8
        )
        # TODO
        # Need 3 distinct operations
        # round((x/scale)+zero_point)
        NOperation(
            graph=self.g,
            type="quantize",
            name=f"{self.node.export_name}_",
            inputs=name_to_tensor[self.node.export_inputs[1]],
            outputs=output_tensor,
        )


class Conv1dExtractor(ModuleInfoExtractor):
    JIT_CLASS_NAME = "__torch__.torch.nn.modules.conv.Conv1d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nnef_weight_ref = None
        self.nnef_bias_ref = None

    def extract_extra_tensor_from_module(
        self, name_to_tensor: T.Dict[str, NTensor]
    ):
        self.nnef_weight_ref = self.add_tensor_to_ngraph(
            self.mod.weight.data, "weight", name_to_tensor
        )
        if self.mod.bias is not None:
            self.nnef_bias_ref = self.add_tensor_to_ngraph(
                self.mod.bias.data, "bias", name_to_tensor
            )

    def extract_operations(self, name_to_tensor):
        out_tensor_name = self.node.export_name
        output_tensor = NTensor(
            graph=self.g,
            name=out_tensor_name,
            dtype=self.mod.weight.data.numpy().dtype.type,
            shape=tuple(self.node.tensor_size)
            if self.node.tensor_size
            else None,
        )
        name_to_tensor[out_tensor_name] = output_tensor

        weight_var = NOperation(
            graph=self.g,
            type="variable",
            name=f"{self.node.export_name}_weight_var",
            inputs=None,
            outputs=self.nnef_weight_ref,
            attribs={
                "label": self.nnef_weight_ref.name,
                "shape": list(self.nnef_weight_ref.shape),
                "dtype": self.nnef_weight_ref.dtype,
            },
        )
        bias_var = NOperation(
            graph=self.g,
            type="variable",
            name=f"{self.node.export_name}_bias_var",
            inputs=None,
            outputs=self.nnef_bias_ref,
            attribs={
                "label": self.nnef_bias_ref.name,
                "shape": list(self.nnef_bias_ref.shape),
                "dtype": self.nnef_bias_ref.dtype,
            },
        )

        NOperation(
            graph=self.g,
            type="conv",
            name=f"{self.node.export_name}_op",
            inputs=(
                name_to_tensor[self.node.export_inputs[1]],
                weight_var.output,
                bias_var.output,
            ),
            outputs=output_tensor,
            attribs={
                "dilation": list(self.mod.dilation),
                "padding": [(self.mod.padding[0], 0)],
                "stride": list(self.mod.stride),
                "groups": self.mod.groups,
                "border": "constant",
            },
        )


class QConvReLU1dExtractor(ModuleInfoExtractor):
    JIT_CLASS_NAME = (
        "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU1d"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nnef_weight_ref = None
        self.nnef_bias_ref = None

    def extract_extra_tensor_from_module(
        self, name_to_tensors: T.Dict[str, NTensor]
    ):
        self.nnef_weight_ref = self.add_quantized_tensor_to_ngraph(
            self.mod.weight(), "weight", name_to_tensors
        )
        bias = self.mod.bias()
        if bias is not None:
            # Warning ! Pytorch does not quantize bias at training time
            # https://github.com/pytorch/pytorch/issues/28952
            # since no clear expectation, we use same scale & zero_point as
            # weight
            qbias = self._quant_from_ref(bias, self.mod.weight())
            self.nnef_bias_ref = self.add_quantized_tensor_to_ngraph(
                qbias, "bias", name_to_tensors
            )
        return name_to_tensors

    def extract_operations(self, name_to_tensor):
        out_tensor_name = self.node.export_name
        output_tensor = NTensor(
            graph=self.g,
            name=out_tensor_name,
            dtype=np.int8,
            shape=tuple(self.node.tensor_size)
            if self.node.tensor_size
            else None,
        )
        name_to_tensor[out_tensor_name] = output_tensor

        weight_var = NOperation(
            graph=self.g,
            type="variable",
            name=f"{self.node.export_name}_weight_var",
            inputs=None,
            outputs=self.nnef_weight_ref,
            attribs={
                "label": self.nnef_weight_ref.name,
                "shape": list(self.nnef_weight_ref.shape),
                "dtype": self.nnef_weight_ref.dtype,
            },
        )
        bias_var = NOperation(
            graph=self.g,
            type="variable",
            name=f"{self.node.export_name}_bias_var",
            inputs=None,
            outputs=self.nnef_bias_ref,
            attribs={
                "label": self.nnef_bias_ref.name,
                "shape": list(self.nnef_bias_ref.shape),
                "dtype": self.nnef_weight_ref.dtype,
            },
        )

        NOperation(
            graph=self.g,
            type="conv",
            name=f"{self.node.export_name}_op",
            inputs=(
                name_to_tensor[self.node.export_inputs[1]],
                weight_var.output,
                bias_var.output,
            ),
            outputs=output_tensor,
            attribs={
                "dilation": list(self.mod.dilation),
                "padding": [(self.mod.padding[0], 0)],
                "stride": list(self.mod.stride),
                "groups": self.mod.groups,
                "border": "constant",
            },
        )
        ReLUExtractor.extract_operations(self, name_to_tensor)


class GraphExtractor:
    def __init__(self, model, args):
        self.model = model
        self._torch_graph_helper = InternalPytorchGraphHelper.parse_model(
            model, args
        )
        datestr = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
        self.g = NGraph(f"net_{datestr}")

    def callmethod_extractor(self, node):
        assert node.kind == "prim::CallMethod"
        dtype_to_extractor = ModuleInfoExtractor.get_registry()

        op_called = self._torch_graph_helper.find_io_by_debug_name(
            node.inputs[0]
        )
        jit_class_name = clean_dtype_name(op_called.dtype)
        if jit_class_name in dtype_to_extractor:
            module = _access_module(op_called.module_path, self.model)
            return dtype_to_extractor[jit_class_name](node, module, self.g)
        else:
            raise NotImplementedError(
                f"hook not yet implemented for {jit_class_name}"
            )

    def _op_nodes_to_nnef_operation(self, node, name_to_tensor, null_ref):
        if node.kind == "prim::CallMethod":
            call_method_extractor = self.callmethod_extractor(node)
            call_method_extractor.extract_extra_tensor_from_module(
                name_to_tensor
            )
            call_method_extractor.extract_operations(name_to_tensor)
            return

        op_type = None
        attributes = {}
        outputs = []
        if node.kind == "aten::unbind":
            out = NTensor(
                self.g,
                node.export_name,
                dtype=_torch_to_nnef_typestr(node.subtype or node.dtype),
                shape=node.tensor_size,
            )
            import ipdb

            ipdb.set_trace()
            name_to_tensor[node.export_name] = out
            outputs = [out]
            op_type = "squeeze"
        else:
            raise NotImplementedError(
                f"NNEF Operation for {node} NOT implmented"
            )

        NOperation(
            graph=self.g,
            type=op_type,
            name=f"{node.export_name}_op",
            inputs=tuple(
                name_to_tensor[inp.export_name] if inp else null_ref
                for inp in self._torch_graph_helper.get_inputs_of_node(node)
            ),
            outputs=tuple(outputs),
            attribs=attributes,
        )

    def build_nnef_graph(
        self,
        input_names: T.List[str],
        output_names: T.List[str],
    ):
        null = NTensor(
            self.g,
            name="",
            shape=(),
            dtype=np.float32,
            data=np.zeros(shape=(), dtype=np.float32),
        )
        name_to_tensor = {}
        for node in chain(
            self._torch_graph_helper.constant_nodes,
            self._torch_graph_helper.inputs_nodes,
        ):
            if node.export_name not in name_to_tensor:
                tensor = NTensor(
                    graph=self.g,
                    name=node.export_name,
                    dtype=_torch_to_nnef_typestr(node.subtype or node.dtype),
                    shape=node.tensor_size,
                )
                name_to_tensor[node.export_name] = tensor
                if isinstance(node, NodeInput):
                    NOperation(
                        graph=self.g,
                        type="external",
                        inputs=None,
                        outputs=tensor,
                        attribs={
                            "shape": list(tensor.shape),
                            "dtype": tensor.dtype,
                        },
                    )
                if isinstance(node, NodeConstant):
                    tensor.data = node.value
        for node in self._torch_graph_helper.operators_nodes:
            # provide to tensor proper shape, dtype
            # tensor.shape, tensor.dtype, tensor.data = shape, dtype, data
            self._op_nodes_to_nnef_operation(
                node, name_to_tensor, null_ref=null
            )

        self.g.inputs = [
            name_to_tensor[_.export_name]
            for _ in self._torch_graph_helper.inputs_nodes
        ]
        assert len(input_names) == len(self.g.inputs)
        for in_tensor, requested_name in zip(self.g.inputs, input_names):
            in_tensor.name = requested_name

        self.g.outputs = [
            name_to_tensor[real_onode]
            for onode in self._torch_graph_helper.outputs_nodes
            for real_onode in onode.export_inputs
        ]
        assert len(output_names) == len(self.g.outputs)
        for out_tensor, requested_name in zip(self.g.outputs, output_names):
            out_tensor.name = requested_name

    def parse(
        self,
        input_names: T.List[str],
        output_names: T.List[str],
    ):
        self.build_nnef_graph(
            input_names,
            output_names,
        )
        return self.g
