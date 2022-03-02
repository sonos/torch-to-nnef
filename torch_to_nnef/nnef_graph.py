import logging
import typing as T
from datetime import datetime
from itertools import chain

import numpy as np
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.op import ModuleInfoExtractor
from torch_to_nnef.op.base import _torch_to_nnef_typestr

from torch_to_nnef.torch_graph import (
    InternalPytorchGraphHelper,
    NodeConstant,
    NodeInput,
    _access_module,
    clean_dtype_name,
)

LOGGER = logging.getLogger(__name__)


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
