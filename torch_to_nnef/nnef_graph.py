import logging
import typing as T
from datetime import datetime
from itertools import chain

import numpy as np
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.op import ModuleInfoExtractor, ReLUExtractor
from torch_to_nnef.op.base import _torch_to_nnef_typestr

from torch_to_nnef.torch_graph import (
    InternalPytorchGraphHelper,
    NodeConstant,
    NodeInput,
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

    def _op_nodes_to_nnef_operation(self, node, name_to_tensor, null_ref):
        self._torch_graph_helper.printall()

        op_type = None
        attributes = {}
        outputs = []

        # map aten name to basic nnef fragment
        op_map = {"unbind": "squeeze"}

        if node.kind.startswith("aten::"):
            out = NTensor(
                self.g,
                node.export_name,
                dtype=_torch_to_nnef_typestr(node.subtype or node.dtype),
                shape=node.tensor_size,
            )
            name_to_tensor[node.export_name] = out
            outputs = [out]
            aten_op_name = node.kind.split("::")[1]
            op_type = op_map[aten_op_name]  # , aten_op_name)
        elif node.kind.startswith("prim"):
            print(node)
            import ipdb

            ipdb.set_trace()
        else:
            raise NotImplementedError(
                f"NNEF Operation for {node} NOT implmented"
            )

        NOperation(
            graph=self.g,
            type=op_type,
            name=f"{node.export_name}_op",
            inputs=tuple(
                name_to_tensor[inp] if inp else null_ref
                for inp in node.export_inputs
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
