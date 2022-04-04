import typing as T
from datetime import datetime

import numpy as np
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.op.custom_extractors import (
    CUSTOMOP_KIND,
    ModuleInfoExtractor,
)
from torch_to_nnef.op.primitive import aten_to_nnef_tensor_and_ops, external
from torch_to_nnef.op.quantized import quantized_node_to_nnef_tensor_and_ops
from torch_to_nnef.torch_graph import (
    MAP_TO_NOP,
    Data,
    TensorVariable,
    TorchModuleTraceHelper,
    _is_container,
)


class GraphExtractor:
    def __init__(self, model, args, renaming_scheme: str = "numeric"):
        self.model = model
        self._torch_graph_helper = TorchModuleTraceHelper(
            model,
            args,
            renaming_scheme=renaming_scheme,
        )
        datestr = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
        self.g = NGraph(f"net_{datestr}")
        self.activated_custom_fragment_keys: T.Set[str] = set()

    def _op_nodes_to_nnef_operation(self, node, name_to_tensor, null_ref):
        if node.kind.startswith("aten::"):
            return aten_to_nnef_tensor_and_ops(
                self.g,
                node,
                name_to_tensor,
                null_ref,
                torch_graph=self._torch_graph_helper,
            )
        if node.kind.startswith("prim::"):
            if node.kind in MAP_TO_NOP:
                return []

        if node.kind.startswith("quantized::"):
            return quantized_node_to_nnef_tensor_and_ops(
                self.g,
                node,
                name_to_tensor,
                null_ref,
                torch_graph=self._torch_graph_helper,
            )
        if node.kind.startswith(CUSTOMOP_KIND):
            return ModuleInfoExtractor.get_by_kind(node.kind).convert_to_nnef(
                self.g,
                node,
                name_to_tensor,
                null_ref,
                torch_graph=self._torch_graph_helper,
            )

        raise NotImplementedError(f"NNEF Operation for {node} NOT implmented")

    def _add_operators(self, name_to_tensor, null_ref):
        def is_missing(node: Data):
            if node.export_name in name_to_tensor:
                return False
            if _is_container(node) and any(
                is_missing(subnode) for subnode in node.data
            ):
                # case where partial data is available for container
                return True
            if not isinstance(node, TensorVariable) or node.data is not None:
                return False
            return True

        operators_nodes = self._torch_graph_helper.op_nodes[:]
        while operators_nodes:
            done_nodes = []
            for node in operators_nodes:
                # node inputs are already realised
                if any(is_missing(in_node) for in_node in node.inputs):
                    continue
                custom_fragments = self._op_nodes_to_nnef_operation(
                    node, name_to_tensor, null_ref=null_ref
                )
                if custom_fragments:
                    self.activated_custom_fragment_keys.update(custom_fragments)
                done_nodes.append(node)
            if len(done_nodes) == 0 and operators_nodes:
                self._torch_graph_helper.printall()
                print(operators_nodes)
                raise RuntimeError("DAG seems impossible to unfold")
            operators_nodes = [
                _ for _ in operators_nodes if _ not in done_nodes
            ]

    def build_nnef_graph(
        self,
        input_names: T.Optional[T.List[str]],
        output_names: T.Optional[T.List[str]],
    ):
        null = NTensor(
            self.g,
            name="",
            shape=(),
            dtype=np.float32,
            data=np.zeros(shape=(), dtype=np.float32),
        )
        name_to_tensor: T.Dict[str, NTensor] = {}
        ginputs = []
        for node in self._torch_graph_helper.inputs:
            ginputs.append(
                external(
                    self.g,
                    node,
                    name_to_tensor,
                )
            )

        self._add_operators(name_to_tensor, null_ref=null)

        self.g.inputs = ginputs
        if input_names is not None:
            assert len(input_names) == len(self.g.inputs)
            for in_tensor, requested_name in zip(self.g.inputs, input_names):
                in_tensor.name = requested_name

        self.g.outputs = [
            name_to_tensor[_.export_name]
            for _ in self._torch_graph_helper.outputs
        ]
        if output_names is not None:
            assert len(output_names) == len(self.g.outputs)
            for out_tensor, requested_name in zip(self.g.outputs, output_names):
                out_tensor.name = requested_name

    def parse(
        self,
        input_names: T.Optional[T.List[str]],
        output_names: T.Optional[T.List[str]],
    ):
        self.build_nnef_graph(
            input_names,
            output_names,
        )
        return self.g
