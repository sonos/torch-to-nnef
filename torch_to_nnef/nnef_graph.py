import typing as T
from datetime import datetime

import numpy as np
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import IRError, TorchToNNEFNotImplementedError
from torch_to_nnef.op.custom_extractors import (
    CUSTOMOP_KIND,
    ModuleInfoExtractor,
)
from torch_to_nnef.op.primitive import (
    aten_to_nnef_tensor_and_ops,
    primitive_ops_registry,
)
from torch_to_nnef.op.quantized import quantized_node_to_nnef_tensor_and_ops
from torch_to_nnef.torch_graph import (
    MAP_TO_NOP,
    Data,
    TensorVariable,
    TorchModuleTracer,
    module_tracer_into_ir_graph,
)
from torch_to_nnef.torch_graph.ir_graph import VariableNamingScheme
from torch_to_nnef.torch_graph.torch_const import ATEN_SIZE_KIND


class TorchToNGraphExtractor:
    """Extract Pytorch Graph and build associated nnef_tools.model.Graph"""

    def __init__(
        self,
        model,
        args,
        renaming_scheme: VariableNamingScheme = VariableNamingScheme.default(),
        check_io_names_qte_match: bool = True,
        nnef_spec_strict: bool = False,
        has_dynamic_axes: bool = False,
        tract_feature_flags: T.Optional[T.Set[str]] = None,
    ):
        self.model = model
        self._torch_ir_graph = module_tracer_into_ir_graph(
            TorchModuleTracer(
                model,
                args=args,
            ),
            renaming_scheme=renaming_scheme,
            is_root_module=True,
        )
        self._check_io_names_qte_match = check_io_names_qte_match
        self._nnef_spec_strict = nnef_spec_strict
        self._has_dynamic_axes = has_dynamic_axes
        self._tract_feature_flags = tract_feature_flags
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
                torch_graph=self._torch_ir_graph,
                nnef_spec_strict=self._nnef_spec_strict,
                has_dynamic_axes=self._has_dynamic_axes,
                tract_feature_flags=self._tract_feature_flags,
            )
        if node.kind.startswith("prim::"):
            if node.kind in MAP_TO_NOP:
                assert len(node.inputs) == 1 and len(node.outputs) == 1
                self._torch_ir_graph.remap_node(node.outputs[0], node.inputs[0])
                return []

        if node.kind.startswith("quantized::"):
            return quantized_node_to_nnef_tensor_and_ops(
                self.g,
                node,
                name_to_tensor,
                null_ref,
                torch_graph=self._torch_ir_graph,
                nnef_spec_strict=self._nnef_spec_strict,
                tract_feature_flags=self._tract_feature_flags,
            )
        if node.kind.startswith(CUSTOMOP_KIND):
            return ModuleInfoExtractor.get_by_kind(node.kind).convert_to_nnef(
                self.g,
                node,
                name_to_tensor,
                null_ref,
                torch_graph=self._torch_ir_graph,
                nnef_spec_strict=self._nnef_spec_strict,
                tract_feature_flags=self._tract_feature_flags,
            )

        raise TorchToNNEFNotImplementedError(
            f"NNEF Operation for {node} NOT implmented"
        )

    def _if_dyn_shape_may_remove_resolved_dim(self, operators_nodes):
        # NOTE: cleanup all resolved output of torch tensor.size(axis) if
        # is dynamic shape to avoid number hard translated

        def forward_clean_values_for_dyn_axes(op_node):
            """forward clean in all child nodes"""
            assert len(op_node.outputs) == 1
            op_node.outputs[0].data = None
            data_node_to_clean = op_node.outputs[0]
            for (
                user_op_node
            ) in self._torch_ir_graph.find_ops_nodes_by_input_node(
                data_node_to_clean
            ):
                assert len(user_op_node.outputs) == 1
                if user_op_node.outputs[0].data is not None:
                    forward_clean_values_for_dyn_axes(user_op_node)

        if self._has_dynamic_axes and not self._nnef_spec_strict:
            for op_node in operators_nodes:
                if op_node.kind == ATEN_SIZE_KIND:
                    forward_clean_values_for_dyn_axes(op_node)

    def _add_operators(self, name_to_tensor, null_ref):
        def is_missing(node: Data):
            if node.export_name in name_to_tensor:
                return False
            if node.is_container and any(
                is_missing(subnode) for subnode in node.data
            ):
                # case where partial data is available for container
                return True
            if not isinstance(node, TensorVariable) or node.data is not None:
                return False
            return True

        operators_nodes = self._torch_ir_graph.op_nodes[:]
        self._if_dyn_shape_may_remove_resolved_dim(operators_nodes)
        while operators_nodes:
            done_nodes = []
            for op_node in operators_nodes:
                # op_node inputs are already realised
                if any(is_missing(_) for _ in op_node.inputs):
                    continue
                custom_fragments = self._op_nodes_to_nnef_operation(
                    op_node, name_to_tensor, null_ref=null_ref
                )
                if custom_fragments:
                    self.activated_custom_fragment_keys.update(custom_fragments)
                done_nodes.append(op_node)
            if len(done_nodes) == 0 and operators_nodes:
                self._torch_ir_graph.printall()
                print(
                    "unable to realise operators with outputs",
                    [out.name for op in operators_nodes for out in op.outputs],
                )
                raise IRError("DAG seems impossible to unfold")
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
        for node in self._torch_ir_graph.inputs:
            op, custom_fragments = primitive_ops_registry.get("external")(
                self.g,
                node,
                name_to_tensor,
                nnef_spec_strict=self._nnef_spec_strict,
            )
            ginputs.append(op)
            if custom_fragments:
                self.activated_custom_fragment_keys.update(custom_fragments)

        self._add_operators(name_to_tensor, null_ref=null)

        self.g.inputs = ginputs
        if input_names is not None:
            if self._check_io_names_qte_match:
                assert len(input_names) == len(
                    self.g.inputs
                ), f"{len(input_names)} == {len(self.g.inputs)}"
            for in_tensor, requested_name in zip(self.g.inputs, input_names):
                in_tensor.name = requested_name

        self.g.outputs = [
            name_to_tensor[_.export_name] for _ in self._torch_ir_graph.outputs
        ]
        if output_names is not None:
            if self._check_io_names_qte_match:
                assert len(output_names) == len(
                    self.g.outputs
                ), f"{len(output_names)} == {len(self.g.outputs)}"
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
