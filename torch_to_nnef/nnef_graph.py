"""Core parsing and NNEF transformation module."""

import logging
import typing as T

import numpy as np
import torch
from nnef_tools.model import Graph as NGraph
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import (
    T2NError,
    T2NErrorIoQuantity,
    T2NErrorIR,
    T2NErrorNotImplemented,
    T2NErrorTorchNotFoundOp,
)
from torch_to_nnef.inference_target import InferenceTarget, TractNNEF
from torch_to_nnef.op.aten import aten_ops_registry, aten_to_nnef_tensor_and_ops
from torch_to_nnef.op.custom_extractors import (
    CUSTOMOP_KIND,
    ModuleInfoExtractor,
)
from torch_to_nnef.op.quantized import quantized_node_to_nnef_tensor_and_ops
from torch_to_nnef.torch_graph import (
    MAP_TO_NOP,
    Data,
    TensorVariable,
    TorchModuleTracer,
    module_tracer_into_ir_graph,
)
from torch_to_nnef.torch_graph.ir_naming import (
    DEFAULT_VARNAME_SCHEME,
    VariableNamingScheme,
)
from torch_to_nnef.torch_graph.ir_op import (
    CacheDataNodeTarget,
    CacheDataToOpsNode,
)
from torch_to_nnef.torch_graph.torch_const import ATEN_SIZE_KIND

LOGGER = logging.getLogger(__name__)


class TorchToNGraphExtractor:
    """Extract PyTorch Graph and build associated nnef_tools.model.Graph."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: T.Tuple[torch.Tensor, ...],
        inference_target: InferenceTarget,
        nnef_variable_naming_scheme: VariableNamingScheme = DEFAULT_VARNAME_SCHEME,  # noqa: E501
        forced_inputs_names: T.Optional[T.List[str]] = None,
        forced_outputs_names: T.Optional[T.List[str]] = None,
        check_io_names_qte_match: bool = True,
    ):
        self.model = model
        LOGGER.info("start to translate PyTorch to internal IR")
        self._torch_ir_graph = module_tracer_into_ir_graph(
            TorchModuleTracer(
                model,
                args=args,
            ),
            forced_inputs_names=forced_inputs_names,
            forced_outputs_names=forced_outputs_names,
            nnef_variable_naming_scheme=nnef_variable_naming_scheme,
            is_root_module=True,
        )
        LOGGER.info("translated PyTorch to internal IR")
        self._forced_inputs_names = forced_inputs_names
        self._forced_outputs_names = forced_outputs_names
        self._check_io_names_qte_match = check_io_names_qte_match
        self._inference_target = inference_target
        self.g = NGraph("network")
        self.activated_custom_fragment_keys: T.Set[str] = set()

    def _op_nodes_to_nnef_operation(self, node, name_to_tensor, null_ref):
        if node.kind.startswith("aten::"):
            return aten_to_nnef_tensor_and_ops(
                self.g,
                node,
                name_to_tensor,
                null_ref,
                torch_graph=self._torch_ir_graph,
                inference_target=self._inference_target,
            )
        if node.kind.startswith("prim::") and node.kind in MAP_TO_NOP:
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
                inference_target=self._inference_target,
            )
        if node.kind.startswith(CUSTOMOP_KIND):
            return ModuleInfoExtractor.get_by_kind(node.kind).convert_to_nnef(
                self.g,
                node,
                name_to_tensor=name_to_tensor,
                null_ref=null_ref,
                torch_graph=self._torch_ir_graph,
                inference_target=self._inference_target,
            )

        raise T2NErrorNotImplemented(
            f"NNEF Operation for {node} NOT implmented"
        )

    def _if_dyn_shape_may_remove_resolved_dim(self, operators_nodes):
        # NOTE: cleanup all resolved output of torch tensor.size(axis) if
        # is dynamic shape to avoid number hard translated

        LOGGER.debug(
            "start to build input cache for forward_clean_values_for_dyn_axes"
        )
        input_data_to_ops_node = CacheDataToOpsNode(
            target=CacheDataNodeTarget.INPUTS,
            ops=self._torch_ir_graph.op_nodes,
        )
        LOGGER.debug("done input cache for forward_clean_values_for_dyn_axes")
        explored_user_op_nodes = set()

        def forward_clean_values_for_dyn_axes(op_node):
            """Forward clean in all child nodes."""
            # LOGGER.debug(f"remove concrete values from {op_node.outputs}")
            for onode in op_node.outputs:
                onode.set_data(None)
            for data_node_to_clean in op_node.outputs:
                try:
                    for user_op_node in input_data_to_ops_node.get(
                        data_node_to_clean
                    ):
                        if user_op_node in explored_user_op_nodes:
                            # LOGGER.debug(f"already explored: {user_op_node}")
                            continue
                        explored_user_op_nodes.add(user_op_node)
                        forward_clean_values_for_dyn_axes(user_op_node)
                except T2NErrorTorchNotFoundOp as exp:
                    if data_node_to_clean not in self._torch_ir_graph.outputs:
                        raise exp

        if (
            isinstance(self._inference_target, TractNNEF)
            and self._inference_target.dynamic_axes
        ):
            for op_node in operators_nodes:
                if op_node.kind == ATEN_SIZE_KIND:
                    forward_clean_values_for_dyn_axes(op_node)
        LOGGER.debug("done all forward_clean_values_for_dyn_axes")

    def _add_operators(self, name_to_tensor, null_ref):
        def is_missing(node: Data):
            if node.export_name in name_to_tensor:
                return False
            if node.is_container and any(
                is_missing(subnode) for subnode in node.data
            ):
                return True

            if not isinstance(node, TensorVariable) or node.data is not None:  # noqa: SIM103
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
                raise T2NErrorIR("DAG seems impossible to unfold")
            operators_nodes = [
                _ for _ in operators_nodes if _ not in done_nodes
            ]

    def build_nnef_graph(self):
        LOGGER.info("start to translate internal IR to NNEF Graph object")
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
            op, custom_fragments = aten_ops_registry.get("external")(
                self.g,
                node,
                name_to_tensor,
                inference_target=self._inference_target,
            )
            ginputs.append(op)
            if custom_fragments:
                self.activated_custom_fragment_keys.update(custom_fragments)

        self._add_operators(name_to_tensor, null_ref=null)

        self.g.inputs = ginputs
        if self._forced_inputs_names is not None:
            assert len(self._forced_inputs_names) > 0
            if self._check_io_names_qte_match and len(
                self._forced_inputs_names
            ) != len(self.g.inputs):
                raise T2NErrorIoQuantity(
                    "miss-aligned quantity of `input_names`: "
                    f"{len(self._forced_inputs_names)}"
                    " and quantity of inputs in NNEF graph: "
                    f"{len(self.g.inputs)}\n"
                    f"\t- with input_names: {self._forced_inputs_names}\n"
                    f"\t- with graph inputs: {self.g.inputs}"
                )
            # still needed since some .remap_node in ._add_operators may araise
            for inode, new_name in zip(
                self.g.inputs, self._forced_inputs_names
            ):
                inode.name = new_name

        self.g.outputs = [
            name_to_tensor[_.export_name] for _ in self._torch_ir_graph.outputs
        ]
        if self._forced_outputs_names is not None:
            # only allow at least 1 output
            assert len(self._forced_outputs_names) > 0
            if self._check_io_names_qte_match and len(
                self._forced_outputs_names
            ) != len(self.g.outputs):
                raise T2NErrorIoQuantity(
                    "miss-aligned quantity of `output_names`: "
                    f"{len(self._forced_outputs_names)}"
                    " and quantity of outputs in NNEF graph: "
                    f"{len(self.g.outputs)}\n"
                    f"\t- with output_names: {self._forced_inputs_names}\n"
                    f"\t- with graph outputs: {self.g.outputs}"
                )
            # still needed since some .remap_node in ._add_operators may araise
            for onode, new_name in zip(
                self.g.outputs, self._forced_outputs_names
            ):
                if onode.name == new_name:
                    continue
                if onode.name in self._forced_inputs_names:
                    raise T2NError(
                        f"input tensor named: '{onode.name}' tryied to "
                        f"be replaced by output named: '{new_name}'."
                        "This is forbidden as it leads to nop for this tensor"
                    )
                onode.name = new_name
        LOGGER.info("translated internal IR to NNEF Graph object sucessfully")

    def parse(self) -> NGraph:
        self.build_nnef_graph()
        return self.g
