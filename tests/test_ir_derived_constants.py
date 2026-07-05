import torch
import torch.nn.functional as F

from torch_to_nnef.tensor.opaque import set_opaque_tensor_in_params_as_ref
from torch_to_nnef.tensor.quant import fp_to_tract_q4_0_with_min_max_calibration
from torch_to_nnef.torch_graph.ir_graph import module_tracer_into_ir_graph
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.torch_const import ATEN_LINEAR, ATEN_SELECT


class PackedExpertLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 4, 5))

    def forward(self, x):
        chunks = x.split([1, 1, 1], dim=0)
        outputs = []
        for expert in range(3):
            outputs.append(F.linear(chunks[expert], self.weight[expert]))
        return torch.cat(outputs, dim=0)


def test_parameter_select_outputs_are_not_materialized_as_constants():
    mod = PackedExpertLinear().eval()
    x = torch.randn(3, 5)

    graph = module_tracer_into_ir_graph(TorchModuleTracer(mod, args=(x,)))

    select_ops = [
        op
        for op in graph.op_nodes
        if op.kind == ATEN_SELECT and op.inputs[0].data is mod.weight
    ]
    assert len(select_ops) == 3
    assert all(op.outputs[0].data is None for op in select_ops)

    linear_weight_nodes = [
        op.inputs[1] for op in graph.op_nodes if op.kind == ATEN_LINEAR
    ]
    assert len(linear_weight_nodes) == 3
    assert all(weight.data is None for weight in linear_weight_nodes)

    packed_weight_nodes = [
        node
        for node in graph.data_nodes
        if getattr(node, "data", None) is mod.weight
    ]
    assert len(packed_weight_nodes) == 1
    assert packed_weight_nodes[0].module_attr


class PackedExpertQuantizedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randn(3, 4, 32)
        q_weight = fp_to_tract_q4_0_with_min_max_calibration(weight)
        self.weight = torch.nn.Parameter(q_weight, requires_grad=False)

    def forward(self, x):
        chunks = x.split([1, 1, 1], dim=0)
        outputs = []
        for expert in range(3):
            weight = self.weight[expert].view(4, 32)
            outputs.append(F.linear(chunks[expert], weight))
        return torch.cat(outputs, dim=0)


def test_opaque_parameter_select_outputs_are_not_materialized_as_constants():
    mod = PackedExpertQuantizedLinear().eval()
    set_opaque_tensor_in_params_as_ref(mod)
    x = torch.randn(3, 32)

    graph = module_tracer_into_ir_graph(TorchModuleTracer(mod, args=(x,)))

    select_ops = [
        op
        for op in graph.op_nodes
        if op.kind == ATEN_SELECT and op.inputs[0].data is mod.weight
    ]
    assert len(select_ops) == 3
    assert all(op.outputs[0].data is None for op in select_ops)

    linear_weight_nodes = [
        op.inputs[1] for op in graph.op_nodes if op.kind == ATEN_LINEAR
    ]
    assert len(linear_weight_nodes) == 3
    assert all(weight.data is None for weight in linear_weight_nodes)
