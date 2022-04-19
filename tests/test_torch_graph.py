"""Tests internals torch graph intermediate representation."""
import torch
from torch import nn

from torch_to_nnef.torch_graph import (
    PythonConstant,
    TensorVariable,
    TorchModuleIRGraph,
    TorchOp,
)


def test_filter_out_useless_nodes():
    mdl = nn.Sequential(nn.Linear(10, 20), nn.GRU(20, 5, batch_first=True))
    args = (torch.rand(1, 100, 10),)
    tth = TorchModuleIRGraph(mdl, args)

    bob_in = PythonConstant(name="bob_in", data=100)
    tth.data_nodes.append(bob_in)
    bob_out = TensorVariable(
        name="bob_out",
        data=None,
        shape=[10, 1],
        dtype=torch.float32,
        quant=None,
    )
    tth.data_nodes.append(bob_out)
    tth.op_nodes.append(
        TorchOp(
            kind="aten::bob",
            module_path="",
            # we link to up node
            inputs=[tth.data_nodes[0], bob_in],
            outputs=[bob_out],
            scope="",
            op_ref=None,
            call_name="bob",
        )
    )
    tth._filter_nodes_not_in_trace_between_inputs_and_outputs()
    assert not any(
        dnode.name in ["bob_in", "bob_out"] for dnode in tth.data_nodes
    )
    assert not any(op_node.kind == "aten::bob" for op_node in tth.op_nodes)
