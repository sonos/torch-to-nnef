"""
torch_graph is intended to extract full representation of pytorch Graph
into a stable intermediate representation suitable to then apply translation
operation to NNEF. This means that not all Pytorch orginal graph is translated.
By example, we ignore part linked to device location informations,
memory specific operation or parameters linked to gradients.

This choice which is different compared to torch.onnx module due to the
absence of control (on our side) over evolution of Pytorch internals.
If some of the Pytorch internals are modified only this module should idealy
be impacted.

Here there is NO notion of dynamic axes all shapes are supposedly defined
based on provided input example.
At latter stage in other modules the dynamic shapes need to be introduced if requested by user.

"""

from torch_to_nnef.torch_graph.ir_data import (
    Data,
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)
from torch_to_nnef.torch_graph.ir_graph import (
    TorchModuleIRGraph,
    module_tracer_into_ir_graph,
)
from torch_to_nnef.torch_graph.ir_helpers import _is_container
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.ir_op import TorchOp
from torch_to_nnef.torch_graph.torch_const import MAP_TO_NOP

__all__ = [
    "Data",
    "FixedTensorList",
    "PythonConstant",
    "TensorVariable",
    "_is_container",
    "TorchModuleTracer",
    "TorchOp",
    "MAP_TO_NOP",
    "module_tracer_into_ir_graph",
    "TorchModuleIRGraph",
]
