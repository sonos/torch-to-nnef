"""`op.custom_extractors` provides mechanism to control extraction to NNEF.

while bypassing PyTorch full expansion of `torch.Module` within `torch_graph`
which by default use torch.jit.trace .

This may be for two main reasons:
    - Some layer such as LSTM/GRU have complex expension which are better
      handled by encapsulation instead of spreading high number of variable
    - Some layer might not be serializable to .jit
    - There might be some edge case where you prefer to keep full control on
      exported NNEF subgraph.

"""

from torch_to_nnef.op.custom_extractors.base import (
    CUSTOMOP_KIND,
    ModuleInfoExtractor,
)

# load default custom registries
from torch_to_nnef.op.custom_extractors.gdn import (  # noqa: F401
    CausalConvUpdateReified,
    GatedDeltaNetRecurrentReified,
)
from torch_to_nnef.op.custom_extractors.moe import (  # noqa: F401
    MoEFFN,
    mark_moe_experts_for_q40,
)
from torch_to_nnef.op.custom_extractors.rnn import (
    GRUExtractor,
    LSTMCellExtractor,
    LSTMExtractor,
    RNNExtractor,
)

__all__ = [
    "CUSTOMOP_KIND",
    "ModuleInfoExtractor",
    "RNNExtractor",
    "LSTMExtractor",
    "LSTMCellExtractor",
    "GRUExtractor",
    "MoEFFN",
    "mark_moe_experts_for_q40",
    "GatedDeltaNetRecurrentReified",
    "CausalConvUpdateReified",
]
