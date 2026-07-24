"""Guards for aten-op aliases that map to a shared handler.

Some PyTorch ops are aliases (``clip`` == ``clamp``). Whether a given
torch/transformers version normalizes them at trace time varies, so these
assert the registry mapping directly (a real export once emitted ``aten::clip``
from a ``.clip(max=...)`` in the Qwen vision reformulation).
"""

from torch_to_nnef.op.aten.activation import OP_REGISTRY


def test_clip_is_registered_as_clamp_alias():
    assert OP_REGISTRY.get("clip") is OP_REGISTRY.get("clamp")
