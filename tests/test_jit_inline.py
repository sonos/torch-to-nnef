"""Tests for `torch_to_nnef.torch_graph.inline_unresolvable_submodules`.

Covers:
- Importable submodules are preserved (no inlining when not needed).
- Non-importable submodules are inlined; output stays bitwise-equal.
- The pass is idempotent on a fully-resolved graph.
- The Silero-VAD JIT integration test (gated on silero_vad availability)
  confirms only the `vad.*` calls are inlined while every `torch.nn.*`
  call is preserved as a `prim::CallMethod` for downstream extractors.
"""

import importlib
from collections import Counter

import pytest
import torch
from torch import nn

from torch_to_nnef.torch_graph import inline_unresolvable_submodules
from torch_to_nnef.torch_graph.torch_const import CALL_KIND


def _walk_nodes(graph_or_block):
    for node in graph_or_block.nodes():
        yield node
        for blk in node.blocks():
            yield from _walk_nodes(blk)


def _callmethod_counts(graph) -> Counter:
    counts = Counter()
    for n in _walk_nodes(graph):
        if n.kind() != CALL_KIND:
            continue
        ty = next(n.inputs()).type()
        try:
            counts[ty.qualified_name()] += 1
        except (RuntimeError, AttributeError):
            counts["<unknown>"] += 1
    return counts


class _PureTorchSubmod(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.lin(x))


class _PureTorchOuter(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _PureTorchSubmod()

    def forward(self, x):
        return self.inner(x) + 1.0


def test_inline_is_noop_when_all_classes_resolvable():
    """No-op path: every CallMethod target lives in an importable module.

    The graph should be untouched; count stays the same.
    """
    m = torch.jit.script(_PureTorchOuter().eval())
    before = _callmethod_counts(m.graph)
    assert before, "test setup expected at least one CallMethod"

    inlined = inline_unresolvable_submodules(m.graph, m)
    after = _callmethod_counts(m.graph)

    assert inlined == 0
    assert before == after


def test_inline_drops_non_importable_targets_and_preserves_torchnn(
    monkeypatch,
):
    """Inline path under a forced ModuleNotFoundError on the inner class.

    Patches `importlib.import_module` so the test-local outer's qualname
    raises. Afterwards every surviving CallMethod must target an
    importable `torch.nn.*` class, and outputs must match.
    """
    m = torch.jit.script(_PureTorchOuter().eval())

    # Find the qualname host module for our test-local class and patch
    # importlib.import_module to raise on that exact path. Real torch.nn.*
    # imports must still succeed.
    inner_call = next(n for n in _walk_nodes(m.graph) if n.kind() == CALL_KIND)
    inner_qname = next(inner_call.inputs()).type().qualified_name()
    parts = [
        p
        for p in inner_qname[len("__torch__.") :].split(".")
        if "___torch_mangle_" not in p
    ]
    inner_mod_path = ".".join(parts[:-1])

    real_import = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == inner_mod_path:
            raise ModuleNotFoundError(f"forced for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "torch_to_nnef.torch_graph.jit_inline.importlib.import_module",
        fake_import_module,
    )

    ref_x = torch.randn(2, 4)
    with torch.no_grad():
        ref_y = m(ref_x)
    inlined = inline_unresolvable_submodules(m.graph, m)
    torch._C._jit_pass_dce(m.graph)
    with torch.no_grad():
        new_y = m(ref_x)

    assert inlined >= 1
    after = _callmethod_counts(m.graph)
    for qn in after:
        assert qn.startswith("__torch__.torch.nn."), (
            f"non-torch.nn.* CallMethod survived inline: {qn}"
        )
    assert torch.allclose(ref_y, new_y)


def test_inline_is_idempotent():
    m = torch.jit.script(_PureTorchOuter().eval())
    n1 = inline_unresolvable_submodules(m.graph, m)
    n2 = inline_unresolvable_submodules(m.graph, m)
    assert (n1, n2) == (0, 0)


@pytest.mark.skipif(
    importlib.util.find_spec("silero_vad") is None,
    reason="silero_vad not installed",
)
def test_inline_silero_vad_jit_real_world():
    """End-to-end on the upstream silero-vad JIT.

    Only `vad.*` modules should be inlined; every `torch.nn.*` call must
    survive; outputs must be bitwise-identical to the unmodified JIT.
    """
    from silero_vad import load_silero_vad

    full = load_silero_vad()
    inner = full._model.eval()
    x = torch.randn(1, 576) * 0.1
    state = torch.zeros(2, 1, 128)
    with torch.no_grad():
        ref_out, ref_state = inner(x, state)

    inlined = inline_unresolvable_submodules(inner.graph, inner)
    torch._C._jit_pass_dce(inner.graph)
    after = _callmethod_counts(inner.graph)

    assert inlined > 0
    for qn in after:
        assert qn.startswith("__torch__.torch.nn."), (
            f"non-torch.nn.* CallMethod target survived: {qn}"
        )

    with torch.no_grad():
        new_out, new_state = inner(x, state)
    assert torch.allclose(ref_out, new_out)
    assert torch.allclose(ref_state, new_state)
