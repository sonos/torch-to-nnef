"""Tests for `harden_jit_for_export`.

The helper bundles the full JIT-only export chain. Per-pass tests live
alongside each pass; this file verifies that the wrapper produces a
graph that's free of the constructs the chain is meant to remove and
that bitwise output equivalence is preserved.
"""

import importlib

import pytest
import torch
from torch import nn

from torch_to_nnef.torch_graph import harden_jit_for_export
from torch_to_nnef.torch_graph.torch_const import GETATTR_KIND, IF_KIND
from torch_to_nnef.utils import torch_version

skipif_torch_lt_20 = pytest.mark.skipif(
    condition=torch_version() < "2.0.0",
    reason=(
        "torch < 2.0's JIT shape analysis + interpret_graph interplay "
        "leaves the helper's data-dep If fold as a no-op"
    ),
)


def _walk(g):
    for n in g.nodes():
        yield n
        for blk in n.blocks():
            yield from _walk(blk)


def _node_count(g, kind):
    return sum(1 for n in _walk(g) if n.kind() == kind)


class _DimDependentBranch(nn.Module):
    """LSTMCell-style dim check that survives the JIT shape passes."""

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x + 1.0


@skipif_torch_lt_20
def test_helper_clears_data_dependent_ifs():
    raw = torch.jit.script(_DimDependentBranch())
    x = torch.randn(2, 3)
    hardened = harden_jit_for_export(raw, (x,))
    assert _node_count(hardened.graph, IF_KIND) == 0


@skipif_torch_lt_20
def test_helper_preserves_output():
    ref = _DimDependentBranch().eval()
    x = torch.randn(2, 3)
    expected = ref(x)

    raw = torch.jit.script(_DimDependentBranch())
    hardened = harden_jit_for_export(raw, (x,))

    got = hardened(x)
    assert torch.allclose(got, expected)


class _Plain(nn.Module):
    def forward(self, x):
        return x + 1.0


def test_helper_is_safe_on_clean_graphs():
    raw = torch.jit.script(_Plain())
    x = torch.randn(3)
    hardened = harden_jit_for_export(raw, (x,))
    assert torch.allclose(hardened(x), raw(x))


class _WithLinear(nn.Module):
    """Has parameters, so a freeze actually replaces GetAttrs with constants."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


def test_helper_freeze_off_keeps_getattrs():
    """`freeze=False` must leave the GetAttr nodes that resolve module attrs."""
    x = torch.randn(2, 4)

    raw = torch.jit.script(_WithLinear().eval())
    n_getattr_pre = _node_count(raw.graph, GETATTR_KIND)
    assert n_getattr_pre > 0  # baseline: scripted module has GetAttrs

    no_freeze = harden_jit_for_export(
        torch.jit.script(_WithLinear().eval()), (x,), freeze=False
    )
    assert _node_count(no_freeze.graph, GETATTR_KIND) == n_getattr_pre

    frozen = harden_jit_for_export(
        torch.jit.script(_WithLinear().eval()), (x,), freeze=True
    )
    assert _node_count(frozen.graph, GETATTR_KIND) == 0


def test_helper_diagnostics_populated():
    """Passing a dict surfaces per-pass fold counts and the freeze flag."""
    raw = torch.jit.script(_DimDependentBranch().eval())
    x = torch.randn(2, 3)

    diag = {}
    harden_jit_for_export(raw, (x,), diagnostics=diag)

    expected_keys = {
        "froze",
        "replace_size_calls_with_constants",
        "fold_constant_scalar_arithmetic",
        "fold_constant_ifs",
        "fold_tuple_index_through_tuple_construct",
        "strip_prim_data",
        "strip_assertion_ifs",
        "fold_data_dependent_ifs",
    }
    assert expected_keys.issubset(diag.keys())
    # Fixture is in eval mode so freeze must succeed.
    assert diag["froze"] is True
    # Every count is a non-negative int. (Exact per-pass counts depend
    # on torch's freeze pre-folding and JIT type cache state, so we
    # don't assert specific numbers; a non-int value would catch a
    # regression where the helper writes the wrong type.)
    for key in expected_keys - {"froze"}:
        assert isinstance(diag[key], int)
        assert diag[key] >= 0


class _NestedSubmod(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _Plain()

    def forward(self, x):
        return self.inner(x) * 2


def test_helper_runs_inline_path_when_freeze_off(monkeypatch):
    """Cover the `inline_unresolvable_submodules` branch in the helper.

    `freeze=False` + an unimportable submodule qualname forces the inline
    pass to fire. We monkeypatch `importlib.import_module` to raise
    `ModuleNotFoundError` for `_NestedSubmod`'s class qualname, which is
    the same condition that surfaces on JIT artifacts whose Python source
    is not on the import path.
    """
    raw = torch.jit.script(_NestedSubmod().eval())
    x = torch.randn(3, 4)

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        # Mimic the silero-VAD condition: any vendored module path is
        # unreachable. For the test we target the inner test class.
        if "test_jit_harden" in name:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    diag = {}
    hardened = harden_jit_for_export(raw, (x,), freeze=False, diagnostics=diag)

    # Beyond the key existing, assert the inline pass actually folded
    # something: the monkeypatched import forces `_NestedSubmod`'s
    # `self.inner(x)` CallMethod to be inlined into the parent graph.
    assert diag["inline_unresolvable_submodules"] > 0
    assert torch.allclose(hardened(x), raw(x))


class _AutoHardenScripted(nn.Module):
    """Module with a dim-check that the parser would refuse without harden."""

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x + 1.0


@skipif_torch_lt_20
def test_export_auto_hardens_scripted_module(tmp_path, caplog):
    """export_model_to_nnef applies harden_jit_for_export on a ScriptModule.

    Verified by the log line and by the export succeeding (without auto-harden,
    the data-dependent If would survive and downstream parsing would fail).
    """
    # Local import to keep the rest of the file independent of TractNNEF.
    from torch_to_nnef import TractNNEF, export_model_to_nnef

    raw = torch.jit.script(_AutoHardenScripted().eval())
    x = torch.randn(2, 3)
    out = tmp_path / "m.nnef.tgz"

    import logging

    with caplog.at_level(logging.INFO, logger="torch_to_nnef.export"):
        export_model_to_nnef(
            model=raw,
            args=(x,),
            file_path_export=out,
            inference_target=TractNNEF(
                version=TractNNEF.latest_version(), check_io=False
            ),
        )

    assert out.exists()
    assert any(
        "auto-applying harden_jit_for_export" in r.message
        for r in caplog.records
    )


@skipif_torch_lt_20
def test_export_auto_harden_opt_out(tmp_path, caplog):
    """auto_harden_jit=False suppresses the helper invocation."""
    from torch_to_nnef import (
        TractNNEF,
        export_model_to_nnef,
        harden_jit_for_export,
    )

    raw = torch.jit.script(_AutoHardenScripted().eval())
    x = torch.randn(2, 3)
    pre_hardened = harden_jit_for_export(raw, (x,))
    out = tmp_path / "m.nnef.tgz"

    import logging

    with caplog.at_level(logging.INFO, logger="torch_to_nnef.export"):
        export_model_to_nnef(
            model=pre_hardened,
            args=(x,),
            file_path_export=out,
            inference_target=TractNNEF(
                version=TractNNEF.latest_version(), check_io=False
            ),
            auto_harden_jit=False,
        )

    assert out.exists()
    assert not any(
        "auto-applying harden_jit_for_export" in r.message
        for r in caplog.records
    )
