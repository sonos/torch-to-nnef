import types

import torch

from torch_to_nnef.remodeler import prepare_subnet_export
from torch_to_nnef.remodeler.adapter import BoundaryAdapter


class _Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_names = ["a"]
        self.output_names = ["o"]

    def forward(self, x):  # pragma: no cover - not used in this test
        return x


def test_dynamic_axes_symbol_renames_applied():
    m = _Dummy().eval()
    # External example with rank 2, dynamic axes declared on input 'a'
    ex = [torch.zeros(1, 2, dtype=torch.float32)]
    dyn = {"a": {0: "B", 1: "U"}}
    # Rename sources: B -> BATCH
    rename = {"BATCH": ["B"]}
    ba = BoundaryAdapter(
        m,
        "s",
        ex,
        dyn,
        collapse_by_input={},
        binds_by_input={},
        renamed_map=rename,
        outputs_keep=[],
    )
    out_dyn = ba.dynamic_shapes_for_export()
    assert out_dyn["a"][0] == "BATCH"
    assert out_dyn["a"][1] == "U"


# ---------------------------------------------------------------------------
# Output collapse_dims tests
# ---------------------------------------------------------------------------


class _BatchedOutput(torch.nn.Module):
    """Returns a [1, C, T] tensor (batch=1 at axis 0)."""

    def __init__(self):
        super().__init__()
        self.input_names = ["x"]
        self.output_names = ["out"]

    def forward(self, x):
        return x.unsqueeze(0)  # add batch dim


def test_output_collapse_dims_squeezes_batch():
    m = _BatchedOutput().eval()
    ex = [torch.zeros(2, 4, dtype=torch.float32)]
    ba = BoundaryAdapter(
        m,
        "s",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=[],
        output_collapse_dims={"out": [0]},
    )
    result = ba(torch.zeros(2, 4))
    assert result.shape == (2, 4), f"Expected (2, 4), got {result.shape}"


def test_output_collapse_dims_no_squeeze_when_not_one():
    """If the axis is not size 1, squeeze must not remove it."""

    class _NoCollapse(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_names = ["x"]
            self.output_names = ["out"]

        def forward(self, x):
            return x  # shape stays [3, 4], axis 0 is not 1

    m = _NoCollapse().eval()
    ex = [torch.zeros(3, 4)]
    ba = BoundaryAdapter(
        m,
        "s",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=[],
        output_collapse_dims={"out": [0]},
    )
    result = ba(torch.zeros(3, 4))
    assert result.shape == (3, 4), f"Expected (3, 4), got {result.shape}"


class _TwoOutputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_names = ["x"]
        self.output_names = ["feat", "length"]

    def forward(self, x):
        return x.unsqueeze(0), torch.tensor([x.shape[1]]).unsqueeze(0)


def test_output_collapse_dims_multiple_outputs():
    m = _TwoOutputs().eval()
    ex = [torch.zeros(2, 4)]
    ba = BoundaryAdapter(
        m,
        "s",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=[],
        output_collapse_dims={"feat": [0], "length": [0]},
    )
    feat, length = ba(torch.zeros(2, 4))
    assert feat.shape == (2, 4), f"feat: expected (2, 4), got {feat.shape}"
    assert length.shape == (1,), f"length: expected (1,), got {length.shape}"


def test_output_collapse_dims_selective():
    """Only squeeze outputs that are configured, leave others untouched."""
    m = _TwoOutputs().eval()
    ex = [torch.zeros(2, 4)]
    ba = BoundaryAdapter(
        m,
        "s",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=[],
        output_collapse_dims={"feat": [0]},  # only feat, not length
    )
    feat, length = ba(torch.zeros(2, 4))
    assert feat.shape == (2, 4)
    assert length.shape == (1, 1), (
        f"length should keep batch: got {length.shape}"
    )


# ---------------------------------------------------------------------------
# outputs_keep validation & filtering tests
# ---------------------------------------------------------------------------


def test_outputs_keep_unknown_name_silently_skipped():
    """outputs_keep with a name not in flat outputs just skips it.

    Validation is done upstream by validate_registry_against_signatures,
    not by BoundaryAdapter itself (to avoid a forward pass at init).
    """
    m = _TwoOutputs().eval()
    ex = [torch.zeros(2, 4)]
    ba = BoundaryAdapter(
        m,
        "dec",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=["feat", "bad_name"],
    )
    # bad_name is simply not matched: only feat is returned
    result = ba(torch.zeros(2, 4))
    assert isinstance(result, tuple)
    assert len(result) == 1


def test_outputs_keep_filters_output_names():
    """output_names property must reflect outputs_keep filtering."""
    m = _TwoOutputs().eval()
    ex = [torch.zeros(2, 4)]
    ba = BoundaryAdapter(
        m,
        "s",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=["feat"],
    )
    assert ba.output_names == ["feat"]


def test_outputs_keep_filters_forward():
    """forward() must return only kept outputs."""
    m = _TwoOutputs().eval()
    ex = [torch.zeros(2, 4)]
    ba = BoundaryAdapter(
        m,
        "s",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=["feat"],
        output_collapse_dims={"feat": [0]},
    )
    result = ba(torch.zeros(2, 4))
    # Should return a single-element tuple (only feat, batch-squeezed)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == (2, 4)


# ---------------------------------------------------------------------------
# Container output flattening via outputs_keep
# ---------------------------------------------------------------------------


class _TupleOutput(torch.nn.Module):
    """Model returning (tensor, (tensor, tensor))."""

    def __init__(self):
        super().__init__()
        self.input_names = ["x"]
        self.output_names = ["logits", "states"]

    def forward(self, x):
        return x + 1, (x + 2, x + 3)


def test_outputs_keep_flattens_container_outputs():
    """outputs_keep should reference flattened names for container outputs."""
    m = _TupleOutput().eval()
    ex = [torch.zeros(2, 4)]
    ba = BoundaryAdapter(
        m,
        "dec",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=["logits", "states_0"],
    )
    assert ba.output_names == ["logits", "states_0"]
    result = ba(torch.zeros(2, 4))
    assert isinstance(result, tuple)
    assert len(result) == 2
    # logits = x + 1, states_0 = x + 2
    assert torch.allclose(result[0], torch.ones(2, 4))
    assert torch.allclose(result[1], torch.full((2, 4), 2.0))


def test_outputs_keep_flattens_all_container_elements():
    """Keeping all flat names returns all flat tensors."""
    m = _TupleOutput().eval()
    ex = [torch.zeros(2, 4)]
    ba = BoundaryAdapter(
        m,
        "dec",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=["logits", "states_0", "states_1"],
    )
    assert ba.output_names == ["logits", "states_0", "states_1"]
    result = ba(torch.zeros(2, 4))
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_outputs_keep_raw_container_name_returns_nothing():
    """Using raw container name 'states' (not flat) matches no output."""
    m = _TupleOutput().eval()
    ex = [torch.zeros(2, 4)]
    ba = BoundaryAdapter(
        m,
        "dec",
        ex,
        {},
        collapse_by_input={},
        outputs_keep=["logits", "states"],
    )
    # 'states' is not a flat name: only 'logits' matches
    result = ba(torch.zeros(2, 4))
    assert isinstance(result, tuple)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Extension symbol-rename tests
# ---------------------------------------------------------------------------


def _fake_registry(*, renamed, extensions):
    """Minimal stand-in exposing the attributes prepare_subnet_export reads."""
    return types.SimpleNamespace(
        eval_symbols_per_input={},
        renamed_symbols_per_subnet=renamed,
        outputs_keep_per_subnet={},
        input_collapse_dims={},
        bind_to_dim={},
        output_collapse_dims={},
        extensions_per_subnet=extensions,
    )


def test_user_extensions_get_symbol_renames_applied():
    """User-provided extensions must honor ``renamed_symbols`` (regression).

    A shape config that renames AUDIO_SIGNAL__TIME -> S while also declaring
    ``extensions: [tract_assert AUDIO_SIGNAL__TIME<=39993]`` used to emit the
    assertion with the un-renamed symbol, referencing a symbol tract never
    declares.
    """
    reg = _fake_registry(
        renamed={"encoder": {"S": ["AUDIO_SIGNAL__TIME"]}},
        extensions={"encoder": ["tract_assert AUDIO_SIGNAL__TIME<=39993"]},
    )
    prepared = prepare_subnet_export(
        model=torch.nn.Identity(),
        test_input=[torch.zeros(1, 128, 16)],
        input_names=["audio_signal"],
        output_names=["outputs"],
        subnet_name="encoder",
        dyn={"audio_signal": {2: "AUDIO_SIGNAL__TIME"}},
        custom_extensions=["tract_assert AUDIO_SIGNAL__TIME >= 1"],
        axis_registry=reg,
    )
    exts = prepared.custom_extensions
    assert "tract_assert S<=39993" in exts
    assert "tract_assert S >= 1" in exts
    assert not any("AUDIO_SIGNAL__TIME" in e for e in exts)
