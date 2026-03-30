import pytest
import torch

from torch_to_nnef.exceptions import T2NErrorInvalidArgument
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


def test_outputs_keep_unknown_name_raises():
    """outputs_keep with a name not in output_names must raise."""
    m = _TwoOutputs().eval()
    ex = [torch.zeros(2, 4)]
    with pytest.raises(T2NErrorInvalidArgument, match="unknown output"):
        BoundaryAdapter(
            m,
            "dec",
            ex,
            {},
            collapse_by_input={},
            outputs_keep=["feat", "bad_name"],
        )


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
