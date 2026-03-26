from torch_to_nnef.nemo_tract.axis_registry import AxisSymbolRegistry
from torch_to_nnef.nemo_tract.registry_utils import (
    validate_registry_against_signatures,
)
from torch_to_nnef.remodeler import IODescriptor, Stage, SubnetSignature


def _sig(name: str, inputs, outputs, axes=None):
    return SubnetSignature(
        name=name,
        stage=Stage.RAW,
        inputs=[IODescriptor(n, s, None, []) for n, s in inputs],
        outputs=[IODescriptor(n, [], None, []) for n in outputs],
        symbol_axes=axes or {},
    )


def test_validate_unknown_input_raises():
    sigs = [_sig("enc", [("x", ["B", 10])], ["y"], axes={"x": {0: "B"}})]
    reg = AxisSymbolRegistry(
        symbols_per_input={"enc.unknown": {0: "B"}},
        rank_per_input={"enc.unknown": 1},
        bind_to_dim={},
        input_collapse_dims={},
        renamed_symbols_per_subnet={},
        outputs_keep_per_subnet={},
    )
    try:
        validate_registry_against_signatures(sigs, reg)
    except ValueError as e:
        assert "unknown inputs" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown inputs")


def test_validate_outputs_keep_subset():
    sigs = [_sig("s", [("a", ["B"])], ["o1", "o2"], axes={"a": {0: "B"}})]
    reg = AxisSymbolRegistry(
        symbols_per_input={"s.a": {0: "B"}},
        rank_per_input={"s.a": 1},
        bind_to_dim={},
        input_collapse_dims={},
        renamed_symbols_per_subnet={},
        outputs_keep_per_subnet={"s": ["o1", "bad"]},
    )
    try:
        validate_registry_against_signatures(sigs, reg)
    except ValueError as e:
        assert "outputs_keep" in str(e)
    else:
        raise AssertionError("expected ValueError for outputs_keep")
