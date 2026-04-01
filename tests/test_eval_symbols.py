import tempfile
from pathlib import Path

import torch
import yaml

from torch_to_nnef.nemo_tract.axis_registry import load_axis_symbol_registry
from torch_to_nnef.remodeler.dyn_axes import (
    apply_eval_symbols,
    remove_eval_symbols_from_dyn,
)


def test_parse_eval_symbols_from_yaml():
    cfg = {
        "decoder": {
            "inputs": {
                "targets": {
                    "original_shape": ["TARGETS__BATCH", "TARGETS__TIME"],
                    "collapse_dims": [],
                    "eval_symbols": {"TARGETS__TIME": 1},
                },
            },
        },
    }
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    ) as f:
        yaml.dump(cfg, f)
        f.flush()
        reg = load_axis_symbol_registry(Path(f.name))

    assert "decoder.targets" in reg.eval_symbols_per_input
    assert reg.eval_symbols_per_input["decoder.targets"] == {"TARGETS__TIME": 1}


def test_parse_eval_symbols_uppercased():
    cfg = {
        "enc": {
            "inputs": {
                "x": {
                    "original_shape": ["B", "T"],
                    "eval_symbols": {"t": 5},
                },
            },
        },
    }
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    ) as f:
        yaml.dump(cfg, f)
        f.flush()
        reg = load_axis_symbol_registry(Path(f.name))

    assert reg.eval_symbols_per_input["enc.x"] == {"T": 5}


def test_apply_eval_symbols_shrink():
    t = torch.randn(3, 42)
    dyn = {"targets": {0: "BATCH", 1: "TIME"}}
    result = apply_eval_symbols(
        test_input=[t],
        input_names=["targets"],
        subnet_name="decoder",
        dyn=dyn,
        eval_symbols={"decoder.targets": {"TIME": 1}},
    )
    assert result[0].shape == (3, 1)
    assert torch.equal(result[0], t[:, :1])
    # _apply_eval_symbols does NOT mutate dyn (that happens later)
    assert dyn == {"targets": {0: "BATCH", 1: "TIME"}}


def test_apply_eval_symbols_expand():
    t = torch.randn(2, 3)
    result = apply_eval_symbols(
        test_input=[t],
        input_names=["x"],
        subnet_name="enc",
        dyn={"x": {0: "B", 1: "T"}},
        eval_symbols={"enc.x": {"T": 7}},
    )
    assert result[0].shape == (2, 7)
    assert torch.equal(result[0][:, :3], t)
    assert (result[0][:, 3:] == 0).all()


def test_apply_eval_symbols_no_match():
    t = torch.randn(3, 10)
    result = apply_eval_symbols(
        test_input=[t],
        input_names=["x"],
        subnet_name="enc",
        dyn={"x": {0: "B", 1: "T"}},
        eval_symbols={"enc.x": {"OTHER": 1}},
    )
    assert result[0].shape == (3, 10)


def test_apply_eval_symbols_noop_same_size():
    t = torch.randn(3, 10)
    result = apply_eval_symbols(
        test_input=[t],
        input_names=["x"],
        subnet_name="enc",
        dyn={"x": {0: "B", 1: "T"}},
        eval_symbols={"enc.x": {"T": 10}},
    )
    assert result[0] is t


def test_apply_eval_symbols_multiple_inputs():
    t1 = torch.randn(3, 42)
    t2 = torch.randn(3, 20)
    result = apply_eval_symbols(
        test_input=[t1, t2],
        input_names=["a", "b"],
        subnet_name="sub",
        dyn={"a": {1: "T"}, "b": {1: "S"}},
        eval_symbols={
            "sub.a": {"T": 1},
            "sub.b": {"S": 5},
        },
    )
    assert result[0].shape == (3, 1)
    assert result[1].shape == (3, 5)


def testremove_eval_symbols_from_dyn():
    dyn = {"targets": {0: "BATCH", 1: "TIME"}, "x": {0: "B"}}
    remove_eval_symbols_from_dyn(
        input_names=["targets", "x"],
        subnet_name="decoder",
        dyn=dyn,
        eval_symbols={"decoder.targets": {"TIME": 1}},
    )
    assert dyn == {"targets": {0: "BATCH"}, "x": {0: "B"}}


def test_remove_eval_symbols_from_dyn_no_match():
    dyn = {"x": {0: "B", 1: "T"}}
    remove_eval_symbols_from_dyn(
        input_names=["x"],
        subnet_name="enc",
        dyn=dyn,
        eval_symbols={"enc.x": {"OTHER": 1}},
    )
    assert dyn == {"x": {0: "B", 1: "T"}}
