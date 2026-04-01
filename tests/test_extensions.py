import tempfile
from pathlib import Path

import yaml

from torch_to_nnef.nemo_tract.axis_registry import (
    AxisSymbolRegistry,
    load_axis_symbol_registry,
)
from torch_to_nnef.nemo_tract.slug_extensions import (
    SLUG_EXTENSIONS,
    get_extensions_for_slug,
)
from torch_to_nnef.remodeler import save_config


def test_parse_extensions_from_yaml():
    cfg = {
        "encoder": {
            "extensions": ["tract_assert AUDIO_SIGNAL__TIME<=39993"],
            "inputs": {
                "audio_signal": {
                    "original_shape": ["BATCH", 128, "TIME"],
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

    assert reg.extensions_per_subnet == {
        "encoder": ["tract_assert AUDIO_SIGNAL__TIME<=39993"],
    }


def test_parse_extensions_invalid_type():
    cfg = {
        "encoder": {
            "extensions": "not a list",
            "inputs": {
                "x": {"original_shape": ["B"]},
            },
        },
    }
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    ) as f:
        yaml.dump(cfg, f)
        f.flush()
        try:
            load_axis_symbol_registry(Path(f.name))
            assert False, "should have raised"
        except Exception as e:
            assert "extensions" in str(e).lower()


def test_extensions_absent_gives_empty():
    cfg = {
        "encoder": {
            "inputs": {
                "x": {"original_shape": ["B", "T"]},
            },
        },
    }
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    ) as f:
        yaml.dump(cfg, f)
        f.flush()
        reg = load_axis_symbol_registry(Path(f.name))

    assert reg.extensions_per_subnet == {}


def test_roundtrip_extensions_via_save_config():
    reg = AxisSymbolRegistry(
        symbols_per_input={"enc.x": {0: "B", 1: "T"}},
        rank_per_input={"enc.x": 2},
        bind_to_dim={},
        input_collapse_dims={},
        renamed_symbols_per_subnet={},
        outputs_keep_per_subnet={},
        original_shape_per_input={"enc.x": ["B", "T"]},
        extensions_per_subnet={
            "enc": ["tract_assert T<=100", "tract_assert T>=1"],
        },
    )
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    ) as f:
        save_config(Path(f.name), reg)
        loaded = load_axis_symbol_registry(Path(f.name))

    assert loaded.extensions_per_subnet == {
        "enc": ["tract_assert T<=100", "tract_assert T>=1"],
    }


def test_slug_extensions_nemotron():
    exts = get_extensions_for_slug("nvidia/nemotron-speech-streaming-en-0.6b")
    assert "encoder" in exts
    assert any("39993" in e for e in exts["encoder"])


def test_slug_extensions_unknown():
    assert get_extensions_for_slug("unknown/model") == {}


def test_slug_registry_is_not_empty():
    assert len(SLUG_EXTENSIONS) > 0
