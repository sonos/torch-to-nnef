import tempfile
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from torch_to_nnef.exceptions import T2NErrorInvalidArgument
from torch_to_nnef.remodeler import save_config
from torch_to_nnef_nemo.axis_registry import (
    AxisSymbolRegistry,
    load_axis_symbol_registry,
)
from torch_to_nnef_nemo.derived_constraints import (
    derive_encoder_time_bound,
    derive_extensions,
)
from torch_to_nnef_nemo.model_loader import (
    NEMOTRON_0_6B,
    PARAKEET_V3_SLUG,
)
from torch_to_nnef_nemo.slug_extensions import (
    SLUG_EXTENSIONS,
    EncoderFingerprint,
    fingerprint_from_asr_model,
    get_extensions_for_slug,
    load_slug_fingerprints,
    resolve_slug_from_asr_model,
)


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
            raise AssertionError("should have raised")
        except T2NErrorInvalidArgument as e:
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


# -- fingerprint resolver ----------------------------------------------------


def _patched_model(
    cfg,
    model_class="EncDecRNNTBPEModel",
    encoder_class="ConformerEncoder",
):
    """Build a minimal stand-in for an ``ASRModel`` instance.

    The resolver reads ``type(asr_model).__name__`` and
    ``type(asr_model.encoder).__name__``, so we dynamically create
    classes with the target names rather than importing NeMo.
    """
    enc_cls = type(encoder_class, (), {})
    model_cls = type(model_class, (), {})
    m = model_cls()
    m.cfg = cfg
    m.encoder = enc_cls()
    return m


def _parakeet_v3_cfg():
    return OmegaConf.create(
        {
            "model_defaults": {"num_tdt_durations": 5},
            "encoder": {
                "feat_in": 128,
                "d_model": 1024,
                "n_layers": 24,
                "subsampling": "dw_striding",
                "subsampling_factor": 8,
                "conv_kernel_size": 9,
                "self_attention_model": "rel_pos",
                "att_context_size": [-1, -1],
                "pos_emb_max_len": 5000,
            },
        }
    )


def _nemotron_0_6b_cfg():
    return OmegaConf.create(
        {
            "model_defaults": {},
            "encoder": {
                "feat_in": 128,
                "d_model": 1024,
                "n_layers": 24,
                "subsampling": "dw_striding",
                "subsampling_factor": 8,
                "conv_kernel_size": 9,
                "self_attention_model": "rel_pos",
                "pos_emb_max_len": 5000,
                "att_context_size": [[70, 13], [70, 6], [70, 1], [70, 0]],
            },
        }
    )


def test_fingerprints_cover_all_slugs():
    """Maintainer guardrail: every SLUG_EXTENSIONS entry has a fingerprint.

    If this fails, run
    ``python -m torch_to_nnef_nemo.tools.refresh_slug_fingerprints``.
    """
    missing = set(SLUG_EXTENSIONS) - set(load_slug_fingerprints())
    assert not missing, f"missing fingerprints for slugs: {missing}"


def test_fingerprint_from_dict_roundtrip():
    for fp in load_slug_fingerprints().values():
        assert EncoderFingerprint.from_dict(fp.to_dict()) == fp


def test_resolve_parakeet_v3_from_fake_cfg():
    m = _patched_model(_parakeet_v3_cfg())
    assert resolve_slug_from_asr_model(m) == "nvidia/parakeet-tdt-0.6b-v3"


def test_resolve_nemotron_from_fake_cfg():
    m = _patched_model(_nemotron_0_6b_cfg())
    assert (
        resolve_slug_from_asr_model(m)
        == "nvidia/nemotron-speech-streaming-en-0.6b"
    )


def test_fingerprint_distinguishes_streaming_from_offline():
    """Flipping att_context_size alone should break the v3 match."""
    cfg = _parakeet_v3_cfg()
    cfg.encoder.att_context_size = [[70, 13], [70, 6], [70, 1], [70, 0]]
    m = _patched_model(cfg)
    # streaming context + num_tdt_durations=5 is not a registered combo
    assert resolve_slug_from_asr_model(m) is None


def test_fingerprint_miss_on_unknown_arch():
    cfg = _parakeet_v3_cfg()
    cfg.encoder.d_model = 768  # arbitrary change
    m = _patched_model(cfg)
    assert resolve_slug_from_asr_model(m) is None


def test_fingerprint_from_asr_model_shape():
    """Smoke: builder returns a fingerprint with the expected shape.

    Independent of the committed JSON.
    """
    m = _patched_model(_parakeet_v3_cfg())
    fp = fingerprint_from_asr_model(m)
    assert fp.model_class == "EncDecRNNTBPEModel"
    assert fp.encoder_class == "ConformerEncoder"
    assert fp.att_context_size == ((-1, -1),)
    assert fp.num_tdt_durations == 5


# -- architecture-derived constraints ---------------------------------------


def test_derived_parakeet_v3_matches_hardcoded_registry():
    """Parity: deriver output equals the committed SLUG_EXTENSIONS entry."""
    m = _patched_model(_parakeet_v3_cfg())
    assert derive_extensions(m) == get_extensions_for_slug(PARAKEET_V3_SLUG)


def test_derived_nemotron_matches_hardcoded_registry():
    """Parity: deriver output equals the committed SLUG_EXTENSIONS entry."""
    m = _patched_model(_nemotron_0_6b_cfg())
    assert derive_extensions(m) == get_extensions_for_slug(NEMOTRON_0_6B)


def test_derived_bound_value_is_39993():
    """Pin the exact numeric bound for the pos_emb_max_len=5000 + factor=8 case.

    Guards against regressions if the preimage formula is touched.
    """
    cfg = _parakeet_v3_cfg()
    assert derive_encoder_time_bound(cfg.encoder) == 39993


def test_derived_scales_with_pos_emb_and_factor():
    cfg = _parakeet_v3_cfg()
    cfg.encoder.pos_emb_max_len = 8000
    cfg.encoder.subsampling_factor = 4
    # (8000 - 1) * 4 + 1 = 31997
    assert derive_encoder_time_bound(cfg.encoder) == 31997


def test_derived_empty_for_unknown_subsampling():
    cfg = _parakeet_v3_cfg()
    cfg.encoder.subsampling = "stacking"
    m = _patched_model(cfg)
    assert derive_extensions(m) == {}


def test_derived_empty_for_unknown_attention():
    cfg = _parakeet_v3_cfg()
    cfg.encoder.self_attention_model = "rotary"
    m = _patched_model(cfg)
    assert derive_extensions(m) == {}


def test_derived_empty_for_model_without_encoder():
    """Models that aren't encoder/decoder ASR should bail gracefully."""

    class _NoEncoderModel:
        pass

    m = _NoEncoderModel()
    m.cfg = OmegaConf.create({"foo": "bar"})
    assert derive_extensions(m) == {}


def test_fingerprint_none_for_model_without_encoder():
    class _NoEncoderModel:
        pass

    m = _NoEncoderModel()
    m.cfg = OmegaConf.create({"foo": "bar"})
    assert fingerprint_from_asr_model(m) is None
    assert resolve_slug_from_asr_model(m) is None
