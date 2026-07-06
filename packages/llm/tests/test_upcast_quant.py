"""Unit tests for the native-quant up-cast decision logic.

The actual dequantization (load-time ``dequantize=True`` for mxfp4/fp8, or
post-load ``model.dequantize()`` for bnb/higgs) needs real quantized weights and
GPU kernels, so only the pure planning / selection / dense-verification is
covered here.
"""

import json

import pytest
import torch
from torch import nn

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef.tensor.offload import (
    ON_DISK_DEVICE_MAP_KEY,
    t2n_load_checkpoint_and_dispatch,
)
from torch_to_nnef.utils import init_empty_weights
from torch_to_nnef_llm import loader
from torch_to_nnef_llm.loader import (
    UPCAST_ANY,
    _finish_upcast,
    _load_time_weight_converters,
    _native_quant_method,
    _normalize_upcast_request,
    _plan_and_inject_upcast,
    _quant_method_of,
    assert_upcast_dense,
    plan_upcast,
    should_upcast,
)


class _LoadTimeConfig:
    """A quant config with a load-time ``dequantize`` flag.

    Like Mxfp4Config / FineGrainedFP8Config.
    """

    def __init__(self, method):
        self.quant_method = method
        self.dequantize = False


class _PostLoadConfig:
    """A quant config with no ``dequantize`` flag (bnb/higgs style)."""

    __slots__ = ("quant_method",)

    def __init__(self, method):
        self.quant_method = method


class _Cfg:
    def __init__(self, quant_config=None):
        if quant_config is not None:
            self.quantization_config = quant_config


class _Model:
    def __init__(self, quant_config=None, hf_quantizer=None):
        self.config = _Cfg(quant_config)
        if hf_quantizer is not None:
            self.hf_quantizer = hf_quantizer


def test_quant_method_normalization():
    assert _quant_method_of(_LoadTimeConfig("mxfp4")) == "mxfp4"

    class _E:
        value = "FbgemmFp8"

    assert _quant_method_of(_PostLoadConfig(_E())) == "fbgemmfp8"
    assert _quant_method_of(None) is None
    assert _native_quant_method(_Model()) is None
    assert _native_quant_method(_Model(_LoadTimeConfig("mxfp4"))) == "mxfp4"


def test_normalize_upcast_request():
    # nothing requested -> None
    assert _normalize_upcast_request(None) is None
    assert _normalize_upcast_request([]) is None
    # valid methods + the "any" sentinel pass through, lowercased. Use methods
    # present across the whole supported transformers range (>= 4.35), so this
    # holds on every cli_transformers_* env (mxfp4/fp8 are 5.x-only).
    assert _normalize_upcast_request(["BitsAndBytes", "gptq"]) == [
        "bitsandbytes",
        "gptq",
    ]
    assert _normalize_upcast_request(["any"]) == [UPCAST_ANY]
    # a bare string is treated as one method, not iterated per-character
    assert _normalize_upcast_request("gptq") == ["gptq"]
    # a typo fails up-front, with the valid list in the message
    with pytest.raises(T2NErrorMisuse, match="unknown upcast_quant"):
        _normalize_upcast_request(["definitely-not-a-real-method"])


def test_normalize_upcast_request_requires_transformers_4_38(monkeypatch):
    import transformers

    # too-old transformers (no quantizer dequant API) fails up-front, clearly
    monkeypatch.setattr(transformers, "__version__", "4.37.0")
    with pytest.raises(T2NErrorMisuse, match="requires transformers >= 4.38"):
        _normalize_upcast_request(["mxfp4"])


def test_should_upcast_matrix():
    assert should_upcast("mxfp4", ["mxfp4"]) is True
    assert should_upcast("mxfp4", ["fp8", "mxfp4"]) is True
    assert should_upcast("mxfp4", [UPCAST_ANY]) is True
    assert should_upcast("mxfp4", ["fp8"]) is False
    assert should_upcast("mxfp4", None) is False
    assert should_upcast(None, ["any"]) is False


def test_plan_not_quantized_or_not_requested():
    assert plan_upcast(None, ["any"]) == ("none", None)
    assert plan_upcast(_LoadTimeConfig("mxfp4"), None) == ("none", None)
    assert plan_upcast(_LoadTimeConfig("mxfp4"), []) == ("none", None)


def test_plan_load_time_format_sets_flag():
    qc = _LoadTimeConfig("mxfp4")
    kind, out = plan_upcast(qc, ["mxfp4"])
    assert kind == "load"
    assert out is qc and qc.dequantize is True  # flag set for from_pretrained


def test_plan_post_load_format():
    kind, method = plan_upcast(_PostLoadConfig("bitsandbytes"), ["any"])
    assert kind == "post" and method == "bitsandbytes"


def test_plan_quantized_but_other_method_requested_errors():
    with pytest.raises(T2NErrorMisuse, match="mxfp4"):
        plan_upcast(_LoadTimeConfig("mxfp4"), ["fp8"])


def test_assert_dense_passes_when_dense():
    assert_upcast_dense(_Model(), ["any"])  # no quant config, no quantizer
    # not requested → skip
    assert_upcast_dense(_Model(_LoadTimeConfig("mxfp4")), None)


def test_assert_dense_raises_on_residual_quant():
    # still reports a native method after up-cast → partial dequant
    with pytest.raises(T2NErrorMisuse, match="did not fully"):
        assert_upcast_dense(_Model(_LoadTimeConfig("mxfp4")), ["any"])
    # a lingering hf_quantizer even if config was cleared: message reads
    # "active quantizer", not "native 'None'"
    with pytest.raises(T2NErrorMisuse, match="active quantizer"):
        assert_upcast_dense(_Model(hf_quantizer=object()), ["any"])


# --- integration helpers: _plan_and_inject_upcast / _finish_upcast ---


class _PostCapableModel:
    """A post-load-dequantizable model (bnb/higgs style)."""

    def __init__(self):
        self.config = _Cfg(_PostLoadConfig("bitsandbytes"))

    def dequantize(self):
        return _Model()  # dense twin (no quantization_config)


class _PostUnsupportedModel:
    """A model whose quantizer has no post-load dequantize (gptq/awq/...)."""

    def __init__(self):
        self.config = _Cfg(_PostLoadConfig("gptq"))

    def dequantize(self):
        raise NotImplementedError("gptq has no implementation of dequantize")


def test_plan_and_inject_load_time(monkeypatch):
    qc = _LoadTimeConfig("mxfp4")
    monkeypatch.setattr(loader, "_peek_quant_config", lambda *a, **k: qc)
    kwargs = {}
    plan = _plan_and_inject_upcast("org/m", ["mxfp4"], kwargs, True, object())
    assert plan == ("load", qc)
    assert kwargs["quantization_config"] is qc and qc.dequantize is True


def test_plan_and_inject_post_and_none(monkeypatch):
    monkeypatch.setattr(
        loader,
        "_peek_quant_config",
        lambda *a, **k: _PostLoadConfig("bitsandbytes"),
    )
    kwargs = {}
    assert _plan_and_inject_upcast("m", ["any"], kwargs, True, object()) == (
        "post",
        "bitsandbytes",
    )
    assert "quantization_config" not in kwargs  # post path injects nothing
    # no request / no source short-circuits without peeking
    assert _plan_and_inject_upcast(None, ["mxfp4"], {}, True, object()) == (
        "none",
        None,
    )
    assert _plan_and_inject_upcast("m", None, {}, True, object()) == (
        "none",
        None,
    )


def test_finish_upcast_post_success():
    out = _finish_upcast(_PostCapableModel(), ("post", "bitsandbytes"), ["any"])
    assert _native_quant_method(out) is None


def test_finish_upcast_post_unsupported_raises_clean_error():
    # the B1 fix: NotImplementedError becomes a clear T2NErrorMisuse
    with pytest.raises(T2NErrorMisuse, match="cannot be dequantized by"):
        _finish_upcast(_PostUnsupportedModel(), ("post", "gptq"), ["any"])


class _FakeAutoConfig:
    def __init__(self, quantization_config):
        self.quantization_config = quantization_config


class _FakeTransformers:
    """Minimal stand-in exposing only what `_peek_quant_config` touches."""

    def __init__(self, quantization_config):
        self._qc = quantization_config

    class _AutoConfig:
        pass

    @property
    def AutoConfig(self):  # noqa: N802 (mirrors transformers.AutoConfig)
        qc = self._qc
        ns = type("AutoConfig", (), {})

        def from_pretrained(*_a, **_k):
            return _FakeAutoConfig(qc)

        ns.from_pretrained = staticmethod(from_pretrained)
        return ns


def test_peek_quant_config_resilient_to_unknown_method():
    from torch_to_nnef_llm.loader import _peek_quant_config

    # a quant_method this transformers version doesn't know makes
    # AutoQuantizationConfig.from_dict raise; the peek must swallow it -> None
    tf = _FakeTransformers({"quant_method": "not-a-real-quant-method"})
    assert _peek_quant_config("some/slug", True, tf) is None


def test_finish_upcast_warns_when_quantized_but_not_requested(caplog):
    import logging

    model = _Model(_LoadTimeConfig("mxfp4"))
    with caplog.at_level(logging.WARNING):
        out = _finish_upcast(model, ("none", None), None)
    assert out is model  # left as-is (opt-in)
    assert any("upcast_quant was not" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("fake_quantizer", "message"),
    [
        (object(), "exposes no weight conversions"),
        (
            type(
                "EmptyConverterQuantizer",
                (),
                {"get_weight_conversions": lambda self: []},
            )(),
            "returned no transformers weight conversions",
        ),
    ],
)
def test_load_time_weight_converters_fail_loudly(
    monkeypatch, fake_quantizer, message
):
    import transformers.quantizers.auto as auto

    monkeypatch.setattr(
        auto.AutoHfQuantizer,
        "from_config",
        staticmethod(lambda *_args, **_kwargs: fake_quantizer),
    )

    with pytest.raises(T2NErrorMisuse, match=message):
        _load_time_weight_converters(("load", _LoadTimeConfig("mxfp4")))


class _ToyWeightConverter:
    source_patterns = ["foo_blocks", "foo_scales"]
    target_patterns = ["foo"]

    def __init__(self):
        self.collected_tensors = {
            "foo_blocks": [],
            "foo_scales": [],
        }

    def rename_source_key(self, source_key):
        for source_pattern in self.source_patterns:
            if source_pattern in source_key:
                return (
                    source_key.replace(source_pattern, "foo"),
                    source_pattern,
                )
        return source_key, None

    def add_tensor(self, _target_key, _source_key, source_pattern, tensor):
        self.collected_tensors[source_pattern].append(tensor)

    def convert(self, layer_name, **_kwargs):
        blocks = self.collected_tensors["foo_blocks"][0]
        scales = self.collected_tensors["foo_scales"][0]
        return {layer_name: blocks + scales}


class _OrderedWeightConverter:
    source_patterns = ["foo_part"]
    target_patterns = ["foo"]

    def __init__(self):
        self.collected_tensors = {"foo_part": []}

    def rename_source_key(self, source_key):
        if source_key.startswith("foo_part."):
            return "foo", "foo_part"
        return source_key, None

    def add_tensor(self, _target_key, _source_key, source_pattern, tensor):
        self.collected_tensors[source_pattern].append(tensor)

    def convert(self, layer_name, **_kwargs):
        return {layer_name: torch.stack(self.collected_tensors["foo_part"])}


def test_t2n_dispatch_rejects_weight_converters_without_device_map(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = nn.Parameter(torch.empty(2, 2))

    with init_empty_weights():
        model = TinyModel()

    with pytest.raises(T2NErrorMisuse, match="require a device_map"):
        t2n_load_checkpoint_and_dispatch(
            model,
            tmp_path,
            device_map=None,
            offload_dir=tmp_path,
            weight_converters=[_ToyWeightConverter()],
        )


def test_t2n_dispatch_applies_weight_converter_across_shards(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = nn.Parameter(torch.empty(2, 2))
            self.bias = nn.Parameter(torch.empty(2))

    with init_empty_weights():
        model = TinyModel()

    shard_1 = tmp_path / "model-00001-of-00002.safetensors"
    shard_2 = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"foo_blocks": torch.ones(2, 2)}, shard_1)
    save_file(
        {
            "foo_scales": torch.full((2, 2), 2.0),
            "bias": torch.zeros(2),
        },
        shard_2,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "foo_blocks": shard_1.name,
                    "foo_scales": shard_2.name,
                    "bias": shard_2.name,
                },
            }
        ),
        encoding="utf-8",
    )

    offload_dir = tmp_path / "offload"
    offload_dir.mkdir()
    t2n_load_checkpoint_and_dispatch(
        model,
        tmp_path,
        device_map=ON_DISK_DEVICE_MAP_KEY,
        offload_dir=offload_dir,
        weight_converters=[_ToyWeightConverter()],
    )

    assert not model.foo.is_meta
    assert torch.equal(model.foo.reload(), torch.full((2, 2), 3.0))
    assert not model.bias.is_meta


def test_t2n_dispatch_preserves_global_converter_key_order(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = nn.Parameter(torch.empty(2, 1))

    with init_empty_weights():
        model = TinyModel()

    shard_1 = tmp_path / "model-00001-of-00002.safetensors"
    shard_2 = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"foo_part.10": torch.tensor([10.0])}, shard_1)
    save_file({"foo_part.2": torch.tensor([2.0])}, shard_2)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "foo_part.10": shard_1.name,
                    "foo_part.2": shard_2.name,
                },
            }
        ),
        encoding="utf-8",
    )

    offload_dir = tmp_path / "offload"
    offload_dir.mkdir()
    t2n_load_checkpoint_and_dispatch(
        model,
        tmp_path,
        device_map=ON_DISK_DEVICE_MAP_KEY,
        offload_dir=offload_dir,
        weight_converters=[_OrderedWeightConverter()],
    )

    assert torch.equal(model.foo.reload(), torch.tensor([[2.0], [10.0]]))


def _mxfp4_blocks(shape, packed_nibbles=0x21):
    return torch.full(shape, packed_nibbles, dtype=torch.uint8)


def _mxfp4_scales(shape, exponent=0):
    return torch.full(shape, 127 + exponent, dtype=torch.uint8)


def test_t2n_dispatch_applies_mxfp4_weight_converter(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    pytest.importorskip("transformers.integrations.mxfp4")
    gpt_oss_config = pytest.importorskip(
        "transformers.models.gpt_oss.configuration_gpt_oss"
    )
    gpt_oss_modeling = pytest.importorskip(
        "transformers.models.gpt_oss.modeling_gpt_oss"
    )
    quant_config_mod = pytest.importorskip(
        "transformers.utils.quantization_config"
    )

    class OneExpertLayer(nn.Module):
        def __init__(self):
            super().__init__()
            config = gpt_oss_config.GptOssConfig(
                hidden_size=32,
                intermediate_size=32,
                num_local_experts=2,
            )
            self.experts = gpt_oss_modeling.GptOssExperts(config)

    with init_empty_weights():
        model = OneExpertLayer()

    save_file(
        {
            "experts.gate_up_proj_blocks": _mxfp4_blocks((2, 64, 1, 16)),
            "experts.gate_up_proj_scales": _mxfp4_scales((2, 64, 1)),
            "experts.down_proj_blocks": _mxfp4_blocks((2, 32, 1, 16)),
            "experts.down_proj_scales": _mxfp4_scales((2, 32, 1)),
            "experts.gate_up_proj_bias": torch.zeros(2, 64),
            "experts.down_proj_bias": torch.zeros(2, 32),
        },
        tmp_path / "model.safetensors",
    )
    quant_config = quant_config_mod.Mxfp4Config(dequantize=True)
    weight_converters, hf_quantizer = _load_time_weight_converters(
        ("load", quant_config)
    )

    offload_dir = tmp_path / "offload"
    offload_dir.mkdir()
    t2n_load_checkpoint_and_dispatch(
        model,
        tmp_path,
        device_map=ON_DISK_DEVICE_MAP_KEY,
        offload_dir=offload_dir,
        weight_converters=weight_converters,
        hf_quantizer=hf_quantizer,
    )

    gate_up_proj = model.experts.gate_up_proj.reload()
    down_proj = model.experts.down_proj.reload()
    assert gate_up_proj.shape == (2, 32, 64)
    assert down_proj.shape == (2, 32, 32)
    assert torch.isfinite(gate_up_proj).all()
    assert torch.isfinite(down_proj).all()
    assert set(gate_up_proj.to(torch.float32).unique().tolist()) == {0.5, 1.0}
    assert set(down_proj.to(torch.float32).unique().tolist()) == {0.5, 1.0}
