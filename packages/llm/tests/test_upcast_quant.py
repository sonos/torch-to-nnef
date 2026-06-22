"""Unit tests for the native-quant up-cast decision logic.

The actual dequantization (load-time ``dequantize=True`` for mxfp4/fp8, or
post-load ``model.dequantize()`` for bnb/higgs) needs real quantized weights and
GPU kernels, so only the pure planning / selection / dense-verification is
covered here.
"""

import pytest

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef_llm import loader
from torch_to_nnef_llm.loader import (
    UPCAST_ANY,
    _finish_upcast,
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
