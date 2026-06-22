"""Unit tests for the native-quant up-cast decision logic.

The actual dequantization (load-time ``dequantize=True`` for mxfp4/fp8, or
post-load ``model.dequantize()`` for bnb/higgs) needs real quantized weights and
GPU kernels, so only the pure planning / selection / dense-verification is
covered here.
"""

import pytest

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef_llm.loader import (
    UPCAST_ANY,
    _native_quant_method,
    _normalize_upcast_request,
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
    # valid methods + the "any" sentinel pass through, lowercased
    assert _normalize_upcast_request(["MXFP4", "fp8"]) == ["mxfp4", "fp8"]
    assert _normalize_upcast_request(["any"]) == [UPCAST_ANY]
    # a typo fails up-front, with the valid list in the message
    with pytest.raises(T2NErrorMisuse, match="unknown upcast_quant"):
        _normalize_upcast_request(["mxpf4"])


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
    # or a lingering hf_quantizer even if config was cleared
    with pytest.raises(T2NErrorMisuse):
        assert_upcast_dense(_Model(hf_quantizer=object()), ["any"])
