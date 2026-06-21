"""Unit tests for the native-quant up-cast selection logic.

The actual dequantization (``model.dequantize()``) needs real quantized weights
and GPU kernels (mxfp4/fp8), so only the pure selection / branching is covered
here; the dequant call itself is delegated to transformers.
"""

import pytest

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef_llm.loader import (
    UPCAST_ANY,
    _native_quant_method,
    maybe_upcast_native_quant,
    should_upcast,
)


class _QConf:
    def __init__(self, method):
        self.quant_method = method


class _Cfg:
    def __init__(self, method=None):
        if method is not None:
            self.quantization_config = _QConf(method)


class _Model:
    """Minimal stand-in. ``dequantize()`` flips a flag and returns a dense twin."""

    def __init__(self, method=None):
        self.config = _Cfg(method)
        self.dequantized = False

    def dequantize(self):
        dense = _Model(method=None)  # dense: no quantization_config
        dense.dequantized = True
        return dense


def test_native_quant_method_detection():
    assert _native_quant_method(_Model("mxfp4")) == "mxfp4"
    assert _native_quant_method(_Model(None)) is None
    # enum-like value with a .value attribute is normalized + lowercased
    class _E:
        value = "FbgemmFp8"
    m = _Model("x")
    m.config.quantization_config.quant_method = _E()
    assert _native_quant_method(m) == "fbgemmfp8"


def test_should_upcast_matrix():
    assert should_upcast("mxfp4", ["mxfp4"]) is True
    assert should_upcast("mxfp4", ["fp8", "mxfp4"]) is True
    assert should_upcast("mxfp4", [UPCAST_ANY]) is True
    assert should_upcast("mxfp4", ["fp8"]) is False
    assert should_upcast("mxfp4", None) is False
    assert should_upcast("mxfp4", []) is False
    assert should_upcast(None, ["any"]) is False  # not quantized


def test_dense_model_is_untouched():
    m = _Model(None)
    assert maybe_upcast_native_quant(m, ["any"]) is m


def test_quantized_but_not_requested_is_kept_with_warning(caplog):
    m = _Model("mxfp4")
    out = maybe_upcast_native_quant(m, None)  # opt-in: not requested
    assert out is m  # unchanged, not dequantized
    assert any("native 'mxfp4'" in r.message for r in caplog.records)


def test_quantized_and_requested_is_dequantized():
    m = _Model("mxfp4")
    out = maybe_upcast_native_quant(m, ["mxfp4"])
    assert out is not m and out.dequantized is True
    assert _native_quant_method(out) is None  # now dense


def test_quantized_with_other_method_requested_errors():
    m = _Model("mxfp4")
    with pytest.raises(T2NErrorMisuse, match="mxfp4"):
        maybe_upcast_native_quant(m, ["fp8"])
