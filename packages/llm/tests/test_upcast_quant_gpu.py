"""GPU-only, opt-in end-to-end checks for native-quant up-cast.

These exercise the *real* dequant paths (not the pure decision logic covered by
``test_upcast_quant.py``): mint a tiny quantized model, load it through
``load_model(..., upcast_quant=[...])``, and assert it comes back dense.

Both quantizer mechanisms are covered:
  - **post-load** (bnb 4-bit → ``model.dequantize()``): any CUDA GPU.
  - **load-time** (fp8 → ``dequantize=True`` at load): needs compute ≥ 8.9
    (e.g. 4090/H100); skips on older GPUs, which auto-dequantize fp8 anyway.

(mxfp4 is the headline format but a *tiny* mxfp4 model can't be serialized:
transformers' mxfp4 packing assumes gpt-oss-scale dims, so the load-time branch
is exercised here via fp8, which shares the exact same code path.)

Triple-gated so CI never runs them by accident:
  1. own file, not collected by the LLM ``tox`` CI;
  2. ``ci_skip`` marker → deselected by the ``-m "not ci_skip"`` lane;
  3. skipped unless ``T2N_RUN_GPU_TESTS=1`` AND CUDA is available.

Run manually on a CUDA box with:
    T2N_RUN_GPU_TESTS=1 pytest tests/test_upcast_quant_gpu.py -v
"""

import os
from pathlib import Path

import pytest

from torch_to_nnef_llm.loader import _native_quant_method, load_model

pytestmark = pytest.mark.ci_skip

_OPT_IN = os.environ.get("T2N_RUN_GPU_TESTS") == "1"


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _cuda_ge_8_9() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (8, 9)
    except Exception:
        return False


def _tiny_llama_dir(tmp_path: Path) -> Path:
    from transformers import AutoModelForCausalLM, LlamaConfig

    cfg = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=512,
        max_position_embeddings=128,
    )
    d = tmp_path / "dense"
    AutoModelForCausalLM.from_config(cfg).save_pretrained(d)
    return d


def _assert_dense_and_runs(model, vocab_size=512):
    import torch

    assert _native_quant_method(model) is None
    assert getattr(model, "hf_quantizer", None) is None
    out = model.to("cpu")(torch.tensor([[1, 2, 3]]))
    assert out.logits.shape[-1] == vocab_size


pytestmark_opt = pytest.mark.skipif(
    not _OPT_IN, reason="opt-in GPU e2e; set T2N_RUN_GPU_TESTS=1"
)


@pytestmark_opt
@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA")
def test_upcast_bnb_4bit_post_load(tmp_path):
    """Post-load dequant branch: bnb-4bit → model.dequantize() → dense."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    pytest.importorskip("bitsandbytes")
    src = _tiny_llama_dir(tmp_path)
    q = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="cuda",
    )
    assert _native_quant_method(q) == "bitsandbytes"
    qdir = tmp_path / "bnb"
    q.save_pretrained(qdir)
    del q
    torch.cuda.empty_cache()

    model = load_model(
        local_dir=qdir, upcast_quant=["bitsandbytes"], force_module_dtype="bf16"
    )
    _assert_dense_and_runs(model)


@pytestmark_opt
@pytest.mark.skipif(
    not _cuda_ge_8_9(),
    reason="fp8 needs compute >= 8.9 (older GPUs auto-dequantize fp8)",
)
def test_upcast_fp8_load_time(tmp_path):
    """Load-time dequant branch: fp8 → dequantize=True at load → dense."""
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.utils.quantization_config import FineGrainedFP8Config

    src = _tiny_llama_dir(tmp_path)
    q = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=FineGrainedFP8Config(), device_map="cuda"
    )
    assert _native_quant_method(q) == "fp8"
    qdir = tmp_path / "fp8"
    q.save_pretrained(qdir)
    del q
    torch.cuda.empty_cache()

    model = load_model(
        local_dir=qdir, upcast_quant=["fp8"], force_module_dtype="bf16"
    )
    _assert_dense_and_runs(model)
