"""GPU-only, opt-in end-to-end check for native-quant up-cast.

This exercises the *real* dequant path (not the pure decision logic covered by
``test_upcast_quant.py``): mint a tiny MXFP4 model, then load it through
``load_model(..., upcast_quant=["mxfp4"])`` and assert it comes back dense.

It is triple-gated so CI never runs it by accident:
  1. lives in its own file, which the LLM ``tox`` CI does not collect (that CI
     only runs ``test_llm_cli.py`` / ``test_load_retry.py``);
  2. marked ``ci_skip`` so the ``-m "not ci_skip"`` lane deselects it;
  3. skipped unless ``T2N_RUN_GPU_TESTS=1`` is explicitly set AND CUDA is
     available — so even a GPU runner won't run it without deliberate opt-in.

Run it manually on a CUDA box with:
    T2N_RUN_GPU_TESTS=1 pytest tests/test_upcast_quant_gpu.py -v
"""

import os

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


@pytest.mark.skipif(
    not _OPT_IN,
    reason="opt-in GPU e2e test; set T2N_RUN_GPU_TESTS=1 to run",
)
@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA")
def test_load_model_upcasts_mxfp4_to_dense(tmp_path):
    """Mint a tiny MXFP4 gpt-oss, then up-cast it to dense float on load."""
    import torch
    from transformers import AutoModelForCausalLM, GptOssConfig
    from transformers.utils.quantization_config import Mxfp4Config

    # tiny gpt-oss (~0.4M params): 2 layers, 4 MoE experts, small dims.
    cfg = GptOssConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=512,
        max_position_embeddings=128,
    )
    dense_dir = tmp_path / "dense"
    AutoModelForCausalLM.from_config(cfg).save_pretrained(dense_dir)

    # mint: quantize to MXFP4 (needs CUDA + triton kernels) and save.
    quant_dir = tmp_path / "mxfp4"
    quantized = AutoModelForCausalLM.from_pretrained(
        dense_dir, quantization_config=Mxfp4Config(), device_map="cuda"
    )
    assert _native_quant_method(quantized) == "mxfp4"
    quantized.save_pretrained(quant_dir)
    del quantized
    torch.cuda.empty_cache()

    # the actual up-cast under test: load with the load-time dequant flag.
    model = load_model(
        local_dir=quant_dir,
        upcast_quant=["mxfp4"],
        force_module_dtype="bf16",
    )

    # it must come back fully dense (no native quant, no lingering quantizer).
    assert _native_quant_method(model) is None
    assert getattr(model, "hf_quantizer", None) is None
    # and still run a forward pass (dense, exportable).
    model = model.to("cpu")
    out = model(torch.tensor([[1, 2, 3]]))
    assert out.logits.shape[-1] == cfg.vocab_size
