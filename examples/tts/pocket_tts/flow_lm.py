"""Export Pocket-TTS' FlowLM as two NNEF graphs (init + step) via t2n.

The FlowLM is the autoregressive transformer that turns text + voice prompt
into a stream of audio latents. PyTorch ships it with an in-place KV cache
(``StreamingMultiheadAttention``) which doesn't trace into a static graph,
so we wrap it with the KV-cache-as-IO pattern (see ``_io_attention.py`` and
``_flow_lm_export.py``):

* ``flow_lm_init.nnef.tgz`` -- once per utterance: token IDs + voice KV
  prefix + caller-supplied position vectors -> transformer hidden state at
  the BOS audio position + EOS logit + populated KV cache.
* ``flow_lm_step.nnef.tgz`` -- once per audio frame: the previous audio
  latent + current KV cache + position vectors -> next transformer hidden
  state + EOS logit + updated KV cache.

The Rust runtime stitches these together with the already-exported
``flow_net.nnef.tgz`` (LSD denoising loop) and ``decoder.nnef.tgz`` (Mimi
SEANet decoder) to produce a 24 kHz waveform from raw text.

Run (mini, fast):
    python flow_lm.py --mini --skip-io-check
With check_io against tract:
    python flow_lm.py --mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pocket_tts.conditioners import text as conditioners_text
from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.mimi_transformer import StreamingTransformer
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.modules.stateful_module import StatefulModule

from examples.tts.pocket_tts._flow_lm_export import FlowLMInit, FlowLMStep
from torch_to_nnef import TractNNEF, export_model_to_nnef


def _build_lut_conditioner_without_tokenizer(
    n_bins: int, dim: int
) -> LUTConditioner:
    """Build a ``LUTConditioner`` whose ``__init__`` skips SentencePiece.

    Pocket-TTS' real conditioner downloads a gated SentencePiece model in
    ``LUTConditioner.__init__``. Tokenization happens in Rust via the
    ``sentencepiece`` crate at runtime, so the exported graph only needs
    the ``embed`` lookup. We patch the SentencePiece constructor to a stub
    around the one-shot init so the rest of the original constructor runs
    unchanged.
    """

    class _StubTokenizer:
        def __init__(self, *_args, **_kwargs):
            pass

        def vocab_size(self):
            return n_bins

    real = conditioners_text.SentencePieceTokenizer
    conditioners_text.SentencePieceTokenizer = _StubTokenizer
    try:
        cond = LUTConditioner(
            n_bins=n_bins, tokenizer_path="", dim=dim, output_dim=dim
        )
    finally:
        conditioners_text.SentencePieceTokenizer = real
    cond.tokenizer = None
    return cond


def build_mini_flow_lm() -> FlowLMModel:
    """Tiny FlowLM mirroring the real Pocket-TTS structure at small scale.

    Real Pocket-TTS: ``d_model=512``, ``num_layers=12``, ``num_heads=8``,
    ``ldim=64``, vocab ``n_bins=4000``, context ``1024``. Here we shrink every
    dim while keeping the streaming-transformer + flow_net architecture so the
    export script runs without HF auth and the gated tokenizer download.
    """
    n_bins, d_model = 100, 16
    num_layers, num_heads = 2, 2
    ldim = 8
    context = 32

    conditioner = _build_lut_conditioner_without_tokenizer(
        n_bins=n_bins, dim=d_model
    )
    transformer = StreamingTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=32,
        context=context,
    )
    flow_net = SimpleMLPAdaLN(
        in_channels=ldim,
        model_channels=16,
        out_channels=ldim,
        cond_channels=d_model,
        num_res_blocks=2,
        num_time_conds=2,
    )
    flow_lm = FlowLMModel(
        conditioner=conditioner,
        flow_net=flow_net,
        transformer=transformer,
        dim=d_model,
        ldim=ldim,
        insert_bos_before_voice=False,
    ).eval()
    for name, mod in flow_lm.transformer.named_modules():
        if isinstance(mod, StatefulModule):
            mod._module_absolute_name = name
    return flow_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-init",
        type=Path,
        default=Path("./pocket_tts_flow_lm_init.nnef.tgz"),
    )
    parser.add_argument(
        "--out-step",
        type=Path,
        default=Path("./pocket_tts_flow_lm_step.nnef.tgz"),
    )
    parser.add_argument("--tract-version", type=str, default=None)
    parser.add_argument(
        "--skip-io-check",
        action="store_true",
        help="Skip tract I/O comparison (only validates the graph emit).",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        default=True,
        help="Tiny random-weights config (default; only mode supported "
        "today). Real-checkpoint export needs HF auth and is the next step.",
    )
    parser.add_argument(
        "--text-tokens",
        type=int,
        default=4,
        help="Number of text tokens to use for the trace shape.",
    )
    parser.add_argument(
        "--past-frames",
        type=int,
        default=4,
        help="Past KV-cache length to use for the step trace shape.",
    )
    args = parser.parse_args()

    flow_lm = build_mini_flow_lm()
    n_layers = len(flow_lm.transformer.layers)
    n_heads = flow_lm.transformer.layers[0].self_attn.num_heads
    head_dim = flow_lm.transformer.layers[0].self_attn.dim_per_head
    ldim = flow_lm.ldim

    tract_version = args.tract_version or TractNNEF.latest_version()
    check_io = not args.skip_io_check
    target = TractNNEF(version=tract_version, check_io=check_io)

    # --- flow_lm_init -------------------------------------------------------
    # Trace shape: T_voice=0 voice prefix (no prompt), ``--text-tokens``
    # text tokens. The caller will swap in a real voice prefix at runtime via
    # ``voice.dat`` -- see ``bake_voice.py``.
    init = FlowLMInit(flow_lm).eval()
    token_ids = torch.randint(0, 100, (1, args.text_tokens), dtype=torch.long)
    init_past_kv = torch.zeros(n_layers, 2, 1, 0, n_heads, head_dim)
    n_q = args.text_tokens + 1
    init_q_pos = torch.arange(n_q, dtype=torch.long)
    init_k_pos = torch.arange(n_q, dtype=torch.long)
    print(f"Exporting flow_lm_init to {args.out_init}")
    export_model_to_nnef(
        model=init,
        args=(token_ids, init_past_kv, init_q_pos, init_k_pos),
        file_path_export=args.out_init,
        inference_target=target,
        input_names=["token_ids", "past_kv", "q_positions", "k_positions"],
        output_names=["transformer_out", "eos_logit", "new_kv"],
        debug_bundle_path=Path("./debug_pocket_tts_flow_lm_init.tgz"),
    )

    # --- flow_lm_step -------------------------------------------------------
    # Trace shape: ``--past-frames`` past KV entries.
    step = FlowLMStep(flow_lm).eval()
    audio = torch.randn(1, ldim, dtype=torch.float32)
    step_past_kv = torch.randn(
        n_layers, 2, 1, args.past_frames, n_heads, head_dim
    )
    step_q_pos = torch.tensor([args.past_frames], dtype=torch.long)
    step_k_pos = torch.arange(args.past_frames + 1, dtype=torch.long)
    print(f"Exporting flow_lm_step to {args.out_step}")
    export_model_to_nnef(
        model=step,
        args=(audio, step_past_kv, step_q_pos, step_k_pos),
        file_path_export=args.out_step,
        inference_target=target,
        input_names=["audio_latent", "past_kv", "q_positions", "k_positions"],
        output_names=["transformer_out", "eos_logit", "new_kv"],
        debug_bundle_path=Path("./debug_pocket_tts_flow_lm_step.tgz"),
    )

    print("done")


if __name__ == "__main__":
    main()
