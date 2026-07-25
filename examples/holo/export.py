"""Export Hcompany/Holo-3.1 (Qwen3.5 dense VLM, ``qwen3_5``) for the Rust demo.

Produces, in ``--out`` dir:
  * ``vision.nnef.tgz``   : the vision tower (dynamic resolution).
  * ``decoder.nnef.tgz``  : a STREAMING hybrid decoder. Unlike the manifest
    joint export, positions + causal mask are runtime INPUTS and every layer's
    state (gated-delta-net conv+recurrent, or attention KV) is threaded
    explicitly, so ONE dynamic graph serves both prefill (S>1, zero states) and
    decode (S=1, carried states) -- what a Rust generation loop needs.
  * ``holo.json``         : shapes + per-layer state layout for the runtime.
  * ``*.bin``             : input_ids + pixel_values + the RoPE cos/sin (prompt
    + a decode-continuation table), flat little-endian. RoPE is computed HOST
    side (interleaved mRoPE does not lower faithfully to tract).

Usage:
    python export.py --dummy --out ./exp     # tiny random model, no download
    python export.py --repo Hcompany/Holo-3.1-0.8B --image shot.png \
        --prompt "Click the search bar" --out ./exp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import (
    AutoProcessor,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef_llm.models.handlers.base import scatter_features_by_mask

# Importing the handler registers the torch-side ``t2n_extra::gated_delta_scan``
# op and exposes the production modules the demo reuses (so the streaming
# decoder cannot drift from the production graph).
from torch_to_nnef_llm.models.handlers.qwen3_5_vl import (  # noqa: E402
    Qwen35ArchitectureHandler,
    Qwen35VisionEncoder,
    _HybridGDNForward,
)

# Single shared instance to reuse the handler's hybrid-state layout helpers
# (_gdn_dims / _state_names / _build_state_inputs), keeping the conv/rec/KV
# shapes + ordering in exactly one place.
_DEC_HANDLER = Qwen35ArchitectureHandler()


def _first_not_none(*values):
    """First non-None value (an explicit check, so an eos id of 0 survives)."""
    return next((v for v in values if v is not None), None)


def _dummy_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        image_token_id=1,
        video_token_id=2,
        vision_start_token_id=3,
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            # head_dim + full rotary so the interleaved mRoPE sections
            # ([11, 11, 10] -> sum 32) index within the rotary freq dim
            # (rotary_dim // 2 = 32). A smaller head_dim makes the in-graph
            # mRoPE gather go out of bounds on tract (torch tolerates it).
            "head_dim": 64,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [11, 11, 10],
            },
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 64,
        },
    )


class StreamingHybridDecoder(torch.nn.Module):
    """Hybrid GDN/attention decoder with runtime positions + explicit state.

    Signature (one graph, prefill and decode):
        in : input_ids[1, S] int64
             cos[1, S, rotary_dim] float32   (RoPE, computed host-side)
             sin[1, S, rotary_dim] float32
             mask[1, 1, S, S+P] float32
             image_embeddings[N_img, H] float32   (empty at decode)
             then per layer, in ``config.layer_types`` order:
               GDN  -> conv_state[1, conv_dim, K-1], rec_state[1, nv, hk, hv]
               attn -> key[1, n_kv, P, hd], value[1, n_kv, P, hd]
        out: logits[1, S, vocab]
             + the same per-layer states, updated

    RoPE (interleaved mRoPE) is computed HOST-side and passed in as cos/sin,
    not computed in-graph: the interleaved-mRoPE strided scatter does not lower
    faithfully to tract, and cos/sin are cheap to precompute (the vision tower
    takes its position embeddings the same way).
    """

    def __init__(self, model):
        super().__init__()
        self.lm = model.model.language_model
        self.lm_head = model.lm_head
        self.text_conf = model.config.text_config
        self.layer_types = list(self.text_conf.layer_types)
        self.image_token_id = model.config.image_token_id

    def forward(self, input_ids, cos, sin, mask, image_embeddings, *states):
        hidden = self.lm.embed_tokens(input_ids)
        hidden = scatter_features_by_mask(
            inputs_embeds=hidden,
            token_mask=input_ids == self.image_token_id,
            features=image_embeddings,
        )
        st = list(states)
        new_states = []
        cursor = 0
        for idx, layer in enumerate(self.lm.layers):
            residual = hidden
            normed = layer.input_layernorm(hidden)
            if self.layer_types[idx] == "linear_attention":
                # Reuse the production handler's GDN/attention compute so the
                # demo graph cannot diverge from the exported production graph.
                mix, c_out, r_out = _HybridGDNForward.gdn(
                    layer.linear_attn, normed, st[cursor], st[cursor + 1]
                )
                new_states += [c_out, r_out]
            else:
                mix, k_out, v_out = _HybridGDNForward.attn(
                    layer.self_attn,
                    normed,
                    cos,
                    sin,
                    st[cursor],
                    st[cursor + 1],
                    mask,
                )
                new_states += [k_out, v_out]
            cursor += 2
            hidden = residual + mix
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
        hidden = self.lm.norm(hidden)
        return (self.lm_head(hidden), *new_states)


def _manifest_layers(text_conf):
    """Per-layer state layout the Rust runtime reads from ``holo.json``.

    Dims come from the handler's ``_gdn_dims`` so the conv_dim formula and the
    GDN/attention head dims live in exactly one place (the handler).
    """
    _, n_v, h_k, h_v, conv_k, conv_dim = _DEC_HANDLER._gdn_dims(text_conf)
    n_kv = text_conf.num_key_value_heads
    head_dim = getattr(
        text_conf,
        "head_dim",
        text_conf.hidden_size // text_conf.num_attention_heads,
    )
    layers = []
    for ltype in _DEC_HANDLER._layer_types(text_conf):
        if ltype == "linear_attention":
            layers.append(
                {
                    "kind": "gdn",
                    "conv_dim": conv_dim,
                    "conv_state_width": conv_k - 1,
                    "num_v_heads": n_v,
                    "key_head_dim": h_k,
                    "value_head_dim": h_v,
                }
            )
        else:
            layers.append(
                {"kind": "attn", "num_kv_heads": n_kv, "head_dim": head_dim}
            )
    return layers


def _zero_states(text_conf, n_past, dtype):
    # The handler owns the state shapes + per-layer ordering; reuse it so the
    # demo cannot build states of the wrong shape.
    return _DEC_HANDLER._build_state_inputs(text_conf, n_past, dtype)[0]


def _build_sample(encoder, model, processor, args):
    """Build the (image, prompt) sample the two graphs are traced/run on.

    Returns pixel_values, input_ids, the host-side RoPE cos/sin (prompt table +
    a decode-continuation table), the causal mask, and the image embeddings.
    """
    conf = model.config
    tc = conf.text_config
    vc = conf.vision_config
    merge = vc.spatial_merge_size
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
    if processor is not None and args.image is not None:
        # Lazy: pillow is only needed for a real screenshot, not for --dummy.
        # pylint: disable-next=import-outside-toplevel
        from PIL import Image

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(args.image).convert("RGB"),
                    },
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        proc = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = proc["input_ids"]
        grid = proc["image_grid_thw"]
        mh, mw = int(grid[0, 1]) // merge, int(grid[0, 2]) // merge
        pixel_values = (
            proc["pixel_values"]
            .reshape(mh, mw, merge, merge, patch_dim)
            .float()
        )
    else:
        h, w = 8, 8
        mh, mw = h // merge, w // merge
        grid = torch.tensor([[1, h, w]], dtype=torch.long)
        num_img = mh * mw
        pixel_values = torch.randn(mh, mw, merge, merge, patch_dim)
        prompt_ids = torch.randint(4, tc.vocab_size, (1, 4))
        input_ids = torch.cat(
            [
                torch.tensor([[conf.vision_start_token_id]]),
                torch.full((1, num_img), conf.image_token_id),
                prompt_ids,
            ],
            dim=1,
        )

    seq = input_ids.shape[1]
    with torch.no_grad():
        image_embeddings = encoder(pixel_values)
    # prompt mRoPE positions via HF (offloads the tricky part off the runtime)
    mm_ids = torch.zeros_like(input_ids, dtype=torch.int)
    mm_ids[input_ids == conf.image_token_id] = 1
    with torch.no_grad():
        position_ids, _ = model.model.get_rope_index(
            input_ids,
            mm_ids,
            image_grid_thw=grid,
            video_grid_thw=None,
            attention_mask=torch.ones_like(input_ids),
        )
    position_ids = position_ids.long()
    # RoPE cos/sin, computed host-side (see StreamingHybridDecoder docstring).
    rotary = model.model.language_model.rotary_emb
    hidden0 = model.model.language_model.embed_tokens(input_ids)
    with torch.no_grad():
        cos, sin = rotary(hidden0, position_ids)  # [1, S, rotary_dim]
    # decode-continuation table: sequential text positions after the prompt
    # (all three mRoPE channels equal), so the runtime just indexes by step.
    n_decode = 128
    start = int(position_ids.max()) + 1
    dec_pos = (
        torch.arange(start, start + n_decode).view(1, 1, -1).expand(3, 1, -1)
    )
    with torch.no_grad():
        cos_tbl, sin_tbl = rotary(hidden0, dec_pos)  # [1, n_decode, rot_dim]
    neg = torch.finfo(torch.float32).min
    mask = torch.triu(torch.full((seq, seq), neg), diagonal=1).view(
        1, 1, seq, seq
    )
    return SimpleNamespace(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_embeddings=image_embeddings,
        cos=cos,
        sin=sin,
        cos_tbl=cos_tbl,
        sin_tbl=sin_tbl,
        mask=mask,
        seq=seq,
        rotary_dim=int(cos.shape[-1]),
        n_decode=n_decode,
        merge=merge,
        patch_dim=patch_dim,
    )


def _write_manifest_and_bins(args, model, s, layout, dec_in, dec_out):
    """Write ``holo.json`` + the flat little-endian sample ``.bin`` files."""
    conf = model.config
    tc = conf.text_config
    manifest = {
        "repo": args.repo or "dummy",
        "encoder_path": "vision.nnef.tgz",
        "decoder_path": "decoder.nnef.tgz",
        "hidden_size": tc.hidden_size,
        "vocab_size": tc.vocab_size,
        "image_token_id": conf.image_token_id,
        "vision_start_token_id": conf.vision_start_token_id,
        "eos_token_id": _first_not_none(
            getattr(conf, "eos_token_id", None),
            getattr(tc, "eos_token_id", None),
        ),
        "spatial_merge_size": s.merge,
        "patch_dim": s.patch_dim,
        "rotary_dim": s.rotary_dim,
        "layers": layout,
        "decoder_input_order": dec_in,
        "decoder_output_order": dec_out,
        "sample": {
            "seq": s.seq,
            "grid_mh": int(s.pixel_values.shape[0]),
            "grid_mw": int(s.pixel_values.shape[1]),
            "num_image_tokens": int((s.input_ids == conf.image_token_id).sum()),
            "n_decode_table": s.n_decode,
        },
    }
    (args.out / "holo.json").write_text(json.dumps(manifest, indent=2))
    # Flat little-endian .bin files: the Rust demo reads them with std only
    # (no npy/npz dependency). Shapes come from holo.json.
    s.input_ids.numpy().astype("<i8").tofile(args.out / "input_ids.bin")
    s.pixel_values.numpy().astype("<f4").tofile(args.out / "pixel_values.bin")
    s.cos.numpy().astype("<f4").tofile(args.out / "cos.bin")
    s.sin.numpy().astype("<f4").tofile(args.out / "sin.bin")
    s.cos_tbl.numpy().astype("<f4").tofile(args.out / "cos_table.bin")
    s.sin_tbl.numpy().astype("<f4").tofile(args.out / "sin_table.bin")
    n_img = int((s.input_ids == conf.image_token_id).sum())
    print(f"[holo] wrote holo.json + *.bin to {args.out}")
    print(f"[holo] seq={s.seq} img_tokens={n_img} layers={len(layout)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--repo", default=None, help="HF repo of a qwen3_5 VLM.")
    ap.add_argument(
        "--dummy",
        action="store_true",
        help="Tiny random model (no download), for CI/plumbing.",
    )
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--prompt", default="Where should I click?")
    ap.add_argument("--out", type=Path, default=Path("./exp"))
    ap.add_argument("--no-check-io", action="store_true")
    ap.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="Also print a torch greedy-decode of N tokens (parity check).",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)  # reproducible dummy weights across runs

    if args.dummy or args.repo is None:
        print("[holo] building tiny dummy qwen3_5 model")
        model = Qwen3_5ForConditionalGeneration(_dummy_config()).eval()
        processor = None
    else:
        print(f"[holo] loading {args.repo}")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.repo, torch_dtype=torch.float32
        ).eval()
        processor = AutoProcessor.from_pretrained(args.repo)
    model.config._attn_implementation = "eager"
    model.model.language_model.config._attn_implementation = "eager"
    tc = model.config.text_config

    encoder = Qwen35VisionEncoder(model.model.visual).eval()
    decoder = StreamingHybridDecoder(model).eval()
    s = _build_sample(encoder, model, processor, args)

    # ---- export vision encoder ----
    enc_path = args.out / "vision.nnef.tgz"
    enc_args = (s.pixel_values,)
    tgt = TractNNEF(
        version=TractNNEF.latest_version(), check_io=not args.no_check_io
    )
    tgt.dynamic_axes = {"pixel_values": {0: "MH", 1: "MW"}}
    print(f"[holo] exporting vision encoder -> {enc_path}")
    export_model_to_nnef(
        model=encoder,
        args=enc_args,
        file_path_export=enc_path,
        inference_target=tgt,
        input_names=["pixel_values"],
        output_names=["out_image_embeddings"],
    )

    # ---- export streaming decoder ----
    in_state_names, out_state_names = _DEC_HANDLER._state_names(tc)
    layout = _manifest_layers(tc)
    states = _zero_states(tc, 0, torch.float32)
    dec_args = (s.input_ids, s.cos, s.sin, s.mask, s.image_embeddings, *states)
    dec_in_names = [
        "input_ids",
        "cos",
        "sin",
        "mask",
        "in_image_embeddings",
    ] + in_state_names
    dec_out_names = ["logits"] + out_state_names
    dec_axes = {
        "input_ids": {1: "S"},
        "cos": {1: "S"},
        "sin": {1: "S"},
        "mask": {2: "S", 3: "SP"},
        "in_image_embeddings": {0: "IMG"},
    }
    # attention KV cache grows along the past axis at decode.
    for name in in_state_names:
        if name.startswith(("cache_key", "cache_value")):
            dec_axes[name] = {2: "P"}
    dec_path = args.out / "decoder.nnef.tgz"
    tgt2 = TractNNEF(
        version=TractNNEF.latest_version(), check_io=not args.no_check_io
    )
    tgt2.dynamic_axes = dec_axes
    print(f"[holo] exporting streaming decoder -> {dec_path}")
    export_model_to_nnef(
        model=decoder,
        args=dec_args,
        file_path_export=dec_path,
        inference_target=tgt2,
        input_names=dec_in_names,
        output_names=dec_out_names,
    )

    _write_manifest_and_bins(
        args, model, s, layout, dec_in_names, dec_out_names
    )

    if args.verify:
        # Torch reference of the exact loop the Rust binary runs (same in-memory
        # weights), so the demo's tokens can be checked for faithfulness.
        ref = _torch_greedy(
            decoder,
            s.input_ids,
            s.cos,
            s.sin,
            s.cos_tbl,
            s.sin_tbl,
            s.image_embeddings,
            tc,
            args.verify,
        )
        print(f"[holo] torch-ref token ids ({args.verify}): {ref}")


def _torch_greedy(
    decoder, input_ids, cos, sin, cos_tbl, sin_tbl, image_embeddings, tc, n_new
):
    """Greedy decode mirroring the Rust loop, for parity checking."""
    neg = torch.finfo(torch.float32).min
    seq = input_ids.shape[1]
    mask = torch.triu(torch.full((seq, seq), neg), diagonal=1).view(
        1, 1, seq, seq
    )
    states = _zero_states(tc, 0, torch.float32)
    with torch.no_grad():
        out = decoder(input_ids, cos, sin, mask, image_embeddings, *states)
    logits, states = out[0], list(out[1:])
    nxt = int(logits[0, -1].argmax())
    gen = [nxt]
    empty_img = torch.zeros((0, tc.hidden_size))
    past = seq
    for step in range(n_new - 1):
        c = cos_tbl[:, step : step + 1]
        s = sin_tbl[:, step : step + 1]
        m = torch.zeros(1, 1, 1, past + 1)
        with torch.no_grad():
            out = decoder(torch.tensor([[nxt]]), c, s, m, empty_img, *states)
        logits, states = out[0], list(out[1:])
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt)
        past += 1
    return gen


if __name__ == "__main__":
    main()
