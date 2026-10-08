"""Export Hcompany/Holo-3.1 (Qwen3.5 dense VLM, ``qwen3_5``) for the Rust demo.

Produces, in ``--out`` dir:
  * ``vision.nnef.tgz``   : the vision tower (dynamic resolution).
  * ``decoder.nnef.tgz``  : a STREAMING hybrid decoder. Unlike the manifest
    joint export, positions + causal mask are runtime INPUTS and every layer's
    state (gated-delta-net conv+recurrent, or attention KV) is threaded
    explicitly, so ONE dynamic graph serves both prefill (S>1, zero states) and
    decode (S=1, carried states), which is what a Rust generation loop needs.
  * ``holo.json``         : shapes + per-layer state layout for the runtime.
  * ``*.bin``             : input_ids + position_ids + pixel_values, flat
    little-endian. RoPE runs IN-GRAPH from position_ids (the mRoPE interleave
    is rewritten as a tract-friendly masked sum); only the position layout for
    the image span comes from the host (get_rope_index).

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

import nnef
import torch
from transformers import (
    AutoProcessor,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef_llm.models.handlers.base import scatter_features_by_mask

# ``--dtype`` presets. f16 lets a bigger checkpoint load + export in half the
# memory (a real Holo model in f32 is large); the Rust demo casts each input to
# whatever dtype the graph expects, so f16 needs no runtime change there.
_DTYPES = {"f32": torch.float32, "f16": torch.float16}

# Importing the handler registers the torch-side ``t2n_extra::gated_delta_scan``
# op and exposes the production modules the demo reuses (so the streaming
# decoder cannot drift from the production graph).
from torch_to_nnef_llm.models.handlers.qwen3_5_vl import (  # noqa: E402
    InGraphRotary,
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


def _eos_token_id(*values):
    """Single eos id for the manifest, since HF configs may carry a list.

    The Rust demo compares one id (`Option<i64>`), and a list reaching the
    manifest fails its parse outright, so unwrap to the first entry. HF's own
    generation stops on any of them; the first is the canonical one.
    """
    value = _first_not_none(*values)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


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
             position_ids[3, 1, S] int64   (the mRoPE t/h/w positions)
             mask[1, 1, S, S+P] float32
             image_embeddings[N_img, H] float32   (empty at decode)
             then per layer, in ``config.layer_types`` order:
               GDN  -> conv_state[1, conv_dim, K-1], rec_state[1, nv, hk, hv]
               attn -> key[1, n_kv, P, hd], value[1, n_kv, P, hd]
        out: logits[1, S, vocab]
             + the same per-layer states, updated

    RoPE (interleaved mRoPE) is computed IN-GRAPH from position_ids via
    ``InGraphRotary`` (a constant-masked-sum interleave that lowers to tract,
    unlike upstream's strided scatter), so the runtime feeds integer positions
    and the graph is fully self-contained. Only the mRoPE position *layout* for
    the image span still comes from the host (transformers' get_rope_index).
    """

    def __init__(self, model):
        super().__init__()
        self.lm = model.model.language_model
        self.lm_head = model.lm_head
        self.rotary = InGraphRotary(self.lm.rotary_emb)
        self.text_conf = model.config.text_config
        self.layer_types = list(self.text_conf.layer_types)
        self.image_token_id = model.config.image_token_id

    def forward(self, input_ids, position_ids, mask, image_embeddings, *states):
        cos, sin = self.rotary(position_ids)
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


def _build_sample(encoder, model, processor, args, dtype):
    """Build the (image, prompt) sample the two graphs are traced/run on.

    Returns pixel_values, input_ids, the host-side RoPE cos/sin (prompt table +
    a decode-continuation table), the causal mask, and the image embeddings, all
    in ``dtype`` (so the traced graph is that dtype).
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
            .to(dtype)
        )
    else:
        h, w = 8, 8
        mh, mw = h // merge, w // merge
        grid = torch.tensor([[1, h, w]], dtype=torch.long)
        num_img = mh * mw
        pixel_values = torch.randn(mh, mw, merge, merge, patch_dim).to(dtype)
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
    # get_rope_index gives the mRoPE position LAYOUT (image tokens get a 2-D
    # grid); the rotary itself runs in-graph, so the sample just carries the
    # integer positions. Decode continues them sequentially (all channels
    # equal) from prompt_max_pos, computed by the runtime.
    position_ids = position_ids.long()
    prompt_max_pos = int(position_ids.max())
    neg = torch.finfo(dtype).min
    mask = (
        torch.triu(torch.full((seq, seq), neg), diagonal=1)
        .view(1, 1, seq, seq)
        .to(dtype)
    )
    return SimpleNamespace(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_embeddings=image_embeddings,
        position_ids=position_ids,
        mask=mask,
        seq=seq,
        prompt_max_pos=prompt_max_pos,
        merge=merge,
        patch_dim=patch_dim,
    )


def _write_manifest_and_inputs(args, model, s, layout, dec_in, dec_out):
    """Write ``holo.json`` + the sample inputs as NNEF ``.dat`` tensors."""
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
        "eos_token_id": _eos_token_id(
            getattr(conf, "eos_token_id", None),
            getattr(tc, "eos_token_id", None),
        ),
        "layers": layout,
        "decoder_input_order": dec_in,
        "decoder_output_order": dec_out,
        "sample": {
            "num_image_tokens": int((s.input_ids == conf.image_token_id).sum()),
            "prompt_max_pos": s.prompt_max_pos,
        },
    }
    (args.out / "holo.json").write_text(json.dumps(manifest, indent=2))
    # Sample inputs as NNEF .dat tensors (self-describing: shape + dtype in the
    # file), read on the Rust side with tract's native `read_tensor`, so no
    # hand-rolled byte parsing and no manifest-driven shapes. Floats are written
    # f32; the Rust side casts each input to whatever dtype the graph expects.
    for name, arr in [
        ("input_ids", s.input_ids.numpy().astype("<i8")),
        ("position_ids", s.position_ids.numpy().astype("<i8")),
        ("pixel_values", s.pixel_values.float().numpy().astype("<f4")),
    ]:
        with open(args.out / f"{name}.dat", "wb") as fh:
            nnef.write_tensor(fh, arr)
    n_img = int((s.input_ids == conf.image_token_id).sum())
    print(f"[holo] wrote holo.json + *.dat to {args.out}")
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
    ap.add_argument(
        "--dtype",
        default="f32",
        choices=sorted(_DTYPES),
        help="Load + export precision. f16 halves memory for big checkpoints.",
    )
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
    dtype = _DTYPES[args.dtype]

    if args.dummy or args.repo is None:
        print(f"[holo] building tiny dummy qwen3_5 model ({args.dtype})")
        model = Qwen3_5ForConditionalGeneration(_dummy_config()).eval()
        model = model.to(dtype)
        processor = None
    else:
        print(f"[holo] loading {args.repo} ({args.dtype})")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.repo, torch_dtype=dtype
        ).eval()
        processor = AutoProcessor.from_pretrained(args.repo)
    model.config._attn_implementation = "eager"
    model.model.language_model.config._attn_implementation = "eager"
    tc = model.config.text_config

    encoder = Qwen35VisionEncoder(model.model.visual).eval()
    decoder = StreamingHybridDecoder(model).eval()
    s = _build_sample(encoder, model, processor, args, dtype)

    # f16 accumulates more drift than f32, so verify against the looser preset
    # every fp16 export uses (the tiny dummy barely diverges regardless).
    tol = (
        TractCheckTolerance.ULTRA
        if dtype == torch.float16
        else TractCheckTolerance.APPROXIMATE
    )

    # ---- export vision encoder ----
    enc_path = args.out / "vision.nnef.tgz"
    enc_args = (s.pixel_values,)
    tgt = TractNNEF(
        version=TractNNEF.latest_version(),
        check_io=not args.no_check_io,
        check_io_tolerance=tol,
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
    states = _zero_states(tc, 0, dtype)
    dec_args = (
        s.input_ids,
        s.position_ids,
        s.mask,
        s.image_embeddings,
        *states,
    )
    dec_in_names = [
        "input_ids",
        "position_ids",
        "mask",
        "in_image_embeddings",
    ] + in_state_names
    dec_out_names = ["logits"] + out_state_names
    dec_axes = {
        "input_ids": {1: "S"},
        "position_ids": {2: "S"},
        "mask": {2: "S", 3: "SP"},
        "in_image_embeddings": {0: "IMG"},
    }
    # attention KV cache grows along the past axis at decode.
    for name in in_state_names:
        if name.startswith(("cache_key", "cache_value")):
            dec_axes[name] = {2: "P"}
    dec_path = args.out / "decoder.nnef.tgz"
    tgt2 = TractNNEF(
        version=TractNNEF.latest_version(),
        check_io=not args.no_check_io,
        check_io_tolerance=tol,
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

    _write_manifest_and_inputs(
        args, model, s, layout, dec_in_names, dec_out_names
    )

    if args.verify:
        # Torch reference of the exact loop the Rust binary runs (same in-memory
        # weights), so the demo's tokens can be checked for faithfulness.
        ref = _torch_greedy(
            decoder,
            s.input_ids,
            s.position_ids,
            s.prompt_max_pos,
            s.image_embeddings,
            tc,
            args.verify,
        )
        print(f"[holo] torch-ref token ids ({args.verify}): {ref}")


def _torch_greedy(
    decoder,
    input_ids,
    position_ids,
    prompt_max_pos,
    image_embeddings,
    tc,
    n_new,
):
    """Greedy decode mirroring the Rust loop, for parity checking."""
    dtype = image_embeddings.dtype
    neg = torch.finfo(dtype).min
    seq = input_ids.shape[1]
    mask = (
        torch.triu(torch.full((seq, seq), neg), diagonal=1)
        .view(1, 1, seq, seq)
        .to(dtype)
    )
    states = _zero_states(tc, 0, dtype)
    with torch.no_grad():
        out = decoder(input_ids, position_ids, mask, image_embeddings, *states)
    logits, states = out[0], list(out[1:])
    nxt = int(logits[0, -1].argmax())
    gen = [nxt]
    empty_img = torch.zeros((0, tc.hidden_size), dtype=dtype)
    past = seq
    for step in range(n_new - 1):
        # text continuation: all three mRoPE channels share the next position
        pos = prompt_max_pos + 1 + step
        pos_ids = torch.tensor([pos, pos, pos]).view(3, 1, 1)
        m = torch.zeros(1, 1, 1, past + 1, dtype=dtype)
        with torch.no_grad():
            out = decoder(torch.tensor([[nxt]]), pos_ids, m, empty_img, *states)
        logits, states = out[0], list(out[1:])
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt)
        past += 1
    return gen


if __name__ == "__main__":
    main()
