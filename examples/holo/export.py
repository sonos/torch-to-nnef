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

import torch
import torch.nn.functional as F
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF

# Importing the handler registers the torch-side ``t2n_extra::gated_delta_scan``
# op and exposes the production vision encoder module.
from torch_to_nnef_llm.models.handlers.qwen3_5_vl import (  # noqa: E402
    Qwen35VisionEncoder,
    _l2norm,
    _rotate_half,
)


def _dummy_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        image_token_id=1,
        video_token_id=2,
        vision_start_token_id=3,
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            # head_dim + full rotary so the interleaved mRoPE sections
            # ([11, 11, 10] -> sum 32) index within the rotary freq dim
            # (rotary_dim // 2 = 32). A smaller head_dim makes the in-graph
            # mRoPE gather go out of bounds on tract (torch tolerates it).
            head_dim=64,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                partial_rotary_factor=1.0,
                mrope_section=[11, 11, 10],
            ),
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=64,
        ),
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

    def _gdn(self, gdn, hidden, conv_state_in, rec_state_in):
        batch, seq, _ = hidden.shape
        conv_k = gdn.conv_kernel_size
        qkv = gdn.in_proj_qkv(hidden).transpose(1, 2)
        z = gdn.in_proj_z(hidden).reshape(batch, seq, -1, gdn.head_v_dim)
        b = gdn.in_proj_b(hidden)
        a = gdn.in_proj_a(hidden)
        conv_dim = qkv.shape[1]
        padded = torch.cat([conv_state_in, qkv], dim=-1)
        conv = F.silu(
            F.conv1d(
                padded, gdn.conv1d.weight, gdn.conv1d.bias, groups=conv_dim
            )
        )
        conv_state_out = padded[:, :, -(conv_k - 1) :]
        mixed = conv.transpose(1, 2)
        q, k, v = torch.split(
            mixed, [gdn.key_dim, gdn.key_dim, gdn.value_dim], -1
        )
        q = q.reshape(batch, seq, gdn.num_k_heads, gdn.head_k_dim)
        k = k.reshape(batch, seq, gdn.num_k_heads, gdn.head_k_dim)
        v = v.reshape(batch, seq, gdn.num_v_heads, gdn.head_v_dim)
        beta = b.sigmoid()
        g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
        rep = gdn.num_v_heads // gdn.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        scale = 1.0 / (gdn.head_k_dim**0.5)
        q_p = (_l2norm(q) * scale).transpose(1, 2)
        k_p = _l2norm(k).transpose(1, 2)
        v_p = v.transpose(1, 2)
        g_p = g.transpose(1, 2)
        beta_p = beta.transpose(1, 2)
        y, rec_state_out = torch.ops.t2n_extra.gated_delta_scan(
            q_p, k_p, v_p, g_p, beta_p, rec_state_in
        )
        core = y.transpose(1, 2).reshape(-1, gdn.head_v_dim)
        core = gdn.norm(core, z.reshape(-1, gdn.head_v_dim)).reshape(
            batch, seq, -1
        )
        return gdn.out_proj(core), conv_state_out, rec_state_out

    def _attn(self, sa, hidden, cos, sin, key_in, value_in, mask):
        batch, seq, _ = hidden.shape
        head_dim = sa.head_dim
        q, gate = torch.chunk(
            sa.q_proj(hidden).view(batch, seq, -1, head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(batch, seq, -1)
        q = sa.q_norm(q.reshape(batch, seq, -1, head_dim)).transpose(1, 2)
        k = sa.k_proj(hidden).view(batch, seq, -1, head_dim)
        k = sa.k_norm(k).transpose(1, 2)
        v = sa.v_proj(hidden).view(batch, seq, -1, head_dim).transpose(1, 2)
        cos2, sin2 = cos.unsqueeze(1), sin.unsqueeze(1)
        rot = cos.shape[-1]
        q = torch.cat(
            [
                q[..., :rot] * cos2 + _rotate_half(q[..., :rot]) * sin2,
                q[..., rot:],
            ],
            dim=-1,
        )
        k = torch.cat(
            [
                k[..., :rot] * cos2 + _rotate_half(k[..., :rot]) * sin2,
                k[..., rot:],
            ],
            dim=-1,
        )
        k = torch.cat([key_in, k], dim=2)
        v = torch.cat([value_in, v], dim=2)
        rep = sa.num_key_value_groups
        keys = k.repeat_interleave(rep, dim=1)
        values = v.repeat_interleave(rep, dim=1)
        w = torch.matmul(q, keys.transpose(2, 3)) * sa.scaling + mask
        w = torch.softmax(w.float(), dim=-1).to(q.dtype)
        o = torch.matmul(w, values).transpose(1, 2).reshape(batch, seq, -1)
        o = o * torch.sigmoid(gate)
        return sa.o_proj(o), k, v

    def forward(self, input_ids, cos, sin, mask, image_embeddings, *states):
        from torch_to_nnef_llm.models.handlers.base import (
            scatter_features_by_mask,
        )

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
                mix, c_out, r_out = self._gdn(
                    layer.linear_attn, normed, st[cursor], st[cursor + 1]
                )
                new_states += [c_out, r_out]
            else:
                mix, k_out, v_out = self._attn(
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


def _state_layout(text_conf):
    layout = []
    in_names, out_names = [], []
    n_kv = text_conf.num_key_value_heads
    head_dim = getattr(
        text_conf,
        "head_dim",
        text_conf.hidden_size // text_conf.num_attention_heads,
    )
    conv_k = text_conf.linear_conv_kernel_dim
    n_v = text_conf.linear_num_value_heads
    h_k = text_conf.linear_key_head_dim
    h_v = text_conf.linear_value_head_dim
    conv_dim = h_k * text_conf.linear_num_key_heads * 2 + h_v * n_v
    for idx, ltype in enumerate(text_conf.layer_types):
        if ltype == "linear_attention":
            layout.append(
                {
                    "kind": "gdn",
                    "conv_dim": conv_dim,
                    "conv_state_width": conv_k - 1,
                    "num_v_heads": n_v,
                    "key_head_dim": h_k,
                    "value_head_dim": h_v,
                }
            )
            in_names += [f"in_conv_state_{idx}", f"in_rec_state_{idx}"]
            out_names += [f"out_conv_state_{idx}", f"out_rec_state_{idx}"]
        else:
            layout.append(
                {
                    "kind": "attn",
                    "num_kv_heads": n_kv,
                    "head_dim": head_dim,
                }
            )
            in_names += [f"in_key_{idx}", f"in_value_{idx}"]
            out_names += [f"out_key_{idx}", f"out_value_{idx}"]
    return layout, in_names, out_names


def _zero_states(text_conf, n_past, dtype):
    layout, _, _ = _state_layout(text_conf)
    out = []
    for lay in layout:
        if lay["kind"] == "gdn":
            out.append(
                torch.zeros(
                    (1, lay["conv_dim"], lay["conv_state_width"]), dtype=dtype
                )
            )
            out.append(
                torch.zeros(
                    (
                        1,
                        lay["num_v_heads"],
                        lay["key_head_dim"],
                        lay["value_head_dim"],
                    ),
                    dtype=dtype,
                )
            )
        else:
            out.append(
                torch.zeros(
                    (1, lay["num_kv_heads"], n_past, lay["head_dim"]),
                    dtype=dtype,
                )
            )
            out.append(
                torch.zeros(
                    (1, lay["num_kv_heads"], n_past, lay["head_dim"]),
                    dtype=dtype,
                )
            )
    return out


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
        from transformers import AutoProcessor

        print(f"[holo] loading {args.repo}")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.repo, torch_dtype=torch.float32
        ).eval()
        processor = AutoProcessor.from_pretrained(args.repo)
    model.config._attn_implementation = "eager"
    model.model.language_model.config._attn_implementation = "eager"
    conf = model.config
    tc = conf.text_config
    vc = conf.vision_config

    encoder = Qwen35VisionEncoder(model.model.visual).eval()
    decoder = StreamingHybridDecoder(model).eval()

    # ---- build a sample (image, prompt) ----
    merge = vc.spatial_merge_size
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
    if processor is not None and args.image is not None:
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
        torch.manual_seed(0)
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
    # prompt mRoPE positions via HF (offloads the tricky part off the Rust side)
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
    rotary_dim = int(cos.shape[-1])
    # decode-continuation table: sequential text positions after the prompt
    # (all three mRoPE channels equal), so the Rust side just indexes by step.
    n_decode = 128
    start = int(position_ids.max()) + 1
    dec_pos = (
        torch.arange(start, start + n_decode).view(1, 1, -1).expand(3, 1, -1)
    )
    with torch.no_grad():
        cos_tbl, sin_tbl = rotary(hidden0, dec_pos)  # [1, n_decode, rotary_dim]
    neg = torch.finfo(torch.float32).min
    mask = torch.triu(torch.full((seq, seq), neg), diagonal=1).view(
        1, 1, seq, seq
    )

    # ---- export vision encoder ----
    enc_path = args.out / "vision.nnef.tgz"
    enc_args = (pixel_values,)
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
    layout, in_state_names, out_state_names = _state_layout(tc)
    states = _zero_states(tc, 0, torch.float32)
    dec_args = (input_ids, cos, sin, mask, image_embeddings, *states)
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
        if name.startswith(("in_key", "in_value")):
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

    # ---- manifest + sample input ----
    mh_s, mw_s = int(pixel_values.shape[0]), int(pixel_values.shape[1])
    manifest = {
        "repo": args.repo or "dummy",
        "encoder_path": "vision.nnef.tgz",
        "decoder_path": "decoder.nnef.tgz",
        "hidden_size": tc.hidden_size,
        "vocab_size": tc.vocab_size,
        "image_token_id": conf.image_token_id,
        "vision_start_token_id": conf.vision_start_token_id,
        "eos_token_id": getattr(conf, "eos_token_id", None)
        or getattr(tc, "eos_token_id", None),
        "spatial_merge_size": merge,
        "patch_dim": patch_dim,
        "rotary_dim": rotary_dim,
        "layers": layout,
        "decoder_input_order": dec_in_names,
        "decoder_output_order": dec_out_names,
        "sample": {
            "seq": seq,
            "grid_mh": mh_s,
            "grid_mw": mw_s,
            "num_image_tokens": int((input_ids == conf.image_token_id).sum()),
            "n_decode_table": n_decode,
        },
    }
    (args.out / "holo.json").write_text(json.dumps(manifest, indent=2))
    # Flat little-endian .bin files: the Rust demo reads them with std only
    # (no npy/npz dependency). Shapes come from holo.json.
    input_ids.numpy().astype("<i8").tofile(args.out / "input_ids.bin")
    pixel_values.numpy().astype("<f4").tofile(args.out / "pixel_values.bin")
    cos.numpy().astype("<f4").tofile(args.out / "cos.bin")
    sin.numpy().astype("<f4").tofile(args.out / "sin.bin")
    cos_tbl.numpy().astype("<f4").tofile(args.out / "cos_table.bin")
    sin_tbl.numpy().astype("<f4").tofile(args.out / "sin_table.bin")
    n_img = int((input_ids == conf.image_token_id).sum())
    print(f"[holo] wrote holo.json + *.bin to {args.out}")
    print(f"[holo] seq={seq} img_tokens={n_img} layers={len(layout)}")

    if args.verify:
        # Torch reference of the exact loop the Rust binary runs (same in-memory
        # weights), so the demo's tokens can be checked for faithfulness.
        ref = _torch_greedy(
            decoder,
            input_ids,
            cos,
            sin,
            cos_tbl,
            sin_tbl,
            image_embeddings,
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
