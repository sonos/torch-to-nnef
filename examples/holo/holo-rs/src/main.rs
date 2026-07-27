//! Single-binary tract demo of the Qwen3.5 dense VLM (Hcompany/Holo-3.1).
//!
//! Loads the two NNEF graphs produced by `../export.py` (the vision tower and
//! the STREAMING hybrid gated-delta-net decoder) plus their `holo.json`
//! manifest, then runs the full pipeline in tract:
//!
//!   pixel_values --[vision encoder]--> image embeddings
//!   (input_ids + image embeddings + positions + zero states)
//!       --[decoder, prefill]--> logits + per-layer states
//!   greedy loop: (next token + carried states) --[decoder, S=1]--> ...
//!
//! Each decoder layer threads its own state: a gated-delta-net layer carries a
//! streaming conv state + a matrix recurrent state; a full-attention layer
//! carries a KV cache. One dynamic graph serves both prefill and decode.
//!
//! Usage:
//!     cargo run --release -- --dir ../exp --max-new-tokens 16

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tract_nnef::prelude::*;
use tract_nnef::tract_core::plan::SimpleState;

const NEG: f32 = -1.0e30;

#[derive(Debug, Deserialize)]
struct Layer {
    kind: String,
    #[serde(default)]
    conv_dim: usize,
    #[serde(default)]
    conv_state_width: usize,
    #[serde(default)]
    num_v_heads: usize,
    #[serde(default)]
    key_head_dim: usize,
    #[serde(default)]
    value_head_dim: usize,
    #[serde(default)]
    num_kv_heads: usize,
    #[serde(default)]
    head_dim: usize,
}

#[derive(Debug, Deserialize)]
struct Sample {
    seq: usize,
    grid_mh: usize,
    grid_mw: usize,
    num_image_tokens: usize,
    prompt_max_pos: i64,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    repo: String,
    encoder_path: String,
    decoder_path: String,
    hidden_size: usize,
    vocab_size: usize,
    spatial_merge_size: usize,
    patch_dim: usize,
    #[serde(default)]
    eos_token_id: Option<i64>,
    layers: Vec<Layer>,
    sample: Sample,
}

struct Args {
    dir: PathBuf,
    max_new_tokens: usize,
}

fn parse_args() -> Args {
    let mut argv = std::env::args().skip(1);
    let mut dir: Option<PathBuf> = None;
    let mut max_new_tokens = 16usize;
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: holo --dir <export_dir> [--max-new-tokens N]");
                std::process::exit(0);
            }
            "--dir" => dir = argv.next().map(PathBuf::from),
            "--max-new-tokens" => {
                max_new_tokens = argv.next().and_then(|v| v.parse().ok()).unwrap_or(16)
            }
            other => {
                eprintln!("unknown flag: {other}");
                std::process::exit(2);
            }
        }
    }
    Args {
        dir: dir.unwrap_or_else(|| {
            eprintln!("--dir <export_dir> is required");
            std::process::exit(2);
        }),
        max_new_tokens,
    }
}

type Plan = std::sync::Arc<TypedSimplePlan>;

fn load_model(path: &Path) -> TractResult<Plan> {
    tract_nnef::nnef()
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()
}

fn read_bin_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_bin_i64(path: &Path) -> std::io::Result<Vec<i64>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn i64_tensor(shape: &[usize], data: Vec<i64>) -> TractResult<Tensor> {
    Ok(tract_ndarray::ArrayD::from_shape_vec(shape.to_vec(), data)?.into())
}

fn f32_tensor(shape: &[usize], data: Vec<f32>) -> TractResult<Tensor> {
    Ok(tract_ndarray::ArrayD::from_shape_vec(shape.to_vec(), data)?.into())
}

/// Per-layer zero states, in `config.layer_types` order, for `n_past` KV depth.
fn zero_states(layers: &[Layer], n_past: usize) -> TractResult<Vec<Tensor>> {
    let mut out = Vec::new();
    for l in layers {
        if l.kind == "gdn" {
            out.push(f32_tensor(
                &[1, l.conv_dim, l.conv_state_width],
                vec![0.0; l.conv_dim * l.conv_state_width],
            )?);
            let n = l.num_v_heads * l.key_head_dim * l.value_head_dim;
            out.push(f32_tensor(
                &[1, l.num_v_heads, l.key_head_dim, l.value_head_dim],
                vec![0.0; n],
            )?);
        } else {
            let n = l.num_kv_heads * n_past * l.head_dim;
            out.push(f32_tensor(
                &[1, l.num_kv_heads, n_past, l.head_dim],
                vec![0.0; n],
            )?);
            out.push(f32_tensor(
                &[1, l.num_kv_heads, n_past, l.head_dim],
                vec![0.0; n],
            )?);
        }
    }
    Ok(out)
}

/// Causal additive mask `[1, 1, s, past + s]` (0 visible, NEG masked).
fn causal_mask(s: usize, past: usize) -> TractResult<Tensor> {
    let total = past + s;
    let mut data = vec![0.0f32; s * total];
    for i in 0..s {
        for j in 0..total {
            // query i sits at absolute position past + i.
            if j > past + i {
                data[i * total + j] = NEG;
            }
        }
    }
    f32_tensor(&[1, 1, s, total], data)
}

fn argmax(slice: &[f32]) -> usize {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in slice.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0
}

/// Cast a tensor to `target` (no-op if it already matches).
fn to_dt(t: Tensor, target: DatumType) -> TractResult<Tensor> {
    if t.datum_type() == target {
        Ok(t)
    } else {
        Ok(t.cast_to_dt(target)?.into_owned())
    }
}

/// Cast each input to the dtype the plan expects for that position, so the
/// f32 tensors this demo builds run against an f32 OR an f16 graph unchanged.
fn cast_inputs(plan: &Plan, inputs: TVec<Tensor>) -> TractResult<TVec<TValue>> {
    let mut out: TVec<TValue> = tvec!();
    for (ix, t) in inputs.into_iter().enumerate() {
        let want = plan.model().input_fact(ix)?.datum_type;
        out.push(to_dt(t, want)?.into_tvalue());
    }
    Ok(out)
}

/// Run the decoder once. `inputs` are in `decoder_input_order`:
/// input_ids, position_ids, mask, image_embeddings, then per-layer states.
/// Returns (logits, new_states).
fn run_decoder(
    decoder: &Plan,
    input_ids: Tensor,
    position_ids: Tensor,
    mask: Tensor,
    image_embeddings: Tensor,
    states: Vec<Tensor>,
) -> TractResult<(Vec<f32>, Vec<Tensor>)> {
    let mut raw: TVec<Tensor> =
        tvec!(input_ids, position_ids, mask, image_embeddings);
    raw.extend(states);
    let outputs = SimpleState::new(decoder)?.run(cast_inputs(decoder, raw)?)?;
    // logits may come back f16 on a half-precision graph; read them as f32.
    let logits = to_dt(outputs[0].clone().into_tensor(), DatumType::F32)?
        .try_as_plain()?
        .as_slice::<f32>()?
        .to_vec();
    let new_states = outputs[1..]
        .iter()
        .map(|t| t.clone().into_tensor())
        .collect();
    Ok((logits, new_states))
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = parse_args();
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(args.dir.join("holo.json"))?)?;
    println!(
        "[holo] repo={} hidden={} vocab={} layers={} (img_tokens={})",
        manifest.repo,
        manifest.hidden_size,
        manifest.vocab_size,
        manifest.layers.len(),
        manifest.sample.num_image_tokens
    );

    let encoder = load_model(&args.dir.join(&manifest.encoder_path))?;
    let decoder = load_model(&args.dir.join(&manifest.decoder_path))?;
    println!("[holo] loaded vision encoder + streaming decoder");

    // ---- inputs from the export sample ----
    let merge = manifest.spatial_merge_size;
    let (mh, mw) = (manifest.sample.grid_mh, manifest.sample.grid_mw);
    let pd = manifest.patch_dim;
    let pixel_values = f32_tensor(
        &[mh, mw, merge, merge, pd],
        read_bin_f32(&args.dir.join("pixel_values.bin"))?,
    )?;
    let seq = manifest.sample.seq;
    let input_ids_vec = read_bin_i64(&args.dir.join("input_ids.bin"))?;
    // The mRoPE t/h/w position layout for the prompt (image tokens get a 2-D
    // grid) is computed host-side by get_rope_index; the rotary itself runs
    // in-graph, so the runtime only feeds integer positions.
    let position_ids_vec = read_bin_i64(&args.dir.join("position_ids.bin"))?;

    // ---- vision encoder ----
    let enc_out = encoder.run(cast_inputs(&encoder, tvec!(pixel_values))?)?;
    let image_embeddings = enc_out[0].clone().into_tensor();
    println!("[holo] image embeddings: {:?}", image_embeddings.shape());

    // ---- decoder prefill ----
    let input_ids = i64_tensor(&[1, seq], input_ids_vec)?;
    let position_ids = i64_tensor(&[3, 1, seq], position_ids_vec)?;
    let mask = causal_mask(seq, 0)?;
    let (mut logits, mut states) = run_decoder(
        &decoder,
        input_ids,
        position_ids,
        mask,
        image_embeddings,
        zero_states(&manifest.layers, 0)?,
    )?;
    let vocab = manifest.vocab_size;
    let mut next = argmax(&logits[(seq - 1) * vocab..seq * vocab]) as i64;

    // ---- greedy decode ----
    let empty_img = f32_tensor(&[0, manifest.hidden_size], vec![])?;
    let mut generated: Vec<i64> = vec![next];
    let mut past = seq;
    for step in 0..args.max_new_tokens {
        if manifest.eos_token_id == Some(next) {
            break;
        }
        // text continuation: all three mRoPE channels share the next position
        let pos = manifest.sample.prompt_max_pos + 1 + step as i64;
        let position_ids = i64_tensor(&[3, 1, 1], vec![pos, pos, pos])?;
        let mask = causal_mask(1, past)?;
        let out = run_decoder(
            &decoder,
            i64_tensor(&[1, 1], vec![next])?,
            position_ids,
            mask,
            empty_img.clone(),
            states,
        )?;
        logits = out.0;
        states = out.1;
        next = argmax(&logits[0..vocab]) as i64;
        generated.push(next);
        past += 1;
    }

    println!(
        "[holo] generated {} token ids: {:?}",
        generated.len(),
        generated
    );
    println!(
        "[holo] (decode a real checkpoint's tokens with its tokenizer to read \
         the UI-grounding coordinates)"
    );
    Ok(())
}
