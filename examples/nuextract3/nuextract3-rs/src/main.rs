//! Single-binary tract demo of NuExtract3 image-to-Markdown generation.
//!
//! Loads the two NNEF graphs produced by `../export.py` (the vision tower and
//! the STREAMING hybrid gated-delta-net decoder) plus their `nuextract3.json`
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
//!     cargo run --release -- --dir ../exp --max-new-tokens 256

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tokenizers::Tokenizer;
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
                eprintln!("Usage: nuextract3 --dir <export_dir> [--max-new-tokens N]");
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

/// Read a sample input written by export.py as a NNEF `.dat` tensor. The file
/// is self-describing (shape + dtype), so tract hands back a typed `Tensor`
/// with no manual byte parsing or manifest-driven shapes.
fn read_dat(path: &Path) -> TractResult<Tensor> {
    let file = std::fs::File::open(path)?;
    tract_nnef::tensors::read_tensor(std::io::BufReader::new(file))
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
    let mut raw: TVec<Tensor> = tvec!(input_ids, position_ids, mask, image_embeddings);
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

fn decode(tokenizer: &Tokenizer, ids: &[i64]) -> Result<String, tokenizers::Error> {
    let ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    tokenizer.decode(&ids, true)
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = parse_args();
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(args.dir.join("nuextract3.json"))?)?;
    let tokenizer_path = args.dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Some(Tokenizer::from_file(tokenizer_path)?)
    } else {
        None
    };
    println!(
        "[nuextract3] repo={} hidden={} vocab={} layers={} (img_tokens={})",
        manifest.repo,
        manifest.hidden_size,
        manifest.vocab_size,
        manifest.layers.len(),
        manifest.sample.num_image_tokens
    );

    let encoder = load_model(&args.dir.join(&manifest.encoder_path))?;
    let decoder = load_model(&args.dir.join(&manifest.decoder_path))?;
    println!("[nuextract3] loaded vision encoder + streaming decoder");

    // ---- inputs from the export sample (self-describing NNEF .dat tensors) ----
    // pixel_values [MH, MW, merge, merge, patch_dim]; input_ids [1, S]; the
    // prompt mRoPE positions [3, 1, S] (image span laid out host-side by
    // get_rope_index; the rotary itself runs in-graph). Shapes come from the
    // tensors, not the manifest.
    let pixel_values = read_dat(&args.dir.join("pixel_values.dat"))?;
    let input_ids = read_dat(&args.dir.join("input_ids.dat"))?;
    let position_ids = read_dat(&args.dir.join("position_ids.dat"))?;
    let seq = input_ids.shape()[1];

    // ---- vision encoder ----
    let enc_out = encoder.run(cast_inputs(&encoder, tvec!(pixel_values))?)?;
    let image_embeddings = enc_out[0].clone().into_tensor();
    println!(
        "[nuextract3] image embeddings: {:?}",
        image_embeddings.shape()
    );

    // ---- decoder prefill ----
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
    let mut generated: Vec<i64> = if args.max_new_tokens == 0 {
        Vec::new()
    } else {
        vec![next]
    };
    let mut past = seq;
    for step in 0..args.max_new_tokens.saturating_sub(1) {
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
        "[nuextract3] generated {} token ids: {:?}",
        generated.len(),
        generated
    );
    if let Some(tokenizer) = tokenizer {
        let markdown = decode(&tokenizer, &generated)?;
        println!("[nuextract3] decoded Markdown:\n{markdown}");
    } else {
        println!("[nuextract3] tokenizer.json not found; dummy exports only print token ids");
    }
    Ok(())
}
