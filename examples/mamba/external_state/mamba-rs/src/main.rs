//! Streaming Mamba inference: load NNEF, tokenize a prompt, thread the
//! conv + ssm states across tokens, greedy-decode N new tokens.
//!
//! Mirrors the `wav-cleaner-rs` layout: reads the sidecar JSON
//! manifest emitted by `export.py` so shapes (L, D, K, N, vocab) are
//! not hard-coded per checkpoint.
//!
//! Usage:
//!     cargo run --release -- \
//!         --model mamba130m.nnef.tgz \
//!         --tokenizer tokenizer.json \
//!         --prompt "Once upon a time," \
//!         --max-new-tokens 20

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use tokenizers::Tokenizer;
use tract_nnef::prelude::*;

#[derive(Debug, Deserialize)]
struct Manifest {
    repo: String,
    num_layers: usize,
    intermediate_size: usize,
    conv_kernel: usize,
    state_size: usize,
    vocab_size: usize,
}

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    max_new_tokens: usize,
}

fn print_usage_and_exit() -> ! {
    eprintln!(
        "Usage: mamba --model <path.nnef.tgz> --tokenizer <tokenizer.json> \
         --prompt <text> [--max-new-tokens N]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut argv = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut max_new_tokens: usize = 20;
    while let Some(flag) = argv.next() {
        let val = match flag.as_str() {
            "-h" | "--help" => print_usage_and_exit(),
            _ => argv.next().unwrap_or_else(|| print_usage_and_exit()),
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(val)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(val)),
            "--prompt" => prompt = Some(val),
            "--max-new-tokens" => {
                max_new_tokens = val.parse().unwrap_or_else(|_| print_usage_and_exit())
            }
            _ => print_usage_and_exit(),
        }
    }
    Args {
        model: model.unwrap_or_else(|| print_usage_and_exit()),
        tokenizer: tokenizer.unwrap_or_else(|| print_usage_and_exit()),
        prompt: prompt.unwrap_or_else(|| print_usage_and_exit()),
        max_new_tokens,
    }
}

fn manifest_path_candidates(nnef_path: &Path) -> Vec<PathBuf> {
    let parent = nnef_path.parent().unwrap_or_else(|| Path::new("."));
    let name = nnef_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let mut stem = name.to_string();
    for suffix in [".tgz", ".nnef", ".json"] {
        if stem.ends_with(suffix) {
            stem.truncate(stem.len() - suffix.len());
        }
    }
    vec![
        parent.join(format!("{stem}.json")),
        parent.join(format!("{name}.json")),
    ]
}

fn load_manifest(nnef_path: &Path) -> Result<Manifest, Box<dyn std::error::Error + Send + Sync>> {
    for cand in manifest_path_candidates(nnef_path) {
        if cand.is_file() {
            let bytes = std::fs::read(&cand)?;
            return Ok(serde_json::from_slice(&bytes)?);
        }
    }
    Err(format!(
        "no manifest next to {} (looked at {:?})",
        nnef_path.display(),
        manifest_path_candidates(nnef_path)
    )
    .into())
}

fn load_model(path: &Path) -> TractResult<std::sync::Arc<TypedSimplePlan>> {
    let nnef = tract_nnef::nnef();
    let model = nnef
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

fn argmax_f32(slice: &[f32]) -> usize {
    let mut best_ix = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_ix = i;
        }
    }
    best_ix
}

fn run_step(
    model: &std::sync::Arc<TypedSimplePlan>,
    token_id: i64,
    conv_states: Tensor,
    ssm_states: Tensor,
) -> TractResult<(Vec<f32>, Tensor, Tensor)> {
    let token: Tensor = tract_ndarray::Array1::from_vec(vec![token_id]).into();
    let outputs = tract_nnef::tract_core::plan::SimpleState::new(model)?.run(tvec!(
        token.into_tvalue(),
        conv_states.into_tvalue(),
        ssm_states.into_tvalue(),
    ))?;
    let logits = outputs[0].try_as_plain()?.as_slice::<f32>()?.to_vec();
    let new_conv = outputs[1].clone().into_tensor();
    let new_ssm = outputs[2].clone().into_tensor();
    Ok((logits, new_conv, new_ssm))
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = parse_args();

    let manifest = load_manifest(&args.model)?;
    println!(
        "[mamba] manifest: repo={} L={} D={} K={} N={} vocab={}",
        manifest.repo,
        manifest.num_layers,
        manifest.intermediate_size,
        manifest.conv_kernel,
        manifest.state_size,
        manifest.vocab_size
    );

    println!("[mamba] loading {}", args.model.display());
    let model = load_model(&args.model)?;

    println!("[mamba] loading tokenizer {}", args.tokenizer.display());
    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;

    let encoding = tokenizer.encode(args.prompt.as_str(), false)?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
    println!("[mamba] prompt tokens ({}): {:?}", prompt_ids.len(), prompt_ids);

    let l = manifest.num_layers;
    let d = manifest.intermediate_size;
    let k = manifest.conv_kernel;
    let n = manifest.state_size;

    let mut conv_states: Tensor =
        tract_ndarray::Array4::<f32>::zeros((l, 1, d, k)).into();
    let mut ssm_states: Tensor =
        tract_ndarray::Array4::<f32>::zeros((l, 1, d, n)).into();

    let mut out_ids: Vec<i64> = prompt_ids.clone();
    let mut last_logits: Vec<f32> = Vec::new();

    let start = Instant::now();
    let mut step_ms: Vec<f64> = Vec::new();

    for &tok in &prompt_ids {
        let t0 = Instant::now();
        let (logits, new_conv, new_ssm) =
            run_step(&model, tok, conv_states, ssm_states)?;
        step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        last_logits = logits;
        conv_states = new_conv;
        ssm_states = new_ssm;
    }

    for _ in 0..args.max_new_tokens {
        let next = argmax_f32(&last_logits) as i64;
        out_ids.push(next);
        let t0 = Instant::now();
        let (logits, new_conv, new_ssm) =
            run_step(&model, next, conv_states, ssm_states)?;
        step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        last_logits = logits;
        conv_states = new_conv;
        ssm_states = new_ssm;
    }

    let total_s = start.elapsed().as_secs_f64();
    let total_steps = step_ms.len();
    let median = {
        let mut sorted = step_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };
    let mean = step_ms.iter().sum::<f64>() / total_steps as f64;

    let ids_u32: Vec<u32> = out_ids.iter().map(|&i| i as u32).collect();
    let decoded = tokenizer.decode(&ids_u32, true)?;

    println!("[mamba] decoded: {decoded}");
    println!(
        "[mamba] {} steps in {:.3}s, median {:.1} ms/step, mean {:.1} ms/step",
        total_steps, total_s, median, mean
    );
    Ok(())
}
