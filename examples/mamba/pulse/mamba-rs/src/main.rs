//! Pulse-mode Mamba streaming.
//!
//! Loads a NNEF artifact whose sequence axis is declared symbolic
//! (`S`), runs it through tract's pulse pipeline (one timestep per
//! pulse), and streams a prompt + greedy decoding through. Internal
//! conv buffers and SSM state are managed by tract; the caller never
//! sees them.
//!
//! Usage:
//!     cargo run --release -- \
//!         --model mamba_pulse.nnef.tgz \
//!         --tokenizer tokenizer.json \
//!         --prompt "Once upon a time," \
//!         --max-new-tokens 20

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use tokenizers::Tokenizer;
use tract_nnef::prelude::*;
use tract_pulse::internal::*;

#[derive(Debug, Deserialize)]
struct Manifest {
    repo: String,
    num_layers: usize,
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
        "Usage: mamba-pulse --model <path.nnef.tgz> --tokenizer <tokenizer.json> \
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
    let name = nnef_path.file_name().and_then(|s| s.to_str()).unwrap_or("");
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

fn load_manifest(
    nnef_path: &Path,
) -> Result<Manifest, Box<dyn std::error::Error + Send + Sync>> {
    for cand in manifest_path_candidates(nnef_path) {
        if cand.is_file() {
            let bytes = std::fs::read(&cand)?;
            return Ok(serde_json::from_slice(&bytes)?);
        }
    }
    Err(format!("no manifest next to {}", nnef_path.display()).into())
}

fn build_pulsed_runnable(
    path: &Path,
) -> TractResult<TypedSimplePlan<TypedModel>> {
    let mut nnef = tract_nnef::nnef();
    nnef.enable_tract_core();
    let typed = nnef.model_for_path(path)?;
    let sym = typed.symbols.sym("S");
    let pulsed = PulsedModel::new(&typed, sym, &1.to_dim())?;
    let typed = pulsed.into_typed()?.into_decluttered()?.into_optimized()?;
    typed.into_runnable()
}

fn argmax_f32(slice: &[f32]) -> usize {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in slice.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = parse_args();
    let manifest = load_manifest(&args.model)?;
    println!(
        "[mamba-pulse] manifest: repo={} L={} vocab={}",
        manifest.repo, manifest.num_layers, manifest.vocab_size
    );

    println!("[mamba-pulse] loading {}", args.model.display());
    let model = build_pulsed_runnable(&args.model)?;

    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;
    let encoding = tokenizer.encode(args.prompt.as_str(), false)?;
    let prompt_ids: Vec<i64> =
        encoding.get_ids().iter().map(|&i| i as i64).collect();
    println!(
        "[mamba-pulse] prompt tokens ({}): {:?}",
        prompt_ids.len(),
        prompt_ids
    );

    // One streaming session = one fresh state. State carries conv +
    // ssm internally across run() calls.
    let plan = std::sync::Arc::new(model);
    let mut state = TypedSimpleState::new(plan.clone())?;

    let mut out_ids: Vec<i64> = prompt_ids.clone();
    let mut last_logits: Vec<f32> = Vec::new();
    let mut step_ms: Vec<f64> = Vec::new();
    let start = Instant::now();

    let run_one = |st: &mut TypedSimpleState<TypedModel, std::sync::Arc<TypedSimplePlan<TypedModel>>>,
                       tok: i64|
     -> TractResult<Vec<f32>> {
        let inp: Tensor =
            tract_ndarray::Array2::from_shape_vec((1, 1), vec![tok])?.into();
        let outputs = st.run(tvec!(inp.into_tvalue()))?;
        Ok(outputs[0].as_slice::<f32>()?.to_vec())
    };

    for &tok in &prompt_ids {
        let t0 = Instant::now();
        last_logits = run_one(&mut state, tok)?;
        step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    for _ in 0..args.max_new_tokens {
        let next = argmax_f32(&last_logits) as i64;
        out_ids.push(next);
        let t0 = Instant::now();
        last_logits = run_one(&mut state, next)?;
        step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
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

    println!("[mamba-pulse] decoded: {decoded}");
    println!(
        "[mamba-pulse] {} steps in {:.3}s, median {:.1} ms/step, mean {:.1} ms/step",
        total_steps, total_s, median, mean
    );
    Ok(())
}
