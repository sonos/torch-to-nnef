//! Minimal WAV cleaner: load NNEF, stream frames through DPDFNet, write a
//! cleaned WAV. Reads the sidecar JSON manifest written by `export.py`
//! so it works for any DPDFNet variant (16 kHz or 48 kHz HR) without
//! recompiling.
//!
//! The NNEF artifact contains the full pipeline: rolling-STFT, DPDFNet
//! inference, iFFT + overlap-add. We thread the four state tensors
//! across frames; everything else lives in the graph.
//!
//! Usage:
//!     cargo run --release -- --model dpdfnet2.nnef.tgz --in noisy.wav --out clean.wav
//!
//! Input WAV must match the variant's sample rate (16 kHz for the
//! `dpdfnet*` variants, 48 kHz for the `*_48khz_hr` ones), mono int16 or
//! float32. After the input ends we feed `--tail-frames` of silence so
//! the OLA buffer flushes the last samples.

use std::path::{Path, PathBuf};
use std::time::Instant;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use serde::Deserialize;
use tract_nnef::prelude::*;

#[derive(Debug, Deserialize)]
struct Manifest {
    variant: String,
    sample_rate: u32,
    n_fft: usize,
    hop_size: usize,
    state_size: usize,
}

fn print_usage_and_exit() -> ! {
    eprintln!(
        "Usage: wav-cleaner --model <path.nnef.tgz> --in <noisy.wav> --out <clean.wav> [--tail-frames N]"
    );
    std::process::exit(2);
}

struct Args {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    tail_frames: usize,
}

fn parse_args() -> Args {
    let mut argv = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut tail_frames: usize = 2;
    while let Some(flag) = argv.next() {
        let val = match flag.as_str() {
            "-h" | "--help" => print_usage_and_exit(),
            _ => argv.next().unwrap_or_else(|| print_usage_and_exit()),
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(val)),
            "--in" => input = Some(PathBuf::from(val)),
            "--out" => output = Some(PathBuf::from(val)),
            "--tail-frames" => {
                tail_frames = val.parse().unwrap_or_else(|_| print_usage_and_exit())
            }
            _ => print_usage_and_exit(),
        }
    }
    Args {
        model: model.unwrap_or_else(|| print_usage_and_exit()),
        input: input.unwrap_or_else(|| print_usage_and_exit()),
        output: output.unwrap_or_else(|| print_usage_and_exit()),
        tail_frames,
    }
}

/// Locate the sidecar manifest next to `nnef_path`.
///
/// `export.py` writes `<variant>.json` next to the artifact; for
/// `dpdfnet2.nnef.tgz` that's `dpdfnet2.json`. We strip the multi-suffix
/// extensions and try the bare-stem path first, then a couple of
/// fallbacks (e.g. the user staged everything with the full suffix
/// chain preserved).
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

fn load_manifest(nnef_path: &Path) -> Result<Manifest, Box<dyn std::error::Error>> {
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

fn load_model(path: &Path) -> TractResult<TypedSimplePlan<TypedModel>> {
    let mut nnef = tract_nnef::nnef();
    nnef.enable_tract_core();
    let model = nnef
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

fn read_wav_mono_f32(
    path: &Path,
    expected_sr: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(format!(
            "expected mono input WAV, got {} channels",
            spec.channels
        )
        .into());
    }
    if spec.sample_rate != expected_sr {
        return Err(format!(
            "expected {} Hz input WAV (matches model variant), got {} Hz",
            expected_sr, spec.sample_rate
        )
        .into());
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32_768.0))
            .collect::<Result<_, _>>()?,
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<_, _>>()?,
        (fmt, bits) => {
            return Err(format!(
                "unsupported WAV sample format ({:?}, {} bits)",
                fmt, bits
            )
            .into());
        }
    };
    Ok(samples)
}

fn write_wav_mono_int16(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32_767.0) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

fn make_input(
    audio_frame: &[f32],
    stft_buf: &[f32],
    nn_state: &[f32],
    ola_buf: &[f32],
) -> TractResult<TVec<TValue>> {
    let frame: Tensor = tract_ndarray::Array1::from_vec(audio_frame.to_vec()).into();
    let stft: Tensor = tract_ndarray::Array1::from_vec(stft_buf.to_vec()).into();
    let nn: Tensor = tract_ndarray::Array1::from_vec(nn_state.to_vec()).into();
    let ola: Tensor = tract_ndarray::Array1::from_vec(ola_buf.to_vec()).into();
    Ok(tvec!(
        frame.into_tvalue(),
        stft.into_tvalue(),
        nn.into_tvalue(),
        ola.into_tvalue(),
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    println!("[wav-cleaner] loading manifest for {}", args.model.display());
    let manifest = load_manifest(&args.model)?;
    println!(
        "[wav-cleaner]   variant={} sr={}Hz hop={} n_fft={} state={}",
        manifest.variant,
        manifest.sample_rate,
        manifest.hop_size,
        manifest.n_fft,
        manifest.state_size
    );

    println!("[wav-cleaner] loading {}", args.model.display());
    let model = load_model(&args.model)?;

    println!("[wav-cleaner] reading {}", args.input.display());
    let mut samples = read_wav_mono_f32(&args.input, manifest.sample_rate)?;
    let original_len = samples.len();
    samples.resize(original_len + args.tail_frames * manifest.hop_size, 0.0);

    let mut stft_buf = vec![0.0f32; manifest.n_fft];
    let mut nn_state = vec![0.0f32; manifest.state_size];
    let mut ola_buf = vec![0.0f32; manifest.n_fft];

    let n_frames = samples.len() / manifest.hop_size;
    let mut out_samples: Vec<f32> = Vec::with_capacity(n_frames * manifest.hop_size);

    let start = Instant::now();
    let mut frame_buf = vec![0.0f32; manifest.hop_size];
    for f in 0..n_frames {
        let begin = f * manifest.hop_size;
        frame_buf.copy_from_slice(&samples[begin..begin + manifest.hop_size]);
        let inputs = make_input(&frame_buf, &stft_buf, &nn_state, &ola_buf)?;
        let outputs = model.run(inputs)?;
        let enhanced = outputs[0].as_slice::<f32>()?;
        out_samples.extend_from_slice(enhanced);
        stft_buf.copy_from_slice(outputs[1].as_slice::<f32>()?);
        nn_state.copy_from_slice(outputs[2].as_slice::<f32>()?);
        ola_buf.copy_from_slice(outputs[3].as_slice::<f32>()?);
    }
    let elapsed = start.elapsed();
    out_samples.truncate(original_len);

    let audio_seconds = original_len as f64 / manifest.sample_rate as f64;
    let elapsed_seconds = elapsed.as_secs_f64();
    let rtfx = audio_seconds / elapsed_seconds.max(1e-9);
    let per_frame_ms = elapsed_seconds * 1000.0 / n_frames as f64;
    println!(
        "[wav-cleaner] cleaned {} frames ({:.2}s audio) in {:.3}s -> {:.2}x real-time, {:.3} ms/frame",
        n_frames, audio_seconds, elapsed_seconds, rtfx, per_frame_ms
    );

    println!("[wav-cleaner] writing {}", args.output.display());
    write_wav_mono_int16(&args.output, &out_samples, manifest.sample_rate)?;
    Ok(())
}
