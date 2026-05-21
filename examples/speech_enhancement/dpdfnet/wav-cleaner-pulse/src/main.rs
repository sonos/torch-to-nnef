//! WAV cleaner using tract's pulse mode end-to-end.
//!
//! Loads a NNEF artifact that declares a streaming axis (`STREAM`) and
//! converts it to a pulsed model. Audio is streamed in chunks of `pulse`
//! samples; tract handles all internal buffering (STFT overlap, conv
//! lookahead, GRU state) automatically.
//!
//! Usage:
//!     cargo run --release -- --model dpdfnet_pulse.nnef.tgz --in noisy.wav --out clean.wav --pulse 320
//!
//! `pulse` must be a multiple of the STFT hop. The first
//! `pulse.delay * pulse` output samples are dropped (warm-up).

use std::path::{Path, PathBuf};
use std::time::Instant;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::sync::Arc;

use tract_nnef::prelude::*;
use tract_pulse::model::{PulsedModel, PulsedModelExt};
use tract_pulse::WithPulse;

const STREAM_SYMBOL: &str = "STREAM";
const SAMPLE_RATE: u32 = 16_000;

struct Args {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    pulse: usize,
}

fn print_usage_and_exit() -> ! {
    eprintln!(
        "Usage: wav-cleaner-pulse --model <path.nnef.tgz> --in <noisy.wav> --out <clean.wav> [--pulse N]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut argv = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut pulse: usize = 320;
    while let Some(flag) = argv.next() {
        let val = match flag.as_str() {
            "-h" | "--help" => print_usage_and_exit(),
            _ => argv.next().unwrap_or_else(|| print_usage_and_exit()),
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(val)),
            "--in" => input = Some(PathBuf::from(val)),
            "--out" => output = Some(PathBuf::from(val)),
            "--pulse" => {
                pulse = val.parse().unwrap_or_else(|_| print_usage_and_exit())
            }
            _ => print_usage_and_exit(),
        }
    }
    Args {
        model: model.unwrap_or_else(|| print_usage_and_exit()),
        input: input.unwrap_or_else(|| print_usage_and_exit()),
        output: output.unwrap_or_else(|| print_usage_and_exit()),
        pulse,
    }
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
            "expected {} Hz input WAV, got {} Hz",
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

fn load_pulsed(
    model_path: &Path,
    pulse: usize,
) -> TractResult<(Arc<TypedRunnableModel>, usize)> {
    let nnef = tract_nnef::nnef().with_pulse();
    let typed = nnef.model_for_path(model_path)?;
    let sym = typed.symbols.sym(STREAM_SYMBOL);
    let pulsed = PulsedModel::new(&typed, sym, &(pulse as i64).to_dim())?;
    let typed_again = pulsed.into_typed()?.into_decluttered()?;
    // Pulse delay (in output samples). Recorded as a 1-element i64 tensor
    // under the `pulse.delay` property. Cow<Tensor> deref into Tensor
    // so `.as_slice::<i64>()` works once cast.
    let delay = typed_again
        .properties
        .get("pulse.delay")
        .and_then(|t| t.cast_to_scalar::<i64>().ok())
        .map(|v| v as usize)
        .unwrap_or(0);
    let runnable = typed_again.into_optimized()?.into_runnable()?;
    Ok((runnable, delay))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    println!(
        "[wav-cleaner-pulse] loading {} with pulse={}",
        args.model.display(),
        args.pulse
    );
    let (model, pulse_delay) = load_pulsed(&args.model, args.pulse)?;
    println!("[wav-cleaner-pulse]   pulse.delay = {} samples", pulse_delay);

    println!("[wav-cleaner-pulse] reading {}", args.input.display());
    let mut samples = read_wav_mono_f32(&args.input, SAMPLE_RATE)?;
    let original_len = samples.len();
    let pad_tail = pulse_delay;
    samples.resize(original_len + pad_tail, 0.0);
    let pulse = args.pulse;
    let needed = samples.len().div_ceil(pulse) * pulse;
    samples.resize(needed, 0.0);

    let n_chunks = samples.len() / pulse;
    let mut out_samples: Vec<f32> = Vec::with_capacity(n_chunks * pulse);

    let start = Instant::now();
    let mut state = model.spawn()?;
    for c in 0..n_chunks {
        let begin = c * pulse;
        let chunk = tract_ndarray::Array2::from_shape_vec(
            (1, pulse),
            samples[begin..begin + pulse].to_vec(),
        )?
        .into_tensor();
        let outputs = state.run(tvec!(chunk.into_tvalue()))?;
        let tensor: &Tensor = &outputs[0];
        let view = tensor.to_plain_array_view::<f32>()?;
        out_samples.extend_from_slice(view.as_slice().unwrap());
    }
    let elapsed = start.elapsed();

    if pulse_delay <= out_samples.len() {
        out_samples.drain(0..pulse_delay);
    }
    out_samples.truncate(original_len);

    let audio_seconds = original_len as f64 / SAMPLE_RATE as f64;
    let elapsed_seconds = elapsed.as_secs_f64();
    let rtfx = audio_seconds / elapsed_seconds.max(1e-9);
    let per_chunk_ms = elapsed_seconds * 1000.0 / n_chunks as f64;
    println!(
        "[wav-cleaner-pulse] {} chunks of {} samples ({:.2}s audio) in {:.3}s -> {:.2}x real-time, {:.3} ms/chunk",
        n_chunks, pulse, audio_seconds, elapsed_seconds, rtfx, per_chunk_ms
    );

    println!("[wav-cleaner-pulse] writing {}", args.output.display());
    write_wav_mono_int16(&args.output, &out_samples, SAMPLE_RATE)?;
    Ok(())
}
