//! Minimal WAV cleaner: load NNEF, stream frames through DPDFNet, write a
//! cleaned WAV.
//!
//! The NNEF artifact (`dpdfnet2.nnef.tgz`) contains the full pipeline:
//! rolling-STFT, DPDFNet inference, iFFT + overlap-add. We thread the four
//! state tensors across frames; everything else lives in the graph.
//!
//! Usage:
//!     cargo run --release -- --model dpdfnet2.nnef.tgz --in noisy.wav --out clean.wav
//!
//! Input WAV must be 16 kHz mono int16. We read frames of `HOP_SIZE` samples,
//! feed them with the four state tensors, and write the model's per-frame
//! output back into a WAV of the same format. After the input ends we feed
//! `--tail-frames` of silence so the OLA buffer flushes the last samples.

use std::path::PathBuf;
use std::time::Instant;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use tract_nnef::prelude::*;

const SAMPLE_RATE: u32 = 16_000;
const HOP_SIZE: usize = 160;
const N_FFT: usize = 320;
const NN_STATE_SIZE: usize = 45_424;

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
    // Tail = win_len / hop_size = 2 frames is enough to flush the OLA buffer.
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

fn load_model(path: &PathBuf) -> TractResult<TypedSimplePlan<TypedModel>> {
    let mut nnef = tract_nnef::nnef();
    nnef.enable_tract_core();
    let model = nnef
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

fn read_wav_mono_f32(path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(format!(
            "expected mono input WAV, got {} channels",
            spec.channels
        )
        .into());
    }
    if spec.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "expected {} Hz input WAV, got {} Hz",
            SAMPLE_RATE, spec.sample_rate
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
    path: &PathBuf,
    samples: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
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
    println!("[wav-cleaner] loading {}", args.model.display());
    let model = load_model(&args.model)?;

    println!("[wav-cleaner] reading {}", args.input.display());
    let mut samples = read_wav_mono_f32(&args.input)?;
    let original_len = samples.len();
    // Right-pad with `tail_frames * HOP_SIZE` zeros so the OLA flushes.
    samples.resize(original_len + args.tail_frames * HOP_SIZE, 0.0);

    let mut stft_buf = vec![0.0f32; N_FFT];
    let mut nn_state = vec![0.0f32; NN_STATE_SIZE];
    let mut ola_buf = vec![0.0f32; N_FFT];

    let n_frames = samples.len() / HOP_SIZE;
    let mut out_samples: Vec<f32> = Vec::with_capacity(n_frames * HOP_SIZE);

    let start = Instant::now();
    let mut frame_buf = vec![0.0f32; HOP_SIZE];
    for f in 0..n_frames {
        let begin = f * HOP_SIZE;
        frame_buf.copy_from_slice(&samples[begin..begin + HOP_SIZE]);
        let inputs = make_input(&frame_buf, &stft_buf, &nn_state, &ola_buf)?;
        let outputs = model.run(inputs)?;
        let enhanced = outputs[0].as_slice::<f32>()?;
        out_samples.extend_from_slice(enhanced);
        stft_buf.copy_from_slice(outputs[1].as_slice::<f32>()?);
        nn_state.copy_from_slice(outputs[2].as_slice::<f32>()?);
        ola_buf.copy_from_slice(outputs[3].as_slice::<f32>()?);
    }
    let elapsed = start.elapsed();

    // Drop the leading frames of OLA warm-up (one frame of zeros lag).
    // We keep `original_len` samples starting at offset 0; tail samples
    // beyond `original_len` are the flush region from the right padding.
    out_samples.truncate(original_len);

    let audio_seconds = original_len as f64 / SAMPLE_RATE as f64;
    let elapsed_seconds = elapsed.as_secs_f64();
    let rtfx = audio_seconds / elapsed_seconds.max(1e-9);
    let per_frame_ms = elapsed_seconds * 1000.0 / n_frames as f64;
    println!(
        "[wav-cleaner] cleaned {} frames ({:.2}s audio) in {:.3}s -> {:.2}x real-time, {:.3} ms/frame",
        n_frames, audio_seconds, elapsed_seconds, rtfx, per_frame_ms
    );

    println!("[wav-cleaner] writing {}", args.output.display());
    write_wav_mono_int16(&args.output, &out_samples)?;
    Ok(())
}
