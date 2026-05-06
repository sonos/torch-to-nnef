//! End-to-end Pocket-TTS demo via tract.
//!
//! Loads the NNEF graphs exported from `examples/tts/pocket_tts/`
//! (`flow_lm_init`, `flow_lm_step`, `flow_net`, plus an audio-decode graph
//! -- either ``mimi_decode.nnef.tgz`` for the full Mimi chain or
//! ``decoder.nnef.tgz`` for the SEANet-only mini config), threads them
//! together with a SentencePiece tokenizer + a voice prompt `.dat`, and
//! writes a 24 kHz WAV.
//!
//! No Python or external decoder process is used at runtime: the binary
//! plus its asset directory (``models/``, ``voices/``, ``tokenizer.model``)
//! is everything needed to synthesise speech.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[cfg(not(target_os = "macos"))]
use anyhow::bail;
use anyhow::{Context, Result, anyhow};
use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use sentencepiece::SentencePieceProcessor;
use tract_core::transform::ModelTransform;
use tract_extra::WithTractExtra;
use tract_nnef::framework::Nnef;
use tract_nnef::prelude::*;
use tract_pulse::WithPulse;
use tract_transformers::WithTractTransformers;

#[cfg(target_os = "macos")]
use tract_metal::MetalTransform;

#[derive(Parser, Debug)]
#[command(about = "Pocket-TTS demo through tract", long_about = None)]
struct Args {
    /// Directory holding the exported NNEF graphs (flow_lm_init.nnef.tgz,
    /// flow_lm_step.nnef.tgz, flow_net.nnef.tgz, plus either
    /// mimi_decode.nnef.tgz for the full Mimi chain or decoder.nnef.tgz for
    /// the SEANet-only mini config).
    #[arg(long, default_value = "models")]
    models: PathBuf,
    /// Voice prompt tensor (output of `bake_voice.py`).
    #[arg(long, default_value = "voices/alba.dat")]
    voice: PathBuf,
    /// SentencePiece tokenizer model (the same one Pocket-TTS ships with
    /// each checkpoint). When omitted, ``--tokens`` is required instead.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Text to synthesise. Requires ``--tokenizer``.
    #[arg(long, conflicts_with = "tokens")]
    text: Option<String>,
    /// Comma-separated raw token IDs (use when no tokenizer is available, or
    /// for the mini demo).
    #[arg(long, value_delimiter = ',')]
    tokens: Option<Vec<i64>>,
    /// Output WAV path.
    #[arg(long, default_value = "out.wav")]
    out: PathBuf,
    /// Maximum number of audio frames to generate (the autoregressive loop
    /// also stops early on EOS).
    #[arg(long, default_value = "32")]
    max_frames: usize,
    /// LSD decode steps per audio frame (call flow_net this many times).
    /// Pocket-TTS' own default is ``1``.
    #[arg(long, default_value = "1")]
    lsd_steps: usize,
    /// EOS logit threshold above which the loop terminates. Pocket-TTS'
    /// own CLI default is ``-4.0`` (raw logit; ``out_eos`` is a Linear
    /// layer, not a sigmoid). With ``mimi_decode.nnef.tgz`` traced at
    /// dynamic ``T_LATENT`` the loop can stop on real EOS.
    #[arg(long, default_value = "-4.0")]
    eos_threshold: f32,
    /// Initial noise temperature for the LSD loop. Pocket-TTS draws the
    /// starting latent from ``Normal(0, sqrt(temp))`` -- the default 0.7
    /// matches its own CLI.
    #[arg(long, default_value = "0.7")]
    temp: f32,
    /// Optional symmetric truncation bound for the initial noise. ``None``
    /// matches Pocket-TTS' default; pass a value to use a truncated normal
    /// in ``[-clamp, +clamp]``.
    #[arg(long)]
    noise_clamp: Option<f32>,
    /// Sample rate to write into the WAV header. Real Pocket-TTS Mimi runs
    /// at 24 kHz; mini exports are unitless.
    #[arg(long, default_value = "24000")]
    sample_rate: u32,
    /// Seed for the noise sampled at each LSD step.
    #[arg(long, default_value = "0")]
    seed: u64,
    /// Latent dim ``ldim`` of the audio latents (= ``in_channels`` /
    /// ``out_channels`` of flow_net = first axis of decoder input). Mini
    /// configs use 8; real Pocket-TTS uses 32.
    #[arg(long, default_value = "8")]
    ldim: usize,
    /// Run the four NNEF graphs through tract's Metal GPU runtime (macOS
    /// only). On any other platform the flag is rejected.
    #[arg(long)]
    gpu: bool,
}

type Runnable =
    Arc<tract_core::internal::SimplePlan<TypedFact, Box<dyn TypedOp>>>;

fn load_graph(nnef: &Nnef, path: &Path, gpu: bool) -> Result<Runnable> {
    let mut model = nnef
        .model_for_path(path)
        .with_context(|| format!("loading NNEF graph from {}", path.display()))?;
    if gpu {
        #[cfg(target_os = "macos")]
        {
            MetalTransform::default()
                .transform(&mut model)
                .with_context(|| format!("applying MetalTransform to {}", path.display()))?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = &mut model;
            bail!("--gpu only supported on macOS (tract Metal runtime)");
        }
    }
    Ok(model.into_optimized()?.into_runnable()?)
}

fn load_voice(path: &Path) -> Result<Tensor> {
    let mut reader = std::fs::File::open(path)
        .with_context(|| format!("opening voice tensor {}", path.display()))?;
    tract_nnef::tensors::read_tensor(&mut reader)
        .with_context(|| format!("reading voice tensor {}", path.display()))
}

fn tokenize(args: &Args) -> Result<Vec<i64>> {
    if let Some(toks) = &args.tokens {
        return Ok(toks.clone());
    }
    let text = args
        .text
        .as_deref()
        .ok_or_else(|| anyhow!("need --text + --tokenizer or --tokens"))?;
    let tokenizer_path = args
        .tokenizer
        .as_deref()
        .ok_or_else(|| anyhow!("--text requires --tokenizer"))?;
    let sp = SentencePieceProcessor::open(tokenizer_path)
        .with_context(|| format!("opening tokenizer {}", tokenizer_path.display()))?;
    Ok(sp
        .encode(text)?
        .into_iter()
        .map(|p| p.id as i64)
        .collect())
}

fn make_position_vec(start: i64, len: usize) -> Tensor {
    let positions: Vec<i64> = (0..len as i64).map(|i| start + i).collect();
    tract_ndarray::Array1::from(positions).into_tensor()
}

fn sample_initial_noise(
    ldim: usize,
    std: f32,
    clamp: Option<f32>,
    rng: &mut StdRng,
) -> Vec<f32> {
    // ``current`` starts as ``Normal(0, std)``, optionally truncated to
    // ``[-clamp, +clamp]`` via rejection (matches PyTorch's
    // ``nn.init.trunc_normal_`` semantics).
    let normal = Normal::new(0.0_f32, std).unwrap();
    (0..ldim)
        .map(|_| match clamp {
            None => normal.sample(rng),
            Some(c) => loop {
                let v = normal.sample(rng);
                if v >= -c && v <= c {
                    break v;
                }
            },
        })
        .collect()
}

fn lsd_decode(
    flow_net: &Runnable,
    cond: &Tensor,
    ldim: usize,
    lsd_steps: usize,
    std: f32,
    clamp: Option<f32>,
    rng: &mut StdRng,
) -> Result<Tensor> {
    let mut current_vec = sample_initial_noise(ldim, std, clamp, rng);
    for i in 0..lsd_steps {
        let s = i as f32 / lsd_steps as f32;
        let t = (i + 1) as f32 / lsd_steps as f32;
        let s_t = tract_ndarray::Array2::from_elem((1, 1), s).into_tensor();
        let t_t = tract_ndarray::Array2::from_elem((1, 1), t).into_tensor();
        let x_t = tract_ndarray::Array2::from_shape_vec((1, ldim), current_vec.clone())
            .map_err(|e| anyhow!("LSD x reshape: {e}"))?
            .into_tensor();
        let out = flow_net.run(tvec!(
            cond.clone().into(),
            s_t.into(),
            t_t.into(),
            x_t.into()
        ))?;
        let flow_dir = out
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("flow_net produced no output"))?;
        let flow_view = flow_dir.to_plain_array_view::<f32>()?;
        for (c, f) in current_vec.iter_mut().zip(flow_view.iter()) {
            *c += *f / lsd_steps as f32;
        }
    }
    Ok(
        tract_ndarray::Array2::from_shape_vec((1, ldim), current_vec)
            .map_err(|e| anyhow!("LSD final reshape: {e}"))?
            .into_tensor(),
    )
}

/// Generation phase: autoregressive loop + LSD per-frame decode + Mimi
/// audio decode. Split out so we can wrap it in ``with_metal_stream``
/// when running on the Metal GPU runtime.
fn run_generation(
    args: &Args,
    flow_lm_init: &Runnable,
    flow_lm_step: &Runnable,
    flow_net: &Runnable,
    audio_decoder: &Runnable,
    voice: Tensor,
    token_tensor: Tensor,
    t_voice: usize,
    token_count: usize,
) -> Result<Vec<f32>> {
    let init_q_pos = make_position_vec(t_voice as i64, token_count + 1);
    let init_k_pos = make_position_vec(0, t_voice + token_count + 1);
    let init_outs = flow_lm_init.run(tvec!(
        token_tensor.into(),
        voice.into(),
        init_q_pos.into(),
        init_k_pos.into()
    ))?;
    let mut transformer_out = init_outs[0].clone();
    let mut eos_logit = init_outs[1].clone();
    let mut past_kv = init_outs[2].clone();
    let mut next_pos = (t_voice + token_count + 1) as i64;

    let ldim = args.ldim;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let std = args.temp.sqrt();
    let mut latents: Vec<Tensor> = Vec::with_capacity(args.max_frames);
    let first_latent = lsd_decode(
        flow_net,
        &transformer_out,
        ldim,
        args.lsd_steps,
        std,
        args.noise_clamp,
        &mut rng,
    )?;
    latents.push(first_latent);

    for frame in 1..args.max_frames {
        let eos_view = eos_logit.to_plain_array_view::<f32>()?;
        let eos_val = *eos_view.iter().next().unwrap_or(&0.0);
        if eos_val > args.eos_threshold {
            println!("EOS at frame {frame} (logit {eos_val:.3} > {})", args.eos_threshold);
            break;
        }
        let prev_latent = latents.last().unwrap().clone();
        let q_pos = make_position_vec(next_pos, 1);
        let k_pos = make_position_vec(0, (next_pos as usize) + 1);
        let outs = flow_lm_step.run(tvec!(
            prev_latent.into(),
            past_kv.clone().into(),
            q_pos.into(),
            k_pos.into()
        ))?;
        transformer_out = outs[0].clone();
        eos_logit = outs[1].clone();
        past_kv = outs[2].clone();
        next_pos += 1;
        let latent = lsd_decode(
            flow_net,
            &transformer_out,
            ldim,
            args.lsd_steps,
            std,
            args.noise_clamp,
            &mut rng,
        )?;
        latents.push(latent);
    }
    println!("generated {} audio latents", latents.len());

    let t_lat = latents.len();
    let mut latent_buf: Vec<f32> = Vec::with_capacity(t_lat * ldim);
    for l in 0..ldim {
        for lat in &latents {
            let v = lat.to_plain_array_view::<f32>()?;
            latent_buf.push(v[[0, l]]);
        }
    }
    let latent_stack =
        tract_ndarray::Array3::from_shape_vec((1, ldim, t_lat), latent_buf)
            .map_err(|e| anyhow!("latent stack reshape: {e}"))?;

    let audio_out = audio_decoder.run(tvec!(latent_stack.into_tensor().into()))?;
    let audio = audio_out
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("decoder produced no output"))?;
    let audio_view = audio.to_plain_array_view::<f32>()?;
    Ok(audio_view.iter().copied().collect())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("loading tract NNEF runtime");
    let nnef = tract_nnef::nnef()
        .with_tract_core()
        .with_pulse()
        .with_tract_extra()
        .with_tract_transformers();

    let init_path = args.models.join("flow_lm_init.nnef.tgz");
    let step_path = args.models.join("flow_lm_step.nnef.tgz");
    let flow_net_path = args.models.join("flow_net.nnef.tgz");
    // Prefer the full Mimi chain (latent -> 24 kHz audio) when present;
    // fall back to the SEANet-only ``decoder.nnef.tgz`` for the mini config.
    let mimi_decode_path = args.models.join("mimi_decode.nnef.tgz");
    let decoder_path = args.models.join("decoder.nnef.tgz");
    let audio_decode_path = if mimi_decode_path.exists() {
        mimi_decode_path
    } else {
        decoder_path
    };
    if args.gpu {
        #[cfg(target_os = "macos")]
        println!("running through tract Metal GPU runtime");
        #[cfg(not(target_os = "macos"))]
        bail!("--gpu only supported on macOS (tract Metal runtime)");
    }
    println!("loading flow_lm_init from {}", init_path.display());
    let flow_lm_init = load_graph(&nnef, &init_path, args.gpu)?;
    println!("loading flow_lm_step from {}", step_path.display());
    let flow_lm_step = load_graph(&nnef, &step_path, args.gpu)?;
    println!("loading flow_net from {}", flow_net_path.display());
    let flow_net = load_graph(&nnef, &flow_net_path, args.gpu)?;
    println!("loading audio decoder from {}", audio_decode_path.display());
    let audio_decoder = load_graph(&nnef, &audio_decode_path, args.gpu)?;

    println!("loading voice prompt from {}", args.voice.display());
    let voice = load_voice(&args.voice)?;
    let voice_shape = voice.shape().to_vec();
    let t_voice = voice_shape
        .get(3)
        .copied()
        .ok_or_else(|| anyhow!("voice tensor must have rank 6 (n_layers,2,B,T,H,D); got {voice_shape:?}"))?;
    println!("voice prefix length: {t_voice}");

    let token_ids: Vec<i64> = tokenize(&args)?;
    println!("token ids: {token_ids:?} ({} tokens)", token_ids.len());
    let token_count = token_ids.len();
    let token_tensor = tract_ndarray::Array2::from_shape_vec((1, token_count), token_ids)
        .map_err(|e| anyhow!("token tensor reshape: {e}"))?
        .into_tensor();

    // -- generation phase (timed for RTFx; wrapped for Metal stream) --------
    let gen_start = Instant::now();
    let samples = if args.gpu {
        #[cfg(target_os = "macos")]
        {
            tract_metal::with_metal_stream(|_| {
                run_generation(
                    &args,
                    &flow_lm_init,
                    &flow_lm_step,
                    &flow_net,
                    &audio_decoder,
                    voice,
                    token_tensor,
                    t_voice,
                    token_count,
                )
            })?
        }
        #[cfg(not(target_os = "macos"))]
        {
            unreachable!("--gpu rejected earlier on non-macOS targets")
        }
    } else {
        run_generation(
            &args,
            &flow_lm_init,
            &flow_lm_step,
            &flow_net,
            &audio_decoder,
            voice,
            token_tensor,
            t_voice,
            token_count,
        )?
    };
    let gen_wall = gen_start.elapsed();
    let audio_secs = samples.len() as f64 / args.sample_rate as f64;
    let wall_secs = gen_wall.as_secs_f64();
    let rtfx = audio_secs / wall_secs;
    println!(
        "decoded {} samples ({:.2} s @ {} Hz) in {:.2} s wall time -- RTFx {:.2}{}",
        samples.len(),
        audio_secs,
        args.sample_rate,
        wall_secs,
        rtfx,
        if args.gpu { " [Metal GPU]" } else { "" },
    );

    // -- write WAV ----------------------------------------------------------
    let spec = WavSpec {
        channels: 1,
        sample_rate: args.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(&args.out, spec)
        .with_context(|| format!("creating WAV at {}", args.out.display()))?;
    for s in &samples {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    println!("wrote {}", args.out.display());
    Ok(())
}
