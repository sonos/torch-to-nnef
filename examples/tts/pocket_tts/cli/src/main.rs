//! End-to-end Pocket-TTS demo via tract.
//!
//! Loads the four NNEF graphs exported from `examples/tts/pocket_tts/`
//! (`flow_lm_init`, `flow_lm_step`, `flow_net`, `decoder`), threads them
//! together with a SentencePiece tokenizer + a voice prompt `.dat`, and
//! writes a 24 kHz WAV.
//!
//! With mini random-weights graphs the output is not coherent speech --
//! the binary is a *plumbing* demo that proves the export -> tract runtime
//! path is wired correctly. Swapping the NNEF graphs for real-checkpoint
//! exports (TODO: needs HF auth + production weight loading) gives real
//! audio without changing a line of Rust.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use sentencepiece::SentencePieceProcessor;
use tract_extra::WithTractExtra;
use tract_nnef::framework::Nnef;
use tract_nnef::prelude::*;
use tract_pulse::WithPulse;
use tract_transformers::WithTractTransformers;

#[derive(Parser, Debug)]
#[command(about = "Pocket-TTS demo through tract", long_about = None)]
struct Args {
    /// Directory holding the four exported NNEF graphs
    /// (flow_lm_init.nnef.tgz, flow_lm_step.nnef.tgz, flow_net.nnef.tgz,
    /// decoder.nnef.tgz).
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
    #[arg(long, default_value = "4")]
    lsd_steps: usize,
    /// EOS logit threshold above which the loop terminates.
    #[arg(long, default_value = "0.5")]
    eos_threshold: f32,
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
    /// Dump the FlowLM-emitted audio latents to this ``.npz`` (key
    /// ``latents``, shape ``(B, ldim, T)``) and skip the SEANet decoder
    /// step. Used by the hybrid ``--full`` run.sh path where the Mimi
    /// audio decode runs in Python via ``decode_audio.py``.
    #[arg(long)]
    dump_latents: Option<PathBuf>,
}

type Runnable =
    Arc<tract_core::internal::SimplePlan<TypedFact, Box<dyn TypedOp>>>;

fn load_graph(nnef: &Nnef, path: &Path) -> Result<Runnable> {
    let model = nnef
        .model_for_path(path)
        .with_context(|| format!("loading NNEF graph from {}", path.display()))?
        .into_runnable()?;
    Ok(model)
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

fn lsd_decode(
    flow_net: &Runnable,
    cond: &Tensor,
    ldim: usize,
    lsd_steps: usize,
    rng: &mut StdRng,
    normal: &Normal<f32>,
) -> Result<Tensor> {
    // ``current`` starts as Gaussian noise, gets refined ``lsd_steps`` times.
    let mut current_vec: Vec<f32> = (0..ldim)
        .map(|_| normal.sample(rng))
        .collect();
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
    let decoder_path = args.models.join("decoder.nnef.tgz");
    println!("loading flow_lm_init from {}", init_path.display());
    let flow_lm_init = load_graph(&nnef, &init_path)?;
    println!("loading flow_lm_step from {}", step_path.display());
    let flow_lm_step = load_graph(&nnef, &step_path)?;
    println!("loading flow_net from {}", flow_net_path.display());
    let flow_net = load_graph(&nnef, &flow_net_path)?;
    let decoder = if args.dump_latents.is_some() {
        println!("(skipping decoder; --dump-latents set)");
        None
    } else {
        println!("loading decoder from {}", decoder_path.display());
        Some(load_graph(&nnef, &decoder_path)?)
    };

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

    // -- init step ----------------------------------------------------------
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

    // -- LSD decode for first audio frame -----------------------------------
    // ``ldim`` is the audio-latent dim (flow_net's in/out_channels),
    // *not* the transformer hidden dim. They happen to differ in the mini
    // config (16 vs 8) so don't confuse them.
    let ldim = args.ldim;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
    let mut latents: Vec<Tensor> = Vec::with_capacity(args.max_frames);
    let first_latent = lsd_decode(&flow_net, &transformer_out, ldim, args.lsd_steps, &mut rng, &normal)?;
    latents.push(first_latent.clone());

    // -- step loop ----------------------------------------------------------
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
        let latent = lsd_decode(&flow_net, &transformer_out, ldim, args.lsd_steps, &mut rng, &normal)?;
        latents.push(latent);
    }
    println!("generated {} audio latents", latents.len());

    // -- stack latents into channels-first (B, ldim, T) ---------------------
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

    // -- ``--dump-latents`` short-circuit (hybrid path) ---------------------
    if let Some(path) = &args.dump_latents {
        write_npz_latents(path, &latent_stack)
            .with_context(|| format!("dumping latents to {}", path.display()))?;
        println!("wrote {} latents=({},{},{})", path.display(), 1, ldim, t_lat);
        return Ok(());
    }

    let decoder = decoder.expect("decoder graph is loaded when not skipping");
    let audio_out = decoder.run(tvec!(latent_stack.into_tensor().into()))?;
    let audio = audio_out
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("decoder produced no output"))?;
    let audio_view = audio.to_plain_array_view::<f32>()?;
    let samples: Vec<f32> = audio_view.iter().copied().collect();
    println!("decoded {} audio samples", samples.len());

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


/// Write a ``(1, ldim, T)`` latent tensor as a numpy ``.npz`` archive
/// containing one array under the key ``latents``. Hand-rolled so we don't
/// take on a numpy / npz crate dep just for this debug dump.
fn write_npz_latents(
    path: &Path,
    latents: &tract_ndarray::Array3<f32>,
) -> Result<()> {
    use std::io::Write;
    let shape = latents.shape();
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {})}}",
        shape[0], shape[1], shape[2]
    );
    // Pad header to 16-byte multiple (np.save requirement).
    let header_padded = {
        let mut h = header;
        // 10 = magic(6) + version(2) + header-len(2)
        let target = ((10 + h.len() + 1 + 15) / 16) * 16;
        let pad = target - 10 - h.len() - 1;
        h.push_str(&" ".repeat(pad));
        h.push('\n');
        h
    };
    let mut npy = Vec::<u8>::with_capacity(latents.len() * 4 + 256);
    npy.extend_from_slice(b"\x93NUMPY"); // magic
    npy.push(1); // major
    npy.push(0); // minor
    npy.extend_from_slice(
        &(u16::try_from(header_padded.len()).expect("header < 64KiB"))
            .to_le_bytes(),
    );
    npy.extend_from_slice(header_padded.as_bytes());
    for &v in latents.iter() {
        npy.extend_from_slice(&v.to_le_bytes());
    }

    let f = std::fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(f);
    let opts: zip::write::SimpleFileOptions = Default::default();
    zip.start_file("latents.npy", opts)?;
    zip.write_all(&npy)?;
    zip.finish()?;
    Ok(())
}
