/// NEMO ASR model inference using tract-nnef
/// Only use full audio inference for now
/// streaming/pulsed inference may be added later
///
/// code adapted from: nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py
/// class ONNXGreedyBatchedRNNTInfer
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tract_core::tract_data::itertools::Itertools;
use tract_ndarray::s;
use tract_nnef::prelude::*;
use tract_nnef::tract_ndarray::Axis;
use tract_transformers::WithTractTransformers;

/// Decoder config struct
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub blank_as_pad: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NemoAsrConfig {
    pub sample_rate: usize,
    pub labels: Vec<String>,
    pub pretrained_name: Option<String>,
    pub decoder: DecoderConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfig {
    max_n_tokens_per_step: Option<usize>,
    force_cpu: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            max_n_tokens_per_step: Some(50),
            force_cpu: false,
        }
    }
}

impl NemoAsrConfig {
    pub fn get_blank_index(&self) -> usize {
        self.labels.len()
    }
}

/// code a struct that represent nemo ASR model inference
/// it load a model from_dir containing model files:
#[derive(Debug, Clone)]
pub struct NemoAsrModel {
    preprocessor_model: TypedRunnableModel<TypedModel>,
    encoder_model: TypedRunnableModel<TypedModel>,
    decoder_joint_model: TypedRunnableModel<TypedModel>,
    model_config: NemoAsrConfig,
    runtime_config: RuntimeConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptItem {
    pub token: String,
    pub logit: f32,
    pub emitted_at_encoder_timestep: usize,
    pub emitted_at_encoder_timestep_iteration: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Transcription {
    pub text: String,
    pub items: Vec<TranscriptItem>,
}

impl Transcription {
    pub fn from_transcript_items(items: Vec<TranscriptItem>) -> Transcription {
        Transcription {
            text: normalize_transcript_text(
                items
                    .iter()
                    .map(|ti| ti.token.as_str())
                    .join("")
                    .replace("▁", " ")
                    .trim(),
            ),
            items,
        }
    }
}

/// Per-sample decoding lane
#[derive(Debug)]
struct Lane {
    encoder_len: usize,
    current_frame: usize,
    last_token: usize,
    last_emitted_token: usize, // for transcript de-dup only
    states: TVec<TValue>,      // each state is [2,1,640] in parakeet
    transcript: Vec<TranscriptItem>,
    n_tokens_added_in_frame: usize,
}

fn normalize_transcript_text(s: &str) -> String {
    // Very conservative cleanup:
    // - remove [] debug-style brackets
    // - collapse excessive whitespace
    let mut out = String::with_capacity(s.len());
    let mut last_space = false;

    for c in s.chars() {
        if c == '[' || c == ']' {
            continue;
        }
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
            }
            last_space = true;
        } else {
            last_space = false;
            out.push(c);
        }
    }

    out.trim().to_string()
}

impl NemoAsrModel {
    fn from_bytes_submodel(
        name: &str,
        runtime_config: &RuntimeConfig,
        model_bytes: &[u8],
    ) -> TractResult<TypedRunnableModel<TypedModel>> {
        let mut model_read = std::io::Cursor::new(model_bytes);
        let nnef = tract_nnef::nnef().with_tract_transformers();

        let transform = nnef
            .get_transform("transformers-detect-all")?
            .context("transformers-detect-all not found")?;

        let mut nn = nnef.model_for_read(&mut model_read)?;

        let mut device = "CPU";
        if !runtime_config.force_cpu {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                use crate::tract_core::transform::ModelTransform;
                use std::str::FromStr;
                nn.properties.insert("GPU".into(), rctensor0(true));
                tract_metal::MetalTransform::from_str("")?.transform(&mut nn)?;
                device = "Metal GPU acceleration ";
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                use tract_core::transform::ModelTransform;
                if tract_cuda::utils::are_culibs_present() {
                    nn.properties.insert("GPU".into(), rctensor0(true));
                    tract_cuda::CudaTransform.transform(&mut nn)?;
                    device = "CUDA GPU acceleration ";
                }
            }
        }
        log::debug!("Using {} for model part: {} inference", device, name);

        let mut nn = nn.into_decluttered()?;
        nn.transform(&*transform)?;

        let model = nn.into_optimized()?.into_runnable()?;
        Ok(model)
    }

    fn from_bytes(
        model_config_bytes: &[u8],
        runtime_config_bytes: Option<&[u8]>,
        pre_model_bytes: &[u8],
        enc_model_bytes: &[u8],
        dec_model_bytes: &[u8],
    ) -> TractResult<NemoAsrModel> {
        log::info!("start loading nemo asr model from bytes");
        let model_config = serde_json::from_slice::<NemoAsrConfig>(model_config_bytes)?;
        let runtime_config = if let Some(rt_conf) = runtime_config_bytes {
            log::info!("found runtime config bytes, loading it");
            serde_json::from_slice::<RuntimeConfig>(rt_conf)?
        } else {
            log::info!("NO runtime config found, using default");
            RuntimeConfig::default()
        };

        let preprocessor_model =
            NemoAsrModel::from_bytes_submodel("preprocessor", &runtime_config, pre_model_bytes)?;
        let encoder_model =
            NemoAsrModel::from_bytes_submodel("encoder", &runtime_config, enc_model_bytes)?;
        let decoder_joint_model =
            NemoAsrModel::from_bytes_submodel("decoder_joint", &runtime_config, dec_model_bytes)?;

        log::info!("all model subparts loaded successfully in tract");
        Ok(NemoAsrModel {
            preprocessor_model,
            encoder_model,
            decoder_joint_model,
            model_config,
            runtime_config,
        })
    }

    pub fn from_dir(path: PathBuf) -> TractResult<NemoAsrModel> {
        let runtime_config_path = path.join("runtime_config.json");
        let model_config_path = path.join("model_config.json");
        let pre_model_path = path.join("preprocessor.nnef.tgz");
        let enc_model_path = path.join("encoder.nnef.tgz");
        let dec_model_path = path.join("decoder_joint.nnef.tgz");

        log::info!("start loading nemo asr model from dir: {:?}", path);

        let runtime_config_bytes = std::fs::read(runtime_config_path).ok();

        let model_config_bytes =
            std::fs::read(model_config_path).expect("Failed to read model config file");
        let pre_model_bytes =
            std::fs::read(pre_model_path).expect("Failed to read preprocessor model file");
        let enc_model_bytes =
            std::fs::read(enc_model_path).expect("Failed to read encoder model file");
        let dec_model_bytes =
            std::fs::read(dec_model_path).expect("Failed to read decoder model file");

        NemoAsrModel::from_bytes(
            model_config_bytes.as_slice(),
            runtime_config_bytes.as_deref(),
            pre_model_bytes.as_slice(),
            enc_model_bytes.as_slice(),
            dec_model_bytes.as_slice(),
        )
    }

    /// Convert a single wav file path to a single input tensor
    fn wav_path_to_tensor(&self, wav_path: &PathBuf) -> TractResult<Tensor> {
        let mut reader = hound::WavReader::open(wav_path).expect("Failed to open WAV file");
        let spec = reader.spec();
        assert_eq!(
            spec.sample_rate, self.model_config.sample_rate as u32,
            "Only 16kHz sample rate is supported"
        );
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect();
        let input_tensor =
            tract_ndarray::Array2::from_shape_vec((1, samples.len()), samples).unwrap();
        Ok(input_tensor.into())
    }

    /// Infer from a wav file path all at once
    pub fn infer_from_wav_paths(&self, wav_paths: &[PathBuf]) -> TractResult<Vec<Transcription>> {
        log::debug!("Loading wav file from path: {:?}", wav_paths);
        let input_tensor_vec = wav_paths
            .iter()
            .map(|wp| self.wav_path_to_tensor(wp).unwrap())
            .collect::<Vec<Tensor>>();
        log::debug!("wav loaded correctly, starting inference");

        log::debug!("prepare input tensor batch");
        let lengths = input_tensor_vec
            .iter()
            .map(|t| t.shape()[1] as i64)
            .collect::<Vec<i64>>();

        // Build input tensor batch {
        let mut input_tensor = tract_ndarray::Array2::<f32>::zeros((
            input_tensor_vec.len(),
            lengths.iter().max().copied().unwrap() as usize,
        ))
        .into_tensor();

        for (ix, itensor) in input_tensor_vec.iter().enumerate() {
            let sample_len = itensor.shape()[1];
            input_tensor
                .to_array_view_mut()?
                .slice_mut(s![ix, 0..sample_len])
                .assign(&itensor.to_array_view::<f32>()?.index_axis(Axis(0), 0));
        }
        // }

        let lengths_tensor: Tensor =
            tract_ndarray::Array1::<i64>::from_shape_vec((wav_paths.len(),), lengths)?
                .into_tensor();
        log::debug!("ready input tensor batch");
        let transcripts = self.infer_from_tensor(input_tensor, lengths_tensor)?;
        Ok(transcripts)
    }

    fn infer_from_tensor(
        &self,
        input_tensor: Tensor,
        length_tensor: Tensor,
    ) -> TractResult<Vec<Transcription>> {
        log::debug!("start inference preprocessor");
        // Preprocessor inference
        let preprocessor_output = self.preprocessor_model.run(tvec!(
            input_tensor.into_tvalue(),
            length_tensor.into_tvalue()
        ))?;
        log::debug!("successfully ran preprocessor");

        // Encoder inference
        log::debug!("start inference encoder");
        let encoder_output = self.encoder_model.run(preprocessor_output)?;
        log::debug!("successfully ran encoder");

        // Decoder inference
        log::debug!("start decoder and joint");
        let t = self.decode_transcripts_from_encoder_output(encoder_output);
        log::debug!("successfully ran decoder and joint");
        t
    }

    fn get_initial_decoder_states(&self, batch_size: usize) -> TractResult<Vec<[usize; 3]>> {
        let mdl: &TypedModel = self.decoder_joint_model.model();

        mdl.input_outlets()?
            .iter()
            .map(|ioutlet| &mdl.nodes()[ioutlet.node])
            .filter(|node| node.name.contains("states")) // only decoder states
            .map(|node| {
                // mdl.with_input_outlets
                let shape = node.outputs[0]
                    .fact
                    .shape
                    .eval_to_usize(&SymbolValues::default().with(
                        &mdl.symbols.get("B").unwrap(),
                        batch_size as i64,
                    ))
                    .context("Failed to get concrete shape for decoder state")?
                    .to_vec();

                let shape: [usize; 3] = shape.try_into().expect("shape must be 3D");
                if shape.contains(&0) {
                    anyhow::bail!(
                        "Decoder state has zero dimension in shape {:?}, cannot create initial state",
                        shape
                    );
                }
                Ok(shape)
            })
            .collect()
    }

    pub fn decode_transcripts_from_encoder_output(
        &self,
        encoder_out: TVec<TValue>,
    ) -> TractResult<Vec<Transcription>> {
        let feats = encoder_out[0].to_array_view::<f32>()?;
        let lens = encoder_out[1].to_array_view::<i64>()?;
        let bsz = feats.shape()[0];

        let blank = self.model_config.get_blank_index();
        let vocab = &self.model_config.labels;

        // Initialize lanes
        let mut lanes: Vec<Lane> = (0..bsz)
            .map(|b| {
                let states = self
                    .decoder_joint_model
                    .model()
                    .inputs
                    .iter()
                    .skip(3)
                    .zip(self.get_initial_decoder_states(1)?)
                    .map(|(_, shape)| {
                        let z = tract_ndarray::Array3::<f32>::zeros(shape);
                        Ok(z.into_tensor().into_tvalue())
                    })
                    .collect::<TractResult<_>>()?;

                Ok(Lane {
                    encoder_len: lens[b] as usize,
                    current_frame: 0,
                    last_token: blank,
                    last_emitted_token: blank,
                    states,
                    transcript: Vec::new(),
                    n_tokens_added_in_frame: 0,
                })
            })
            .collect::<TractResult<_>>()?;

        loop {
            let active: Vec<usize> = lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| {
                    l.current_frame < l.encoder_len
                        && self
                            .runtime_config
                            .max_n_tokens_per_step
                            .is_none_or(|m| l.n_tokens_added_in_frame < m)
                })
                .map(|(i, _)| i)
                .collect();

            if active.is_empty() {
                break;
            }

            let enc_frames = tract_ndarray::concatenate(
                Axis(0),
                &active
                    .iter()
                    .map(|&i| {
                        feats.slice(s![
                            i..i + 1,
                            ..,
                            lanes[i].current_frame..lanes[i].current_frame + 1
                        ])
                    })
                    .collect::<Vec<_>>(),
            )?
            .into_tensor()
            .into_tvalue();

            let labels = tract_ndarray::Array2::<i32>::from_shape_vec(
                (active.len(), 1),
                active.iter().map(|&i| lanes[i].last_token as i32).collect(),
            )?
            .into_tensor()
            .into_tvalue();

            let target_lens = tract_ndarray::Array1::<i32>::from_elem(active.len(), 1)
                .into_tensor()
                .into_tvalue();

            let packed_states = Self::state_pack_lanes(&lanes, &active)?;

            let mut inputs = tvec!(enc_frames, labels, target_lens);
            inputs.extend(packed_states);

            log::debug!("Decoding step with {} active lanes", active.len());
            log::debug!(
                "lanes steps: {:?}",
                lanes.iter().map(|l| l.current_frame).collect::<Vec<_>>()
            );
            log::debug!(
                "lanes n_tokens_added_in_frame: {:?}",
                lanes
                    .iter()
                    .map(|l| l.n_tokens_added_in_frame)
                    .collect::<Vec<_>>()
            );
            let outs = self.decoder_joint_model.run(inputs)?;
            log::debug!("Decoder joint step done");

            let logp = outs[0].to_array_view::<f32>()?;

            for (k, &lane_ix) in active.iter().enumerate() {
                let row = logp.index_axis(Axis(0), k);
                let (tok, tok_prob) = row
                    .iter()
                    .take(vocab.len() + 1)
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                let lane = &mut lanes[lane_ix];
                if tok == blank {
                    // If the joint emits additional logits after vocab+1, those logits
                    // encode how many frames to skip/advance. We take the argmax index
                    // and ensure we advance at least 1 frame (guard against 0).
                    let n_skip_steps = if row.len() > vocab.len() + 1 {
                        let (idx, _val) = row
                            .iter()
                            .skip(vocab.len() + 1)
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .unwrap();
                        // `idx` is 0-based within the skip logits; interpret conservatively:
                        // require at least 1 frame advance.
                        idx
                    } else {
                        1usize
                    };

                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "lane {} blank -> applied_skip={}, row_len={}, vocab+1={}",
                            lane_ix,
                            n_skip_steps,
                            row.len(),
                            vocab.len() + 1
                        );
                    }

                    lane.current_frame += n_skip_steps;
                    lane.n_tokens_added_in_frame = if n_skip_steps > 0 {
                        // Predictor input label should track the last predicted non-blank label
                        for (sid, st) in outs[2..].iter().enumerate() {
                            lane.states[sid] = Self::state_take_lane(st, k)?;
                        }
                        0
                    } else {
                        lane.n_tokens_added_in_frame
                    };

                    // Reset last_token on blank
                    lane.last_emitted_token = blank;
                } else {
                    if tok != lane.last_emitted_token {
                        lane.transcript.push(TranscriptItem {
                            token: vocab[tok].clone(),
                            emitted_at_encoder_timestep: lane.current_frame,
                            emitted_at_encoder_timestep_iteration: 0,
                            logit: *tok_prob,
                        });
                        lane.last_emitted_token = tok;
                    }
                    lane.n_tokens_added_in_frame += 1;

                    lane.last_token = tok;

                    // Predictor input label should track the last predicted non-blank label
                    for (sid, st) in outs[2..].iter().enumerate() {
                        lane.states[sid] = Self::state_take_lane(st, k)?;
                    }
                }
            }
            if let Some(max_n_tok) = self.runtime_config.max_n_tokens_per_step {
                for lane in lanes.iter_mut() {
                    if lane.n_tokens_added_in_frame >= max_n_tok {
                        lane.current_frame += 1;
                        lane.n_tokens_added_in_frame = 0;
                        log::debug!(
                            "Lane reached max tokens per step {}, advancing frame to {}",
                            max_n_tok,
                            lane.current_frame
                        );
                    }
                }
            }
        }

        Ok(lanes
            .into_iter()
            .map(|l| Transcription::from_transcript_items(l.transcript))
            .collect())
    }

    fn state_take_lane(state: &TValue, b: usize) -> TractResult<TValue> {
        let v = state.to_array_view::<f32>()?;
        Ok(v.index_axis(Axis(1), b)
            .insert_axis(Axis(1))
            .to_owned()
            .into_tensor()
            .into_tvalue())
    }

    fn state_pack_lanes(lanes: &[Lane], active: &[usize]) -> TractResult<TVec<TValue>> {
        let n_states = lanes[active[0]].states.len();
        let mut packed = tvec!();
        for sid in 0..n_states {
            let views: Vec<_> = active
                .iter()
                .map(|&lx| lanes[lx].states[sid].to_array_view::<f32>())
                .collect::<TractResult<_>>()?;
            let cat = tract_ndarray::concatenate(Axis(1), &views)?;
            packed.push(cat.into_tensor().into_tvalue());
        }
        Ok(packed)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::Path;
    use test_log::test;

    fn workspace_root() -> PathBuf {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

        for dir in manifest_dir.ancestors() {
            let cargo_toml = dir.join("Cargo.toml");
            if cargo_toml.exists() && is_workspace_root(&cargo_toml) {
                return dir.to_path_buf();
            }
        }

        panic!("Workspace root not found");
    }

    fn is_workspace_root(cargo_toml: &Path) -> bool {
        // Minimal heuristic: workspace root has a [workspace] table
        std::fs::read_to_string(cargo_toml)
            .map(|s| s.contains("[workspace]"))
            .unwrap_or(false)
    }

    fn assets_dir() -> PathBuf {
        workspace_root().join("assets")
    }

    fn truncate_with_ellipsis(s: &str, max_chars: usize) -> String {
        let mut chars = s.chars();

        let truncated: String = chars.by_ref().take(max_chars).collect();

        if chars.next().is_some() {
            format!("{truncated}...")
        } else {
            truncated
        }
    }

    #[test]
    fn test_load_and_decode_audio() -> TractResult<()> {
        println!("Assets dir: {:?}\n", assets_dir());
        let asr = NemoAsrModel::from_dir(assets_dir().join("model"))?;
        println!("Loaded ASR model successfully");
        let transcripts = asr.infer_from_wav_paths(&[
            assets_dir().join("2086-149220-0033.wav"),
            assets_dir().join("data_smoke_test_LDC93S1.wav"),
        ])?;
        let max_chars = 200;
        for (i, t) in transcripts.iter().enumerate() {
            println!(
                "Transcription[{}]: '{}'",
                i,
                &truncate_with_ellipsis(&t.text, max_chars)
            );
            if i == 0 {
                println!("Full items: {:#?}", t.items);
            }
        }
        // This code works if only 1 sample in batch
        // but output garbage text when multiple samples in batch
        Ok(())
    }
}
