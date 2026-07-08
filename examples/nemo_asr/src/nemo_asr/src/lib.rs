/// NEMO ASR model inference using tract-nnef
/// Only use full audio inference for now
/// streaming/pulsed inference may be added later
///
/// code adapted from: nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py
/// class ONNXGreedyBatchedRNNTInfer
use anyhow::ensure;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tract_rs::prelude::tract_ndarray::prelude::*;
use tract_rs::{prelude::*, runtime_for_name};

type Res<T> = anyhow::Result<T>;

/// Decoder config struct
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub blank_as_pad: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecodingConfig {
    // Present for TDT models (token-and-duration). Absent for plain RNNT
    // (e.g. multilingual Nemotron), where advance is blank-driven.
    #[serde(default)]
    pub durations: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NemoAsrConfig {
    pub sample_rate: usize,
    pub labels: Vec<String>,
    pub pretrained_name: Option<String>,
    pub decoder: DecoderConfig,
    pub decoding: DecodingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfig {
    max_n_tokens_per_step: Option<usize>,
    force_cpu: bool,
    encoder_per_batch: bool,
    dump_intermediate_io_path: Option<PathBuf>,
    // Language id fed to a prompt-fused encoder (multilingual Nemotron).
    // Only used when the encoder declares a `lang_id` input; ignored
    // otherwise (e.g. Parakeet). Defaults to 101 (the "auto" slot).
    #[serde(default = "default_lang_id")]
    default_lang_id: i64,
}

fn default_lang_id() -> i64 {
    101
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            max_n_tokens_per_step: Some(50),
            force_cpu: false,
            encoder_per_batch: false,
            // if set to Some(path), will dump encoder inputs/outputs
            dump_intermediate_io_path: None,
            default_lang_id: default_lang_id(),
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
    preprocessor_model: Runnable,
    encoder_model: Runnable,
    decoder_joint_model: Runnable,
    model_config: NemoAsrConfig,
    runtime_config: RuntimeConfig,
    decoder_state_inputs_facts: Vec<Fact>,
    // Set when the encoder has a prompt-fused `lang_id` input; the value is
    // appended to the encoder inputs at run time. `None` for plain encoders.
    encoder_lang_id: Option<i64>,
    // Some RNNT decoder_joint exports keep a `target_length` input; when present
    // we pass a per-lane length of 1 (one predicted token). Absent when the
    // export drops it.
    decoder_joint_wants_target_length: bool,
    // Output indices of the predictor states (names containing "states"), and
    // the logits output ("outputs"). Some exports interleave a non-state output
    // (e.g. `prednet_lengths`), so states are located by name, not position.
    decoder_joint_logits_output_index: usize,
    decoder_joint_state_output_indices: Vec<usize>,
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
            text: items
                .iter()
                .map(|ti| ti.token.as_str())
                .join("")
                .replace("▁", " ")
                .trim()
                .into(),
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
    states: Vec<Value>,
    transcript: Vec<TranscriptItem>,
    symbols_added: usize,
    need_loop: bool,
}

/// dump tensor to a npy file
fn dump_value_vec_to_file(values: &[Value], dir_path: PathBuf) -> anyhow::Result<()> {
    std::fs::create_dir_all(&dir_path)?;

    for (i, value) in values.iter().enumerate() {
        // Find the first available filename
        let mut idx = 0;
        let file_path = loop {
            let path = dir_path.join(format!("tensor_{}_{}.npy", i, idx));
            if !path.exists() {
                break path;
            }

            idx += 1;
        };

        match value.datum_type()? {
            DatumType::TRACT_DATUM_TYPE_F32 => {
                let view = value.view::<f32>()?;
                ndarray_npy::write_npy(&file_path, &view.to_owned())?;
            }
            DatumType::TRACT_DATUM_TYPE_I64 => {
                let view = value.view::<i64>()?;
                ndarray_npy::write_npy(&file_path, &view.to_owned())?;
            }
            other => {
                anyhow::bail!("Unsupported dtype for npy dump: {:?}", other);
            }
        }
    }

    Ok(())
}

impl NemoAsrModel {
    fn load_submodel(bytes: &[u8]) -> Res<Model> {
        let nnef = tract_rs::nnef()?.with_tract_transformers()?;
        let mut nn = nnef.load_buffer(bytes)?;
        nn.transform("transformers-detect-all")?;
        Ok(nn)
    }

    fn from_bytes(
        model_config_bytes: &[u8],
        runtime_config_bytes: Option<&[u8]>,
        pre_model_bytes: &[u8],
        enc_model_bytes: &[u8],
        dec_model_bytes: &[u8],
    ) -> Res<NemoAsrModel> {
        let runtime_config: RuntimeConfig = if let Some(rt_conf) = runtime_config_bytes {
            log::info!("found runtime config bytes, loading it");
            serde_json::from_reader(rt_conf)?
        } else {
            log::info!("NO runtime config found, using default");
            RuntimeConfig::default()
        };

        let model_config: NemoAsrConfig = serde_json::from_reader(model_config_bytes)?;

        let decoder_joint_model = Self::load_submodel(dec_model_bytes)?;
        let mut decoder_state_inputs_facts = vec![];
        let mut decoder_joint_wants_target_length = false;
        for ix in 0..decoder_joint_model.input_count()? {
            let name = decoder_joint_model.input_name(ix)?;
            if name.contains("states") {
                let fact = decoder_joint_model.input_fact(ix)?;
                ensure!(fact.rank()? == 3);
                decoder_state_inputs_facts.push(fact);
            } else if name == "target_length" {
                decoder_joint_wants_target_length = true;
            }
        }
        let mut decoder_joint_logits_output_index = 0;
        let mut decoder_joint_state_output_indices = vec![];
        for ix in 0..decoder_joint_model.output_count()? {
            let name = decoder_joint_model.output_name(ix)?;
            if name.contains("states") {
                decoder_joint_state_output_indices.push(ix);
            } else if name == "outputs" {
                decoder_joint_logits_output_index = ix;
            }
        }
        ensure!(
            decoder_joint_state_output_indices.len() == decoder_state_inputs_facts.len(),
            "decoder_joint state output/input count mismatch"
        );

        // Detect a prompt-fused encoder: an extra `lang_id` input carries the
        // language for multilingual Nemotron. Plain encoders (Parakeet) lack it.
        let mut encoder_model = Self::load_submodel(enc_model_bytes)?;
        let mut encoder_lang_id = None;
        for ix in 0..encoder_model.input_count()? {
            if encoder_model.input_name(ix)? == "lang_id" {
                encoder_lang_id = Some(runtime_config.default_lang_id);
                log::info!(
                    "encoder has a `lang_id` input; feeding lang_id={}",
                    runtime_config.default_lang_id
                );
            }
        }
        // The encoder runs one sample at a time (per-sample path), so pin its
        // batch symbol to 1. Cache-aware streaming encoders (Nemotron) have
        // pre-encode shape expressions in `BATCH` that tract cannot resolve
        // from a symbolic batch at plan time; concretizing avoids
        // "Undetermined symbol: BATCH". Harmless when the symbol is absent
        // (e.g. Parakeet) or already concrete.
        if !runtime_config.encoder_per_batch {
            encoder_model.concretize_symbols([("BATCH", 1i64)])?;
        }

        let mut rt = runtime_for_name("default")?;
        if !runtime_config.force_cpu {
            if let Ok(r) = runtime_for_name("cuda") {
                rt = r;
            }
            if let Ok(r) = runtime_for_name("metal") {
                rt = r;
            }
        }

        Ok(NemoAsrModel {
            model_config,
            runtime_config,
            preprocessor_model: rt.prepare(Self::load_submodel(pre_model_bytes)?)?,
            encoder_model: rt.prepare(encoder_model)?,
            decoder_joint_model: rt.prepare(decoder_joint_model)?,
            decoder_state_inputs_facts,
            encoder_lang_id,
            decoder_joint_wants_target_length,
            decoder_joint_logits_output_index,
            decoder_joint_state_output_indices,
        })
    }

    pub fn from_dir(path: PathBuf) -> Res<NemoAsrModel> {
        Self::from_dir_with_runtime_config(path, None)
    }

    pub fn from_dir_with_runtime_config(
        path: PathBuf,
        runtime_config_as_u8: Option<Vec<u8>>,
    ) -> Res<NemoAsrModel> {
        let runtime_config_path = path.join("runtime_config.json");
        let model_config_path = path.join("model_config.json");
        let pre_model_path = path.join("preprocessor.nnef.tgz");
        let enc_model_path = path.join("encoder.nnef.tgz");
        let dec_model_path = path.join("decoder_joint.nnef.tgz");
        log::info!("start loading nemo asr model from dir: {:?}", path);

        let runtime_config_bytes = runtime_config_as_u8.or(std::fs::read(runtime_config_path).ok());
        let model_config_bytes =
            std::fs::read(model_config_path).expect("Failed to read model config file");
        let pre_model_bytes =
            std::fs::read(pre_model_path).expect("Failed to read preprocessor model file");
        let enc_model_bytes =
            std::fs::read(enc_model_path).expect("Failed to read encoder model file");
        let dec_model_bytes =
            std::fs::read(dec_model_path).expect("Failed to read decoder model file");

        Self::from_bytes(
            &model_config_bytes,
            runtime_config_bytes.as_deref(),
            &pre_model_bytes,
            &enc_model_bytes,
            &dec_model_bytes,
        )
    }

    /// Convert a single wav file path to a single input tensor
    fn wav_path_to_tensor(&self, wav_path: &PathBuf) -> Res<Value> {
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
        let input_tensor = Array2::from_shape_vec((1, samples.len()), samples).unwrap();
        input_tensor.try_into()
    }

    /// Infer from a wav file path all at once
    pub fn infer_from_wav_paths(&self, wav_paths: &[PathBuf]) -> Res<Vec<Transcription>> {
        log::debug!("Loading wav file from path: {:?}", wav_paths);
        let input_tensor_vec = wav_paths
            .iter()
            .map(|wp| self.wav_path_to_tensor(wp).unwrap())
            .collect::<Vec<Value>>();
        log::debug!("wav loaded correctly, starting inference");

        log::debug!("prepare input tensor batch");
        let lengths = input_tensor_vec
            .iter()
            .map(|t| Ok(t.shape()?[1] as i64))
            .collect::<Res<Vec<i64>>>()?;

        // Build input tensor batch {
        let mut input_tensor = Array2::<f32>::zeros((
            input_tensor_vec.len(),
            lengths.iter().max().copied().unwrap() as usize,
        ));

        for (ix, itensor) in input_tensor_vec.iter().enumerate() {
            let sample_len = itensor.shape()?[1];
            input_tensor
                .slice_mut(s![ix, 0..sample_len])
                .assign(&itensor.view()?.index_axis(Axis(0), 0));
        }
        // }

        let lengths_tensor: Value =
            Array1::<i64>::from_shape_vec((wav_paths.len(),), lengths)?.try_into()?;
        log::debug!("ready input tensor batch");
        let transcripts = self.infer_from_tensor(input_tensor.try_into()?, lengths_tensor)?;
        Ok(transcripts)
    }

    /// Build the `lang_id` input tensor ([1] i64) when the encoder is
    /// prompt-fused, else `None`.
    fn lang_id_value(&self) -> Res<Option<Value>> {
        match self.encoder_lang_id {
            Some(id) => Ok(Some(
                Array1::<i64>::from_shape_vec((1,), vec![id])?.try_into()?,
            )),
            None => Ok(None),
        }
    }

    fn run_encoder(&self, preprocessor_output: Vec<Value>) -> Res<Vec<Value>> {
        if let Some(dump_dir) = &self.runtime_config.dump_intermediate_io_path {
            dump_value_vec_to_file(
                &preprocessor_output.to_vec(),
                dump_dir.join("encoder_inputs"),
            )?;
        }

        let encoder_output = if self.runtime_config.encoder_per_batch {
            log::debug!("running encoder in full batch mode");
            let mut inputs = preprocessor_output;
            if let Some(lang) = self.lang_id_value()? {
                inputs.push(lang);
            }
            self.encoder_model.run(inputs)?
        } else {
            log::debug!("running encoder in batch mode");
            let features = preprocessor_output[0].view::<f32>()?;
            let lengths = preprocessor_output[1].view::<i64>()?;
            let batch_size = features.shape()[0];
            let mut encoder_out = vec![];
            let mut encoder_len = vec![];
            for b in 0..batch_size {
                // Slice one sample
                let feat_b: Value = features.slice_axis(Axis(0), (b..b + 1).into()).try_into()?;

                let len_b: Value = lengths.slice_axis(Axis(0), (b..b + 1).into()).try_into()?;
                // Run encoder for a single sample (append lang_id when fused)
                let mut inputs = vec![feat_b, len_b];
                if let Some(lang) = self.lang_id_value()? {
                    inputs.push(lang);
                }
                let encoder_output_sample = self.encoder_model.run(inputs)?;
                encoder_out.push(encoder_output_sample[0].view::<f32>()?.to_owned());
                encoder_len.push(encoder_output_sample[1].view::<i64>()?.to_owned());
            }

            vec![
                tract_ndarray::concatenate(
                    Axis(0),
                    &encoder_out.iter().map(|a| a.view()).collect::<Vec<_>>(),
                )?
                .try_into()?,
                tract_ndarray::concatenate(
                    Axis(0),
                    &encoder_len.iter().map(|a| a.view()).collect::<Vec<_>>(),
                )?
                .try_into()?,
            ]
        };
        if let Some(dump_dir) = &self.runtime_config.dump_intermediate_io_path {
            dump_value_vec_to_file(&encoder_output.to_vec(), dump_dir.join("encoder_outputs"))?;
        }
        log::debug!("successfully ran encoder");
        Ok(encoder_output)
    }

    fn infer_from_tensor(
        &self,
        input_tensor: Value,
        length_tensor: Value,
    ) -> Res<Vec<Transcription>> {
        log::debug!("start inference preprocessor");

        // Preprocessor inference
        let preprocessor_output = self.preprocessor_model.run([input_tensor, length_tensor])?;
        log::debug!("successfully ran preprocessor");

        // Encoder inference
        log::debug!("start inference encoder");
        let encoder_output = self.run_encoder(preprocessor_output)?;
        log::debug!("successfully ran encoder");

        log::debug!("start running decoder and joint");
        let transcripts = self.decode_transcripts_from_encoder_output(encoder_output)?;

        log::debug!("successfully ran decoder and joint");
        Ok(transcripts)
    }

    fn get_initial_decoder_states(&self, batch_size: usize) -> Res<Vec<[usize; 3]>> {
        // t2n names the batch symbol "BATCH" (tie_batch_symbols); the decoder
        // state facts are [2, BATCH, 640].
        let values = [("BATCH", batch_size as i64)];

        let mut shapes = vec![];
        for fact in &self.decoder_state_inputs_facts {
            let mut shape = [0; 3];
            for (ix, s) in shape.iter_mut().enumerate() {
                let dim = fact.dim(ix)?.eval(values)?.to_int64()? as usize;
                *s = dim;
            }
            shapes.push(shape);
        }
        Ok(shapes)
    }

    pub fn decode_transcripts_from_encoder_output(
        &self,
        encoder_out: Vec<Value>,
    ) -> Res<Vec<Transcription>> {
        let feats = encoder_out[0].view::<f32>()?;
        let lens = encoder_out[1].view::<i64>()?;
        let bsz = feats.shape()[0];

        let blank = self.model_config.get_blank_index();
        let vocab = &self.model_config.labels;
        let durations = &self.model_config.decoding.durations;

        // Initialize lanes
        let mut lanes: Vec<Lane> = (0..bsz)
            .map(|b| {
                let states = self
                    .get_initial_decoder_states(1)?
                    .into_iter()
                    .map(|shape| Array3::<f32>::zeros(shape).try_into())
                    .collect::<Res<_>>()?;
                Ok(Lane {
                    encoder_len: lens[b] as usize,
                    current_frame: 0,
                    last_token: blank,
                    states,
                    transcript: Vec::new(),
                    symbols_added: 0,
                    need_loop: true,
                })
            })
            .collect::<Res<_>>()?;

        let vocab_sz = vocab.len() + 1;
        let dur_sz = durations.len();

        // --- Main decoding loop ---
        loop {
            let active: Vec<usize> = lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| {
                    l.current_frame < l.encoder_len
                        && (l.need_loop
                            || self
                                .runtime_config
                                .max_n_tokens_per_step
                                .is_none_or(|m| l.symbols_added < m))
                })
                .map(|(i, _)| i)
                .collect();

            if active.is_empty() {
                break;
            }

            // Encoder frames [B, C, 1]
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
            .try_into()?;

            let labels = Array2::<i32>::from_shape_vec(
                (active.len(), 1),
                active.iter().map(|&i| lanes[i].last_token as i32).collect(),
            )?
            .try_into()?;

            let packed_states = Self::state_pack_lanes(&lanes, &active)?;
            let mut inputs = vec![enc_frames, labels];
            if self.decoder_joint_wants_target_length {
                // one predicted token per lane -> length 1
                let target_length: Value =
                    Array1::<i32>::from_elem((active.len(),), 1i32).try_into()?;
                inputs.push(target_length);
            }
            inputs.extend(packed_states);

            let outs = self.decoder_joint_model.run(inputs)?;
            let logits = outs[self.decoder_joint_logits_output_index].view::<f32>()?;
            log::debug!("Decoder joint step done");

            for (k, &lane_ix) in active.iter().enumerate() {
                let row = logits.index_axis(Axis(0), k);

                // --- Token ---
                let (tok, tok_logp) = row
                    .iter()
                    .take(vocab_sz)
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                // --- Advance (skip) ---
                // TDT reads a duration head appended after the vocab logits.
                // Plain RNNT has none: advance one encoder frame on blank,
                // stay (emit) otherwise.
                let skip = if durations.is_empty() {
                    usize::from(tok == blank)
                } else {
                    let (dur_idx, _) = row
                        .iter()
                        .skip(vocab_sz)
                        .take(dur_sz)
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap();
                    durations[dur_idx]
                };
                let lane = &mut lanes[lane_ix];

                lane.need_loop = skip == 0;
                lane.symbols_added += 1;

                // Emit token
                if tok != blank {
                    lane.transcript.push(TranscriptItem {
                        token: vocab[tok].clone(),
                        emitted_at_encoder_timestep: lane.current_frame,
                        emitted_at_encoder_timestep_iteration: lane.symbols_added - 1,
                        logit: *tok_logp,
                    });

                    lane.last_token = tok;

                    // Update predictor states (located by output name)
                    for (sid, &oidx) in self.decoder_joint_state_output_indices.iter().enumerate() {
                        lane.states[sid] = Self::state_take_lane(&outs[oidx], k)?;
                    }
                }

                if skip > 0 {
                    lane.current_frame += skip;
                    lane.symbols_added = 0;
                }
                lane.need_loop = true;
            }

            // max_symbols guard
            if let Some(max_symbols) = self.runtime_config.max_n_tokens_per_step {
                for lane in lanes.iter_mut() {
                    if lane.symbols_added >= max_symbols {
                        lane.current_frame += 1; // force advance by 1 frame
                        lane.symbols_added = 0;
                        lane.need_loop = true;
                    }
                }
            }
        }

        Ok(lanes
            .into_iter()
            .map(|l| Transcription::from_transcript_items(l.transcript))
            .collect())
    }

    fn state_take_lane(state: &Value, b: usize) -> Res<Value> {
        let v = state.view::<f32>()?;
        v.index_axis(Axis(1), b).insert_axis(Axis(1)).try_into()
    }

    fn state_pack_lanes(lanes: &[Lane], active: &[usize]) -> Res<Vec<Value>> {
        let n_states = lanes[active[0]].states.len();
        let mut packed = vec![];
        for sid in 0..n_states {
            let views: Vec<_> = active
                .iter()
                .map(|&lx| lanes[lx].states[sid].view::<f32>())
                .collect::<Res<_>>()?;
            let cat = tract_ndarray::concatenate(Axis(1), &views)?;
            packed.push(cat.try_into()?);
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
    fn test_load_and_decode_audio() -> Res<()> {
        println!("Assets dir: {:?}\n", assets_dir());
        let asr = NemoAsrModel::from_dir(assets_dir().join("model"))?;
        println!("Loaded ASR model successfully");
        let transcripts = asr.infer_from_wav_paths(&[
            assets_dir().join("2086-149220-0033.wav"),
            assets_dir().join("data_smoke_test_LDC93S1.wav"),
        ])?;
        let max_chars = 500;
        for (i, t) in transcripts.iter().enumerate() {
            println!(
                "Transcription[{}]: '{}'",
                i,
                &truncate_with_ellipsis(&t.text, max_chars)
            );
            // if i == 0 {
            //     println!("Full items: {:#?}", t.items);
            // }
        }
        Ok(())
    }
}
