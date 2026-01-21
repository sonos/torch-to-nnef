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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RuntimeConfig {
    max_symbols_per_step: Option<usize>,
    force_cpu: bool,
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
                .to_string(),
            items,
        }
    }
}

impl NemoAsrModel {
    fn from_bytes_submodel(
        runtime_config: &RuntimeConfig,
        model_bytes: &[u8],
    ) -> TractResult<TypedRunnableModel<TypedModel>> {
        let mut model_read = std::io::Cursor::new(model_bytes);
        let nnef = tract_nnef::nnef().with_tract_transformers();

        let transform = nnef
            .get_transform("transformers-detect-all")?
            .context("transformers-detect-all not found")?;

        let mut nn = nnef.model_for_read(&mut model_read)?;

        if !runtime_config.force_cpu {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                use crate::tract_core::transform::ModelTransform;
                use std::str::FromStr;
                nn.properties.insert("GPU".into(), rctensor0(true));
                tract_metal::MetalTransform::from_str("")?.transform(&mut nn)?;
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                use tract_core::transform::ModelTransform;
                if tract_cuda::utils::are_culibs_present() {
                    nn.properties.insert("GPU".into(), rctensor0(true));
                    tract_cuda::CudaTransform.transform(&mut nn)?;
                }
            }
        }

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
            serde_json::from_slice::<RuntimeConfig>(rt_conf)?
        } else {
            RuntimeConfig::default()
        };

        let preprocessor_model =
            NemoAsrModel::from_bytes_submodel(&runtime_config, pre_model_bytes)?;
        let encoder_model = NemoAsrModel::from_bytes_submodel(&runtime_config, enc_model_bytes)?;
        let decoder_joint_model =
            NemoAsrModel::from_bytes_submodel(&runtime_config, dec_model_bytes)?;

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
        log::info!("Loading wav file from path: {:?}", wav_paths);
        let input_tensor_vec = wav_paths
            .iter()
            .map(|wp| self.wav_path_to_tensor(wp).unwrap())
            .collect::<Vec<Tensor>>();
        log::info!("wav loaded correctly, starting inference");

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
        log::info!("start inference preprocessor");
        // Preprocessor inference
        let preprocessor_output = self.preprocessor_model.run(tvec!(
            input_tensor.into_tvalue(),
            length_tensor.into_tvalue()
        ))?;
        log::info!("successfully ran preprocessor");

        // Encoder inference
        log::info!("start inference encoder");
        let encoder_output = self.encoder_model.run(preprocessor_output)?;
        log::info!("successfully ran encoder");

        // Decoder inference
        log::info!("start decoder and joint");
        let t = self.decode_transcripts_from_encoder_output(encoder_output);
        log::info!("successfully ran decoder and joint");
        t
    }

    fn get_initial_decoder_states(&self, batch_size: usize) -> TractResult<TVec<TValue>> {
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
                let state_tensor: Tensor = tract_ndarray::Array3::<f32>::zeros(shape).into();
                Ok(state_tensor.into_tvalue())
            })
            .collect()
    }

    fn decode_transcripts_from_encoder_output(
        &self,
        encoder_output: TVec<TValue>,
    ) -> TractResult<Vec<Transcription>> {
        // Copy of part in ../example.py in rust (post encoder)
        let vocab = &self.model_config.labels;
        let batch_size = encoder_output[0].to_array_view::<f32>()?.shape()[0];
        let blank_index = self.model_config.get_blank_index();

        // out_len as vec of usize from encoder output[1] i64
        let out_len = encoder_output[1]
            .to_array_view::<i64>()?
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<usize>>();

        // heuristic: max
        // output length is 2x max encoder output length
        let max_output_length = out_len.iter().max().copied().unwrap() * 2;

        let total_n_labels = vocab.len() + 1; // +1 for blank
        let target_lengths: Tensor =
            tract_ndarray::Array1::<i32>::from_elem(batch_size, 1i32).into();
        let mut transcript_items: Vec<Vec<TranscriptItem>> = vec![Vec::new(); out_len.len()];
        let mut input_states = self.get_initial_decoder_states(batch_size)?;
        let mut last_turn_token_ixes: Vec<usize> = vec![blank_index; batch_size];
        // tracking current_frames per batch item (avoid looping)
        let mut current_frames: Vec<usize> = vec![0; batch_size];
        let mut blank_mask: Vec<bool> = vec![true; batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];

        // TODO: drop each sample in batch that exceed max length
        // currently we continue to slice last frame
        // if exceed max length for the related samples

        for ix in 0..max_output_length {
            // use current_frame for each sample in batch
            // instead of slicing full batch at 1 time step
            let encoder_output_view = encoder_output[0].to_array_view::<f32>()?;
            let enc_frame_vec: Vec<tract_ndarray::ArrayView2<f32>> = current_frames
                .iter()
                .enumerate()
                .map(|(b, current_frame)| {
                    let c_frame = if *current_frame >= out_len[b] {
                        // if exceed max length, just slice the last frame
                        out_len[b] - 1
                    } else {
                        *current_frame
                    };
                    encoder_output_view.slice(s![b, .., c_frame..c_frame + 1])
                })
                .collect();
            let enc_frame = tract_ndarray::stack(Axis(0), &enc_frame_vec)?;
            let mut symbols_added = 0;

            // Update blank mask with time mask
            // Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            // Forcibly mask with "blank" tokens, for all sample where current time step time_ix > seq_len
            for b in 0..finished.len() {
                finished[b] = current_frames[b] >= out_len[b];
                blank_mask[b] = finished[b];
            }

            if finished.iter().all(|f| *f) {
                break;
            }

            while self
                .runtime_config
                .max_symbols_per_step
                .is_none_or(|m| m > symbols_added)
            {
                let last_label_tensor = tract_ndarray::Array2::<i32>::from_shape_vec(
                    (batch_size, 1),
                    last_turn_token_ixes
                        .iter()
                        .map(|&x| x as i32)
                        .collect::<Vec<i32>>(),
                )?
                .into_tensor();
                let mut inps = tvec!(
                    enc_frame.clone().into_tvalue(),
                    last_label_tensor.into_tvalue(),
                    target_lengths.clone().into_tvalue(),
                );
                inps.extend(input_states.clone());
                log::debug!("run nn decoding_joint step {}", ix);
                let outs = self.decoder_joint_model.run(inps)?;
                log::debug!("finished nn decoding_joint step {}", ix);
                // outs → (outputs, target_length, output_states_1, output_states_2)

                // get max logprob for all samples in batch
                let logp = &outs[0]; // logprobs
                let logp_arr = logp.to_array_view::<f32>()?;
                for b in 0..batch_size {
                    let logp_b = logp_arr.index_axis(Axis(0), b);
                    let (selected_token_ix, _max_val) = logp_b
                        .iter()
                        .take(total_n_labels)
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .context("Failed to get max logprob")?;

                    blank_mask[b] = selected_token_ix == blank_index;
                    if blank_mask[b] {
                        // use token index from previous turn if blank is max
                        if logp_b.len() > total_n_labels {
                            // get how many turn to skip next
                            let (selected_jump_ix, _max_val) = logp_b
                                .iter()
                                .skip(total_n_labels)
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                .context("Failed to get max next turn ix")?;
                            current_frames[b] += selected_jump_ix;
                        } else {
                            current_frames[b] += 1;
                        }
                    } else {
                        last_turn_token_ixes[b] = selected_token_ix;
                        // collect hypothesis
                        transcript_items[b].push(TranscriptItem {
                            token: vocab.get(selected_token_ix).unwrap().to_string(),
                            emitted_at_encoder_timestep: current_frames[b],
                            emitted_at_encoder_timestep_iteration: symbols_added,
                        })
                    }
                }

                if blank_mask.iter().zip(&finished).all(|(b, f)| *b || *f) {
                    break;
                }

                // set next step states
                // follow ONNX python implementation where states are maintained to prior state if blank state
                // so far it was observed to have no difference in output with/without this reassignment
                input_states = TVec::from_vec(
                    outs[2..]
                        .iter()
                        .enumerate()
                        .map(|(state_id, t)| {
                            // force back the states where blank was selected
                            let mut new_arr = t.to_array_view::<f32>().unwrap().to_owned();
                            let prev_arr = input_states[state_id].to_array_view::<f32>().unwrap();

                            for (bix, is_blank) in blank_mask.iter().enumerate() {
                                if *is_blank {
                                    new_arr
                                        .index_axis_mut(Axis(1), bix)
                                        .assign(&prev_arr.index_axis(Axis(1), bix));
                                }
                            }

                            new_arr.into_tensor().into_tvalue()
                        })
                        .collect(),
                );
                symbols_added += 1;
            }
            log::debug!("completed decoding of encoder step {}", ix);
        }

        let transcripts: Vec<Transcription> = transcript_items
            .into_iter()
            .map(Transcription::from_transcript_items)
            .collect();
        Ok(transcripts)
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
            // assets_dir().join("data_smoke_test_LDC93S1.wav"),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H03_FIO089_0023485_0026102.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004b_H02_MEE014_0177063_0179451.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H02_FIO084_0063941_0066293.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_EN2002b_H00_FEO070_0042617_0044950.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004c_H02_MEE014_0089550_0091768.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H02_FIO084_0073867_0076040.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009d_H00_FIE088_0019852_0022001.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004c_H01_FEE013_0082509_0084553.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H01_FIO087_0173696_0175737.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004b_H00_MEO015_0070826_0072814.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009c_H02_FIO084_0090140_0092106.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004b_H02_MEE014_0037334_0039212.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_EN2002b_H00_FEO070_0027547_0029410.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004b_H02_MEE014_0030387_0032173.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_EN2002c_H01_FEO072_0095852_0097625.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009d_H02_FIO084_0036376_0038147.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H01_FIO087_0192775_0194536.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_TS3003b_H01_MTD011UID_0059215_0060963.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_EN2002d_H01_FEO072_0044186_0045922.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H02_FIO084_0079506_0081211.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H02_FIO084_0077721_0079412.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004b_H02_MEE014_0138917_0140603.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004b_H02_MEE014_0140693_0142376.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H00_FIE088_0163714_0165367.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H02_FIO084_0067027_0068629.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004c_H01_FEE013_0014472_0016044.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H00_FIE088_0188477_0190040.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009b_H00_FIE088_0194549_0196111.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_IS1009d_H03_FIO089_0077358_0078888.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004b_H01_FEE013_0167575_0169104.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_ES2004d_H02_MEE014_0018731_0020259.wav").into(),
            // Path::new("/Users/julien.balian/SONOS/src/torch-to-nnef/docs/examples/nemo_asr/src/nemo_asr_py/audio_cache/ami/test/AMI_TS3003b_H00_MTD009PM_0007726_0009245.wav").into()
        ])?;
        let max_chars = 200;
        for (i, t) in transcripts.iter().enumerate() {
            println!(
                "Transcription[{}]: '{}'",
                i,
                &truncate_with_ellipsis(&t.text, max_chars)
            );
        }
        // This code works if only 1 sample in batch
        // but output garbage text when multiple samples in batch
        Ok(())
    }
}
