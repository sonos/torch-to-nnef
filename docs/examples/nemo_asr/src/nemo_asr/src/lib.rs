/// NEMO ASR model inference using tract-nnef
/// Only use full audio inference for now
/// streaming/pulsed inference may be added later
///
/// code adapted from: nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py
/// class ONNXGreedyBatchedRNNTInfer
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tract_core::plan::SimpleState;
use tract_core::tract_data::itertools::Itertools;
use tract_ndarray::s;
use tract_nnef::prelude::*;
use tract_nnef::tract_ndarray::Axis;

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

impl NemoAsrConfig {
    pub fn get_blank_index(&self) -> usize {
        return self.labels.len();
    }
}

/// code a struct that represent nemo ASR model inference
/// it load a model from_dir containing model files:
#[derive(Debug, Clone)]
pub struct NemoAsrModel {
    preprocessor_model: TypedRunnableModel<TypedModel>,
    encoder_model: TypedRunnableModel<TypedModel>,
    decoder_joint_model: TypedRunnableModel<TypedModel>,
    config: NemoAsrConfig,
    max_symbols_per_step: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptItem {
    pub token: String,
    pub timestep: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Transcription {
    pub text: String,
    pub items: Vec<TranscriptItem>,
}

impl NemoAsrModel {
    fn from_bytes(
        config_bytes: &[u8],
        pre_model_bytes: &[u8],
        enc_model_bytes: &[u8],
        dec_model_bytes: &[u8],
    ) -> TractResult<NemoAsrModel> {
        let config = serde_json::from_slice::<NemoAsrConfig>(config_bytes)?;
        let mut pre_read = std::io::Cursor::new(pre_model_bytes);
        let preprocessor_model = tract_nnef::nnef()
            .with_tract_core()
            .model_for_read(&mut pre_read)?
            .into_optimized()?
            .into_runnable()?;

        let mut enc_read = std::io::Cursor::new(enc_model_bytes);
        let encoder_model = tract_nnef::nnef()
            .with_tract_core()
            .model_for_read(&mut enc_read)?
            .into_optimized()?
            .into_runnable()?;

        let mut dec_read = std::io::Cursor::new(dec_model_bytes);
        let decoder_joint_model = tract_nnef::nnef()
            .with_tract_core()
            .model_for_read(&mut dec_read)?
            .into_optimized()?
            .into_runnable()?;
        Ok(NemoAsrModel {
            preprocessor_model,
            encoder_model,
            decoder_joint_model,
            config,
            max_symbols_per_step: None,
        })
    }

    pub fn from_dir(path: PathBuf) -> TractResult<NemoAsrModel> {
        let config_path = path.join("model_config.json");
        let pre_model_path = path.join("preprocessor.nnef.tgz");
        let enc_model_path = path.join("encoder.nnef.tgz");
        let dec_model_path = path.join("decoder_joint.nnef.tgz");

        let config_bytes = std::fs::read(config_path).expect("Failed to read model config file");
        let pre_model_bytes =
            std::fs::read(pre_model_path).expect("Failed to read preprocessor model file");
        let enc_model_bytes =
            std::fs::read(enc_model_path).expect("Failed to read encoder model file");
        let dec_model_bytes =
            std::fs::read(dec_model_path).expect("Failed to read decoder model file");

        NemoAsrModel::from_bytes(
            config_bytes.as_slice(),
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
            spec.sample_rate, self.config.sample_rate as u32,
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
        let input_tensor_vec = wav_paths
            .iter()
            .map(|wp| self.wav_path_to_tensor(wp).unwrap())
            .collect::<Vec<Tensor>>();

        let lengths: Tensor = tract_ndarray::Array1::<i64>::from_shape_vec(
            (wav_paths.len(),),
            input_tensor_vec
                .iter()
                .map(|t| t.shape()[1] as i64)
                .collect::<Vec<i64>>(),
        )?
        .into_tensor();

        let input_tensor = tract_ndarray::concatenate(
            Axis(0),
            &input_tensor_vec
                .iter()
                .map(|t| t.to_array_view::<f32>().unwrap())
                .collect::<Vec<_>>(),
        )?
        .into_tensor();
        let transcripts = self.infer_from_tensor(input_tensor, lengths)?;
        Ok(transcripts)
    }

    fn infer_from_tensor(
        &self,
        input_tensor: Tensor,
        length_tensor: Tensor,
    ) -> TractResult<Vec<Transcription>> {
        // Preprocessor inference
        let preprocessor_output = self.preprocessor_model.run(tvec!(
            input_tensor.into_tvalue(),
            length_tensor.into_tvalue()
        ))?;

        // Encoder inference
        let encoder_output = self.encoder_model.run(preprocessor_output)?;

        // Decoder inference
        self.decode_logits(encoder_output)
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

    fn decode_logits(&self, encoder_output: TVec<TValue>) -> TractResult<Vec<Transcription>> {
        // Copy of part in ../example.py in rust (post encoder)
        let vocab = &self.config.labels;
        let batch_size = encoder_output[0].to_array_view::<f32>()?.shape()[0];
        let blank_index = self.config.get_blank_index();
        let out_len = encoder_output[1].to_array_view::<i64>()?;

        // heuristic: max
        // output length is 2x max encoder output length
        let max_output_length = out_len.iter().max().copied().unwrap() * 2;

        let total_n_labels = vocab.len() + 1; // +1 for blank
        let target_lengths: Tensor =
            tract_ndarray::Array1::<i32>::from_elem(batch_size, 1i32).into();

        let mut hyp: Vec<Vec<usize>> = vec![Vec::new(); out_len.len()];
        let mut input_states = self.get_initial_decoder_states(batch_size)?;
        let mut state = SimpleState::new(self.decoder_joint_model.clone())?;
        let mut timesteps: Vec<Vec<usize>> = vec![Vec::new(); out_len.len()];
        let mut last_turn_token_ixes: Vec<usize> = vec![blank_index; batch_size];
        // tracking current_frames per batch item (avoid looping)
        let mut current_frames: Vec<usize> = vec![0; batch_size];
        let mut blank_mask: Vec<bool> = vec![true; batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];

        // TODO: drop each sample in batch that exceed max length
        // currently we continue to slice last frame
        // if exceed max length for the related samples

        for _ in 0..max_output_length {
            // use current_frame for each sample in batch
            // instead of slicing full batch at 1 time step
            let encoder_output_view = encoder_output[0].to_array_view::<f32>()?;
            let enc_frame_vec: Vec<tract_ndarray::ArrayView2<f32>> = (0..batch_size)
                .zip(current_frames.iter())
                .map(|(b, current_frame)| {
                    let c_frame = if *current_frame > out_len[b] as usize - 1 {
                        // if exceed max length, just slice the last frame
                        out_len[b] as usize - 1
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
                finished[b] = current_frames[b] as i64 >= out_len[b];
                blank_mask[b] = finished[b];
            }

            if finished.iter().all(|f| *f) {
                break;
            }

            while self.max_symbols_per_step.is_none_or(|m| m > symbols_added) {
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
                let outs = state.run(inps)?;
                // outs → (outputs, target_length, output_states_1, output_states_2)

                // get max logprob for all samples in batch
                let logp = &outs[0]; // logprobs
                let logp_arr = logp.to_array_view::<f32>()?;
                for b in 0..batch_size {
                    let logp_b = logp_arr.index_axis(Axis(0), b);
                    let (max_ix, _max_val) = logp_b
                        .iter()
                        .take(total_n_labels)
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .context("Failed to get max logprob")?;

                    // print!("{}", vocab.get(max_ix).unwrap_or(&"<blank>".to_string()));
                    blank_mask[b] = max_ix == blank_index;
                    if blank_mask[b] {
                        // use token index from previous turn if blank is max
                        if logp_b.len() > total_n_labels {
                            // get how many turn to skip next
                            let (max_ix, _max_val) = logp_b
                                .iter()
                                .skip(total_n_labels)
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                .context("Failed to get max next turn ix")?;
                            current_frames[b] += max_ix;
                        } else {
                            current_frames[b] += 1;
                        }
                    } else {
                        last_turn_token_ixes[b] = max_ix;
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

                            for b in 0..batch_size {
                                if blank_mask[b] {
                                    new_arr
                                        .index_axis_mut(Axis(0), b)
                                        .assign(&prev_arr.index_axis(Axis(0), b));
                                }
                            }

                            new_arr.into_tensor().into_tvalue()
                        })
                        .collect(),
                );

                // collect hypothesis
                for (b, &tok_ix) in last_turn_token_ixes.iter().enumerate() {
                    if !blank_mask[b] {
                        hyp[b].push(tok_ix);
                        timesteps[b].push(current_frames[b] as usize);
                    }
                }
                symbols_added += 1;
            }
        }

        let transcripts: Vec<Transcription> = hyp
            .iter()
            .zip(timesteps.iter())
            .map(|(tokens, times)| Transcription {
                text: tokens
                    .iter()
                    .map(|&ix| vocab.get(ix).unwrap().as_str().replace("▁", " "))
                    .join(""),
                items: tokens
                    .iter()
                    .zip(times.iter())
                    .map(|(&ix, &t)| TranscriptItem {
                        token: vocab.get(ix).unwrap().to_string(),
                        timestep: t,
                    })
                    .collect(),
            })
            .collect();
        Ok(transcripts)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_load_and_decode_audio() -> TractResult<()> {
        let asr = NemoAsrModel::from_dir(PathBuf::from("./model"))?;
        println!("Loaded ASR model successfully");
        // EXPECTED in py: ,▁I▁don't▁wish▁to▁see▁it▁any▁more,▁observed▁Phoebe,▁turning▁away▁her▁eyes.▁It▁is▁certainly▁very▁like▁the▁old▁portrait.
        // OBSERVED in rs: , I don't wish to see it any more, observed Phoe, turning away her eyes.. It is certainly very like the oldrait.
        let transcripts = asr.infer_from_wav_paths(&[PathBuf::from("./2086-149220-0033.wav")])?;
        println!("Transcription: '{}'", &transcripts[0].text);
        println!("ITEMS: '{:?}'", &transcripts[0].items);
        Ok(())
    }
}
