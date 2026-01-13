/// NEMO ASR model inference using tract-nnef
/// Only use full audio inference for now
/// streaming/pulsed inference may be added later
///
/// code adapted from: nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py
/// class ONNXGreedyBatchedRNNTInfer
use anyhow::Context;
use serde::Deserialize;
use std::path::PathBuf;
use tract_core::plan::SimpleState;
use tract_nnef::prelude::*;
use tract_nnef::tract_ndarray::Axis;

use wasm_bindgen::prelude::*;

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
    pub fn is_blank_as_pad(&self) -> bool {
        self.decoder.blank_as_pad
    }

    pub fn get_blank_index(&self) -> usize {
        let blank_tok = if self.is_blank_as_pad() {
            "<pad>"
        } else {
            "<blank>"
        };

        self.labels
            .iter()
            .position(|x| x == blank_tok)
            .unwrap_or(self.labels.len())
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

impl NemoAsrModel {
    fn load_from_bytes(
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
            max_symbols_per_step: Some(100),
        })
    }

    pub fn load_from_path(path: PathBuf) -> TractResult<NemoAsrModel> {
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

        NemoAsrModel::load_from_bytes(
            config_bytes.as_slice(),
            pre_model_bytes.as_slice(),
            enc_model_bytes.as_slice(),
            dec_model_bytes.as_slice(),
        )
    }

    fn wav_path_to_tensor(&self, wav_path: PathBuf) -> TractResult<Tensor> {
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
    pub fn infer_from_wav_path(&self, wav_path: PathBuf) -> TractResult<String> {
        let input_tensor = self.wav_path_to_tensor(wav_path)?;
        let text = self.infer_from_tensor(input_tensor)?;
        Ok(text)
    }

    fn infer_from_tensor(&self, input_tensor: Tensor) -> TractResult<String> {
        let length = input_tensor.shape()[1];
        let length_tensor: Tensor = tract_ndarray::arr1(&[length as i64]).into();

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
                dbg!("Creating initial decoder state: {} with shape {:?}", &node.name, shape);
                let state_tensor: Tensor = tract_ndarray::Array3::<f32>::zeros(shape).into();
                Ok(state_tensor.into_tvalue())
            })
            .collect()
    }

    fn decode_logits(&self, encoder_output: TVec<TValue>) -> TractResult<String> {
        // Copy of part in ../example.py in rust (post encoder)
        let mut t: usize = 0;
        let max_output_length = encoder_output[1]
            .to_array_view::<i32>()?
            .iter()
            .max()
            .copied()
            .unwrap();
        let mut hyp: Vec<usize> = vec![];

        let vocab = &self.config.labels;
        let vocab_len = vocab.len();
        let blank_index = self.config.get_blank_index();

        let mut state = SimpleState::new(self.decoder_joint_model.clone())?;

        // given encoder of Parakeet v3 return:
        //   outputs ━━━ B,1024,(S+7)/8,F32
        //   encoded_lengths ━━━ B,I32
        //
        //   l1275
        let batch_size = encoder_output[0].to_array_view::<f32>()?.shape()[0];
        let mut input_states = self.get_initial_decoder_states(batch_size)?;

        // target_lengths = torch.ones(batchsize, dtype = torch.int32);
        let target_lengths: Tensor = tract_ndarray::Array1::from_elem(batch_size, 1i32).into();

        let mut last_label: Tensor =
            tract_ndarray::Array2::from_elem((batch_size, 1), blank_index as i64).into();

        let mut blank_mask: Tensor = tract_ndarray::Array1::from_elem(batch_size, true).into();

        for time_ix in 0..max_output_length {
            // Get logits for time step t
            let enc_frame = encoder_output[0]
                .to_array_view::<f32>()?
                .index_axis(Axis(2), t)
                .to_owned();
            let mut not_blank = false;
            let mut symbols_added = 0;

            while not_blank && self.max_symbols_per_step.is_none_or(|m| m > symbols_added) {
                // TODO: understand why g is needed
                // if time_ix == 0 && symbols_added == 0 {
                //     hyp.push(blank_index); // initial blank
                // }

                // joint_out, hidden_prime = self.run_decoder_joint(f, g, target_lengths, *hidden)
                let mut inps = tvec!(
                    enc_frame.into_tvalue(),
                    last_label.clone().into_tvalue(),
                    target_lengths.clone().into_tvalue(),
                );
                inps.extend(input_states.clone());
                let outs = state.run(inps)?;
                // outs → (outputs, target_length, output_states_1, output_states_2)
                dbg!("Decoder joint outputs len: {}", outs.len());
                break; // placeholder to avoid infinite loop
            }
        }
        Ok("decoded transcription".to_string())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_load_and_decode_audio() -> TractResult<()> {
        let asr = NemoAsrModel::load_from_path(PathBuf::from("./model"))?;
        println!("Loaded ASR model successfully");
        let text = asr
            .infer_from_wav_path(PathBuf::from("./2086-149220-0033.wav"))
            .unwrap();
        println!("Transcription: '{}'", &text);
        Ok(())
    }
}
