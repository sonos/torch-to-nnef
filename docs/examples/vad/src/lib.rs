use tract_rs::{
    State,
    prelude::{
        tract_ndarray::{Array1, Array2, ArrayView2, Axis, s},
        *,
    },
};
use wasm_bindgen::prelude::*;

type Res<T> = anyhow::Result<T>;

extern crate web_sys;

#[wasm_bindgen]
struct VadClassifier {
    preprocessor_model: Runnable,
    encoder_model: Runnable,
    decoder_model: Runnable,
    vad_session: Option<VadSession>,
}

#[wasm_bindgen]
impl VadClassifier {
    fn load_internal() -> Res<VadClassifier> {
        let rt = runtime_for_name("default")?;
        let preprocessor_model_bytes = include_bytes!("../model/preprocessor.nnef.tgz");
        let nnef = tract_rs::nnef()?;
        let preprocessor_model = rt.prepare(nnef.load_buffer(preprocessor_model_bytes)?)?;

        let enc_model_bytes = include_bytes!("../model/encoder.pulse10.nnef.tgz");
        let encoder_model = rt.prepare(nnef.load_buffer(enc_model_bytes)?)?;

        let dec_model_bytes = include_bytes!("../model/decoder.nnef.tgz");
        let decoder_model = rt.prepare(nnef.load_buffer(dec_model_bytes)?)?;
        web_sys::console::log_1(&"model loaded/optimized".into());
        Ok(VadClassifier {
            preprocessor_model,
            encoder_model,
            decoder_model,
            vad_session: None,
        })
    }

    fn predict_speech_presence_internal(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        if self.vad_session.is_none() {
            self.vad_session = Some(VadSession::new(
                &self.preprocessor_model,
                &self.encoder_model,
                &self.decoder_model,
            )?);
        }
        let session = self.vad_session.as_mut().unwrap();
        session.predict_speech_presence(raw_audio_data)
    }

    pub fn predict_speech_presence(
        &mut self,
        js_raw_audio_data: &web_sys::js_sys::Float32Array,
    ) -> Result<JsValue, JsError> {
        let prediction_res = self.predict_speech_presence_internal(js_raw_audio_data.to_vec());
        let pred = prediction_res.map_err(|err| JsError::new(format!("{:?}", err).as_str()))?;
        Ok(pred.into())
    }

    pub fn load() -> VadClassifier {
        web_sys::console::log_1(&"try loading".into());
        let result = VadClassifier::load_internal()
            .map_err(|err| JsError::new(format!("{:?}", err).as_str()))
            .expect("unable to load");
        result
    }
}

struct VadSession {
    preprocessor_model: Runnable,
    encoder_state: State,
    decoder_model: Runnable,
    audio_buffer: Vec<f32>,
    current_buffer_fill: usize,
    last_score: f32,
    encoder_frame_buffer: Array2<f32>,
}

impl VadSession {
    fn new(preprocessor: &Runnable, encoder: &Runnable, decoder: &Runnable) -> Res<VadSession> {
        let n_encoder_frames_to_aggregate_over = 10;
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_state: encoder.spawn_state()?,
            decoder_model: decoder.clone(),
            audio_buffer: vec![0.0; 160 * 2], // 20ms of audio at 16kHz
            current_buffer_fill: 0,
            last_score: 0.0,
            encoder_frame_buffer: Array2::<f32>::zeros((128, n_encoder_frames_to_aggregate_over)),
        })
    }

    fn predict_speech_presence(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        // Assumed to be called every 20ms or more

        // web_sys::console::debug_1(&"start predict voice presence".into());
        assert!(self.audio_buffer.len() > raw_audio_data.len());

        // prep audio data {
        // roll data
        let start = self.audio_buffer.len() - raw_audio_data.len();
        let end = self.audio_buffer.len();
        self.audio_buffer.copy_within(start..end, 0);
        // add fresh data
        self.audio_buffer[..raw_audio_data.len()].copy_from_slice(&raw_audio_data);

        if self.current_buffer_fill < self.audio_buffer.len() {
            // not enough data yet, return last score
            self.current_buffer_fill += raw_audio_data.len();
            return Ok(self.last_score);
        } else {
            self.current_buffer_fill = 0; // reset fill count after we have enough data
        }

        let nd_audio_data = Array1::from_vec(self.audio_buffer.to_vec()).insert_axis(Axis(0));
        let audio_data_value: Value = nd_audio_data.try_into()?;
        // }
        // run the model on the input
        let pre_result = self.preprocessor_model.run(vec![audio_data_value])?;
        let enc_result = self.encoder_state.run(pre_result)?;

        // aggregate encoder results over multiple frames given:
        // - a pulse of 2
        // - n_encoder_frames_to_aggregate_over=10
        // this means we will have slightly wrong metrics until we reach 5x this place.
        let decoder_input = self.slide_encoder_window(enc_result)?;
        let dec_result = self.decoder_model.run(decoder_input)?;
        // web_sys::console::debug_1(&"model prediction done".into());
        // find and display the max value with its index
        let score: ArrayView2<f32> = dec_result[0].view::<f32>()?.into_dimensionality()?;
        self.last_score = *score.get((0, 1)).unwrap();
        Ok(self.last_score)
    }

    fn slide_encoder_window(&mut self, enc_result: Vec<Value>) -> Res<Vec<Value>> {
        let enc_frame: ArrayView2<f32> = enc_result[0].view::<f32>()?.into_dimensionality()?;
        let n: i32 = enc_frame.shape()[1].try_into()?;
        // roll the buffer to the left by n frame
        let temp = &self.encoder_frame_buffer.slice(s![.., n..]).to_owned();
        self.encoder_frame_buffer
            .slice_mut(s![.., ..-n])
            .assign(temp);
        // add the new n frames to the end of the buffer
        self.encoder_frame_buffer
            .slice_mut(s![.., -n..])
            .assign(&enc_frame.insert_axis(Axis(1)));
        let val: Value = self.encoder_frame_buffer.clone().try_into()?;
        Ok(vec![val])
    }
}
