use anyhow::{bail, ensure};
use tract_rs::{
    State,
    prelude::{
        tract_ndarray::{Array1, Array2, Axis, s},
        *,
    },
};
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;

type Res<T> = anyhow::Result<T>;

extern crate web_sys;

#[inline]
#[cfg(feature = "log-vad")]
fn clog(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}
#[inline]
#[cfg(not(feature = "log-vad"))]
fn clog(_: &str) {}

#[inline]
fn fmt_shape(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

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
        // Better panic messages in the browser console.
        console_error_panic_hook::set_once();
        clog("loading runtime 'default'");
        let rt = runtime_for_name("default")?;
        let preprocessor_model_bytes = include_bytes!("../model/preprocessor.nnef.tgz");
        clog("creating NNEF loader");
        let nnef = tract_rs::nnef()?
            .with_tract_core()? // required for core ops used in models
            .with_pulse()?; // required for pulse-encoded encoder model
        clog("preparing preprocessor model");
        let preprocessor_model = rt.prepare(nnef.load_buffer(preprocessor_model_bytes)?)?;

        let enc_model_bytes = include_bytes!("../model/encoder.pulsed.nnef.tgz");
        clog("preparing encoder model");
        let encoder_model = rt.prepare(nnef.load_buffer(enc_model_bytes)?)?;

        let dec_model_bytes = include_bytes!("../model/decoder.nnef.tgz");
        clog("preparing decoder model");
        let decoder_model = rt.prepare(nnef.load_buffer(dec_model_bytes)?)?;
        clog("model loaded/optimized");
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

    pub fn load() -> Result<VadClassifier, JsError> {
        // Install panic hook as early as possible in public entrypoint too.
        console_error_panic_hook::set_once();
        clog("try loading");
        VadClassifier::load_internal().map_err(|err| JsError::new(&format!("{:?}", err)))
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
    const EXPECTED_PULSE_FRAMES: usize = 8;
    const ENCODER_INPUT_FRAME_SIZE: usize = 160; // 10ms at 16kHz
    const ENCODER_INPUT_NEEDED_IN_AUDIO_SAMPLES: usize =
        Self::EXPECTED_PULSE_FRAMES * Self::ENCODER_INPUT_FRAME_SIZE;

    fn new(preprocessor: &Runnable, encoder: &Runnable, decoder: &Runnable) -> Res<VadSession> {
        let n_encoder_frames_to_aggregate_over = 10;
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_state: encoder.spawn_state()?,
            decoder_model: decoder.clone(),
            // allows for stable stft features extraction
            audio_buffer: vec![0.0; Self::ENCODER_INPUT_NEEDED_IN_AUDIO_SAMPLES + 512],
            // 512 extra for STFT context frames
            current_buffer_fill: 0,
            last_score: 0.0,
            encoder_frame_buffer: Array2::<f32>::zeros((128, n_encoder_frames_to_aggregate_over)),
        })
    }

    fn predict_speech_presence(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        // Assumed to be called every 10ms * EXPECTED_PULSE_FRAMES or more

        // prep audio data {
        // roll data into a fixed-size ring buffer (keep most recent samples)
        let l = self.audio_buffer.len();
        let n = raw_audio_data.len();
        clog(&format!(
            "RB before: buf_len={l}, incoming={n}, current_fill={}",
            self.current_buffer_fill
        ));
        if n >= l {
            // Incoming chunk is larger than buffer: keep only the last l samples
            self.audio_buffer.copy_from_slice(&raw_audio_data[n - l..n]);
            clog(
                "WARNING: incoming chunk bigger than buffer; dropping oldest samples and keeping only the last part that fits",
            );
        } else {
            // Shift older data left by n, append new samples at the end
            self.audio_buffer.copy_within(n..l, 0);
            self.audio_buffer[l - n..].copy_from_slice(&raw_audio_data);
            clog("RB after shift+append done");
        }

        if (self.current_buffer_fill + n) < Self::ENCODER_INPUT_NEEDED_IN_AUDIO_SAMPLES {
            // not enough data yet, return last score
            self.current_buffer_fill += n;
            clog(&format!(
                "RB filling: current_fill={} / {}",
                self.current_buffer_fill,
                Self::ENCODER_INPUT_NEEDED_IN_AUDIO_SAMPLES
            ));
            return Ok(self.last_score);
        }
        self.current_buffer_fill = 0;

        // Prepare strict-length input matching expected STFT context
        // let max = self
        //     .audio_buffer
        //     .clone()
        //     .into_iter()
        //     .reduce(f32::max)
        //     .unwrap_or(0.);
        // if max > 1.0 || max < -1.0 {
        //     clog(&format!(
        //         "WARNING: audio sample value {} exceeds expected [-1.0, 1.0] range; ensure proper normalization",
        //         max
        //     ));
        // }
        let audio_buffer_arr = Array1::from_vec(self.audio_buffer.clone()).insert_axis(Axis(0));
        let audio_buffer_value: Value = audio_buffer_arr.try_into()?;
        let mut pre_result = self.preprocessor_model.run(vec![audio_buffer_value])?;
        // Ensure encoder sees a stable pulse length (e.g., 2 frames per call)
        let pre_feat = pre_result[0].view::<f32>()?;
        let frames = pre_feat.shape()[1];
        clog(&format!("PRE frames={frames}"));
        ensure!(
            frames >= Self::EXPECTED_PULSE_FRAMES,
            "number of frames from preprocessor ({frames}) less than expected pulse frames ({}); cannot slice stable window for encoder",
            Self::EXPECTED_PULSE_FRAMES
        );
        let start = frames - Self::EXPECTED_PULSE_FRAMES;
        clog(&format!(
            "PRE slicing last {} frames (start={start})",
            Self::EXPECTED_PULSE_FRAMES
        ));
        let sliced = pre_feat.slice(s![.., start..]).to_owned();
        let new_value: Value = sliced.try_into()?;
        pre_result[0] = new_value;
        clog("ENC run");
        let enc_result = self.encoder_state.run(pre_result)?;

        let decoder_input = self.slide_encoder_window(enc_result)?;
        clog("DEC run");
        let dec_result = self.decoder_model.run(decoder_input)?;
        // Handle decoder output of various shapes; average class-1 across time/batch axes
        let dec_dyn: Array1<f32> = dec_result[0]
            .view::<f32>()?
            .into_dimensionality()?
            .to_owned();
        let shape = dec_dyn.shape().to_vec();
        clog(&format!("DEC dyn shape={}", fmt_shape(&shape)));
        let len_all = dec_dyn.len();
        ensure!(len_all == 2, "DEC <2 elements; keep last score");
        self.last_score = *dec_dyn.get(1).unwrap_or(&0.0);
        Ok(self.last_score)
    }

    fn slide_encoder_window(&mut self, enc_result: Vec<Value>) -> Res<Vec<Value>> {
        // Accept any incoming encoder shape, squeeze to 2D [features, frames]
        clog("SLIDE start");
        let enc_view_dyn = enc_result[0]
            .view::<f32>()?
            .into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
        let frames_usize = *enc_view_dyn.shape().last().unwrap_or(&1);
        let features = enc_view_dyn.len() / frames_usize;
        clog(&format!(
            "ENC raw dyn shape={}, features={}, frames={}",
            fmt_shape(enc_view_dyn.shape()),
            features,
            frames_usize
        ));
        let mut enc_frame = enc_view_dyn
            .to_owned()
            .into_shape((features, frames_usize))
            .map_err(|e| anyhow::anyhow!(format!("reshape encoder frame: {e}")))?;
        // Enforce expected pulse frames for downstream window aggregation
        if frames_usize > Self::EXPECTED_PULSE_FRAMES {
            let start = frames_usize - Self::EXPECTED_PULSE_FRAMES;
            clog(&format!(
                "ENC slicing last {} frames (start={start})",
                Self::EXPECTED_PULSE_FRAMES
            ));
            enc_frame = enc_frame.slice_move(s![.., start..]);
        } else if frames_usize < Self::EXPECTED_PULSE_FRAMES {
            bail!(
                "Encoder output frames ({frames_usize}) less than expected pulse frames ({}); cannot slide window",
                Self::EXPECTED_PULSE_FRAMES
            );
        }
        let n: i32 = Self::EXPECTED_PULSE_FRAMES.try_into()?;
        // roll the buffer to the left by n frame
        clog(&format!(
            "WIN roll: window_shape={}, n={}",
            fmt_shape(self.encoder_frame_buffer.shape()),
            n
        ));
        let temp = &self.encoder_frame_buffer.slice(s![.., n..]).to_owned();
        self.encoder_frame_buffer
            .slice_mut(s![.., ..-n])
            .assign(temp);
        // add the new n frames to the end of the buffer
        self.encoder_frame_buffer
            .slice_mut(s![.., -n..])
            .assign(&enc_frame);
        let val: Value = self.encoder_frame_buffer.clone().try_into()?;
        Ok(vec![val])
    }
}
