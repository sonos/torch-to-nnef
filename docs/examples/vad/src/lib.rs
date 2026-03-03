#![cfg_attr(target_arch = "wasm32", allow(unused, dead_code))]

use anyhow::{bail, ensure};
use tract_rs::{
    State,
    prelude::{
        tract_ndarray::{Array1, Array2, Axis, IndexLonger, s},
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
    // Pulsed encoder model (stateful)
    encoder_model_pulsed: Runnable,
    // Batch encoder model (stateless)
    encoder_model_batch: Runnable,
    decoder_model: Runnable,
    // Sessions are created lazily on first use for each mode
    vad_session_pulsed: Option<VadSessionPulsed>,
    vad_session_batch: Option<VadSessionBatch>,
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

        // Load pulsed-encoded encoder for stream/pulse mode
        let enc_model_pulsed_bytes = include_bytes!("../model/encoder.pulsed.nnef.tgz");
        clog("preparing encoder model (pulsed)");
        let encoder_model_pulsed = rt.prepare(nnef.load_buffer(enc_model_pulsed_bytes)?)?;

        // Load batch (non-pulsed) encoder for stream/batch mode
        let enc_model_batch_bytes = include_bytes!("../model/encoder.nnef.tgz");
        clog("preparing encoder model (batch)");
        let encoder_model_batch = rt.prepare(nnef.load_buffer(enc_model_batch_bytes)?)?;

        let dec_model_bytes = include_bytes!("../model/decoder.nnef.tgz");
        clog("preparing decoder model");
        let decoder_model = rt.prepare(nnef.load_buffer(dec_model_bytes)?)?;
        clog("model loaded/optimized");
        Ok(VadClassifier {
            preprocessor_model,
            encoder_model_pulsed,
            encoder_model_batch,
            decoder_model,
            vad_session_pulsed: None,
            vad_session_batch: None,
        })
    }

    fn predict_speech_presence_internal(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        if self.vad_session_pulsed.is_none() {
            self.vad_session_pulsed = Some(VadSessionPulsed::new(
                &self.preprocessor_model,
                &self.encoder_model_pulsed,
                &self.decoder_model,
            )?);
        }
        let session = self.vad_session_pulsed.as_mut().unwrap();
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

    // Batch mode API: stateless encoder, large rolling audio buffer
    pub fn predict_speech_presence_batch(
        &mut self,
        js_raw_audio_data: &web_sys::js_sys::Float32Array,
    ) -> Result<JsValue, JsError> {
        // Install panic hook as early as possible in public entrypoint too.
        console_error_panic_hook::set_once();
        if self.vad_session_batch.is_none() {
            // Derive pulse delay from pulsed encoder to align batch outputs
            let pulse_delay: usize = match self.encoder_model_pulsed.property("pulse.delay") {
                Ok(d) => d
                    .view::<i64>()
                    .ok()
                    .map(|v| v.index(0).to_owned() as usize)
                    .unwrap_or(0usize),
                Err(_) => 0usize,
            };
            self.vad_session_batch = Some(
                VadSessionBatch::new(
                    &self.preprocessor_model,
                    &self.encoder_model_batch,
                    &self.decoder_model,
                    pulse_delay,
                )
                .map_err(|err| JsError::new(&format!("{:?}", err)))?,
            );
        }
        let session = self.vad_session_batch.as_mut().unwrap();
        let pred = session
            .predict_speech_presence(js_raw_audio_data.to_vec())
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        Ok(pred.into())
    }

    pub fn load() -> Result<VadClassifier, JsError> {
        // Install panic hook as early as possible in public entrypoint too.
        console_error_panic_hook::set_once();
        clog("try loading");
        VadClassifier::load_internal().map_err(|err| JsError::new(&format!("{:?}", err)))
    }

    // Reset internal streaming sessions so that a new decode (e.g., Run File)
    // starts from a clean state for both pulsed and batch modes.
    pub fn reset_sessions(&mut self) {
        self.vad_session_pulsed = None;
        self.vad_session_batch = None;
    }

    // Expose pulsed parameters and readiness for UI coordination
    pub fn get_pulse_delay(&mut self) -> Result<usize, JsError> {
        // Ensure pulsed session exists to read properties
        if self.vad_session_pulsed.is_none() {
            self.vad_session_pulsed = Some(
                VadSessionPulsed::new(
                    &self.preprocessor_model,
                    &self.encoder_model_pulsed,
                    &self.decoder_model,
                )
                .map_err(|err| JsError::new(&format!("{:?}", err)))?,
            );
        }
        let s = self.vad_session_pulsed.as_ref().unwrap();
        Ok(s.pulse_delay)
    }

    pub fn get_decoder_pool_len(&mut self) -> Result<usize, JsError> {
        if self.vad_session_pulsed.is_none() {
            self.vad_session_pulsed = Some(
                VadSessionPulsed::new(
                    &self.preprocessor_model,
                    &self.encoder_model_pulsed,
                    &self.decoder_model,
                )
                .map_err(|err| JsError::new(&format!("{:?}", err)))?,
            );
        }
        let s = self.vad_session_pulsed.as_ref().unwrap();
        Ok(s.encoder_frame_buffer.shape()[1])
    }

    pub fn is_pulsed_ready(&mut self) -> Result<bool, JsError> {
        if self.vad_session_pulsed.is_none() {
            self.vad_session_pulsed = Some(
                VadSessionPulsed::new(
                    &self.preprocessor_model,
                    &self.encoder_model_pulsed,
                    &self.decoder_model,
                )
                .map_err(|err| JsError::new(&format!("{:?}", err)))?,
            );
        }
        let s = self.vad_session_pulsed.as_ref().unwrap();
        let need_frames = s.pulse_delay + s.encoder_frame_buffer.shape()[1];
        Ok(s.stable_frames_ready >= need_frames)
    }
}

struct VadSessionPulsed {
    preprocessor_model: Runnable,
    encoder_state: State,
    decoder_model: Runnable,
    audio_buffer: Vec<f32>,
    current_buffer_fill: usize,
    last_score: f32,
    encoder_frame_buffer: Array2<f32>,
    pulse_delay: usize,
    warmup_done: bool,
    decoded_emissions: usize,
    stable_frames_ready: usize,
    #[cfg(test)]
    last_pre_feat: Option<Array2<f32>>,
    #[cfg(test)]
    last_pre_sliced: Option<Array2<f32>>,
    #[cfg(test)]
    last_enc_out: Option<Array2<f32>>,
    #[cfg(test)]
    last_enc_block: Option<Array2<f32>>,
    #[cfg(test)]
    last_encoder_window: Option<Array2<f32>>,
    #[cfg(test)]
    last_logits: Option<Array1<f32>>,
    #[cfg(test)]
    last_prob: Option<f32>,
}

impl VadSessionPulsed {
    const EXPECTED_PULSE_FRAMES: usize = 4;
    const ENCODER_INPUT_FRAME_SIZE: usize = 160; // 10ms at 16kHz
    const ENCODER_INPUT_NEEDED_IN_AUDIO_SAMPLES: usize =
        Self::EXPECTED_PULSE_FRAMES * Self::ENCODER_INPUT_FRAME_SIZE;
    // Match batch receptive field to stabilize STFT features for pulsed path too
    const RECEPTIVE_FIELD_FRAMES: usize = 75;
    const RECEPTIVE_FIELD_SAMPLES: usize =
        Self::RECEPTIVE_FIELD_FRAMES * Self::ENCODER_INPUT_FRAME_SIZE;
    // Model emits [non_speech, speech] logits; flip if needed
    const SPEECH_CLASS_INDEX: usize = 1;
    // Suppress first few decoder emissions to avoid startup spike

    fn new(
        preprocessor: &Runnable,
        encoder: &Runnable,
        decoder: &Runnable,
    ) -> Res<VadSessionPulsed> {
        // Pool ~100ms (10 frames) at the decoder for stability
        let n_encoder_frames_to_aggregate_over = 10;
        let pulse_delay_arr = encoder.property("pulse.delay")?; // ensure pulse property exists for sanity check
        let pulse_delay: i64 = pulse_delay_arr.view::<i64>()?.index(0).to_owned();
        assert!(Self::EXPECTED_PULSE_FRAMES <= n_encoder_frames_to_aggregate_over);
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_state: encoder.spawn_state()?,
            decoder_model: decoder.clone(),
            // Use long receptive field to stabilize last-4 preprocessor frames like batch mode
            audio_buffer: vec![0.0; Self::RECEPTIVE_FIELD_SAMPLES],
            // 512 extra for STFT context frames
            current_buffer_fill: 0,
            last_score: f32::NAN,
            encoder_frame_buffer: Array2::<f32>::zeros((128, n_encoder_frames_to_aggregate_over)),
            pulse_delay: pulse_delay.try_into().unwrap_or(0usize),
            warmup_done: false,
            decoded_emissions: 0,
            stable_frames_ready: 0,
            #[cfg(test)]
            last_pre_feat: None,
            #[cfg(test)]
            last_pre_sliced: None,
            #[cfg(test)]
            last_enc_out: None,
            #[cfg(test)]
            last_enc_block: None,
            #[cfg(test)]
            last_encoder_window: None,
            #[cfg(test)]
            last_logits: None,
            #[cfg(test)]
            last_prob: None,
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
        self.current_buffer_fill += n;
        if self.current_buffer_fill < Self::ENCODER_INPUT_NEEDED_IN_AUDIO_SAMPLES {
            // not enough data yet, return last score
            clog(&format!(
                "RB filling: current_fill={} / {}",
                self.current_buffer_fill,
                Self::ENCODER_INPUT_NEEDED_IN_AUDIO_SAMPLES
            ));
            return Ok(self.last_score);
        }
        self.current_buffer_fill = 0;

        // Prepare strict-length input matching expected STFT context
        let max = self
            .audio_buffer
            .clone()
            .into_iter()
            .reduce(f32::max)
            .unwrap_or(0.);
        if !(-1.0..=1.0).contains(&max) {
            bail!(format!(
                "WARNING: audio sample value {} exceeds expected [-1.0, 1.0] range; ensure proper normalization",
                max
            ));
        }
        let audio_buffer_arr = Array1::from_vec(self.audio_buffer.clone());
        let audio_buffer_value_1d: Value = audio_buffer_arr.clone().try_into()?;
        let audio_buffer_value_2d: Value = audio_buffer_arr.insert_axis(Axis(0)).try_into()?;
        // Provide dynamic length input expected by preprocessor: [time] as i64
        let audio_len: i64 = self.audio_buffer.len() as i64;
        let len_arr = Array1::<i64>::from_vec(vec![audio_len]);
        let len_value: Value = len_arr.try_into()?;
        // Try running preprocessor with 1-D input; if that fails, retry with [1, T]
        let mut pre_result = match self
            .preprocessor_model
            .run(vec![audio_buffer_value_1d.clone(), len_value.clone()])
        {
            Ok(r) => r,
            Err(_) => self
                .preprocessor_model
                .run(vec![audio_buffer_value_2d.clone(), len_value.clone()])?,
        };
        // Ensure encoder sees a stable pulse length window
        let pre_any = pre_result[0].view::<f32>()?;
        // Squeeze batch axis if present
        let pre_feat = if pre_any.shape().len() == 3 && pre_any.shape()[0] == 1 {
            pre_any
                .into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix3>()?
                .index_axis(Axis(0), 0)
                .to_owned()
        } else {
            pre_any
                .into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix2>()?
                .to_owned()
        };
        #[cfg(test)]
        {
            self.last_pre_feat = Some(pre_feat.clone());
        }
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
        #[cfg(test)]
        {
            self.last_pre_sliced = Some(sliced.clone());
        }
        let new_value: Value = sliced.try_into()?;
        pre_result[0] = new_value;
        clog("ENC run");
        // Remove length tensor before encoder
        if pre_result.len() > 1 {
            pre_result.remove(1);
        }
        let enc_result = self.encoder_state.run(pre_result)?;
        #[cfg(test)]
        {
            // Capture raw encoder output as [features, frames]
            let enc_view_dyn = enc_result[0]
                .view::<f32>()?
                .into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
            let frames_usize = *enc_view_dyn.shape().last().unwrap_or(&1);
            let features = enc_view_dyn.len() / frames_usize;
            if features > 0 && frames_usize > 0 {
                if let Ok(enc2d) = enc_view_dyn.to_owned().into_shape((features, frames_usize)) {
                    self.last_enc_out = Some(enc2d);
                }
            }
        }

        let decoder_input = self.slide_encoder_window(enc_result)?;
        // Frame-accurate warmup: require encoder right-context (pulse.delay, in frames)
        // plus a fully populated decoder window before first decode.
        let need_frames = self.pulse_delay + self.encoder_frame_buffer.shape()[1];
        if self.stable_frames_ready < need_frames {
            clog(&format!(
                "GATE frames_ready={} need={}",
                self.stable_frames_ready, need_frames
            ));
            return Ok(self.last_score);
        }
        clog("DEC run");
        let dec_result = self.decoder_model.run(decoder_input)?;
        let dec_view = dec_result[0].view::<f32>()?;
        let logits: Array1<f32> = dec_view.into_dimensionality()?.to_owned();
        ensure!(logits.len() == 2, "Decoder output must have 2 logits");
        let mut l0 = logits[0];
        let mut l1 = logits[1];
        if !l0.is_finite() || !l1.is_finite() {
            // Guard against non-finite logits; keep previous score
            return Ok(self.last_score);
        }
        let m = l0.max(l1);
        l0 -= m;
        l1 -= m;
        let e0 = l0.exp();
        let e1 = l1.exp();
        let s = e0 + e1;
        let p1 = if s.is_finite() && s != 0.0 {
            e1 / s
        } else {
            self.last_score
        };
        let p0 = if s.is_finite() && s != 0.0 {
            e0 / s
        } else {
            1.0 - p1
        };
        #[cfg(feature = "log-vad")]
        if self.decoded_emissions < 20 {
            let win_len = self.encoder_frame_buffer.shape()[1];
            clog(&format!(
                "DBG logits: l0={:.4} l1={:.4} p0={:.4} p1={:.4} | win_len={}",
                l0, l1, p0, p1, win_len
            ));
        }
        // Speech corresponds to class index 1 per model labels
        let p_speech = p1;
        self.last_score = p_speech;
        #[cfg(test)]
        {
            self.last_logits = Some(logits);
            self.last_prob = Some(p_speech);
        }

        // Suppress first few emissions to avoid startup spike
        self.decoded_emissions += 1;
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
        let enc_all = enc_view_dyn
            .to_owned()
            .into_shape_with_order((features, frames_usize))
            .map_err(|e| anyhow::anyhow!(format!("reshape encoder frame: {e}")))?;
        // Select frames aligned with pulsed delay: prefer last 4 ending at (T - delay)
        let enc_frame = if frames_usize >= self.pulse_delay + Self::EXPECTED_PULSE_FRAMES {
            let start = frames_usize - self.pulse_delay - Self::EXPECTED_PULSE_FRAMES;
            clog(&format!(
                "ENC slicing delayed {} frames (start={start}, delay={})",
                Self::EXPECTED_PULSE_FRAMES,
                self.pulse_delay
            ));
            enc_all.slice_move(s![.., start..start + Self::EXPECTED_PULSE_FRAMES])
        } else if frames_usize >= Self::EXPECTED_PULSE_FRAMES {
            // Early warmup: fall back to absolute last 4 frames
            let start = frames_usize - Self::EXPECTED_PULSE_FRAMES;
            clog(&format!(
                "ENC slicing tail {} frames (start={start})",
                Self::EXPECTED_PULSE_FRAMES
            ));
            enc_all.slice_move(s![.., start..])
        } else {
            // Not enough frames yet from pulsed encoder; keep window unchanged and do not advance readiness.
            let val: Value = self.encoder_frame_buffer.clone().try_into()?;
            return Ok(vec![val]);
        };
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
        #[cfg(test)]
        {
            self.last_enc_block = Some(enc_frame.clone());
            self.last_encoder_window = Some(self.encoder_frame_buffer.clone());
        }
        // Track stable frames produced so far for warmup gating.
        self.stable_frames_ready = self
            .stable_frames_ready
            .saturating_add(Self::EXPECTED_PULSE_FRAMES);
        let val: Value = self.encoder_frame_buffer.clone().try_into()?;
        Ok(vec![val])
    }
}

// Batch-mode session: stateless encoder, large rolling audio buffer matching receptive field
struct VadSessionBatch {
    preprocessor_model: Runnable,
    encoder_model: Runnable,
    decoder_model: Runnable,
    audio_buffer: Vec<f32>,
    last_score: f32,
    encoder_frame_buffer: Array2<f32>,
    // Count of stable encoder frames produced so far (for warmup gating)
    stable_frames_ready: usize,
    // Use pulse delay from pulsed model to align batch outputs
    pulse_delay: usize,
    #[cfg(test)]
    last_pre_feat: Option<Array2<f32>>,
    #[cfg(test)]
    last_pre_sliced: Option<Array2<f32>>,
    #[cfg(test)]
    last_enc_out: Option<Array2<f32>>,
    #[cfg(test)]
    last_enc_block: Option<Array2<f32>>,
    #[cfg(test)]
    last_encoder_window: Option<Array2<f32>>,
    #[cfg(test)]
    last_logits: Option<Array1<f32>>,
    #[cfg(test)]
    last_prob: Option<f32>,
}

impl VadSessionBatch {
    // For batch mode, use same decoder window params as pulsed
    const EXPECTED_PULSE_FRAMES: usize = 4; // process in 4-frame steps
    const ENCODER_INPUT_FRAME_SIZE: usize = 160; // 10ms at 16kHz
    // Encoder receptive field per request: 75 frames (750ms)
    const RECEPTIVE_FIELD_FRAMES: usize = 75;
    const RECEPTIVE_FIELD_SAMPLES: usize =
        Self::RECEPTIVE_FIELD_FRAMES * Self::ENCODER_INPUT_FRAME_SIZE;

    fn new(
        preprocessor: &Runnable,
        encoder: &Runnable,
        decoder: &Runnable,
        pulse_delay: usize,
    ) -> Res<VadSessionBatch> {
        let n_encoder_frames_to_aggregate_over = 10; // pool ~100ms at decoder
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_model: encoder.clone(),
            decoder_model: decoder.clone(),
            // Big rolling buffer initialized with zeros to satisfy receptive field
            audio_buffer: vec![0.0; Self::RECEPTIVE_FIELD_SAMPLES],
            last_score: 0.0,
            encoder_frame_buffer: Array2::<f32>::zeros((128, n_encoder_frames_to_aggregate_over)),
            stable_frames_ready: 0,
            pulse_delay,
            #[cfg(test)]
            last_pre_feat: None,
            #[cfg(test)]
            last_pre_sliced: None,
            #[cfg(test)]
            last_enc_out: None,
            #[cfg(test)]
            last_enc_block: None,
            #[cfg(test)]
            last_encoder_window: None,
            #[cfg(test)]
            last_logits: None,
            #[cfg(test)]
            last_prob: None,
        })
    }

    fn predict_speech_presence(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        // Roll new audio into big receptive-field buffer
        let l = self.audio_buffer.len();
        let n = raw_audio_data.len();
        if n >= l {
            // Keep only the last l samples if block is too large
            self.audio_buffer.copy_from_slice(&raw_audio_data[n - l..n]);
        } else {
            self.audio_buffer.copy_within(n..l, 0);
            self.audio_buffer[l - n..].copy_from_slice(&raw_audio_data);
        }

        // Sanity on ranges
        let max = self
            .audio_buffer
            .iter()
            .copied()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        if max > 1.0 {
            bail!(format!(
                "WARNING: audio sample abs value {} exceeds expected [-1.0, 1.0] range; ensure proper normalization",
                max
            ));
        }

        // Run preprocessor on full receptive-field buffer
        let audio_arr = Array1::from_vec(self.audio_buffer.clone());
        let audio_val_1d: Value = audio_arr.clone().try_into()?;
        let audio_val_2d: Value = audio_arr.insert_axis(Axis(0)).try_into()?;
        let audio_len: i64 = self.audio_buffer.len() as i64;
        let len_val: Value = Array1::<i64>::from_vec(vec![audio_len]).try_into()?;
        let mut pre_result = match self
            .preprocessor_model
            .run(vec![audio_val_1d, len_val.clone()])
        {
            Ok(r) => r,
            Err(_) => self
                .preprocessor_model
                .run(vec![audio_val_2d, len_val.clone()])?,
        };

        // Ensure encoder sees 2D [features, frames] (squeeze batch axis if present)
        let pre_any = pre_result[0].view::<f32>()?;
        clog(&format!(
            "BATCH PRE raw shape={}",
            fmt_shape(pre_any.shape())
        ));
        let pre_feat = if pre_any.shape().len() == 3 && pre_any.shape()[0] == 1 {
            pre_any
                .into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix3>()?
                .index_axis(Axis(0), 0)
                .to_owned()
        } else {
            pre_any
                .into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix2>()?
                .to_owned()
        };
        clog(&format!(
            "BATCH PRE squeezed shape={}",
            fmt_shape(pre_feat.shape())
        ));
        #[cfg(test)]
        {
            self.last_pre_feat = Some(pre_feat.clone());
        }
        let pre_val: Value = pre_feat.try_into()?;
        pre_result[0] = pre_val;
        // Remove dynamic length tensor before encoder if present
        if pre_result.len() > 1 {
            pre_result.remove(1);
        }

        // Run batch encoder on full feature sequence (stateless)
        clog("BATCH ENC run (try 2D)");
        let enc_result = match self.encoder_model.run(pre_result.clone()) {
            Ok(r) => r,
            Err(e2d) => {
                clog(&format!("BATCH ENC retry with batch axis: {}", e2d));
                // Retry with explicit batch axis [1, F, T]
                let pre_any = pre_result[0].view::<f32>()?;
                let pre2 =
                    pre_any.into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix2>()?;
                let with_b: Value = pre2.insert_axis(Axis(0)).try_into()?;
                let mut vv = pre_result.clone();
                vv[0] = with_b;
                self.encoder_model.run(vv)?
            }
        };

        // Extract aligned 4-frame block directly from encoder output, ending at T - delay - 1 when possible
        let enc_view_dyn = enc_result[0]
            .view::<f32>()?
            .into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
        let frames_usize = *enc_view_dyn.shape().last().unwrap_or(&1);
        let features = enc_view_dyn.len() / frames_usize;
        let enc_all = enc_view_dyn
            .to_owned()
            .into_shape_with_order((features, frames_usize))
            .map_err(|e| anyhow::anyhow!(format!("reshape encoder frame: {e}")))?;
        #[cfg(test)]
        {
            self.last_enc_out = Some(enc_all.clone());
        }
        ensure!(
            features == 128,
            "expected 128 encoder features, got {}",
            features
        );
        ensure!(
            frames_usize >= Self::EXPECTED_PULSE_FRAMES,
            "encoder produced too few frames: {}",
            frames_usize
        );
        let block = if frames_usize >= self.pulse_delay + Self::EXPECTED_PULSE_FRAMES {
            let start = frames_usize - self.pulse_delay - Self::EXPECTED_PULSE_FRAMES;
            enc_all.slice_move(s![.., start..start + Self::EXPECTED_PULSE_FRAMES])
        } else {
            enc_all.slice_move(s![
                ..,
                frames_usize - Self::EXPECTED_PULSE_FRAMES..frames_usize
            ])
        };
        #[cfg(test)]
        {
            self.last_enc_block = Some(block.clone());
            if let Some(pre) = &self.last_pre_feat {
                let t = pre.shape()[1];
                let start = if t >= self.pulse_delay + Self::EXPECTED_PULSE_FRAMES {
                    t - self.pulse_delay - Self::EXPECTED_PULSE_FRAMES
                } else if t >= Self::EXPECTED_PULSE_FRAMES {
                    t - Self::EXPECTED_PULSE_FRAMES
                } else {
                    0
                };
                let pre_blk = pre
                    .slice(s![.., start..start + Self::EXPECTED_PULSE_FRAMES])
                    .to_owned();
                self.last_pre_sliced = Some(pre_blk);
            }
        }
        // Slide into decoder window buffer
        let n: i32 = Self::EXPECTED_PULSE_FRAMES.try_into()?;
        let temp = &self.encoder_frame_buffer.slice(s![.., n..]).to_owned();
        self.encoder_frame_buffer
            .slice_mut(s![.., ..-n])
            .assign(temp);
        self.encoder_frame_buffer
            .slice_mut(s![.., -n..])
            .assign(&block);
        #[cfg(test)]
        {
            self.last_encoder_window = Some(self.encoder_frame_buffer.clone());
        }
        let dec_in = {
            let val: Value = self.encoder_frame_buffer.clone().try_into()?;
            vec![val]
        };
        // Track produced stable frames (advance by processed step) only when a delayed block was added
        self.stable_frames_ready = self
            .stable_frames_ready
            .saturating_add(Self::EXPECTED_PULSE_FRAMES);

        // Warmup: require encoder right-context (pulse_delay) plus a fully populated decoder window
        let need_frames = self.pulse_delay + self.encoder_frame_buffer.shape()[1];
        if self.stable_frames_ready < need_frames {
            return Ok(self.last_score);
        }

        // Decode single logit vector
        let dec_result = self.decoder_model.run(dec_in)?;
        let dec_view = dec_result[0].view::<f32>()?;
        let logits: Array1<f32> = dec_view.into_dimensionality()?.to_owned();
        ensure!(logits.len() == 2, "Decoder output must have 2 logits");
        let mut l0 = logits[0];
        let mut l1 = logits[1];
        if !l0.is_finite() || !l1.is_finite() {
            return Ok(self.last_score);
        }
        let m = l0.max(l1);
        l0 -= m;
        l1 -= m;
        let e0 = l0.exp();
        let e1 = l1.exp();
        let s = e0 + e1;
        let p1 = if s.is_finite() && s != 0.0 {
            e1 / s
        } else {
            self.last_score
        };
        self.last_score = p1;
        #[cfg(test)]
        {
            self.last_logits = Some(logits);
            self.last_prob = Some(p1);
        }
        Ok(self.last_score)
    }

    fn slide_encoder_window(
        win: &mut Array2<f32>,
        enc_result: Vec<Value>,
        pulse_delay: usize,
    ) -> Res<Vec<Value>> {
        let enc_view_dyn = enc_result[0]
            .view::<f32>()?
            .into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
        let frames_usize = *enc_view_dyn.shape().last().unwrap_or(&1);
        let features = enc_view_dyn.len() / frames_usize;
        let enc_all = enc_view_dyn
            .to_owned()
            .into_shape_with_order((features, frames_usize))
            .map_err(|e| anyhow::anyhow!(format!("reshape encoder frame: {e}")))?;
        // Select frames aligned with pulsed delay: prefer last 4 ending at (T - delay).
        // During early warmup when there are not enough frames to honor the delay,
        // fall back to the absolute last 4 frames (gating will prevent decode until ready).
        // Empirical alignment: pulsed encoder stable output lags batch by ~1 frame.
        // Align batch frames to end at (T - delay - 1).
        let align_shift: usize = 1;
        let enc_frame = if frames_usize >= pulse_delay + Self::EXPECTED_PULSE_FRAMES + align_shift {
            let start = frames_usize - pulse_delay - Self::EXPECTED_PULSE_FRAMES - align_shift;
            enc_all.slice_move(s![.., start..start + Self::EXPECTED_PULSE_FRAMES])
        } else if frames_usize >= Self::EXPECTED_PULSE_FRAMES {
            enc_all.slice_move(s![
                ..,
                frames_usize - Self::EXPECTED_PULSE_FRAMES..frames_usize
            ])
        } else {
            bail!(
                "Batch encoder frames ({frames_usize}) less than required step ({}); cannot build decoder window yet",
                Self::EXPECTED_PULSE_FRAMES
            );
        };
        let n: i32 = Self::EXPECTED_PULSE_FRAMES.try_into()?;
        let temp = &win.slice(s![.., n..]).to_owned();
        win.slice_mut(s![.., ..-n]).assign(temp);
        win.slice_mut(s![.., -n..]).assign(&enc_frame);
        let val: Value = win.clone().try_into()?;
        Ok(vec![val])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::Path;

    fn write_npy_f32(path: &std::path::Path, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
        // Minimal NPY v1.0 writer for little-endian f32, C-order
        // Magic header
        let mut f = File::create(path)?;
        f.write_all(b"\x93NUMPY")?; // magic
        f.write_all(&[1, 0])?; // v1.0
        // Build header dict
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", dims.join(", "))
        };
        let mut header_dict = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
            shape_str
        );
        // Pad header to 16-byte alignment, ending with newline
        let preamble_len = 10 + 2; // magic(6)+ver(2)+hlen(2)
        let mut header_len = header_dict.len() + 1; // +1 for newline
        let padding = (16 - ((preamble_len + header_len) % 16)) % 16;
        header_dict.push_str(&" ".repeat(padding));
        header_dict.push('\n');
        header_len = header_dict.len();
        if header_len > u16::MAX as usize {
            panic!("npy header too large");
        }
        let hlen_le = (header_len as u16).to_le_bytes();
        f.write_all(&hlen_le)?;
        f.write_all(header_dict.as_bytes())?;
        // Write data in little-endian
        for &v in data {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    fn write_npy_arr2(path: &std::path::Path, a: &Array2<f32>) -> std::io::Result<()> {
        let buf: Vec<f32> = a.iter().copied().collect();
        write_npy_f32(path, &buf, a.shape())
    }

    fn write_npy_arr1(path: &std::path::Path, a: &Array1<f32>) -> std::io::Result<()> {
        let buf: Vec<f32> = a.iter().copied().collect();
        write_npy_f32(path, &buf, a.shape())
    }

    #[test]
    fn silence_pulsed_vs_batch_probs_below_6_percent() -> anyhow::Result<()> {
        // Locate silence asset (try both expected names)
        let p1 = Path::new("assets/audio/silence_16_khz.wav");
        let p2 = Path::new("assets/audio/silence_16k.wav");
        let wav_path = if p1.exists() { p1 } else { p2 };
        assert!(
            wav_path.exists(),
            "silence wav not found at {:?} or {:?}",
            p1,
            p2
        );

        // Load mono PCM16 -> f32 in [-1, 1]
        let mut reader = hound::WavReader::open(wav_path)?;
        let spec = reader.spec();
        assert_eq!(
            spec.sample_rate, 16_000,
            "expected 16kHz sample rate, got {}",
            spec.sample_rate
        );
        let mut samples: Vec<f32> = Vec::with_capacity(reader.duration() as usize);
        if spec.bits_per_sample <= 16 && spec.sample_format == hound::SampleFormat::Int {
            for s in reader.samples::<i16>() {
                let v = s? as f32 / 32768.0;
                samples.push(v.clamp(-1.0, 1.0));
            }
        } else if spec.sample_format == hound::SampleFormat::Float {
            for s in reader.samples::<f32>() {
                let v = s?;
                samples.push(v.clamp(-1.0, 1.0));
            }
        } else {
            panic!(
                "unsupported WAV format: {:?} bits={}",
                spec.sample_format, spec.bits_per_sample
            );
        }

        // Build VAD components
        let clf = VadClassifier::load_internal()?;
        let mut pulsed = VadSessionPulsed::new(
            &clf.preprocessor_model,
            &clf.encoder_model_pulsed,
            &clf.decoder_model,
        )?;
        // Use pulsed encoder's pulse.delay for batch alignment
        let pulse_delay: usize = match clf.encoder_model_pulsed.property("pulse.delay") {
            Ok(d) => d
                .view::<i64>()
                .ok()
                .map(|v| v.index(0).to_owned() as usize)
                .unwrap_or(0usize),
            Err(_) => 0usize,
        };
        let mut batch = VadSessionBatch::new(
            &clf.preprocessor_model,
            &clf.encoder_model_batch,
            &clf.decoder_model,
            pulse_delay,
        )?;

        // Stream in 4-frame pulses (640 samples at 16kHz)
        let step = VadSessionPulsed::ENCODER_INPUT_NEEDED_IN_AUDIO_SAMPLES; // 4*160
        let mut pulsed_scores: Vec<f32> = Vec::new();
        let mut batch_scores: Vec<f32> = Vec::new();
        let mut i = 0usize;
        while i < samples.len() {
            let end = (i + step).min(samples.len());
            let chunk = samples[i..end].to_vec();
            let p_pulsed = pulsed.predict_speech_presence(chunk.clone())?;
            let p_batch = batch.predict_speech_presence(chunk)?;
            pulsed_scores.push(p_pulsed);
            batch_scores.push(p_batch);
            i = end;
        }

        // Consider only post-warmup (finite) values
        let pulsed_finite: Vec<f32> = pulsed_scores
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        let batch_finite: Vec<f32> = batch_scores
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();

        assert!(
            !pulsed_finite.is_empty(),
            "no finite pulsed predictions observed (warmup never completed?)"
        );
        assert!(
            !batch_finite.is_empty(),
            "no finite batch predictions observed (warmup never completed?)"
        );

        // Print last 20 values from each sequence
        let n_print = 20usize;
        let pulsed_tail_start = pulsed_finite.len().saturating_sub(n_print);
        let batch_tail_start = batch_finite.len().saturating_sub(n_print);
        println!(
            "pulsed last {} p(speech): {:?}",
            n_print,
            &pulsed_finite[pulsed_tail_start..]
        );
        println!(
            "batch  last {} p(speech): {:?}",
            n_print,
            &batch_finite[batch_tail_start..]
        );

        // Assert all post-warmup probabilities are below 6%
        let thr = 0.06f32;
        for (idx, v) in pulsed_finite.iter().enumerate() {
            assert!(
                *v <= thr,
                "pulsed p(speech) at step {} = {:.5} exceeds 6%",
                idx,
                v
            );
        }
        for (idx, v) in batch_finite.iter().enumerate() {
            assert!(
                *v <= thr,
                "batch p(speech) at step {} = {:.5} exceeds 6%",
                idx,
                v
            );
        }

        Ok(())
    }

    #[test]
    fn silence_debug_dump_pulsed_vs_batch() -> anyhow::Result<()> {
        // Locate silence asset (try both expected names)
        let p1 = Path::new("assets/audio/silence_16_khz.wav");
        let p2 = Path::new("assets/audio/silence_16k.wav");
        let wav_path = if p1.exists() { p1 } else { p2 };
        assert!(wav_path.exists(), "missing silence wav");

        let mut reader = hound::WavReader::open(wav_path)?;
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000);
        let mut samples: Vec<f32> = Vec::with_capacity(reader.duration() as usize);
        if spec.sample_format == hound::SampleFormat::Int {
            for s in reader.samples::<i16>() {
                samples.push((s? as f32 / 32768.0).clamp(-1.0, 1.0));
            }
        } else {
            for s in reader.samples::<f32>() {
                let v = s?;
                samples.push(v.clamp(-1.0, 1.0));
            }
        }

        // Build VAD components
        let clf = VadClassifier::load_internal()?;
        let mut pulsed = VadSessionPulsed::new(
            &clf.preprocessor_model,
            &clf.encoder_model_pulsed,
            &clf.decoder_model,
        )?;
        let pulse_delay: usize = match clf.encoder_model_pulsed.property("pulse.delay") {
            Ok(d) => d
                .view::<i64>()
                .ok()
                .map(|v| v.index(0).to_owned() as usize)
                .unwrap_or(0usize),
            Err(_) => 0usize,
        };
        let mut batch = VadSessionBatch::new(
            &clf.preprocessor_model,
            &clf.encoder_model_batch,
            &clf.decoder_model,
            pulse_delay,
        )?;

        let step = VadSessionPulsed::ENCODER_INPUT_NEEDED_IN_AUDIO_SAMPLES; // 4*160
        let mut i = 0usize;
        let mut step_idx = 0usize;
        let base = Path::new("target/vad_dumps/silence");
        fs::create_dir_all(base)?;
        while i < samples.len() {
            let end = (i + step).min(samples.len());
            let chunk = samples[i..end].to_vec();
            let _ = pulsed.predict_speech_presence(chunk.clone())?;
            let _ = batch.predict_speech_presence(chunk)?;

            let step_dir = base.join(format!("step_{:04}", step_idx));
            fs::create_dir_all(&step_dir)?;

            // Pulsed snapshots
            if let Some(a) = &pulsed.last_pre_feat {
                write_npy_arr2(&step_dir.join("pulsed_pre_full.npy"), a)?;
            }
            if let Some(a) = &pulsed.last_pre_sliced {
                write_npy_arr2(&step_dir.join("pulsed_pre_4.npy"), a)?;
            }
            if let Some(a) = &pulsed.last_enc_out {
                write_npy_arr2(&step_dir.join("pulsed_enc_full.npy"), a)?;
            }
            if let Some(a) = &pulsed.last_enc_block {
                write_npy_arr2(&step_dir.join("pulsed_enc_4.npy"), a)?;
            }
            if let Some(a) = &pulsed.last_encoder_window {
                write_npy_arr2(&step_dir.join("pulsed_dec_in.npy"), a)?;
            }
            if let Some(a) = &pulsed.last_logits {
                write_npy_arr1(&step_dir.join("pulsed_logits.npy"), a)?;
            }
            if let Some(p) = pulsed.last_prob {
                write_npy_f32(&step_dir.join("pulsed_prob.npy"), &[p], &[1])?;
            }

            // Batch snapshots
            if let Some(a) = &batch.last_pre_feat {
                write_npy_arr2(&step_dir.join("batch_pre_full.npy"), a)?;
            }
            if let Some(a) = &batch.last_pre_sliced {
                write_npy_arr2(&step_dir.join("batch_pre_4.npy"), a)?;
            }
            if let Some(a) = &batch.last_enc_out {
                write_npy_arr2(&step_dir.join("batch_enc_full.npy"), a)?;
            }
            if let Some(a) = &batch.last_enc_block {
                write_npy_arr2(&step_dir.join("batch_enc_4.npy"), a)?;
            }
            if let Some(a) = &batch.last_encoder_window {
                write_npy_arr2(&step_dir.join("batch_dec_in.npy"), a)?;
            }
            if let Some(a) = &batch.last_logits {
                write_npy_arr1(&step_dir.join("batch_logits.npy"), a)?;
            }
            if let Some(p) = batch.last_prob {
                write_npy_f32(&step_dir.join("batch_prob.npy"), &[p], &[1])?;
            }

            // Print quick diffs where shapes match (current step)
            let diff = |a: &Array2<f32>, b: &Array2<f32>, name: &str| {
                if a.shape() == b.shape() {
                    let mut mae = 0f32;
                    let mut maxd = 0f32;
                    let mut n = 0usize;
                    for (x, y) in a.iter().zip(b.iter()) {
                        let d = (x - y).abs();
                        mae += d;
                        if d > maxd {
                            maxd = d;
                        }
                        n += 1;
                    }
                    if n > 0 {
                        mae /= n as f32;
                    }
                    println!(
                        "step {} diff {}: mae={:.6} max={:.6}",
                        step_idx, name, mae, maxd
                    );
                }
            };
            if let (Some(a), Some(b)) = (&pulsed.last_pre_sliced, &batch.last_pre_sliced) {
                diff(a, b, "pre_4");
            }
            if let (Some(a), Some(b)) = (&pulsed.last_enc_block, &batch.last_enc_block) {
                diff(a, b, "enc_4");
            }
            if let (Some(a), Some(b)) = (&pulsed.last_encoder_window, &batch.last_encoder_window) {
                diff(a, b, "dec_in_window");
            }
            if let (Some(a), Some(b)) = (&pulsed.last_logits, &batch.last_logits)
                && a.shape() == b.shape()
            {
                let d0 = (a[0] - b[0]).abs();
                let d1 = (a[1] - b[1]).abs();
                println!("step {} diff logits: d0={:.6} d1={:.6}", step_idx, d0, d1);
            }

            i = end;
            step_idx += 1;
        }

        Ok(())
    }
}
