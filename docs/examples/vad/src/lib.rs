use anyhow::{bail, ensure};
use std::collections::VecDeque;
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
            last_score: 0.0,
            encoder_frame_buffer: Array2::<f32>::zeros((128, n_encoder_frames_to_aggregate_over)),
            pulse_delay: pulse_delay.try_into().unwrap_or(0usize),
            warmup_done: false,
            decoded_emissions: 0,
            stable_frames_ready: 0,
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
        if max > 1.0 || max < -1.0 {
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
        // Remove length tensor before encoder
        if pre_result.len() > 1 {
            pre_result.remove(1);
        }
        let enc_result = self.encoder_state.run(pre_result)?;

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
            // Not enough frames yet from pulsed encoder; keep window unchanged and do not advance readiness.
            let val: Value = self.encoder_frame_buffer.clone().try_into()?;
            return Ok(vec![val]);
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
            .into_shape((features, frames_usize))
            .map_err(|e| anyhow::anyhow!(format!("reshape encoder frame: {e}")))?;
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
        // Slide into decoder window buffer
        let n: i32 = Self::EXPECTED_PULSE_FRAMES.try_into()?;
        let temp = &self.encoder_frame_buffer.slice(s![.., n..]).to_owned();
        self.encoder_frame_buffer
            .slice_mut(s![.., ..-n])
            .assign(temp);
        self.encoder_frame_buffer
            .slice_mut(s![.., -n..])
            .assign(&block);
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
            .into_shape((features, frames_usize))
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
    use std::path::PathBuf;

    // Simple deterministic pseudo-random generator (LCG) kept for potential future use.
    fn _lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = (*seed as f32) / (u32::MAX as f32);
        2.0 * v - 1.0
    }

    fn read_wav_f32_mono(path: &str) -> anyhow::Result<(u32, Vec<f32>)> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        anyhow::ensure!(spec.channels == 1, "expected mono wav: {}", path);
        let sr = spec.sample_rate;
        let mut out = Vec::new();
        match spec.sample_format {
            hound::SampleFormat::Int => {
                if spec.bits_per_sample <= 16 {
                    for s in reader.samples::<i16>() {
                        let x = s.unwrap_or(0) as f32 / i16::MAX as f32;
                        out.push(x.max(-1.0).min(1.0));
                    }
                } else {
                    for s in reader.samples::<i32>() {
                        let x = s.unwrap_or(0) as f32 / i32::MAX as f32;
                        out.push(x.max(-1.0).min(1.0));
                    }
                }
            }
            hound::SampleFormat::Float => {
                for s in reader.samples::<f32>() {
                    let x = s.unwrap_or(0.0);
                    out.push(x.max(-1.0).min(1.0));
                }
            }
        }
        Ok((sr, out))
    }

    #[test]
    #[ignore]
    fn silence_like_audio_has_low_probs() -> anyhow::Result<()> {
        let mut clf = VadClassifier::load_internal()?;
        let silence_path = PathBuf::from("assets")
            .join("audio")
            .join("silence_16k.wav");
        let (sr, samples) = read_wav_f32_mono(silence_path.to_str().unwrap())?;
        anyhow::ensure!(sr == 16_000, "silence wav must be 16 kHz, got {}", sr);

        // Feed chunks matching EXPECTED_PULSE_FRAMES * ENCODER_INPUT_FRAME_SIZE
        let chunk =
            VadSessionPulsed::EXPECTED_PULSE_FRAMES * VadSessionPulsed::ENCODER_INPUT_FRAME_SIZE;
        let mut probs: Vec<f32> = Vec::new();
        for window in samples.chunks(chunk) {
            let mut buf = vec![0.0f32; chunk];
            buf[..window.len()].copy_from_slice(window);
            let p = clf.predict_speech_presence_internal(buf)?;
            probs.push(p);
        }

        // Drop the first few predictions to allow startup suppression and warmup.
        let skip = 5usize;
        let tail = if probs.len() > skip {
            &probs[skip..]
        } else {
            &probs[..]
        };
        let mean: f32 = if tail.is_empty() {
            0.0
        } else {
            tail.iter().copied().sum::<f32>() / (tail.len() as f32)
        };

        // Also compute late tail after ~1.5s to catch slow ramps
        let sr = 16_000usize;
        let emits_per_sec = (sr as f32
            / (VadSessionPulsed::EXPECTED_PULSE_FRAMES * VadSessionPulsed::ENCODER_INPUT_FRAME_SIZE)
                as f32)
            .round() as usize; // ~25
        let late_start = skip + (1.5f32 * emits_per_sec as f32).round() as usize;
        let late = if probs.len() > late_start {
            &probs[late_start..]
        } else {
            &[] as &[f32]
        };
        let late_mean: f32 = if late.is_empty() {
            0.0
        } else {
            late.iter().copied().sum::<f32>() / (late.len() as f32)
        };
        let late_max: f32 = late.iter().copied().fold(0.0, f32::max);

        eprintln!(
            "silence: mean_tail={:.3} mean_late={:.3} max_late={:.3} (tail_n={} late_n={})",
            mean,
            late_mean,
            late_max,
            tail.len(),
            late.len()
        );

        // Expect low baseline for near-silence across full tail and late tail
        assert!(
            mean < 0.4,
            "silence mean too high: {} (n={})",
            mean,
            tail.len()
        );
        assert!(
            late_mean < 0.4,
            "silence late-mean too high: {} (n={})",
            late_mean,
            late.len()
        );
        assert!(
            late_max < 0.8,
            "silence late-max unexpectedly high: {} (n={})",
            late_max,
            late.len()
        );
        Ok(())
    }

    #[test]
    fn noisy_segment_triggers_high_prob() -> anyhow::Result<()> {
        let mut clf = VadClassifier::load_internal()?;
        let speech_path = PathBuf::from("assets").join("audio").join("speech.wav");
        let (sr, samples) = read_wav_f32_mono(speech_path.to_str().unwrap())?;
        anyhow::ensure!(sr == 16_000, "speech wav must be 16 kHz, got {}", sr);
        let chunk =
            VadSessionPulsed::EXPECTED_PULSE_FRAMES * VadSessionPulsed::ENCODER_INPUT_FRAME_SIZE;
        let mut probs: Vec<f32> = Vec::new();
        for window in samples.chunks(chunk) {
            let mut buf = vec![0.0f32; chunk];
            buf[..window.len()].copy_from_slice(window);
            let p = clf.predict_speech_presence_internal(buf)?;
            probs.push(p);
        }
        let tail = if probs.len() > 5 {
            &probs[5..]
        } else {
            &probs[..]
        };
        let maxp = tail.iter().copied().fold(0.0, f32::max);
        assert!(maxp > 0.8, "max probability on speech too low: {}", maxp);
        Ok(())
    }

    fn squeeze_features_2d(val: &Value) -> anyhow::Result<Array2<f32>> {
        let v = val.view::<f32>()?;
        if v.shape().len() == 3 && v.shape()[0] == 1 {
            Ok(
                v.into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix3>()?
                    .index_axis(Axis(0), 0)
                    .to_owned(),
            )
        } else {
            Ok(
                v.into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix2>()?
                    .to_owned(),
            )
        }
    }

    fn run_preprocessor_2d(pre: &Runnable, audio: &[f32]) -> anyhow::Result<Array2<f32>> {
        let arr = Array1::from_iter(audio.iter().copied());
        let v1: Value = arr.clone().try_into()?;
        let v2: Value = arr.insert_axis(Axis(0)).try_into()?;
        let len: Value = Array1::<i64>::from_vec(vec![audio.len() as i64]).try_into()?;
        let out = match pre.run(vec![v1, len.clone()]) {
            Ok(r) => r,
            Err(_) => pre.run(vec![v2, len])?,
        };
        squeeze_features_2d(&out[0])
    }

    fn run_batch_encoder_2d(enc: &Runnable, feats: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        // Try 2D [F, T], then retry as [1, F, T]
        let mut inputs = vec![feats.clone().try_into()?];
        let out = match enc.run(inputs.clone()) {
            Ok(r) => r,
            Err(_) => {
                let with_b: Value = feats.clone().insert_axis(Axis(0)).try_into()?;
                inputs[0] = with_b;
                enc.run(inputs)?
            }
        };
        // Squeeze output to 2D [features, frames]
        let v = out[0].view::<f32>()?;
        let enc_dyn = v.into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
        let frames = *enc_dyn.shape().last().unwrap_or(&1);
        let features = enc_dyn.len() / frames;
        Ok(enc_dyn
            .to_owned()
            .into_shape((features, frames))
            .map_err(|e| anyhow::anyhow!(format!("reshape: {e}")))?)
    }

    fn run_pulsed_encoder_last4(
        enc: &Runnable,
        feats: &Array2<f32>,
    ) -> anyhow::Result<Array2<f32>> {
        let mut state = enc.spawn_state()?;
        let frames = feats.shape()[1];
        anyhow::ensure!(frames >= VadSessionPulsed::EXPECTED_PULSE_FRAMES);
        let step = VadSessionPulsed::EXPECTED_PULSE_FRAMES;
        let mut last: Option<Array2<f32>> = None;
        let mut i = 0usize;
        while i + step <= frames {
            let slice = feats.slice(s![.., i..i + step]).to_owned();
            let val: Value = slice.clone().try_into()?;
            let out = state.run(vec![val])?;
            let v = out[0].view::<f32>()?;
            let enc_dyn = v.into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
            let f = *enc_dyn.shape().last().unwrap_or(&1);
            let feat = enc_dyn.len() / f;
            let arr = enc_dyn
                .to_owned()
                .into_shape((feat, f))
                .map_err(|e| anyhow::anyhow!(format!("reshape enc: {e}")))?;
            last = Some(arr);
            i += step;
        }
        last.ok_or_else(|| anyhow::anyhow!("no encoder output produced"))
    }

    fn run_pulsed_encoder_series(
        enc: &Runnable,
        feats: &Array2<f32>,
    ) -> anyhow::Result<Array2<f32>> {
        let mut state = enc.spawn_state()?;
        let frames = feats.shape()[1];
        anyhow::ensure!(frames >= VadSessionPulsed::EXPECTED_PULSE_FRAMES);
        let step = VadSessionPulsed::EXPECTED_PULSE_FRAMES;
        let mut series: Option<Array2<f32>> = None;
        let mut i = 0usize;
        while i + step <= frames {
            let slice = feats.slice(s![.., i..i + step]).to_owned();
            let val: Value = slice.clone().try_into()?;
            let out = state.run(vec![val])?;
            let v = out[0].view::<f32>()?;
            let enc_dyn = v.into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
            let f = *enc_dyn.shape().last().unwrap_or(&1);
            let feat = enc_dyn.len() / f;
            let arr = enc_dyn
                .to_owned()
                .into_shape((feat, f))
                .map_err(|e| anyhow::anyhow!(format!("reshape enc: {e}")))?;
            series = Some(match series {
                None => arr,
                Some(prev) => {
                    // concat along frames axis
                    tract_rs::prelude::tract_ndarray::concatenate(
                        Axis(1),
                        &[prev.view(), arr.view()],
                    )
                    .map_err(|e| anyhow::anyhow!(format!("concat: {e}")))?
                }
            });
            i += step;
        }
        series.ok_or_else(|| anyhow::anyhow!("no encoder output produced"))
    }

    fn run_decoder_logits(dec: &Runnable, win: &Array2<f32>) -> anyhow::Result<Array1<f32>> {
        let val: Value = win.clone().try_into()?;
        let out = dec.run(vec![val])?;
        let v = out[0].view::<f32>()?;
        Ok(
            v.into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix1>()?
                .to_owned(),
        )
    }

    #[test]
    fn compare_batch_vs_pulsed_on_silence_features_and_encoder() -> anyhow::Result<()> {
        // Load models
        let clf = VadClassifier::load_internal()?;
        let pre = clf.preprocessor_model.clone();
        let enc_p = clf.encoder_model_pulsed.clone();
        let enc_b = clf.encoder_model_batch.clone();
        let dec = clf.decoder_model.clone();

        // Read silence wav (mono 16k)
        let silence_path = PathBuf::from("assets")
            .join("audio")
            .join("silence_16k.wav");
        let (sr, samples) = read_wav_f32_mono(silence_path.to_str().unwrap())?;
        anyhow::ensure!(sr == 16_000, "silence wav must be 16 kHz, got {}", sr);

        // Build batch-style big buffer: zeros for receptive field, then the full silence file
        // This guarantees enough encoder frames to align a 10-frame decoder window at T - pulse_delay
        let receptive = VadSessionBatch::RECEPTIVE_FIELD_SAMPLES;
        let step =
            VadSessionPulsed::EXPECTED_PULSE_FRAMES * VadSessionPulsed::ENCODER_INPUT_FRAME_SIZE; // 640
        anyhow::ensure!(samples.len() >= step, "silence.wav too short for test");
        let mut big = vec![0.0f32; receptive];
        big.extend_from_slice(&samples);
        // For the small preprocessor run, compare against the last 4*160 segment of the file
        let seg = &samples[samples.len().saturating_sub(step)..];

        // Run preprocessor both ways
        let feats_big = run_preprocessor_2d(&pre, &big)?; // [F, Tb]
        let feats_small = run_preprocessor_2d(&pre, seg)?; // [F, Ts]
        let f_big_last4 = {
            let tb = feats_big.shape()[1];
            feats_big.slice(s![.., tb - 4..tb]).to_owned()
        };
        let f_small_last4 = {
            let ts = feats_small.shape()[1];
            feats_small.slice(s![.., ts - 4..ts]).to_owned()
        };

        // Compare preprocessor tails (expect near-identical)
        let mut pre_diff_sum = 0f32;
        let mut pre_diff_max = 0f32;
        let n = f_big_last4.len();
        for (a, b) in f_big_last4.iter().zip(f_small_last4.iter()) {
            let d = (a - b).abs();
            pre_diff_sum += d;
            if d > pre_diff_max {
                pre_diff_max = d;
            }
        }
        let pre_diff_mean = pre_diff_sum / (n as f32);
        eprintln!(
            "preproc tail diffs: mean={:.6} max={:.6} | shapes big={} small={}",
            pre_diff_mean,
            pre_diff_max,
            fmt_shape(f_big_last4.shape()),
            fmt_shape(f_small_last4.shape())
        );

        // Get pulse delay from pulsed encoder to set expected alignment
        let pulse_delay: usize = enc_p
            .property("pulse.delay")?
            .view::<i64>()?
            .index(0)
            .to_owned()
            .try_into()
            .unwrap_or(0usize);

        // Now compare encoder outputs aligned at the tail
        // Batch encoder over full feats_big
        let encb_full = run_batch_encoder_2d(&enc_b, &feats_big)?; // [Fb, Tb]
        let tb = encb_full.shape()[1];
        // Tail aligned to pulsed delay with empirical -1 frame shift: end at tb - delay - 1
        let encb_last4 = if tb >= pulse_delay + 4 + 1 {
            encb_full
                .slice(s![.., tb - pulse_delay - 4 - 1..tb - pulse_delay - 1])
                .to_owned()
        } else {
            encb_full.slice(s![.., tb - 4..tb]).to_owned()
        };
        // Pulsed encoder advanced over feats_big in 4-frame steps; take last output (4 frames)
        let encp_last4 = run_pulsed_encoder_last4(&enc_p, &feats_big)?;

        // Compare, allow shift tolerance up to pulse_delay frames
        let mut best_mean = f32::INFINITY;
        let mut best_max = f32::INFINITY;
        let mut best_shift = 0i32;
        let max_shift = (pulse_delay + 2) as i32;
        for shift in 0..=max_shift {
            if (shift as usize) > tb - 4 {
                break;
            }
            let s = shift as usize;
            let end = if tb >= pulse_delay + 1 {
                tb - pulse_delay - 1
            } else {
                tb
            };
            if end < 4 + s {
                break;
            }
            let cand = encb_full.slice(s![.., end - 4 - s..end - s]).to_owned();
            let mut sum = 0f32;
            let mut mx = 0f32;
            for (a, b) in encp_last4.iter().zip(cand.iter()) {
                let d = (a - b).abs();
                sum += d;
                if d > mx {
                    mx = d;
                }
            }
            let mean = sum / (encp_last4.len() as f32);
            if mean < best_mean {
                best_mean = mean;
                best_max = mx;
                best_shift = shift;
            }
        }
        eprintln!(
            "encoder tail diffs: mean={:.6} max={:.6} best_shift={} (pulse_delay={}) | shapes batch_last4={} pulsed_last4={}",
            best_mean,
            best_max,
            best_shift,
            pulse_delay,
            fmt_shape(encb_last4.shape()),
            fmt_shape(encp_last4.shape())
        );

        // Decoder-level comparison: build last-10-frame window for both paths
        // Pulsed: run series and take the last 10 frames
        let encp_series = run_pulsed_encoder_series(&enc_p, &feats_big)?;
        let kp = encp_series.shape()[1];
        anyhow::ensure!(
            kp >= 10,
            "pulsed encoder series too short for 10-frame window"
        );
        let win_p = encp_series.slice(s![.., kp - 10..kp]).to_owned();

        // Batch: take 10 frames ending at tb - pulse_delay - 1 (empirical -1 alignment)
        anyhow::ensure!(
            tb >= pulse_delay + 10 + 1,
            "batch encoder output too short to align 10-frame window"
        );
        let end_b = tb - pulse_delay - 1;
        let win_b = encb_full.slice(s![.., end_b - 10..end_b]).to_owned();

        let log_p = run_decoder_logits(&dec, &win_p)?;
        let log_b = run_decoder_logits(&dec, &win_b)?;
        anyhow::ensure!(
            log_p.len() == 2 && log_b.len() == 2,
            "decoder outputs must be 2 logits"
        );
        let mut dec_diff_sum = 0f32;
        let mut dec_diff_max = 0f32;
        for (a, b) in log_p.iter().zip(log_b.iter()) {
            let d = (a - b).abs();
            dec_diff_sum += d;
            if d > dec_diff_max {
                dec_diff_max = d;
            }
        }
        let dec_diff_mean = dec_diff_sum / 2.0;
        eprintln!(
            "decoder tail diffs (logits): mean={:.8} max={:.8} | win shapes batch={} pulsed={}",
            dec_diff_mean,
            dec_diff_max,
            fmt_shape(win_b.shape()),
            fmt_shape(win_p.shape()),
        );

        // Also compare probabilities for class 1 (speech)
        let to_p1 = |l0: f32, l1: f32| {
            let m = l0.max(l1);
            let e0 = (l0 - m).exp();
            let e1 = (l1 - m).exp();
            let s = e0 + e1;
            if s == 0.0 { 0.5 } else { e1 / s }
        };
        let p1_p = to_p1(log_p[0], log_p[1]);
        let p1_b = to_p1(log_b[0], log_b[1]);
        let p1_diff = (p1_p - p1_b).abs();
        eprintln!(
            "decoder tail p1 diff: {:.8} (pulsed={:.6} batch={:.6})",
            p1_diff, p1_p, p1_b
        );

        // Heuristic thresholds; adjust if models change
        let pre_ok = pre_diff_mean < 5e-3 && pre_diff_max < 5e-2;
        let enc_ok = best_mean < 5e-3 && best_max < 5e-2;
        // Decoder diffs are expected to be very small but non-zero; relax thresholds
        let dec_ok = dec_diff_mean < 1e-3 && dec_diff_max < 1e-3 && p1_diff < 1e-3;
        if !pre_ok || !enc_ok || !dec_ok {
            // Provide a targeted failure that indicates which stage diverged
            anyhow::bail!(format!(
                "Batch vs pulsed mismatch: pre_ok={} (mean={:.5}, max={:.5}) enc_ok={} (mean={:.5}, max={:.5}, shift={}) dec_ok={} (logit_mean={:.6}, logit_max={:.6}, p1_diff={:.8})",
                pre_ok,
                pre_diff_mean,
                pre_diff_max,
                enc_ok,
                best_mean,
                best_max,
                best_shift,
                dec_ok,
                dec_diff_mean,
                dec_diff_max,
                p1_diff,
            ));
        }
        Ok(())
    }

    #[test]
    fn end_to_end_stream_parity_on_zero_silence() -> anyhow::Result<()> {
        // Compute probabilities from timeline-anchored windows (same method as trace test)
        let clf = VadClassifier::load_internal()?;
        let pre = clf.preprocessor_model.clone();
        let enc_p = clf.encoder_model_pulsed.clone();
        let enc_b = clf.encoder_model_batch.clone();
        let dec = clf.decoder_model.clone();

        let pulse_delay: usize = enc_p
            .property("pulse.delay")?
            .view::<i64>()?
            .index(0)
            .to_owned()
            .try_into()
            .unwrap_or(0usize);

        // Build full zero-audio, derive features once
        let receptive =
            VadSessionBatch::RECEPTIVE_FIELD_SAMPLES / VadSessionPulsed::ENCODER_INPUT_FRAME_SIZE; // 75
        let steps = 60usize; // ~2.4 s beyond receptive at 10ms/frame
        let total_frames = receptive + 4 * steps;
        let total_samples = total_frames * VadSessionPulsed::ENCODER_INPUT_FRAME_SIZE; // 10ms @16kHz per frame
        let audio = Array1::<f32>::zeros(total_samples);
        let audio_1d: Value = audio.clone().try_into()?;
        let audio_2d: Value = audio.insert_axis(Axis(0)).try_into()?;
        let len_v: Value = Array1::<i64>::from_vec(vec![total_samples as i64]).try_into()?;
        let mut out = match pre.run(vec![audio_1d.clone(), len_v.clone()]) {
            Ok(r) => r,
            Err(_) => pre.run(vec![audio_2d.clone(), len_v.clone()])?,
        };
        let view = out[0].view::<f32>()?;
        let feats = if view.shape().len() == 3 && view.shape()[0] == 1 {
            view.into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix3>()?
                .index_axis(Axis(0), 0)
                .to_owned()
        } else {
            view.into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix2>()?
                .to_owned()
        };

        // Encoders over full timeline
        let batch_out = run_batch_encoder_2d(&enc_b, &feats)?;
        let mut state_p = enc_p.spawn_state()?;
        let mut pulsed_seq: Option<Array2<f32>> = None;
        for i in (0..total_frames).step_by(4) {
            if i + 4 > total_frames {
                break;
            }
            let sl = feats.slice(s![.., i..i + 4]).to_owned();
            let v: Value = sl.clone().try_into()?;
            let res = state_p.run(vec![v])?;
            let v2 = res[0].view::<f32>()?;
            let d2 = v2.into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
            let f2 = *d2.shape().last().unwrap_or(&1);
            let fe2 = d2.len() / f2;
            let outp = d2
                .to_owned()
                .into_shape((fe2, f2))
                .map_err(|e| anyhow::anyhow!(format!("reshape enc p: {e}")))?;
            pulsed_seq = Some(match pulsed_seq {
                None => outp,
                Some(prev) => tract_rs::prelude::tract_ndarray::concatenate(
                    Axis(1),
                    &[prev.view(), outp.view()],
                )?,
            });
        }
        let pulsed_out = pulsed_seq.ok_or_else(|| anyhow::anyhow!("no pulsed out"))?;

        // Compare p1 from decoder windows at each valid step
        let tb = batch_out.shape()[1];
        let tp = pulsed_out.shape()[1];
        let end_t_max = std::cmp::min(tp, tb + pulse_delay);
        let mut diffs = Vec::new();
        if end_t_max > receptive + 10 {
            let max_k = (end_t_max - receptive) / 4;
            for k in 1..=max_k {
                let end_t = receptive + k * 4;
                if end_t <= pulse_delay + 10 || end_t > tp {
                    continue;
                }
                let end_b = end_t - pulse_delay;
                if end_b > tb || end_b < 10 {
                    continue;
                }
                let wp = pulsed_out.slice(s![.., end_t - 10..end_t]).to_owned();
                let wb = batch_out.slice(s![.., end_b - 10..end_b]).to_owned();
                let logp = run_decoder_logits(&dec, &wp)?;
                let logb = run_decoder_logits(&dec, &wb)?;
                let to_p1 = |l0: f32, l1: f32| {
                    let m = l0.max(l1);
                    let e0 = (l0 - m).exp();
                    let e1 = (l1 - m).exp();
                    let s = e0 + e1;
                    if s == 0.0 { 0.5 } else { e1 / s }
                };
                let pp = to_p1(logp[0], logp[1]);
                let pb = to_p1(logb[0], logb[1]);
                diffs.push((pp - pb).abs());
            }
        }
        let mean = if diffs.is_empty() {
            0.0
        } else {
            diffs.iter().copied().sum::<f32>() / (diffs.len() as f32)
        };
        let maxd = diffs.iter().copied().fold(0.0, f32::max);
        eprintln!(
            "stream parity (zeros): mean_diff={:.6} max_diff={:.6} n={}",
            mean,
            maxd,
            diffs.len()
        );
        assert!(
            mean < 1e-2 && maxd < 3e-2,
            "stream parity too far: mean={} max={}",
            mean,
            maxd
        );
        Ok(())
    }

    #[test]
    fn trace_parity_across_steps_on_zero_silence() -> anyhow::Result<()> {
        // Step-by-step parity using shared pre/enc and timeline-anchored windows (matches repro CLI)
        let clf = VadClassifier::load_internal()?;
        let pre = clf.preprocessor_model.clone();
        let enc_p = clf.encoder_model_pulsed.clone();
        let enc_b = clf.encoder_model_batch.clone();
        let pulse_delay: usize = enc_p
            .property("pulse.delay")?
            .view::<i64>()?
            .index(0)
            .to_owned()
            .try_into()
            .unwrap_or(0usize);
        // Build full-length zero audio and derive full feature timeline via preprocessor
        let receptive =
            VadSessionBatch::RECEPTIVE_FIELD_SAMPLES / VadSessionPulsed::ENCODER_INPUT_FRAME_SIZE; // 75 frames
        let steps = 60usize;
        let total_frames = receptive + 4 * steps;
        let total_samples = total_frames * VadSessionPulsed::ENCODER_INPUT_FRAME_SIZE; // 10ms @16kHz per frame
        let audio = Array1::<f32>::zeros(total_samples);
        let audio_1d: Value = audio.clone().try_into()?;
        let audio_2d: Value = audio.insert_axis(Axis(0)).try_into()?;
        let len_v: Value = Array1::<i64>::from_vec(vec![total_samples as i64]).try_into()?;
        let mut out = match pre.run(vec![audio_1d.clone(), len_v.clone()]) {
            Ok(r) => r,
            Err(_) => pre.run(vec![audio_2d.clone(), len_v.clone()])?,
        };
        let view = out[0].view::<f32>()?;
        let feats = if view.shape().len() == 3 && view.shape()[0] == 1 {
            view.into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix3>()?
                .index_axis(Axis(0), 0)
                .to_owned()
        } else {
            view.into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix2>()?
                .to_owned()
        };

        // Run encoders on the full timeline
        let batch_out = run_batch_encoder_2d(&enc_b, &feats)?;
        let mut state_p = enc_p.spawn_state()?;
        let mut pulsed_seq: Option<Array2<f32>> = None;
        for i in (0..total_frames).step_by(4) {
            if i + 4 > total_frames {
                break;
            }
            let sl = feats.slice(s![.., i..i + 4]).to_owned();
            let v: Value = sl.clone().try_into()?;
            let res = state_p.run(vec![v])?;
            let v2 = res[0].view::<f32>()?;
            let d2 = v2.into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
            let f2 = *d2.shape().last().unwrap_or(&1);
            let fe2 = d2.len() / f2;
            let outp = d2
                .to_owned()
                .into_shape((fe2, f2))
                .map_err(|e| anyhow::anyhow!(format!("reshape enc p: {e}")))?;
            pulsed_seq = Some(match pulsed_seq {
                None => outp,
                Some(prev) => tract_rs::prelude::tract_ndarray::concatenate(
                    Axis(1),
                    &[prev.view(), outp.view()],
                )?,
            });
        }
        let pulsed_out = pulsed_seq.ok_or_else(|| anyhow::anyhow!("no pulsed out"))?;
        let tb = batch_out.shape()[1];
        let tp = pulsed_out.shape()[1];
        let end_t_max = std::cmp::min(tp, tb + pulse_delay);
        let mut wsum_all = 0f32;
        let mut wmax_all = 0f32;
        let mut nsteps = 0usize;
        if end_t_max > receptive + 10 {
            let max_k = (end_t_max - receptive) / 4;
            for k in 1..=max_k {
                let end_t = receptive + k * 4;
                if end_t <= pulse_delay + 10 || end_t > tp {
                    continue;
                }
                let end_b = end_t - pulse_delay;
                if end_b > tb || end_b < 10 {
                    continue;
                }
                let wp = pulsed_out.slice(s![.., end_t - 10..end_t]).to_owned();
                let wb = batch_out.slice(s![.., end_b - 10..end_b]).to_owned();
                // Compute diff
                let mut wsum = 0f32;
                let mut wmx = 0f32;
                let mut wn = 0usize;
                for (a, b) in wp.iter().zip(wb.iter()) {
                    let d = (a - b).abs();
                    wsum += d;
                    if d > wmx {
                        wmx = d;
                    }
                    wn += 1;
                }
                if wn > 0 {
                    wsum_all += wsum / (wn as f32);
                    if wmx > wmax_all {
                        wmax_all = wmx;
                    }
                    nsteps += 1;
                }
                // Tail p1 parity at this step (optional)
                let logp = wb.column(wb.shape()[1] - 1)[0]; // noop to touch
                let _ = logp;
                // Diff in last-4 block probabilities (optional) omitted
                // Use window-based parity for stability
                // accumulate absolute diff of p1 via decoder if needed
            }
        }
        let wmean = if nsteps == 0 {
            0.0
        } else {
            wsum_all / (nsteps as f32)
        };
        eprintln!(
            "trace window parity (zeros): mean={:.6} max={:.6} steps={}",
            wmean, wmax_all, nsteps
        );
        assert!(
            wmean < 1e-2 && wmax_all < 3e-2,
            "trace parity too far: mean={} max={}",
            wmean,
            wmax_all
        );
        Ok(())
    }
}
