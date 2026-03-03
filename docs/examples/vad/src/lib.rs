#![cfg_attr(target_arch = "wasm32", allow(unused, dead_code))]

use anyhow::{bail, ensure};
use tract_rs::{
    State,
    prelude::{
        tract_ndarray::{Array1, Array2, Axis, IndexLonger, s},
        *,
    },
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

type Res<T> = anyhow::Result<T>;

#[cfg(target_arch = "wasm32")]
extern crate web_sys;

#[inline]
#[cfg(all(feature = "log-vad", target_arch = "wasm32"))]
fn clog(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}
#[inline]
#[cfg(not(all(feature = "log-vad", target_arch = "wasm32")))]
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

// Shared VAD constants
const VAD_ENCODER_INPUT_FRAME_SIZE: usize = 160; // 10ms at 16kHz

impl VadSessionCommon for VadSessionPulsed {
    fn decoder_model(&self) -> &Runnable {
        &self.decoder_model
    }
    fn encoder_frame_buffer(&self) -> &Array2<f32> {
        &self.encoder_frame_buffer
    }
    fn encoder_frame_buffer_mut(&mut self) -> &mut Array2<f32> {
        &mut self.encoder_frame_buffer
    }
    fn pulse_delay(&self) -> usize {
        self.pulse_delay
    }
    fn get_last_score(&self) -> f32 {
        self.last_score
    }
    fn set_last_score(&mut self, v: f32) {
        self.last_score = v;
    }
    fn stable_frames_ready(&self) -> usize {
        self.stable_frames_ready
    }
    fn add_stable_frames(&mut self, n: usize) {
        self.stable_frames_ready = self.stable_frames_ready.saturating_add(n);
    }
    fn on_decoded(&mut self, logits: &Array1<f32>, p1: f32) {
        #[cfg(test)]
        {
            self.dbg.set_logits_and_prob(logits, p1);
        }
        self.decoded_emissions = self.decoded_emissions.saturating_add(1);
    }
}

// Common utilities shared by VAD sessions
#[cfg(test)]
#[derive(Default, Clone)]
struct SessionDebug {
    last_pre_feat: Option<Array2<f32>>,
    last_pre_sliced: Option<Array2<f32>>,
    last_enc_out: Option<Array2<f32>>,
    last_enc_block: Option<Array2<f32>>,
    last_encoder_window: Option<Array2<f32>>,
    last_logits: Option<Array1<f32>>,
    last_prob: Option<f32>,
}

#[cfg(test)]
impl SessionDebug {
    fn set_pre_feat(&mut self, a: &Array2<f32>) {
        self.last_pre_feat = Some(a.clone());
    }
    fn set_pre_sliced(&mut self, a: &Array2<f32>) {
        self.last_pre_sliced = Some(a.clone());
    }
    fn set_enc_out(&mut self, a: &Array2<f32>) {
        self.last_enc_out = Some(a.clone());
    }
    fn set_enc_block(&mut self, a: &Array2<f32>) {
        self.last_enc_block = Some(a.clone());
    }
    fn set_encoder_window(&mut self, a: &Array2<f32>) {
        self.last_encoder_window = Some(a.clone());
    }
    fn set_logits_and_prob(&mut self, logits: &Array1<f32>, p: f32) {
        self.last_logits = Some(logits.clone());
        self.last_prob = Some(p);
    }
}

trait VadSessionCommon {
    fn decoder_model(&self) -> &Runnable;
    fn encoder_frame_buffer(&self) -> &Array2<f32>;
    fn encoder_frame_buffer_mut(&mut self) -> &mut Array2<f32>;
    fn pulse_delay(&self) -> usize;
    fn get_last_score(&self) -> f32;
    fn set_last_score(&mut self, v: f32);
    fn stable_frames_ready(&self) -> usize;
    fn add_stable_frames(&mut self, n: usize);
    fn on_decoded(&mut self, _logits: &Array1<f32>, _p1: f32) {
        // default no-op; implementers may record debug/test data
    }

    fn warmup_needed_frames(&self) -> usize {
        self.pulse_delay() + self.encoder_frame_buffer().shape()[1]
    }

    fn warmup_ready(&self) -> bool {
        self.stable_frames_ready() >= self.warmup_needed_frames()
    }

    fn slide_window_append(&mut self, block: &Array2<f32>, step_frames: usize) -> Res<Vec<Value>> {
        let n: i32 = step_frames.try_into()?;
        {
            let win = self.encoder_frame_buffer_mut();
            let temp = &win.slice(s![.., n..]).to_owned();
            win.slice_mut(s![.., ..-n]).assign(temp);
            win.slice_mut(s![.., -n..]).assign(block);
        }
        self.add_stable_frames(step_frames);
        let val: Value = self.encoder_frame_buffer().clone().try_into()?;
        Ok(vec![val])
    }

    fn decode_from_input(&mut self, decoder_input: Vec<Value>) -> Res<f32> {
        clog("DEC run");
        let dec_result = self.decoder_model().run(decoder_input)?;
        let dec_view = dec_result[0].view::<f32>()?;
        let logits: Array1<f32> = dec_view.into_dimensionality()?.to_owned();
        ensure!(logits.len() == 2, "Decoder output must have 2 logits");
        let mut l0 = logits[0];
        let mut l1 = logits[1];
        // Guard against non-finite logits; keep previous score
        if !l0.is_finite() || !l1.is_finite() {
            return Ok(self.get_last_score());
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
            self.get_last_score()
        };
        #[cfg(feature = "log-vad")]
        {
            let win_len = self.encoder_frame_buffer().shape()[1];
            let p0 = if s.is_finite() && s != 0.0 {
                e0 / s
            } else {
                1.0 - p1
            };
            clog(&format!(
                "DBG logits: l0={:.4} l1={:.4} p0={:.4} p1={:.4} | win_len={}",
                l0, l1, p0, p1, win_len
            ));
        }
        self.set_last_score(p1);
        self.on_decoded(&logits, p1);
        Ok(self.get_last_score())
    }
}

// Shared helpers
fn roll_into_ring(buf: &mut [f32], incoming: &[f32]) {
    let l = buf.len();
    let n = incoming.len();
    if n >= l {
        buf.copy_from_slice(&incoming[n - l..n]);
    } else {
        buf.copy_within(n..l, 0);
        buf[l - n..].copy_from_slice(incoming);
    }
}

fn validate_audio_range_11(buf: &[f32]) -> Res<()> {
    let mut max = 0.0f32;
    for &v in buf.iter() {
        let a = v.abs();
        if a > max {
            max = a;
        }
    }
    if max > 1.0 {
        bail!(format!(
            "WARNING: audio sample abs value {} exceeds expected [-1.0, 1.0] range; ensure proper normalization",
            max
        ));
    }
    Ok(())
}

fn run_preprocessor_2d(preprocessor: &Runnable, audio: &[f32]) -> Res<Array2<f32>> {
    let audio_arr = Array1::from_vec(audio.to_vec());
    let audio_val_1d: Value = audio_arr.clone().try_into()?;
    let audio_val_2d: Value = audio_arr.insert_axis(Axis(0)).try_into()?;
    let audio_len: i64 = audio.len() as i64;
    let len_val: Value = Array1::<i64>::from_vec(vec![audio_len]).try_into()?;
    let pre_result = match preprocessor.run(vec![audio_val_1d, len_val.clone()]) {
        Ok(r) => r,
        Err(_) => preprocessor.run(vec![audio_val_2d, len_val.clone()])?,
    };
    let pre_any = pre_result[0].view::<f32>()?;
    // Squeeze optional batch axis and ensure 2D [features, frames]
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
    Ok(pre_feat)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
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
    // configuration
    pulse_frames: usize,
    frame_size: usize,
}

// Internal API usable by tests and wasm shims
impl VadClassifier {
    fn load_internal(pulse_frames: usize) -> Res<VadClassifier> {
        // Better panic messages in the browser console.
        console_error_panic_hook::set_once();
        clog("loading runtime 'default'");
        let rt = runtime_for_name("default")?;
        let preprocessor_model_bytes = include_bytes!("../model/preprocessor.nnef.tgz");
        clog("creating NNEF loader");
        let mut nnef = tract_rs::nnef()?.with_tract_core()?; // core ops for models
        nnef.enable_pulse()?; // allow pulsing the batch graph
        clog("preparing preprocessor model");
        let preprocessor_model = rt.prepare(nnef.load_buffer(preprocessor_model_bytes)?)?;

        // Load batch (non-pulsed) encoder for stream/batch mode, then derive pulsed from it
        let enc_model_batch_bytes = include_bytes!("../model/encoder.nnef.tgz");
        let enc_model = nnef.load_buffer(enc_model_batch_bytes)?;
        let mut pulsed_encoder = enc_model.clone();
        clog("preparing encoder model (batch)");
        let encoder_model_batch = rt.prepare(enc_model)?;
        // Derive pulsed-encoded encoder from the same batch graph using pulse transform
        clog(&format!(
            "pulsifying encoder model (derived from batch): pulse_frames={}",
            pulse_frames
        ));
        pulsed_encoder.pulse(
            "AUDIO_SIGNAL__TIME",
            pulse_frames.max(1).to_string().as_str(),
        )?;
        let encoder_model_pulsed = rt.prepare(pulsed_encoder)?;

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
            pulse_frames: pulse_frames.max(1),
            frame_size: VAD_ENCODER_INPUT_FRAME_SIZE,
        })
    }

    // Factorized helpers
    fn compute_pulse_delay_from_encoder(&self) -> usize {
        match self.encoder_model_pulsed.property("pulse.delay") {
            Ok(d) => d
                .view::<i64>()
                .ok()
                .map(|v| v.index(0).to_owned() as usize)
                .unwrap_or(0usize),
            Err(_) => 0usize,
        }
    }

    fn ensure_pulsed_session(&mut self) -> Res<&mut VadSessionPulsed> {
        let pulse_delay = self.compute_pulse_delay_from_encoder();
        if self.vad_session_pulsed.is_none() {
            self.vad_session_pulsed = Some(VadSessionPulsed::new(
                &self.preprocessor_model,
                &self.encoder_model_pulsed,
                &self.decoder_model,
                self.pulse_frames,
                self.frame_size,
                pulse_delay,
            )?);
        }
        Ok(self.vad_session_pulsed.as_mut().unwrap())
    }

    fn ensure_batch_session(&mut self) -> Res<&mut VadSessionBatch> {
        if self.vad_session_batch.is_none() {
            let pulse_delay = self.compute_pulse_delay_from_encoder();
            self.vad_session_batch = Some(VadSessionBatch::new(
                &self.preprocessor_model,
                &self.encoder_model_batch,
                &self.decoder_model,
                pulse_delay,
                self.pulse_frames,
                self.frame_size,
            )?);
        }
        Ok(self.vad_session_batch.as_mut().unwrap())
    }

    fn predict_speech_presence_internal(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        let session = self.ensure_pulsed_session()?;
        session.predict_speech_presence(raw_audio_data)
    }
}

// JS-facing API exposed only on wasm32
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl VadClassifier {
    #[wasm_bindgen]
    pub fn predict_speech_presence(
        &mut self,
        js_raw_audio_data: &web_sys::js_sys::Float32Array,
    ) -> Result<JsValue, JsError> {
        let prediction_res = self.predict_speech_presence_internal(js_raw_audio_data.to_vec());
        let pred = prediction_res.map_err(|err| JsError::new(format!("{:?}", err).as_str()))?;
        Ok(pred.into())
    }

    // Batch mode API: stateless encoder, large rolling audio buffer
    #[wasm_bindgen]
    pub fn predict_speech_presence_batch(
        &mut self,
        js_raw_audio_data: &web_sys::js_sys::Float32Array,
    ) -> Result<JsValue, JsError> {
        console_error_panic_hook::set_once();
        let session = self
            .ensure_batch_session()
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        let pred = session
            .predict_speech_presence(js_raw_audio_data.to_vec())
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        Ok(pred.into())
    }

    #[wasm_bindgen]
    pub fn load() -> Result<VadClassifier, JsError> {
        console_error_panic_hook::set_once();
        clog("try loading");
        VadClassifier::load_internal(4).map_err(|err| JsError::new(&format!("{:?}", err)))
    }

    #[wasm_bindgen]
    pub fn load_with_pulse(pulse_frames: usize) -> Result<VadClassifier, JsError> {
        console_error_panic_hook::set_once();
        VadClassifier::load_internal(pulse_frames).map_err(|e| JsError::new(&format!("{:?}", e)))
    }

    // Expose configuration
    #[wasm_bindgen]
    pub fn get_pulse_frames(&self) -> usize {
        self.pulse_frames
    }

    #[wasm_bindgen]
    pub fn get_frame_size(&self) -> usize {
        self.frame_size
    }

    // Reset internal streaming sessions so that a new decode starts from a clean state.
    #[wasm_bindgen]
    pub fn reset_sessions(&mut self) {
        self.vad_session_pulsed = None;
        self.vad_session_batch = None;
    }

    // Expose pulsed parameters and readiness for UI coordination
    #[wasm_bindgen]
    pub fn get_pulse_delay(&mut self) -> Result<usize, JsError> {
        let s = self
            .ensure_pulsed_session()
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        Ok(s.pulse_delay)
    }

    #[wasm_bindgen]
    pub fn get_decoder_pool_len(&mut self) -> Result<usize, JsError> {
        let s = self
            .ensure_pulsed_session()
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        Ok(s.encoder_frame_buffer.shape()[1])
    }

    #[wasm_bindgen]
    pub fn is_pulsed_ready(&mut self) -> Result<bool, JsError> {
        let s = self
            .ensure_pulsed_session()
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
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
    pulse_frames: usize,
    frame_size: usize,
    warmup_done: bool,
    decoded_emissions: usize,
    stable_frames_ready: usize,
    #[cfg(test)]
    dbg: SessionDebug,
}

impl VadSessionPulsed {
    // Match batch receptive field to stabilize STFT features for pulsed path too
    // Model emits [non_speech, speech] logits; flip if needed
    const SPEECH_CLASS_INDEX: usize = 1;
    // Suppress first few decoder emissions to avoid startup spike

    fn new(
        preprocessor: &Runnable,
        encoder: &Runnable,
        decoder: &Runnable,
        pulse_frames: usize,
        frame_size: usize,
        pulse_delay: usize,
    ) -> Res<VadSessionPulsed> {
        // Pool ~100ms (10 frames) at the decoder for stability
        let n_encoder_frames_to_aggregate_over = 10;
        assert!(pulse_frames <= n_encoder_frames_to_aggregate_over);
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_state: encoder.spawn_state()?,
            decoder_model: decoder.clone(),
            // Keep a rolling window sufficient for delay + decoder window + one pulse
            audio_buffer: vec![0.0; pulse_delay * frame_size],
            // 512 extra for STFT context frames
            current_buffer_fill: 0,
            last_score: f32::NAN,
            encoder_frame_buffer: Array2::<f32>::zeros((128, n_encoder_frames_to_aggregate_over)),
            pulse_delay,
            pulse_frames: pulse_frames.max(1),
            frame_size,
            warmup_done: false,
            decoded_emissions: 0,
            stable_frames_ready: 0,
            #[cfg(test)]
            dbg: SessionDebug::default(),
        })
    }

    fn predict_speech_presence(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        // Assumed to be called every 10ms * pulse_frames or more
        self.roll_receptive_buffer(&raw_audio_data);

        // Preprocess full 2D and slice last-N frames for pulsed encoder
        let pre_full = self.preprocess_full_2d()?;
        let pre_slice = self.select_pre_slice(&pre_full);
        let sliced_value: Value = pre_slice.try_into()?;

        clog(&format!(
            "PULSED ENC run with new audio data qte: {}",
            raw_audio_data.len()
        ));
        let enc_result = self.encoder_state.run(vec![sliced_value])?;
        #[cfg(test)]
        {
            let enc_view_dyn = enc_result[0]
                .view::<f32>()?
                .into_dimensionality::<tract_rs::prelude::tract_ndarray::IxDyn>()?;
            let frames_usize = *enc_view_dyn.shape().last().unwrap_or(&1);
            let features = enc_view_dyn.len() / frames_usize;
            if features > 0 && frames_usize > 0 {
                if let Ok(enc2d) = enc_view_dyn
                    .to_owned()
                    .into_shape_with_order((features, frames_usize))
                {
                    self.dbg.set_enc_out(&enc2d);
                }
            }
        }
        let decoder_input = self.slide_encoder_window(enc_result)?;
        if !self.warmup_ready() {
            clog(&format!(
                "GATE frames_ready={} need={}",
                self.stable_frames_ready,
                self.warmup_needed_frames()
            ));
            return Ok(self.last_score);
        }

        self.decode_from_input(decoder_input)
    }

    // Helpers to keep predict_speech_presence concise
    fn roll_receptive_buffer(&mut self, raw_audio_data: &[f32]) {
        let n = raw_audio_data.len();
        clog(&format!(
            "RB before: buf_len={}, incoming={}, current_fill={}",
            self.audio_buffer.len(),
            n,
            self.current_buffer_fill
        ));
        roll_into_ring(&mut self.audio_buffer, raw_audio_data);
    }

    fn step_samples(&self) -> usize {
        self.pulse_frames * self.frame_size
    }

    fn preprocess_full_2d(&mut self) -> Res<Array2<f32>> {
        validate_audio_range_11(&self.audio_buffer)?;
        let pre_feat = run_preprocessor_2d(&self.preprocessor_model, &self.audio_buffer)?;
        #[cfg(test)]
        {
            self.dbg.set_pre_feat(&pre_feat);
        }
        Ok(pre_feat)
    }

    fn select_pre_slice(&mut self, pre_feat: &Array2<f32>) -> Array2<f32> {
        let frames = pre_feat.shape()[1];
        clog(&format!("PRE frames={frames}"));
        let start = frames.saturating_sub(self.pulse_frames);
        clog(&format!(
            "PRE slicing last {} frames (start={start})",
            self.pulse_frames
        ));
        let sliced = pre_feat.slice(s![.., start..]).to_owned();
        #[cfg(test)]
        {
            self.dbg.set_pre_sliced(&sliced);
        }
        sliced
    }

    fn warmup_needed_frames(&self) -> usize {
        self.pulse_delay + self.encoder_frame_buffer.shape()[1]
    }

    fn warmup_ready(&self) -> bool {
        self.stable_frames_ready >= self.warmup_needed_frames()
    }

    fn decode_from_input(&mut self, decoder_input: Vec<Value>) -> Res<f32> {
        clog("DEC run");
        let dec_result = self.decoder_model.run(decoder_input)?;
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
        #[cfg(feature = "log-vad")]
        if self.decoded_emissions < 20 {
            let win_len = self.encoder_frame_buffer.shape()[1];
            let p0 = if s.is_finite() && s != 0.0 {
                e0 / s
            } else {
                1.0 - p1
            };
            clog(&format!(
                "DBG logits: l0={:.4} l1={:.4} p0={:.4} p1={:.4} | win_len={}",
                l0, l1, p0, p1, win_len
            ));
        }
        self.last_score = p1;
        #[cfg(test)]
        {
            self.dbg.last_logits = Some(logits);
            self.dbg.last_prob = Some(p1);
        }
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
        let enc_frame = if frames_usize >= self.pulse_delay + self.pulse_frames {
            let start = frames_usize - self.pulse_delay - self.pulse_frames;
            clog(&format!(
                "ENC slicing delayed {} frames (start={start}, delay={})",
                self.pulse_frames, self.pulse_delay
            ));
            enc_all.slice_move(s![.., start..start + self.pulse_frames])
        } else if frames_usize >= self.pulse_frames {
            // Early warmup: fall back to absolute last 4 frames
            let start = frames_usize - self.pulse_frames;
            clog(&format!(
                "ENC slicing tail {} frames (start={start})",
                self.pulse_frames
            ));
            enc_all.slice_move(s![.., start..])
        } else {
            // Not enough frames yet from pulsed encoder; keep window unchanged and do not advance readiness.
            let val: Value = self.encoder_frame_buffer.clone().try_into()?;
            return Ok(vec![val]);
        };
        let n: i32 = self.pulse_frames.try_into()?;
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
            self.dbg.set_enc_block(&enc_frame);
            let win = self.encoder_frame_buffer.clone();
            self.dbg.set_encoder_window(&win);
        }
        // Track stable frames produced so far for warmup gating.
        self.stable_frames_ready = self.stable_frames_ready.saturating_add(self.pulse_frames);
        let val: Value = self.encoder_frame_buffer.clone().try_into()?;
        Ok(vec![val])
    }
}

// Share common utilities via trait
impl VadSessionCommon for VadSessionBatch {
    fn decoder_model(&self) -> &Runnable {
        &self.decoder_model
    }
    fn encoder_frame_buffer(&self) -> &Array2<f32> {
        &self.encoder_frame_buffer
    }
    fn encoder_frame_buffer_mut(&mut self) -> &mut Array2<f32> {
        &mut self.encoder_frame_buffer
    }
    fn pulse_delay(&self) -> usize {
        self.pulse_delay
    }
    fn get_last_score(&self) -> f32 {
        self.last_score
    }
    fn set_last_score(&mut self, v: f32) {
        self.last_score = v;
    }
    fn stable_frames_ready(&self) -> usize {
        self.stable_frames_ready
    }
    fn add_stable_frames(&mut self, n: usize) {
        self.stable_frames_ready = self.stable_frames_ready.saturating_add(n);
    }
    fn on_decoded(&mut self, logits: &Array1<f32>, p1: f32) {
        #[cfg(test)]
        {
            self.dbg.last_logits = Some(logits.clone());
            self.dbg.last_prob = Some(p1);
        }
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
    pulse_frames: usize,
    frame_size: usize,
    #[cfg(test)]
    dbg: SessionDebug,
}

impl VadSessionBatch {
    // For batch mode, use same decoder window params as pulsed

    fn new(
        preprocessor: &Runnable,
        encoder: &Runnable,
        decoder: &Runnable,
        pulse_delay: usize,
        pulse_frames: usize,
        frame_size: usize,
    ) -> Res<VadSessionBatch> {
        let n_encoder_frames_to_aggregate_over = 10; // pool ~100ms at decoder
        // NOTE: using  buffer_frames * frame_size here is WRONG
        // THIS LEAD TO DEGRADATION IN BATCH MODE BECAUSE THE PREPROCESSOR FEATURES
        // WERE NOT PROPERLY ALIGNED WITH THE ENCODER FRAMES FED TO THE DECODER
        // let buffer_frames = pulse_delay
        //     .saturating_add(n_encoder_frames_to_aggregate_over)
        //     .saturating_add(pulse_frames.max(1));
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_model: encoder.clone(),
            decoder_model: decoder.clone(),
            // Big rolling buffer initialized with zeros to satisfy receptive field
            audio_buffer: vec![0.0; pulse_delay * frame_size],
            last_score: 0.0,
            encoder_frame_buffer: Array2::<f32>::zeros((128, n_encoder_frames_to_aggregate_over)),
            stable_frames_ready: 0,
            pulse_delay,
            pulse_frames: pulse_frames.max(1),
            frame_size,
            #[cfg(test)]
            dbg: SessionDebug::default(),
        })
    }

    fn predict_speech_presence(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        self.roll_receptive_buffer(&raw_audio_data);
        let pre_feat = self.preprocess_full_2d()?;
        let enc_all = self.encode_full(&pre_feat)?;
        let block = self.select_aligned_block_from_enc(&enc_all, &pre_feat);
        let dec_in = self.build_decoder_input_from_block(&block)?;
        if !self.warmup_ready() {
            return Ok(self.last_score);
        }
        self.decode_from_input(dec_in)
    }

    fn roll_receptive_buffer(&mut self, data: &[f32]) {
        roll_into_ring(&mut self.audio_buffer, data);
    }

    fn preprocess_full_2d(&mut self) -> Res<Array2<f32>> {
        validate_audio_range_11(&self.audio_buffer)?;
        let pre_feat = run_preprocessor_2d(&self.preprocessor_model, &self.audio_buffer)?;
        #[cfg(test)]
        {
            self.dbg.set_pre_feat(&pre_feat);
        }
        Ok(pre_feat)
    }

    fn encode_full(&mut self, pre_feat: &Array2<f32>) -> Res<Array2<f32>> {
        clog("BATCH ENC run");
        let pre_val_2d: Value = pre_feat.clone().try_into()?;
        let enc_result = self.encoder_model.run(vec![pre_val_2d])?;
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
            self.dbg.set_enc_out(&enc_all);
        }
        ensure!(
            features == 128,
            "expected 128 encoder features, got {}",
            features
        );
        ensure!(
            frames_usize >= self.pulse_frames,
            "encoder produced too few frames: {}",
            frames_usize
        );
        Ok(enc_all)
    }

    fn select_aligned_block_from_enc(
        &mut self,
        enc_all: &Array2<f32>,
        pre_feat: &Array2<f32>,
    ) -> Array2<f32> {
        let frames_usize = enc_all.shape()[1];
        // Empirically, pulsed encoder stable output lags batch by ~1 frame
        let align_shift: usize = 1;
        let block = if frames_usize >= self.pulse_delay + self.pulse_frames + align_shift {
            let start = frames_usize - self.pulse_delay - self.pulse_frames - align_shift;
            enc_all
                .slice(s![.., start..start + self.pulse_frames])
                .to_owned()
        } else {
            enc_all
                .slice(s![.., frames_usize - self.pulse_frames..frames_usize])
                .to_owned()
        };
        #[cfg(test)]
        {
            self.dbg.set_enc_block(&block);
            let t = pre_feat.shape()[1];
            let start = if t >= self.pulse_delay + self.pulse_frames + align_shift {
                t - self.pulse_delay - self.pulse_frames - align_shift
            } else if t >= self.pulse_frames {
                t - self.pulse_frames
            } else {
                0
            };
            let pre_blk = pre_feat
                .slice(s![.., start..start + self.pulse_frames])
                .to_owned();
            self.dbg.set_pre_sliced(&pre_blk);
        }
        block
    }

    fn build_decoder_input_from_block(&mut self, block: &Array2<f32>) -> Res<Vec<Value>> {
        let dec_in = self.slide_window_append(block, self.pulse_frames)?;
        #[cfg(test)]
        {
            let win = self.encoder_frame_buffer.clone();
            self.dbg.set_encoder_window(&win);
        }
        Ok(dec_in)
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
        let wav_path = Path::new("assets/audio/silence_16_khz.wav");
        assert!(wav_path.exists(), "silence wav not found at {:?}", wav_path);

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
        let clf = VadClassifier::load_internal(4)?;
        let mut pulsed = VadSessionPulsed::new(
            &clf.preprocessor_model,
            &clf.encoder_model_pulsed,
            &clf.decoder_model,
            4,
            clf.frame_size,
            clf.compute_pulse_delay_from_encoder(),
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
            4,
            clf.frame_size,
        )?;

        // Stream in 4-frame pulses (640 samples at 16kHz)
        let step = pulsed.step_samples();
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
        let wav_path = Path::new("assets/audio/silence_16k.wav");
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
        self::write_npy_f32(
            Path::new("target/vad_dumps/silence_samples.npy"),
            &samples,
            &[samples.len()],
        )?;

        // Build VAD components
        let clf = VadClassifier::load_internal(4)?;
        let pre_feats_arr = run_preprocessor_2d(&clf.preprocessor_model, &samples)?;
        let new = pre_feats_arr.clone();
        let pre_shape = new.shape();
        let pre_feats = pre_feats_arr.into_raw_vec_and_offset().0;
        // up preprocessor for clean debug dumps
        self::write_npy_f32(
            Path::new("target/vad_dumps/silence_pre_feats.npy"),
            &pre_feats,
            pre_shape,
        )?;
        let mut pulsed = VadSessionPulsed::new(
            &clf.preprocessor_model,
            &clf.encoder_model_pulsed,
            &clf.decoder_model,
            4,
            clf.frame_size,
            clf.compute_pulse_delay_from_encoder(),
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
            4,
            clf.frame_size,
        )?;

        let step = pulsed.step_samples();
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
            if let Some(a) = &pulsed.dbg.last_pre_feat {
                write_npy_arr2(&step_dir.join("pulsed_pre_full.npy"), a)?;
            }
            if let Some(a) = &pulsed.dbg.last_pre_sliced {
                write_npy_arr2(&step_dir.join("pulsed_pre_4.npy"), a)?;
            }
            if let Some(a) = &pulsed.dbg.last_enc_out {
                write_npy_arr2(&step_dir.join("pulsed_enc_full.npy"), a)?;
            }
            if let Some(a) = &pulsed.dbg.last_enc_block {
                write_npy_arr2(&step_dir.join("pulsed_enc_4.npy"), a)?;
            }
            if let Some(a) = &pulsed.dbg.last_encoder_window {
                write_npy_arr2(&step_dir.join("pulsed_dec_in.npy"), a)?;
            }
            if let Some(a) = &pulsed.dbg.last_logits {
                write_npy_arr1(&step_dir.join("pulsed_logits.npy"), a)?;
            }
            if let Some(p) = pulsed.dbg.last_prob {
                write_npy_f32(&step_dir.join("pulsed_prob.npy"), &[p], &[1])?;
            }

            // Batch snapshots
            if let Some(a) = &batch.dbg.last_pre_feat {
                write_npy_arr2(&step_dir.join("batch_pre_full.npy"), a)?;
            }
            if let Some(a) = &batch.dbg.last_pre_sliced {
                write_npy_arr2(&step_dir.join("batch_pre_4.npy"), a)?;
            }
            if let Some(a) = &batch.dbg.last_enc_out {
                write_npy_arr2(&step_dir.join("batch_enc_full.npy"), a)?;
            }
            if let Some(a) = &batch.dbg.last_enc_block {
                write_npy_arr2(&step_dir.join("batch_enc_4.npy"), a)?;
            }
            if let Some(a) = &batch.dbg.last_encoder_window {
                write_npy_arr2(&step_dir.join("batch_dec_in.npy"), a)?;
            }
            if let Some(a) = &batch.dbg.last_logits {
                write_npy_arr1(&step_dir.join("batch_logits.npy"), a)?;
            }
            if let Some(p) = batch.dbg.last_prob {
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
            if let (Some(a), Some(b)) = (&pulsed.dbg.last_pre_sliced, &batch.dbg.last_pre_sliced) {
                diff(a, b, "pre_4");
            }
            if let (Some(a), Some(b)) = (&pulsed.dbg.last_enc_block, &batch.dbg.last_enc_block) {
                diff(a, b, "enc_4");
            }
            if let (Some(a), Some(b)) = (
                &pulsed.dbg.last_encoder_window,
                &batch.dbg.last_encoder_window,
            ) {
                diff(a, b, "dec_in_window");
            }
            if let (Some(a), Some(b)) = (&pulsed.dbg.last_logits, &batch.dbg.last_logits)
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
