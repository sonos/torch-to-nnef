#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code, unused))]

mod audio;
mod session;
mod session_batch;
mod session_pulsed;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use tract_rs::prelude::*;

tract_rs::impl_ndarray_interop!();

use audio::clog;
use session_batch::VadSessionBatch;
use session_pulsed::VadSessionPulsed;

pub(crate) type Res<T> = anyhow::Result<T>;

// Preprocessor window: 1 second of 16 kHz audio (fixed input shape in NNEF).
pub(crate) const PREPROCESSOR_INPUT_SAMPLES: usize = 16000;
// Samples per LFR'd feature frame (10 ms at 16 kHz).
const VAD_ENCODER_INPUT_FRAME_SIZE: usize = 160;
// FSMN emits per-frame softmax posteriors; `silence_pdf_ids = [0]` in the
// funasr/fsmn-vad config so `p(speech) = 1 - probs[0]`.
pub(crate) const SILENCE_PDF_IDX: usize = 0;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
struct VadClassifier {
    preprocessor_model: Runnable,
    encoder_model_pulsed: Runnable,
    encoder_model_batch: Runnable,
    vad_session_pulsed: Option<VadSessionPulsed>,
    vad_session_batch: Option<VadSessionBatch>,
    pulse_frames: usize,
    frame_size: usize,
}

impl VadClassifier {
    fn load_internal(pulse_frames: usize) -> Res<VadClassifier> {
        console_error_panic_hook::set_once();
        clog("loading runtime 'default'");
        let rt = runtime_for_name("default")?;

        let preprocessor_bytes = include_bytes!("../model/preprocessor.nnef.tgz");
        let mut nnef = tract_rs::nnef()?.with_tract_core()?;
        nnef.enable_pulse()?;

        clog("preparing preprocessor model");
        let preprocessor_model = rt.prepare(nnef.load_buffer(preprocessor_bytes)?)?;

        let encoder_bytes = include_bytes!("../model/encoder.nnef.tgz");
        let enc_model = nnef.load_buffer(encoder_bytes)?;
        let mut enc_pulsed = enc_model.clone();
        clog("preparing encoder model (batch)");
        let encoder_model_batch = rt.prepare(enc_model)?;
        clog(&format!("pulsifying encoder: pulse_frames={}", pulse_frames));
        enc_pulsed.transform(
            Pulse::new(pulse_frames.max(1).to_string()).symbol("ENCODER__TIME"),
        )?;
        let encoder_model_pulsed = rt.prepare(enc_pulsed)?;
        clog("model loaded/optimized");

        Ok(VadClassifier {
            preprocessor_model,
            encoder_model_pulsed,
            encoder_model_batch,
            vad_session_pulsed: None,
            vad_session_batch: None,
            pulse_frames: pulse_frames.max(1),
            frame_size: VAD_ENCODER_INPUT_FRAME_SIZE,
        })
    }

    fn ensure_pulsed_session(&mut self) -> Res<&mut VadSessionPulsed> {
        if self.vad_session_pulsed.is_none() {
            self.vad_session_pulsed = Some(VadSessionPulsed::new(
                &self.preprocessor_model,
                &self.encoder_model_pulsed,
                self.pulse_frames,
                self.frame_size,
                PREPROCESSOR_INPUT_SAMPLES,
            )?);
        }
        Ok(self.vad_session_pulsed.as_mut().unwrap())
    }

    fn ensure_batch_session(&mut self) -> Res<&mut VadSessionBatch> {
        if self.vad_session_batch.is_none() {
            self.vad_session_batch = Some(VadSessionBatch::new(
                &self.preprocessor_model,
                &self.encoder_model_batch,
                PREPROCESSOR_INPUT_SAMPLES,
                self.frame_size,
            )?);
        }
        Ok(self.vad_session_batch.as_mut().unwrap())
    }

    fn predict_speech_presence_internal(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        self.ensure_pulsed_session()?
            .predict_speech_presence(raw_audio_data)
    }
}

// JS-facing API (wasm32 only). Kept minimal; the old pulse-delay / decoder
// pool accessors disappeared with the decoder stage and are no longer needed
// — the JS already handles their absence via optional-chaining fallbacks.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl VadClassifier {
    #[wasm_bindgen]
    pub fn predict_speech_presence(
        &mut self,
        js_raw_audio_data: &web_sys::js_sys::Float32Array,
    ) -> Result<JsValue, JsError> {
        let pred = self
            .predict_speech_presence_internal(js_raw_audio_data.to_vec())
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        Ok(pred.into())
    }

    #[wasm_bindgen]
    pub fn predict_speech_presence_batch(
        &mut self,
        js_raw_audio_data: &web_sys::js_sys::Float32Array,
    ) -> Result<JsValue, JsError> {
        console_error_panic_hook::set_once();
        let pred = self
            .ensure_batch_session()
            .and_then(|s| s.predict_speech_presence(js_raw_audio_data.to_vec()))
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        Ok(pred.into())
    }

    #[wasm_bindgen]
    pub fn load() -> Result<VadClassifier, JsError> {
        console_error_panic_hook::set_once();
        VadClassifier::load_internal(4).map_err(|err| JsError::new(&format!("{:?}", err)))
    }

    #[wasm_bindgen]
    pub fn load_with_pulse(pulse_frames: usize) -> Result<VadClassifier, JsError> {
        console_error_panic_hook::set_once();
        VadClassifier::load_internal(pulse_frames).map_err(|e| JsError::new(&format!("{:?}", e)))
    }

    #[wasm_bindgen]
    pub fn get_pulse_frames(&self) -> usize {
        self.pulse_frames
    }

    #[wasm_bindgen]
    pub fn get_frame_size(&self) -> usize {
        self.frame_size
    }

    #[wasm_bindgen]
    pub fn reset_sessions(&mut self) {
        self.vad_session_pulsed = None;
        self.vad_session_batch = None;
    }

    #[wasm_bindgen]
    pub fn is_pulsed_ready(&mut self) -> Result<bool, JsError> {
        let s = self
            .ensure_pulsed_session()
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        Ok(s.warmup_ready())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn read_wav_mono_16k(path: &Path) -> anyhow::Result<Vec<f32>> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000, "{} must be 16kHz mono", path.display());
        let mut samples: Vec<f32> = Vec::with_capacity(reader.duration() as usize);
        if spec.sample_format == hound::SampleFormat::Int {
            for s in reader.samples::<i16>() {
                samples.push((s? as f32 / 32768.0).clamp(-1.0, 1.0));
            }
        } else {
            for s in reader.samples::<f32>() {
                samples.push(s?.clamp(-1.0, 1.0));
            }
        }
        Ok(samples)
    }

    fn build_sessions(pulse_frames: usize) -> anyhow::Result<(VadSessionPulsed, VadSessionBatch, usize)> {
        let clf = VadClassifier::load_internal(pulse_frames)?;
        let pulsed = VadSessionPulsed::new(
            &clf.preprocessor_model,
            &clf.encoder_model_pulsed,
            pulse_frames,
            clf.frame_size,
            PREPROCESSOR_INPUT_SAMPLES,
        )?;
        let batch = VadSessionBatch::new(
            &clf.preprocessor_model,
            &clf.encoder_model_batch,
            PREPROCESSOR_INPUT_SAMPLES,
            clf.frame_size,
        )?;
        let step = pulsed.step_samples();
        Ok((pulsed, batch, step))
    }

    fn stream_scores(
        samples: &[f32],
        pulsed: &mut VadSessionPulsed,
        batch: &mut VadSessionBatch,
        step: usize,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let mut pulsed_scores: Vec<f32> = Vec::new();
        let mut batch_scores: Vec<f32> = Vec::new();
        let mut i = 0usize;
        while i < samples.len() {
            let end = (i + step).min(samples.len());
            let chunk = samples[i..end].to_vec();
            pulsed_scores.push(pulsed.predict_speech_presence(chunk.clone())?);
            batch_scores.push(batch.predict_speech_presence(chunk)?);
            i = end;
        }
        Ok((pulsed_scores, batch_scores))
    }

    fn filter_finite(v: &[f32]) -> Vec<f32> {
        v.iter().copied().filter(|x| x.is_finite()).collect()
    }

    fn tail<'a>(v: &'a [f32], n: usize) -> &'a [f32] {
        &v[v.len().saturating_sub(n)..]
    }

    #[test]
    fn silence_pulsed_vs_batch_below_threshold() -> anyhow::Result<()> {
        let wav_path = Path::new("assets/audio/silence_16k.wav");
        assert!(wav_path.exists(), "silence wav not found at {:?}", wav_path);

        let samples = read_wav_mono_16k(wav_path)?;
        let (mut pulsed, mut batch, step) = build_sessions(4)?;
        let (pulsed_scores, batch_scores) = stream_scores(&samples, &mut pulsed, &mut batch, step)?;

        let pulsed_finite = filter_finite(&pulsed_scores);
        let batch_finite = filter_finite(&batch_scores);
        assert!(!pulsed_finite.is_empty() && !batch_finite.is_empty());

        println!("pulsed last 20 p(speech): {:?}", tail(&pulsed_finite, 20));
        println!("batch  last 20 p(speech): {:?}", tail(&batch_finite, 20));

        // FSMN-VAD + MelSpectrogram featurizer has a higher silence baseline
        // than marblenet + kaldi fbank; threshold chosen well below typical
        // speech score (>0.95) so misdetection is still caught.
        let thr = 0.3f32;
        for (idx, v) in pulsed_finite.iter().enumerate() {
            assert!(*v <= thr, "pulsed p(speech) at step {} = {:.5} exceeds {}", idx, v, thr);
        }
        for (idx, v) in batch_finite.iter().enumerate() {
            assert!(*v <= thr, "batch p(speech) at step {} = {:.5} exceeds {}", idx, v, thr);
        }
        Ok(())
    }

    #[test]
    fn speech_pulsed_vs_batch_above_95_percent() -> anyhow::Result<()> {
        let wav_path = Path::new("assets/audio/speech.wav");
        assert!(wav_path.exists(), "missing speech wav at {:?}", wav_path);
        let samples = read_wav_mono_16k(wav_path)?;
        let (mut pulsed, mut batch, step) = build_sessions(4)?;
        let (pulsed_scores, batch_scores) = stream_scores(&samples, &mut pulsed, &mut batch, step)?;

        let pulsed_finite = filter_finite(&pulsed_scores);
        let batch_finite = filter_finite(&batch_scores);
        assert!(!pulsed_finite.is_empty() && !batch_finite.is_empty());

        let argmax = |v: &[f32]| v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &x)| (i, x))
            .unwrap_or((0, 0.0));
        let (idx_p, max_p) = argmax(&pulsed_finite);
        let (idx_b, max_b) = argmax(&batch_finite);
        println!("peak pulsed: {:.4} @{} | peak batch: {:.4} @{}", max_p, idx_p, max_b, idx_b);

        assert!(max_p >= 0.95, "pulsed peak {:.5} below 95%", max_p);
        assert!(max_b >= 0.95, "batch peak {:.5} below 95%", max_b);
        // Peak plateaus at ~1.0 over a range of frames; argmax is sensitive to
        // tiny feature/warmup differences between pulsed and batch.
        let idx_diff = idx_p.abs_diff(idx_b);
        assert!(idx_diff <= 5, "peak index mismatch too large: diff={}", idx_diff);
        Ok(())
    }
}
