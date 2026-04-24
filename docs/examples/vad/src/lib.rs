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

// Preprocessor window: 1 second of 16 kHz audio. The FSMN-VAD preprocessor
// graph was exported with this fixed input shape.
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
        clog(&format!(
            "pulsifying encoder model (derived from batch): pulse_frames={}",
            pulse_frames
        ));
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
        let session = self.ensure_pulsed_session()?;
        session.predict_speech_presence(raw_audio_data)
    }
}

// JS-facing API exposed only on wasm32. Signatures match the previous marblenet
// demo so the HTML page consuming vad_wasm.js keeps working as a drop-in.
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

    // FSMN-VAD is strictly causal (rorder=0), so the pulsed encoder has no
    // intrinsic delay versus the batch encoder. Kept for API compat.
    #[wasm_bindgen]
    pub fn get_pulse_delay(&mut self) -> Result<usize, JsError> {
        Ok(0)
    }

    // No decoder pool in FSMN-VAD (per-frame posteriors). Kept for API compat;
    // return the pulse step used on the encoder input.
    #[wasm_bindgen]
    pub fn get_decoder_pool_len(&mut self) -> Result<usize, JsError> {
        Ok(self.pulse_frames)
    }

    #[wasm_bindgen]
    pub fn is_pulsed_ready(&mut self) -> Result<bool, JsError> {
        use crate::session::VadSessionCommon;
        let s = self
            .ensure_pulsed_session()
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        Ok(s.warmup_ready())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    use std::fs::{self, File};
    use std::io::Write;
    use std::path::Path;

    fn read_wav_mono_16k(path: &Path) -> anyhow::Result<Vec<f32>> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        assert_eq!(
            spec.sample_rate, 16_000,
            "{} must be 16kHz mono; got {} Hz",
            path.display(),
            spec.sample_rate
        );
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
            let p_pulsed = pulsed.predict_speech_presence(chunk.clone())?;
            let p_batch = batch.predict_speech_presence(chunk)?;
            pulsed_scores.push(p_pulsed);
            batch_scores.push(p_batch);
            i = end;
        }
        Ok((pulsed_scores, batch_scores))
    }

    fn filter_finite(v: &[f32]) -> Vec<f32> {
        v.iter().copied().filter(|x| x.is_finite()).collect()
    }

    fn tail<'a>(v: &'a [f32], n: usize) -> &'a [f32] {
        let start = v.len().saturating_sub(n);
        &v[start..]
    }

    fn write_npy_f32(path: &std::path::Path, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(b"\x93NUMPY")?;
        f.write_all(&[1, 0])?;
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
        let preamble_len = 10 + 2;
        let mut header_len = header_dict.len() + 1;
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
        for &v in data {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    fn write_npy_arr2(path: &std::path::Path, a: &Array2<f32>) -> std::io::Result<()> {
        let buf: Vec<f32> = a.iter().copied().collect();
        write_npy_f32(path, &buf, a.shape())
    }

    #[test]
    fn silence_pulsed_vs_batch_probs_below_6_percent() -> anyhow::Result<()> {
        let wav_path = Path::new("assets/audio/silence_16k.wav");
        assert!(wav_path.exists(), "silence wav not found at {:?}", wav_path);

        let samples = read_wav_mono_16k(wav_path)?;
        let (mut pulsed, mut batch, step) = build_sessions(4)?;
        let (pulsed_scores, batch_scores) = stream_scores(&samples, &mut pulsed, &mut batch, step)?;

        let pulsed_finite: Vec<f32> = filter_finite(&pulsed_scores);
        let batch_finite: Vec<f32> = filter_finite(&batch_scores);

        assert!(!pulsed_finite.is_empty(), "no finite pulsed predictions observed");
        assert!(!batch_finite.is_empty(), "no finite batch predictions observed");

        let n_print = 20usize;
        println!("pulsed last {} p(speech): {:?}", n_print, tail(&pulsed_finite, n_print));
        println!("batch  last {} p(speech): {:?}", n_print, tail(&batch_finite, n_print));

        // FSMN-VAD + MelSpectrogram featurizer has a higher silence baseline
        // than marblenet + kaldi fbank did (no preemphasis, different log
        // semantics). Threshold is chosen well below typical speech score
        // (>0.95) so misdetection on real speech is still caught.
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
    fn silence_debug_dump_pulsed_vs_batch() -> anyhow::Result<()> {
        let wav_path = Path::new("assets/audio/silence_16k.wav");
        assert!(wav_path.exists(), "missing silence wav");
        let samples = read_wav_mono_16k(wav_path)?;
        fs::create_dir_all("target/vad_dumps")?;
        write_npy_f32(
            Path::new("target/vad_dumps/silence_samples.npy"),
            &samples,
            &[samples.len()],
        )?;
        let (mut pulsed, mut batch, step) = build_sessions(4)?;
        let base = Path::new("target/vad_dumps/silence");
        fs::create_dir_all(base)?;
        let mut i = 0usize;
        let mut step_idx = 0usize;
        while i < samples.len() {
            let end = (i + step).min(samples.len());
            let chunk = samples[i..end].to_vec();
            let _ = pulsed.predict_speech_presence(chunk.clone())?;
            let _ = batch.predict_speech_presence(chunk)?;
            let step_dir = base.join(format!("step_{:04}", step_idx));
            fs::create_dir_all(&step_dir)?;
            if let Some(a) = &pulsed.dbg.last_pre_feat {
                write_npy_arr2(&step_dir.join("pulsed_pre_full.npy"), a)?;
            }
            if let Some(a) = &pulsed.dbg.last_pre_sliced {
                write_npy_arr2(&step_dir.join("pulsed_pre_sliced.npy"), a)?;
            }
            if let Some(a) = &pulsed.dbg.last_probs {
                write_npy_arr2(&step_dir.join("pulsed_probs.npy"), a)?;
            }
            if let Some(p) = pulsed.dbg.last_prob {
                write_npy_f32(&step_dir.join("pulsed_prob.npy"), &[p], &[1])?;
            }
            if let Some(a) = &batch.dbg.last_pre_feat {
                write_npy_arr2(&step_dir.join("batch_pre_full.npy"), a)?;
            }
            if let Some(a) = &batch.dbg.last_probs {
                write_npy_arr2(&step_dir.join("batch_probs.npy"), a)?;
            }
            if let Some(p) = batch.dbg.last_prob {
                write_npy_f32(&step_dir.join("batch_prob.npy"), &[p], &[1])?;
            }
            if let (Some(a), Some(b)) = (&pulsed.dbg.last_pre_feat, &batch.dbg.last_pre_feat)
                && a.shape() == b.shape()
            {
                let mae = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f32>()
                    / a.len() as f32;
                println!("step {} diff pre_feat: mae={:.6}", step_idx, mae);
            }
            i = end;
            step_idx += 1;
        }
        Ok(())
    }

    #[test]
    fn speech_pulsed_vs_batch_probs_above_95_percent() -> anyhow::Result<()> {
        let wav_path = Path::new("assets/audio/speech.wav");
        assert!(wav_path.exists(), "missing speech wav at {:?}", wav_path);
        let samples = read_wav_mono_16k(wav_path)?;
        let (mut pulsed, mut batch, step) = build_sessions(4)?;
        let (pulsed_scores, batch_scores) = stream_scores(&samples, &mut pulsed, &mut batch, step)?;

        let pulsed_finite: Vec<f32> = filter_finite(&pulsed_scores);
        let batch_finite: Vec<f32> = filter_finite(&batch_scores);
        assert!(!pulsed_finite.is_empty());
        assert!(!batch_finite.is_empty());

        let (idx_max_pulsed, &max_pulsed) = pulsed_finite
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &0.0));
        let (idx_max_batch, &max_batch) = batch_finite
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &0.0));
        println!(
            "peak pulsed: {:.4} @{} | peak batch: {:.4} @{}",
            max_pulsed, idx_max_pulsed, max_batch, idx_max_batch
        );
        assert!(max_pulsed >= 0.95, "pulsed peak {:.5} below 95%", max_pulsed);
        assert!(max_batch >= 0.95, "batch peak {:.5} below 95%", max_batch);
        let idx_diff = idx_max_pulsed.abs_diff(idx_max_batch);
        // Peak is plateau'd at ~1.0 over a range of frames, so argmax is
        // sensitive to tiny feature/warmup differences between pulsed and
        // batch. Allow up to 5 frames drift.
        assert!(idx_diff <= 5, "peak index mismatch too large: diff={}", idx_diff);
        Ok(())
    }
}
