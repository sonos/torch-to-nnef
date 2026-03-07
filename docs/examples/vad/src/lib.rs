#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code, unused))]

mod audio;
mod session;
mod session_batch;
mod session_pulsed;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use tract_rs::prelude::{tract_ndarray::IndexLonger, *};

use audio::clog;
use session_batch::VadSessionBatch;
use session_pulsed::VadSessionPulsed;
use crate::session::VadSessionCommon;

pub(crate) type Res<T> = anyhow::Result<T>;

// Shared VAD constants
const VAD_ENCODER_INPUT_FRAME_SIZE: usize = 160; // 10ms at 16kHz

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
                VAD_ENCODER_INPUT_FRAME_SIZE,
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
        Ok(s.pulse_delay())
    }

    #[wasm_bindgen]
    pub fn get_decoder_pool_len(&mut self) -> Result<usize, JsError> {
        let s = self
            .ensure_pulsed_session()
            .map_err(|err| JsError::new(&format!("{:?}", err)))?;
        Ok(s.encoder_frame_buffer().shape()[1])
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
    use audio::run_preprocessor;
    use tract_rs::prelude::tract_ndarray::{Array1, Array2};

    use std::fs::{self, File};
    use std::io::Write;
    use std::path::Path;

    fn write_npy_f32(path: &std::path::Path, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
        // Minimal NPY v1.0 writer for little-endian f32, C-order
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
        let wav_path = Path::new("assets/audio/silence_16k.wav");
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
        let pulse_delay = clf.compute_pulse_delay_from_encoder();
        let mut pulsed = VadSessionPulsed::new(
            &clf.preprocessor_model,
            &clf.encoder_model_pulsed,
            &clf.decoder_model,
            4,
            clf.frame_size,
            pulse_delay,
        )?;
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
        fs::create_dir_all("target/vad_dumps")?;
        write_npy_f32(
            Path::new("target/vad_dumps/silence_samples.npy"),
            &samples,
            &[samples.len()],
        )?;

        // Build VAD components
        let clf = VadClassifier::load_internal(4)?;
        let pre_feats_arr = run_preprocessor(&clf.preprocessor_model, &samples)?;
        let pre_shape: Vec<usize> = pre_feats_arr.shape().to_vec();
        let pre_feats = pre_feats_arr.into_raw_vec_and_offset().0;
        write_npy_f32(
            Path::new("target/vad_dumps/silence_pre_feats.npy"),
            &pre_feats,
            &pre_shape,
        )?;
        let pulse_delay = clf.compute_pulse_delay_from_encoder();
        let mut pulsed = VadSessionPulsed::new(
            &clf.preprocessor_model,
            &clf.encoder_model_pulsed,
            &clf.decoder_model,
            4,
            clf.frame_size,
            pulse_delay,
        )?;
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

    #[test]
    fn speech_pulsed_vs_batch_probs_above_95_percent() -> anyhow::Result<()> {
        // Expect a clean speech segment to yield high p(speech) near the tail
        let wav_path = Path::new("assets/audio/speech.wav");
        assert!(wav_path.exists(), "missing speech wav at {:?}", wav_path);

        let mut reader = hound::WavReader::open(wav_path)?;
        let spec = reader.spec();
        assert_eq!(
            spec.sample_rate, 16_000,
            "speech.wav must be 16kHz mono; got {} Hz",
            spec.sample_rate
        );
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
        let clf = VadClassifier::load_internal(4)?;
        let pulse_delay = clf.compute_pulse_delay_from_encoder();
        let mut pulsed = VadSessionPulsed::new(
            &clf.preprocessor_model,
            &clf.encoder_model_pulsed,
            &clf.decoder_model,
            4,
            clf.frame_size,
            pulse_delay,
        )?;
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

        // Consider only post-warmup (finite) values and check the tail
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

        // Check last 20 values from each sequence
        let n_check = 20usize;
        let pulsed_tail_start = pulsed_finite.len().saturating_sub(n_check);
        let batch_tail_start = batch_finite.len().saturating_sub(n_check);
        let pulsed_tail = &pulsed_finite[pulsed_tail_start..];
        let batch_tail = &batch_finite[batch_tail_start..];
        println!("pulsed last {} p(speech): {:?}", n_check, pulsed_tail);
        println!("batch  last {} p(speech): {:?}", n_check, batch_tail);

        let thr = 0.95f32;
        for (idx, v) in pulsed_tail.iter().enumerate() {
            assert!(
                *v >= thr,
                "pulsed p(speech) at tail idx {} = {:.5} below 95%",
                idx,
                v
            );
        }
        for (idx, v) in batch_tail.iter().enumerate() {
            assert!(
                *v >= thr,
                "batch p(speech) at tail idx {} = {:.5} below 95%",
                idx,
                v
            );
        }

        Ok(())
    }
}
