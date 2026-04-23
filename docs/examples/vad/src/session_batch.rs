use anyhow::ensure;
use ndarray::{Array1, Array2};
#[cfg(test)]
use ndarray::s;
use tract_rs::prelude::*;

use crate::Res;
use crate::{Ndarray as _, Tract as _};
use crate::audio::{clog, roll_into_ring, run_preprocessor, validate_audio_range_11};
#[cfg(test)]
use crate::session::SessionDebug;
use crate::session::VadSessionCommon;

pub(crate) struct VadSessionBatch {
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
    #[cfg(test)]
    pub(crate) dbg: SessionDebug,
}

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
    fn pulse_frames(&self) -> usize {
        self.pulse_frames
    }
    fn enc_align_shift(&self) -> usize { 1 }
    #[cfg(test)]
    fn on_enc_block(&mut self, block: &Array2<f32>) {
        self.dbg.set_enc_block(block);
    }
    #[cfg(test)]
    fn on_enc_window(&mut self, win: &Array2<f32>) {
        self.dbg.set_encoder_window(win);
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
    fn on_decoded(&mut self, _logits: &Array1<f32>, _p1: f32) {
        #[cfg(test)]
        {
            self.dbg.last_logits = Some(_logits.clone());
            self.dbg.last_prob = Some(_p1);
        }
    }
}

impl VadSessionBatch {
    pub(crate) fn new(
        preprocessor: &Runnable,
        encoder: &Runnable,
        decoder: &Runnable,
        pulse_delay: usize,
        pulse_frames: usize,
        frame_size: usize,
    ) -> Res<VadSessionBatch> {
        let n_encoder_frames_to_aggregate_over = 10; // pool ~100ms at decoder
        // Keep the pulsed step <= decoder pool to avoid window overruns
        assert!(
            pulse_frames <= n_encoder_frames_to_aggregate_over,
            "pulse_frames must be <= {} (decoder pool)",
            n_encoder_frames_to_aggregate_over
        );
        // NOTE: audio_buffer is sized by pulse_delay * frame_size (NOT by the decoder
        // window size). Sizing by the window led to misaligned preprocessor features.
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_model: encoder.clone(),
            decoder_model: decoder.clone(),
            // Rolling buffer initialized with zeros to satisfy the receptive field
            audio_buffer: vec![0.0; pulse_delay * frame_size],
            last_score: 0.0,
            encoder_frame_buffer: Array2::<f32>::zeros((128, n_encoder_frames_to_aggregate_over)),
            stable_frames_ready: 0,
            pulse_delay,
            pulse_frames: pulse_frames.max(1),
            #[cfg(test)]
            dbg: SessionDebug::default(),
        })
    }

    pub(crate) fn predict_speech_presence(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        roll_into_ring(&mut self.audio_buffer, &raw_audio_data);
        let pre_feat = self.preprocess_full()?;
        let enc_all = self.encode_full(&pre_feat)?;
        #[cfg(test)]
        {
            // Record the preprocessor slice aligned to the same window as the encoder block.
            let t = pre_feat.shape()[1];
            let align_shift = self.enc_align_shift();
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
        let dec_in = self.build_dec_input(&enc_all)?;
        if !self.warmup_ready() {
            return Ok(self.last_score);
        }
        self.decode_from_input(dec_in)
    }

    fn preprocess_full(&mut self) -> Res<Array2<f32>> {
        validate_audio_range_11(&self.audio_buffer)?;
        let pre_feat = run_preprocessor(&self.preprocessor_model, &self.audio_buffer)?;
        #[cfg(test)]
        {
            self.dbg.set_pre_feat(&pre_feat);
        }
        Ok(pre_feat)
    }

    fn encode_full(&mut self, pre_feat: &Array2<f32>) -> Res<Array2<f32>> {
        clog("BATCH ENC run");
        let pre_val_2d: Tensor = pre_feat.clone().tract()?;
        let enc_result = self.encoder_model.run(vec![pre_val_2d])?;
        let enc_all = enc_result[0].ndarray2::<f32>()?.to_owned();
        #[cfg(test)]
        {
            self.dbg.set_enc_out(&enc_all);
        }
        ensure!(
            enc_all.shape()[0] == 128,
            "expected 128 encoder features, got {}",
            enc_all.shape()[0]
        );
        ensure!(
            enc_all.shape()[1] >= self.pulse_frames,
            "encoder produced too few frames: {}",
            enc_all.shape()[1]
        );
        Ok(enc_all)
    }

}
