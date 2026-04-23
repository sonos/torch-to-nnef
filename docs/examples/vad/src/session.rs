use anyhow::ensure;
use ndarray::{Array1, Array2, s};
use tract_rs::prelude::*;

use crate::Res;
use crate::audio::{clog, select_enc_block};
use crate::{Ndarray as _, Tract as _};

#[cfg(test)]
#[derive(Default, Clone)]
pub(crate) struct SessionDebug {
    pub(crate) last_pre_feat: Option<Array2<f32>>,
    pub(crate) last_pre_sliced: Option<Array2<f32>>,
    pub(crate) last_enc_out: Option<Array2<f32>>,
    pub(crate) last_enc_block: Option<Array2<f32>>,
    pub(crate) last_encoder_window: Option<Array2<f32>>,
    pub(crate) last_logits: Option<Array1<f32>>,
    pub(crate) last_prob: Option<f32>,
}

#[cfg(test)]
impl SessionDebug {
    pub(crate) fn set_pre_feat(&mut self, a: &Array2<f32>) {
        self.last_pre_feat = Some(a.clone());
    }
    pub(crate) fn set_pre_sliced(&mut self, a: &Array2<f32>) {
        self.last_pre_sliced = Some(a.clone());
    }
    pub(crate) fn set_enc_out(&mut self, a: &Array2<f32>) {
        self.last_enc_out = Some(a.clone());
    }
    pub(crate) fn set_enc_block(&mut self, a: &Array2<f32>) {
        self.last_enc_block = Some(a.clone());
    }
    pub(crate) fn set_encoder_window(&mut self, a: &Array2<f32>) {
        self.last_encoder_window = Some(a.clone());
    }
    pub(crate) fn set_logits_and_prob(&mut self, logits: &Array1<f32>, p: f32) {
        self.last_logits = Some(logits.clone());
        self.last_prob = Some(p);
    }
}

pub(crate) trait VadSessionCommon {
    fn decoder_model(&self) -> &Runnable;
    fn encoder_frame_buffer(&self) -> &Array2<f32>;
    fn encoder_frame_buffer_mut(&mut self) -> &mut Array2<f32>;
    fn pulse_delay(&self) -> usize;
    fn pulse_frames(&self) -> usize;
    fn get_last_score(&self) -> f32;
    fn set_last_score(&mut self, v: f32);
    fn stable_frames_ready(&self) -> usize;
    fn add_stable_frames(&mut self, n: usize);
    fn on_decoded(&mut self, _logits: &Array1<f32>, _p1: f32) {
        // default no-op; implementers may record debug/test data
    }
    fn on_enc_block(&mut self, _block: &Array2<f32>) {}
    fn on_enc_window(&mut self, _win: &Array2<f32>) {}
    // Pulsed encoder output lags batch by ~1 frame; batch overrides to 1.
    fn enc_align_shift(&self) -> usize { 0 }

    fn warmup_needed_frames(&self) -> usize {
        self.pulse_delay() + self.encoder_frame_buffer().shape()[1]
    }

    fn warmup_ready(&self) -> bool {
        self.stable_frames_ready() >= self.warmup_needed_frames()
    }

    fn slide_window_append(&mut self, block: &Array2<f32>, step_frames: usize) -> Res<Vec<Tensor>> {
        let n: i32 = step_frames.try_into()?;
        {
            let win = self.encoder_frame_buffer_mut();
            let temp = &win.slice(s![.., n..]).to_owned();
            win.slice_mut(s![.., ..-n]).assign(temp);
            win.slice_mut(s![.., -n..]).assign(block);
        }
        self.add_stable_frames(step_frames);
        let val: Tensor = self.encoder_frame_buffer().clone().tract()?;
        Ok(vec![val])
    }

    fn build_dec_input(&mut self, enc_all: &Array2<f32>) -> Res<Vec<Tensor>> {
        let Some(block) = select_enc_block(enc_all, self.pulse_frames(), self.pulse_delay(), self.enc_align_shift()) else {
            let val: Tensor = self.encoder_frame_buffer().clone().tract()?;
            return Ok(vec![val]);
        };
        #[cfg(test)]
        self.on_enc_block(&block);
        let dec_in = self.slide_window_append(&block, self.pulse_frames())?;
        #[cfg(test)]
        {
            let win = self.encoder_frame_buffer().clone();
            self.on_enc_window(&win);
        }
        Ok(dec_in)
    }

    fn decode_from_input(&mut self, decoder_input: Vec<Tensor>) -> Res<f32> {
        clog("DEC run");
        let dec_result = self.decoder_model().run(decoder_input)?;
        let logits: Array1<f32> = dec_result[0].ndarray1::<f32>()?.to_owned();
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
