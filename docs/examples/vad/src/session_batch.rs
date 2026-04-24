use ndarray::Array3;
#[cfg(test)]
use ndarray::s;
use tract_rs::prelude::*;

use crate::Res;
use crate::audio::{clog, roll_into_ring, run_preprocessor, validate_audio_range_11};
use crate::session::{VadSessionCommon, tail_p_speech};
#[cfg(test)]
use crate::session::SessionDebug;
use crate::{Ndarray as _, Tract as _};

pub(crate) struct VadSessionBatch {
    preprocessor_model: Runnable,
    encoder_model: Runnable,
    audio_buffer: Vec<f32>,
    last_score: f32,
    stable_frames_ready: usize,
    frame_size: usize,
    #[cfg(test)]
    pub(crate) dbg: SessionDebug,
}

impl VadSessionCommon for VadSessionBatch {
    fn audio_buffer(&self) -> &[f32] { &self.audio_buffer }
    fn audio_buffer_mut(&mut self) -> &mut [f32] { &mut self.audio_buffer }
    fn get_last_score(&self) -> f32 { self.last_score }
    fn set_last_score(&mut self, v: f32) { self.last_score = v; }
    fn stable_frames_ready(&self) -> usize { self.stable_frames_ready }
    fn add_stable_frames(&mut self, n: usize) {
        self.stable_frames_ready = self.stable_frames_ready.saturating_add(n);
    }
}

impl VadSessionBatch {
    pub(crate) fn new(
        preprocessor: &Runnable,
        encoder: &Runnable,
        preprocessor_samples: usize,
        frame_size: usize,
    ) -> Res<Self> {
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_model: encoder.clone(),
            audio_buffer: vec![0.0; preprocessor_samples],
            last_score: f32::NAN,
            stable_frames_ready: 0,
            frame_size,
            #[cfg(test)]
            dbg: SessionDebug::default(),
        })
    }

    pub(crate) fn predict_speech_presence(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        let n_new = raw_audio_data.len();
        roll_into_ring(&mut self.audio_buffer, &raw_audio_data);
        self.add_stable_frames(n_new / self.frame_size);

        validate_audio_range_11(&self.audio_buffer)?;
        let feats = run_preprocessor(&self.preprocessor_model, &self.audio_buffer)?;
        #[cfg(test)]
        {
            let feat2 = feats.slice(s![0, .., ..]).to_owned();
            self.dbg.set_pre_feat(&feat2);
        }

        clog("BATCH ENC run");
        let feat_tensor: Tensor = feats.clone().tract()?;
        let probs_out = self.encoder_model.run(vec![feat_tensor])?;
        let probs_view = probs_out[0].ndarray::<f32>()?;
        let probs: Array3<f32> = probs_view.into_dimensionality()?.to_owned();
        #[cfg(test)]
        {
            let prob2 = probs.slice(s![0, .., ..]).to_owned();
            self.dbg.set_probs(&prob2);
        }

        // Gate early scores while the rolling audio buffer is mostly zero:
        // the receptive field of the last-frame probability then mixes zero
        // and real-audio features, producing noisy transitions. Same frame
        // count as the pulsed session so both are fair to compare.
        if !self.warmup_ready() {
            return Ok(self.last_score);
        }
        let p = tail_p_speech(&probs);
        self.last_score = p;
        #[cfg(test)]
        self.dbg.set_prob(p);
        Ok(p)
    }
}
