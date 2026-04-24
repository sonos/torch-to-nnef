use ndarray::{Array3, s};
use tract_rs::{State, prelude::*};

use crate::audio::{clog, roll_into_ring, run_preprocessor, validate_audio_range_11};
use crate::session::{VadSessionCommon, tail_p_speech};
#[cfg(test)]
use crate::session::SessionDebug;
use crate::Res;
use crate::{Ndarray as _, Tract as _};

pub(crate) struct VadSessionPulsed {
    preprocessor_model: Runnable,
    encoder_state: State,
    audio_buffer: Vec<f32>,
    last_score: f32,
    pulse_frames: usize,
    frame_size: usize,
    stable_frames_ready: usize,
    #[cfg(test)]
    pub(crate) dbg: SessionDebug,
}

impl VadSessionCommon for VadSessionPulsed {
    fn audio_buffer(&self) -> &[f32] { &self.audio_buffer }
    fn audio_buffer_mut(&mut self) -> &mut [f32] { &mut self.audio_buffer }
    fn get_last_score(&self) -> f32 { self.last_score }
    fn set_last_score(&mut self, v: f32) { self.last_score = v; }
    fn stable_frames_ready(&self) -> usize { self.stable_frames_ready }
    fn add_stable_frames(&mut self, n: usize) {
        self.stable_frames_ready = self.stable_frames_ready.saturating_add(n);
    }
}

impl VadSessionPulsed {
    pub(crate) fn new(
        preprocessor: &Runnable,
        encoder: &Runnable,
        pulse_frames: usize,
        frame_size: usize,
        preprocessor_samples: usize,
    ) -> Res<Self> {
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_state: encoder.spawn_state()?,
            audio_buffer: vec![0.0; preprocessor_samples],
            last_score: f32::NAN,
            pulse_frames: pulse_frames.max(1),
            frame_size,
            stable_frames_ready: 0,
            #[cfg(test)]
            dbg: SessionDebug::default(),
        })
    }

    pub(crate) fn step_samples(&self) -> usize { self.pulse_frames * self.frame_size }

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

        // Feed the newest `pulse_frames` LFR'd feature frames into the pulsed
        // encoder. The encoder's internal delay line maintains the left context
        // (FSMN lorder=20) across steps.
        let t = feats.shape()[1];
        let start = t.saturating_sub(self.pulse_frames);
        let slice_arr = feats.slice(s![.., start.., ..]).to_owned();
        #[cfg(test)]
        {
            let sliced2 = slice_arr.slice(s![0, .., ..]).to_owned();
            self.dbg.set_pre_sliced(&sliced2);
        }

        clog("PULSED ENC run");
        let slice_t: Tensor = slice_arr.tract()?;
        let probs_out = self.encoder_state.run(vec![slice_t])?;
        let probs_view = probs_out[0].ndarray::<f32>()?;
        let probs: Array3<f32> = probs_view.into_dimensionality()?.to_owned();
        #[cfg(test)]
        {
            let prob2 = probs.slice(s![0, .., ..]).to_owned();
            self.dbg.set_probs(&prob2);
        }

        if !self.warmup_ready() {
            clog(&format!(
                "GATE frames_ready={} need={}",
                self.stable_frames_ready,
                self.warmup_needed_frames()
            ));
            return Ok(self.last_score);
        }

        let p = tail_p_speech(&probs);
        self.last_score = p;
        #[cfg(test)]
        self.dbg.set_prob(p);
        Ok(p)
    }
}
