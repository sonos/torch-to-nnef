use ndarray::{Array3, s};
use tract_rs::{State, prelude::*};

use crate::Res;
use crate::audio::{clog, roll_into_ring, run_preprocessor, validate_audio_range_11};
use crate::session::{WARMUP_NEEDED_FRAMES, tail_p_speech};
use crate::{Ndarray as _, Tract as _};

pub(crate) struct VadSessionPulsed {
    preprocessor_model: Runnable,
    encoder_state: State,
    audio_buffer: Vec<f32>,
    last_score: f32,
    pulse_frames: usize,
    frame_size: usize,
    stable_frames_ready: usize,
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
        })
    }

    pub(crate) fn warmup_ready(&self) -> bool {
        self.stable_frames_ready >= WARMUP_NEEDED_FRAMES
    }

    pub(crate) fn step_samples(&self) -> usize {
        self.pulse_frames * self.frame_size
    }

    pub(crate) fn predict_speech_presence(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        let n_new = raw_audio_data.len();
        roll_into_ring(&mut self.audio_buffer, &raw_audio_data);
        self.stable_frames_ready = self
            .stable_frames_ready
            .saturating_add(n_new / self.frame_size);

        validate_audio_range_11(&self.audio_buffer)?;
        let feats = run_preprocessor(&self.preprocessor_model, &self.audio_buffer)?;

        // Feed the newest `pulse_frames` LFR'd feature frames into the pulsed
        // encoder. Its internal delay line maintains the left context (FSMN
        // lorder=20) across steps.
        let t = feats.shape()[1];
        let start = t.saturating_sub(self.pulse_frames);
        let slice_arr = feats.slice(s![.., start.., ..]).to_owned();

        clog("PULSED ENC run");
        let slice_t: Tensor = slice_arr.tract()?;
        let probs_out = self.encoder_state.run(vec![slice_t])?;
        let probs_view = probs_out[0].ndarray::<f32>()?;
        let probs: Array3<f32> = probs_view.into_dimensionality()?.to_owned();

        if !self.warmup_ready() {
            return Ok(self.last_score);
        }
        let p = tail_p_speech(&probs);
        self.last_score = p;
        Ok(p)
    }
}
