use tract_rs::{
    State,
    prelude::{
        tract_ndarray::{Array1, Array2, s},
        *,
    },
};

use crate::audio::{clog, roll_into_ring, run_preprocessor, validate_audio_range_11};
use crate::session::VadSessionCommon;
#[cfg(test)]
use crate::session::SessionDebug;
use crate::Res;
use anyhow::ensure;

pub(crate) struct VadSessionPulsed {
    preprocessor_model: Runnable,
    encoder_state: State,
    decoder_model: Runnable,
    audio_buffer: Vec<f32>,
    last_score: f32,
    pub(crate) encoder_frame_buffer: Array2<f32>,
    pub(crate) pulse_delay: usize,
    pulse_frames: usize,
    frame_size: usize,
    decoded_emissions: usize,
    pub(crate) stable_frames_ready: usize,
    #[cfg(test)]
    pub(crate) dbg: SessionDebug,
}

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
    fn pulse_frames(&self) -> usize {
        self.pulse_frames
    }
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
            self.dbg.set_logits_and_prob(_logits, _p1);
        }
        self.decoded_emissions = self.decoded_emissions.saturating_add(1);
    }

    // For the pulsed encoder, the emitted frames already reflect the internal delay.
    // We should not subtract `pulse_delay` again when selecting the block to append
    // to the decoder's window. Use the most recent `pulse_frames` directly.
    fn build_dec_input(&mut self, enc_all: &Array2<f32>) -> Res<Vec<Value>> {
        let frames = enc_all.shape()[1];
        if frames < self.pulse_frames() {
            let val: Value = self.encoder_frame_buffer().clone().try_into()?;
            return Ok(vec![val]);
        }
        let start = frames - self.pulse_frames();
        let block = enc_all.slice(s![.., start..]).to_owned();
        #[cfg(test)]
        self.dbg.set_enc_block(&block);
        let dec_in = self.slide_window_append(&block, self.pulse_frames())?;
        #[cfg(test)]
        {
            let win = self.encoder_frame_buffer().clone();
            self.dbg.set_encoder_window(&win);
        }
        Ok(dec_in)
    }
}

impl VadSessionPulsed {
    pub(crate) fn new(
        preprocessor: &Runnable,
        encoder: &Runnable,
        decoder: &Runnable,
        pulse_frames: usize,
        frame_size: usize,
        pulse_delay: usize,
    ) -> Res<VadSessionPulsed> {
        // Pool ~100ms (10 frames) at the decoder for stability
        let n_encoder_frames_to_aggregate_over = 10;
        // Keep the pulsed step <= decoder pool to avoid window overruns
        ensure!(
            pulse_frames <= n_encoder_frames_to_aggregate_over,
            "pulse_frames must be <= {} (decoder pool)",
            n_encoder_frames_to_aggregate_over
        );
        Ok(Self {
            preprocessor_model: preprocessor.clone(),
            encoder_state: encoder.spawn_state()?,
            decoder_model: decoder.clone(),
            audio_buffer: vec![0.0; pulse_delay * frame_size],
            last_score: f32::NAN,
            encoder_frame_buffer: Array2::<f32>::zeros((128, n_encoder_frames_to_aggregate_over)),
            pulse_delay,
            pulse_frames: pulse_frames.max(1),
            frame_size,
            decoded_emissions: 0,
            stable_frames_ready: 0,
            #[cfg(test)]
            dbg: SessionDebug::default(),
        })
    }

    pub(crate) fn predict_speech_presence(&mut self, raw_audio_data: Vec<f32>) -> Res<f32> {
        roll_into_ring(&mut self.audio_buffer, &raw_audio_data);
        clog(&format!(
            "PULSED ENC run with new audio data qte: {}",
            raw_audio_data.len()
        ));

        let pre_full = self.preprocess_full()?;
        let pre_slice = self.select_pre_slice(&pre_full);
        let sliced_value: Value = pre_slice.try_into()?;

        let enc_result = self.encoder_state.run(vec![sliced_value])?;
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

    pub(crate) fn step_samples(&self) -> usize {
        self.pulse_frames * self.frame_size
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

    fn slide_encoder_window(&mut self, enc_result: Vec<Value>) -> Res<Vec<Value>> {
        clog("SLIDE start");
        let enc_all = enc_result[0]
            .view::<f32>()?
            .into_dimensionality::<tract_rs::prelude::tract_ndarray::Ix2>()?
            .to_owned();
        clog(&format!(
            "ENC features={}, frames={}",
            enc_all.shape()[0],
            enc_all.shape()[1]
        ));
        #[cfg(test)]
        self.dbg.set_enc_out(&enc_all);
        self.build_dec_input(&enc_all)
    }
}
