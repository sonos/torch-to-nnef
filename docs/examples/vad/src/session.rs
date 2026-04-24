use ndarray::{Array1, Array2, Array3};

use crate::SILENCE_PDF_IDX;

#[cfg(test)]
#[derive(Default, Clone)]
pub(crate) struct SessionDebug {
    pub(crate) last_pre_feat: Option<Array2<f32>>,
    pub(crate) last_pre_sliced: Option<Array2<f32>>,
    pub(crate) last_probs: Option<Array2<f32>>,
    pub(crate) last_logits: Option<Array1<f32>>,
    pub(crate) last_prob: Option<f32>,
}

#[cfg(test)]
impl SessionDebug {
    pub(crate) fn set_pre_feat(&mut self, a: &Array2<f32>) { self.last_pre_feat = Some(a.clone()); }
    pub(crate) fn set_pre_sliced(&mut self, a: &Array2<f32>) { self.last_pre_sliced = Some(a.clone()); }
    pub(crate) fn set_probs(&mut self, a: &Array2<f32>) { self.last_probs = Some(a.clone()); }
    pub(crate) fn set_prob(&mut self, p: f32) { self.last_prob = Some(p); }
}

pub(crate) trait VadSessionCommon {
    fn audio_buffer(&self) -> &[f32];
    fn audio_buffer_mut(&mut self) -> &mut [f32];
    fn get_last_score(&self) -> f32;
    fn set_last_score(&mut self, v: f32);
    fn stable_frames_ready(&self) -> usize;
    fn add_stable_frames(&mut self, n: usize);

    // FSMN has 4 stacked layers with lorder=20 each. The pulsed encoder's
    // delay-line state for each layer only carries 19 frames, so with 4 layers
    // it takes ~4 * 19 = 76 accumulated frames of *real audio* before every
    // layer's receptive field is filled (no more synthetic leading zeros in
    // the delay lines). We round up to 80 for safety. Below this, pulsed and
    // batch can still differ on the first few emitted frames.
    fn warmup_needed_frames(&self) -> usize { 80 }
    fn warmup_ready(&self) -> bool {
        self.stable_frames_ready() >= self.warmup_needed_frames()
    }
}

/// Extract p(speech) from the tail of an FSMN softmax output `(1, T, D)`.
/// `silence_pdf_ids = [0]` so `p(speech) = 1 - probs[0, T-1, 0]`.
pub(crate) fn tail_p_speech(probs: &Array3<f32>) -> f32 {
    let t = probs.shape()[1];
    if t == 0 {
        return 0.0;
    }
    let p_sil = probs[[0, t - 1, SILENCE_PDF_IDX]];
    (1.0 - p_sil).clamp(0.0, 1.0)
}
