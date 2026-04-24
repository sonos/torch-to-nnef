use ndarray::Array3;

use crate::SILENCE_PDF_IDX;

// FSMN has 4 stacked layers with lorder=20 each. The pulsed encoder's delay
// line for each layer only carries 19 frames, so with 4 layers it takes
// ~4 * 19 = 76 accumulated frames of real audio before every layer's receptive
// field is filled (no more synthetic leading zeros). We round up to 80.
pub(crate) const WARMUP_NEEDED_FRAMES: usize = 80;

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
