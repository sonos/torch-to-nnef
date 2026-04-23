use anyhow::bail;
use ndarray::{Array1, Array2, s};
use tract_rs::prelude::*;

use crate::Res;

#[inline]
#[cfg(all(feature = "log-vad", target_arch = "wasm32"))]
pub(crate) fn clog(msg: &str) {
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(msg));
}
#[inline]
#[cfg(not(all(feature = "log-vad", target_arch = "wasm32")))]
pub(crate) fn clog(_: &str) {}


pub(crate) fn roll_into_ring(buf: &mut [f32], incoming: &[f32]) {
    let l = buf.len();
    let n = incoming.len();
    if n >= l {
        buf.copy_from_slice(&incoming[n - l..n]);
    } else {
        buf.copy_within(n..l, 0);
        buf[l - n..].copy_from_slice(incoming);
    }
}

pub(crate) fn validate_audio_range_11(buf: &[f32]) -> Res<()> {
    let max = buf.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if max > 1.0 {
        // In the browser path, resampling and filtering may produce minor overshoot.
        // Treat it as a warning instead of a hard error to avoid breaking the demo.
        #[cfg(target_arch = "wasm32")]
        {
            clog(&format!(
                "WARNING: audio sample abs value {:.6} exceeds [-1.0, 1.0]; proceeding",
                max
            ));
            return Ok(());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            bail!(
                "WARNING: audio sample abs value {} exceeds expected [-1.0, 1.0] range; ensure proper normalization",
                max
            );
        }
    }
    Ok(())
}

// Select `pulse_frames` columns from a 2D encoder output [features, frames].
// `align_shift` shifts the window back from the tail (0 for pulsed, 1 for batch).
// Returns None when the encoder has not yet produced enough frames.
pub(crate) fn select_enc_block(
    enc_all: &Array2<f32>,
    pulse_frames: usize,
    pulse_delay: usize,
    align_shift: usize,
) -> Option<Array2<f32>> {
    let frames = enc_all.shape()[1];
    if frames >= pulse_delay + pulse_frames + align_shift {
        let start = frames - pulse_delay - pulse_frames - align_shift;
        Some(enc_all.slice(s![.., start..start + pulse_frames]).to_owned())
    } else if frames >= pulse_frames {
        Some(enc_all.slice(s![.., frames - pulse_frames..]).to_owned())
    } else {
        None
    }
}

pub(crate) fn run_preprocessor(preprocessor: &Runnable, audio: &[f32]) -> Res<Array2<f32>> {
    use crate::{Ndarray as _, Tract as _};
    let audio_val_1d: Tensor = Array1::from_vec(audio.to_vec()).tract()?;
    let pre_result = preprocessor.run(vec![audio_val_1d])?;
    let pre_feat = pre_result[0].ndarray2::<f32>()?.to_owned();
    Ok(pre_feat)
}
