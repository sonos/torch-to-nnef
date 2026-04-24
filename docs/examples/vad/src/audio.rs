use anyhow::bail;
use ndarray::Array3;
use tract_rs::prelude::*;

use crate::Res;
use crate::{Ndarray as _, Tract as _};

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

/// Run the FSMN-VAD preprocessor on a fixed-length audio buffer.
///
/// Input: `audio` of length `PREPROCESSOR_INPUT_SAMPLES` (1 second at 16 kHz).
/// Output: `(1, T, lfr_m * n_mels)` = `(1, ~95, 400)` LFR'd + CMVN'd features.
pub(crate) fn run_preprocessor(preprocessor: &Runnable, audio: &[f32]) -> Res<Array3<f32>> {
    let arr = ndarray::Array2::from_shape_vec((1, audio.len()), audio.to_vec())?;
    let tensor: Tensor = arr.tract()?;
    let out = preprocessor.run(vec![tensor])?;
    let view = out[0].ndarray::<f32>()?;
    let arr3: Array3<f32> = view.into_dimensionality()?.to_owned();
    Ok(arr3)
}
