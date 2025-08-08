use tract_nnef::{
    prelude::*,
    tract_ndarray::{Array1, ArrayView2, Axis},
};
use wasm_bindgen::prelude::*;

extern crate web_sys;

#[wasm_bindgen]
struct VadClassifier {
    encoder_model: TypedRunnableModel<TypedModel>,
    decoder_model: TypedRunnableModel<TypedModel>,
}

#[wasm_bindgen]
impl VadClassifier {
    fn load_internal() -> TractResult<VadClassifier> {
        let enc_model_bytes = include_bytes!("../vad_marblenet.encoder.nnef.tgz");
        let mut enc_read = std::io::Cursor::new(enc_model_bytes);
        let encoder_model = tract_nnef::nnef()
            .with_tract_core()
            .model_for_read(&mut enc_read)?
            .into_optimized()?
            .into_runnable()?;

        let dec_model_bytes = include_bytes!("../vad_marblenet.decoder.nnef.tgz");
        let mut dec_read = std::io::Cursor::new(dec_model_bytes);
        let decoder_model = tract_nnef::nnef()
            .with_tract_core()
            .model_for_read(&mut dec_read)?
            .into_optimized()?
            .into_runnable()?;
        web_sys::console::log_1(&"model loaded/optimized".into());
        Ok(VadClassifier {
            encoder_model,
            decoder_model,
        })
    }

    fn predict_speech_presence_internal(&self, raw_audio_data: Vec<f32>) -> TractResult<f32> {
        // web_sys::console::debug_1(&"start predict voice presence".into());

        // prep audio data
        let nd_audio_data = Array1::from_vec(raw_audio_data)
            .insert_axis(Axis(0))
            .into_tvalue();
        // run the model on the input
        let enc_result = self.encoder_model.run(tvec!(nd_audio_data))?;
        let dec_result = self.decoder_model.run(enc_result)?;
        // web_sys::console::debug_1(&"model prediction done".into());
        // find and display the max value with its index
        let score: ArrayView2<f32> = dec_result[0]
            .to_array_view::<f32>()?
            .into_dimensionality()?;
        Ok(*score.get((0, 0)).unwrap())
    }

    pub fn predict_speech_presence(
        &self,
        js_raw_audio_data: &web_sys::js_sys::Float32Array,
    ) -> Result<JsValue, JsError> {
        let prediction_res = self.predict_speech_presence_internal(js_raw_audio_data.to_vec());
        let pred = prediction_res.map_err(|err| JsError::new(format!("{:?}", err).as_str()))?;
        Ok(pred.into())
    }

    pub fn load() -> VadClassifier {
        web_sys::console::log_1(&"try loading".into());
        let result = VadClassifier::load_internal()
            .map_err(|err| JsError::new(format!("{:?}", err).as_str()))
            .expect("unable to load");
        result
    }
}
#[cfg(test)]
mod test {
    #[test]
    fn test_load_and_request() {
        todo!();
    }
}
