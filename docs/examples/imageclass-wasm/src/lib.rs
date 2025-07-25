use base64::{Engine as _, engine::general_purpose};
use serde::Serialize;
use tract_nnef::prelude::*;
use wasm_bindgen::prelude::*;

extern crate web_sys;

#[derive(Serialize)]
pub struct Prediction {
    score: f32,
    class_id: i32,
    label: String,
}

#[wasm_bindgen]
struct ImageClassifier {
    model: TypedRunnableModel<TypedModel>,
    classes: Vec<String>,
}

#[wasm_bindgen]
impl ImageClassifier {
    fn load_internal() -> TractResult<ImageClassifier> {
        let model_bytes = include_bytes!("../efficientnet_b0_batchable.nnef.tgz");
        let mut read = std::io::Cursor::new(model_bytes);
        let classes_bytes = include_bytes!("../classes.txt");
        let classes_txt = String::from_utf8_lossy(classes_bytes);
        let classes: Vec<String> = classes_txt.split("\n").map(|s| s.to_string()).collect();
        let model = tract_nnef::nnef()
            .with_tract_core()
            .model_for_read(&mut read)?
            // optimize the model
            .into_optimized()?
            // make the model runnable and fix its inputs and outputs
            .into_runnable()?;
        web_sys::console::log_1(&"model loaded/optimized".into());
        Ok(ImageClassifier { model, classes })
    }

    fn predict_class_internal(&self, image_base64: &str) -> TractResult<Prediction> {
        web_sys::console::log_1(&"start predict class".into());

        let blob = general_purpose::STANDARD.decode(
            &image_base64
                .split_once(";")
                .unwrap()
                .1
                .split_once(",")
                .unwrap()
                .1,
        )?;
        web_sys::console::log_1(&"loaded blob".into());
        // open image, resize it and make a Tensor out of it
        let image = image::load_from_memory(&blob)?.to_rgb8();
        web_sys::console::log_1(&"image path loaded".into());
        // scale to model input dimension
        let resized =
            image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
        web_sys::console::log_1(&"image resized".into());
        let image = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, x, y)| {
            let mean = [0.485, 0.456, 0.406][c];
            let std = [0.229, 0.224, 0.225][c];
            (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
        })
        .into_tensor();
        // run the model on the input
        let result = self.model.run(tvec!(image.into()))?;
        web_sys::console::log_1(&"model prediction done".into());

        // find and display the max value with its index
        let best = result[0]
            .to_array_view::<f32>()?
            .iter()
            .cloned()
            .zip(0..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .expect("missing result");
        Ok(Prediction {
            score: best.0,
            class_id: best.1,
            label: self.classes[best.1 as usize].clone(),
        })
    }

    pub fn predict_class(&self, image_base64: &str) -> Result<JsValue, JsError> {
        let prediction_res = self.predict_class_internal(image_base64);
        let pred = prediction_res.map_err(|err| JsError::new(format!("{:?}", err).as_str()))?;
        Ok(serde_wasm_bindgen::to_value(&pred)?)
    }
    pub fn load() -> ImageClassifier {
        web_sys::console::log_1(&"try loading".into());
        let result = ImageClassifier::load_internal()
            .map_err(|err| JsError::new(format!("{:?}", err).as_str()))
            .expect("unable to load");
        result
    }
}
