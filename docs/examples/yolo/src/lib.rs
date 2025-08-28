use std::fmt::Display;

use base64::{Engine as _, engine::general_purpose};
use serde::Serialize;
use tract_nnef::{
    prelude::*,
    tract_ndarray::{Axis, Ix3},
};
use wasm_bindgen::prelude::*;

extern crate web_sys;

#[derive(Serialize)]
pub enum Keypoint {
    Nose,
    LeftEye,
    RightEye,
    LeftEar,
    RightEar,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
}

impl Display for Keypoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Keypoint::Nose => "Nose",
            Keypoint::LeftEye => "LeftEye",
            Keypoint::RightEye => "RightEye",
            Keypoint::LeftEar => "LeftEar",
            Keypoint::RightEar => "RightEar",
            Keypoint::LeftShoulder => "LeftShoulder",
            Keypoint::RightShoulder => "RightShoulder",
            Keypoint::LeftElbow => "LeftElbow",
            Keypoint::RightElbow => "RightElbow",
            Keypoint::LeftWrist => "LeftWrist",
            Keypoint::RightWrist => "RightWrist",
            Keypoint::LeftHip => "LeftHip",
            Keypoint::RightHip => "RightHip",
            Keypoint::LeftKnee => "LeftKnee",
            Keypoint::RightKnee => "RightKnee",
            Keypoint::LeftAnkle => "LeftAnkle",
            Keypoint::RightAnkle => "RightAnkle",
        };
        write!(f, "{s}")
    }
}

impl From<String> for Keypoint {
    fn from(s: String) -> Self {
        match s.as_str() {
            "Nose" => Keypoint::Nose,
            "LeftEye" => Keypoint::LeftEye,
            "RightEye" => Keypoint::RightEye,
            "LeftEar" => Keypoint::LeftEar,
            "RightEar" => Keypoint::RightEar,
            "LeftShoulder" => Keypoint::LeftShoulder,
            "RightShoulder" => Keypoint::RightShoulder,
            "LeftElbow" => Keypoint::LeftElbow,
            "RightElbow" => Keypoint::RightElbow,
            "LeftWrist" => Keypoint::LeftWrist,
            "RightWrist" => Keypoint::RightWrist,
            "LeftHip" => Keypoint::LeftHip,
            "RightHip" => Keypoint::RightHip,
            "LeftKnee" => Keypoint::LeftKnee,
            "RightKnee" => Keypoint::RightKnee,
            "LeftAnkle" => Keypoint::LeftAnkle,
            "RightAnkle" => Keypoint::RightAnkle,
            _ => panic!("Unknown keypoint string: {}", s),
        }
    }
}

impl From<usize> for Keypoint {
    fn from(index: usize) -> Self {
        match index {
            0 => Keypoint::Nose,
            1 => Keypoint::LeftEye,
            2 => Keypoint::RightEye,
            3 => Keypoint::LeftEar,
            4 => Keypoint::RightEar,
            5 => Keypoint::LeftShoulder,
            6 => Keypoint::RightShoulder,
            7 => Keypoint::LeftElbow,
            8 => Keypoint::RightElbow,
            9 => Keypoint::LeftWrist,
            10 => Keypoint::RightWrist,
            11 => Keypoint::LeftHip,
            12 => Keypoint::RightHip,
            13 => Keypoint::LeftKnee,
            14 => Keypoint::RightKnee,
            15 => Keypoint::LeftAnkle,
            16 => Keypoint::RightAnkle,
            _ => panic!("Unknown keypoint index: {}", index),
        }
    }
}

#[derive(Serialize)]
pub struct HumanPosePrediction {
    keypoints: Vec<KeypointPrediction>,
    // bounding box elements {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    // }
    confidence: f32,
}

#[derive(Serialize)]
pub struct KeypointPrediction {
    keypoint: Keypoint,
    x: f32,
    y: f32,
    visibility: f32,
}

impl KeypointPrediction {
    pub fn new(keypoint: Keypoint, x: f32, y: f32, visibility: f32) -> KeypointPrediction {
        KeypointPrediction {
            keypoint,
            x,
            y,
            visibility,
        }
    }

    pub fn from(keypoint_index: usize, tensor: &[f32]) -> Self {
        KeypointPrediction {
            x: tensor[0],
            y: tensor[1],
            visibility: tensor[2],
            keypoint: Keypoint::from(keypoint_index),
        }
    }
}

impl HumanPosePrediction {
    pub fn from_slice(tensor: &[f32]) -> Self {
        // data pointer[0 - 3] - bounding box's x, y, w, and h values.
        // data pointer[4] - confidence.
        // data pointer[5 - 55] - keypoints(17x3)
        let keypoints = (0..17)
            .map(|i| KeypointPrediction::from(i, &tensor[5 + i * 3..5 + i * 3 + 3]))
            .collect::<Vec<KeypointPrediction>>();
        HumanPosePrediction {
            x: tensor[0],
            y: tensor[1],
            w: tensor[2],
            h: tensor[3],
            confidence: tensor[4],
            keypoints,
        }
    }
}

static IMG_WIDTH: usize = 640;
static IMG_HEIGHT: usize = 640;
#[wasm_bindgen]
struct YoloPoser {
    model: TypedRunnableModel<TypedModel>,
    min_detection_threshold: f32,
    n_tops: usize,
}

#[wasm_bindgen]
impl YoloPoser {
    fn load_internal(min_detection_threshold: f32, n_tops: usize) -> TractResult<YoloPoser> {
        let model_bytes = include_bytes!("../yolo11n-pose.nnef.tgz");
        let mut read = std::io::Cursor::new(model_bytes);
        // let classes_bytes = include_bytes!("../classes.txt");
        // let classes_txt = String::from_utf8_lossy(classes_bytes);
        // let classes: Vec<String> = classes_txt.split("\n").map(|s| s.to_string()).collect();
        let model = tract_nnef::nnef()
            .with_tract_core()
            .model_for_read(&mut read)?
            // optimize the model
            .into_optimized()?
            // make the model runnable and fix its inputs and outputs
            .into_runnable()?;
        web_sys::console::log_1(&"model loaded/optimized".into());
        Ok(YoloPoser {
            model,
            min_detection_threshold,
            n_tops,
        })
    }

    fn predict_keypoints_internal(
        &self,
        image_base64: &str,
    ) -> TractResult<Vec<HumanPosePrediction>> {
        web_sys::console::log_1(&"start predict class".into());

        let blob = general_purpose::STANDARD.decode(
            image_base64
                .split_once(";")
                .unwrap()
                .1
                .split_once(",")
                .unwrap()
                .1,
        )?;
        web_sys::console::log_1(&"loaded blob".into());
        // open image, resize it and make a Tensor out of it
        let image = image::load_from_memory(&blob)?;
        web_sys::console::log_1(&"image path loaded".into());
        // scale to model input dimension
        let resized = image
            .resize(
                IMG_WIDTH as u32,
                IMG_HEIGHT as u32,
                ::image::imageops::FilterType::Triangle,
            )
            .to_rgb8();
        web_sys::console::log_1(&"image resized".into());
        let image =
            tract_ndarray::Array4::from_shape_fn((1, 3, IMG_WIDTH, IMG_HEIGHT), |(_, c, x, y)| {
                let val = if resized.width() <= x as u32 || resized.height() <= y as u32 {
                    0.0
                } else {
                    resized[(x as _, y as _)][c] as f32
                };
                val / 255.0
            })
            .into_tensor();
        web_sys::console::log_1(&"normalized image".into());
        // run the model on the input
        let result = self.model.run(tvec!(image.into()))?;
        web_sys::console::log_1(&"model prediction done".into());

        let raw_preds = result[0]
            .to_array_view::<f32>()?
            .into_dimensionality::<Ix3>()?
            .permuted_axes((0, 2, 1))
            .to_owned();

        // filter out low confidence predictions
        let confidences_ok = raw_preds
            .axis_iter(Axis(1))
            .map(|row| row.get((0, 4)).unwrap_or(&0.0).to_owned())
            .filter(|confidence| *confidence > self.min_detection_threshold)
            .collect::<Vec<_>>();

        // select top-k predictions
        web_sys::console::log_1(&format!("found {} predictions", confidences_ok.len()).into());
        let topk_indices = if confidences_ok.len() > self.n_tops {
            let mut indices: Vec<usize> = (0..confidences_ok.len()).collect();
            indices.select_nth_unstable_by(self.n_tops, |&a, &b| {
                confidences_ok[b]
                    .partial_cmp(&confidences_ok[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            indices[..self.n_tops].to_vec()
        } else {
            (0..confidences_ok.len()).collect()
        };

        let preds = topk_indices
            .into_iter()
            .map(|i| raw_preds.index_axis(Axis(1), i).to_owned())
            .map(|row| HumanPosePrediction::from_slice(row.as_slice().unwrap()))
            .collect::<Vec<HumanPosePrediction>>();
        web_sys::console::log_1(&"predictions parsed".into());
        Ok(preds)
    }

    pub fn predict_keypoints(&self, image_base64: &str) -> Result<JsValue, JsError> {
        let prediction_res = self.predict_keypoints_internal(image_base64);
        let pred = prediction_res.map_err(|err| JsError::new(format!("{:?}", err).as_str()))?;
        Ok(serde_wasm_bindgen::to_value(&pred)?)
    }

    pub fn load(min_detection_threshold: f32, n_tops: usize) -> YoloPoser {
        web_sys::console::log_1(&"try loading".into());
        let result = YoloPoser::load_internal(min_detection_threshold, n_tops)
            .map_err(|err| JsError::new(format!("{:?}", err).as_str()))
            .expect("unable to load");
        result
    }
}
