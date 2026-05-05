use std::{collections::HashMap, fmt::Display, io::Cursor};

use base64::{Engine as _, engine::general_purpose};
use lazy_static::lazy_static;
use serde::Serialize;
use tract_nnef::{
    prelude::*,
    tract_ndarray::{Axis, Ix3},
};
use wasm_bindgen::prelude::*;

extern crate web_sys;

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

lazy_static! {
    pub static ref KEYPOINT_EDGES_TO_COLORS: HashMap<(Keypoint, Keypoint), image::Rgb<u8>> = {
        let mut m = HashMap::new();
        m.insert(
            (Keypoint::Nose, Keypoint::LeftEye),
            image::Rgb([255u8, 0u8, 0u8]),
        );
        m.insert(
            (Keypoint::Nose, Keypoint::RightEye),
            image::Rgb([0u8, 255u8, 0u8]),
        );
        m.insert(
            (Keypoint::LeftEye, Keypoint::LeftEar),
            image::Rgb([0u8, 0u8, 255u8]),
        );
        m.insert(
            (Keypoint::RightEye, Keypoint::RightEar),
            image::Rgb([255u8, 255u8, 0u8]),
        );
        m.insert(
            (Keypoint::Nose, Keypoint::LeftShoulder),
            image::Rgb([0u8, 255u8, 255u8]),
        );
        m.insert(
            (Keypoint::Nose, Keypoint::RightShoulder),
            image::Rgb([255u8, 0u8, 255u8]),
        );
        m.insert(
            (Keypoint::LeftShoulder, Keypoint::RightShoulder),
            image::Rgb([192u8, 192u8, 192u8]),
        );
        m.insert(
            (Keypoint::LeftShoulder, Keypoint::LeftElbow),
            image::Rgb([128u8, 0u8, 0u8]),
        );
        m.insert(
            (Keypoint::RightShoulder, Keypoint::RightElbow),
            image::Rgb([0u8, 128u8, 0u8]),
        );
        m.insert(
            (Keypoint::LeftElbow, Keypoint::LeftWrist),
            image::Rgb([0u8, 0u8, 128u8]),
        );
        m.insert(
            (Keypoint::RightElbow, Keypoint::RightWrist),
            image::Rgb([128u8, 128u8, 0u8]),
        );
        m.insert(
            (Keypoint::LeftShoulder, Keypoint::LeftHip),
            image::Rgb([0u8, 128u8, 128u8]),
        );
        m.insert(
            (Keypoint::RightShoulder, Keypoint::RightHip),
            image::Rgb([128u8, 0u8, 128u8]),
        );
        m.insert(
            (Keypoint::LeftHip, Keypoint::RightHip),
            image::Rgb([64u8, 64u8, 64u8]),
        );
        m.insert(
            (Keypoint::LeftHip, Keypoint::LeftKnee),
            image::Rgb([192u8, 64u8, 64u8]),
        );
        m.insert(
            (Keypoint::RightHip, Keypoint::RightKnee),
            image::Rgb([64u8, 192u8, 64u8]),
        );
        m.insert(
            (Keypoint::LeftKnee, Keypoint::LeftAnkle),
            image::Rgb([64u8, 64u8, 192u8]),
        );
        m.insert(
            (Keypoint::RightKnee, Keypoint::RightAnkle),
            image::Rgb([255u8, 165u8, 0u8]),
        );
        m
    };
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

    pub fn from(keypoint_index: usize, tensor: &[f32], scale_x: f32, scale_y: f32) -> Self {
        KeypointPrediction {
            x: tensor[0] * scale_x,
            y: tensor[1] * scale_y,
            visibility: tensor[2],
            keypoint: Keypoint::from(keypoint_index),
        }
    }
}

impl HumanPosePrediction {
    pub fn from_slice(tensor: &[f32], scale_x: f32, scale_y: f32) -> Self {
        // data pointer[0 - 3] - bounding box's x, y, w, and h values.
        // data pointer[4] - confidence.
        // data pointer[5 - 55] - keypoints(17x3)
        let keypoints = (0..17)
            .map(|i| {
                KeypointPrediction::from(i, &tensor[5 + i * 3..5 + i * 3 + 3], scale_x, scale_y)
            })
            .collect::<Vec<KeypointPrediction>>();
        HumanPosePrediction {
            x: tensor[0] * scale_x,
            y: tensor[1] * scale_y,
            w: tensor[2] * scale_x,
            h: tensor[3] * scale_y,
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
        // maybe need to permute width and height
        // img shape received in ultralytics/nn/tasks.py(155)predict(): torch.Size([1, 3, 640, 576])
        // so (1, 3, height, width)
        let image_tensor =
            tract_ndarray::Array4::from_shape_fn((1, 3, IMG_HEIGHT, IMG_WIDTH), |(_, c, y, x)| {
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
        let result = self.model.run(tvec!(image_tensor.into()))?;
        web_sys::console::log_1(&"model prediction done".into());

        let raw_preds = result[0]
            .to_array_view::<f32>()?
            .into_dimensionality::<Ix3>()?
            .permuted_axes((0, 2, 1))
            .to_owned();

        // filter out low confidence predictions
        web_sys::console::log_1(&format!("raw_preds shape: {:?}", raw_preds.shape()).into());
        let confidences_ok = raw_preds
            .axis_iter(Axis(1))
            .enumerate()
            .map(|(ix, row)| (ix, row.get((0, 4)).unwrap_or(&0.0).to_owned()))
            .filter(|(_, confidence)| *confidence > self.min_detection_threshold)
            .collect::<Vec<_>>();

        let topk_indices = if confidences_ok.len() > self.n_tops {
            let mut indices: Vec<usize> = (0..confidences_ok.len()).collect();
            indices.select_nth_unstable_by(self.n_tops, |&a, &b| {
                confidences_ok[b]
                    .1
                    .partial_cmp(&confidences_ok[a].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            indices[..self.n_tops].to_vec()
        } else {
            (0..confidences_ok.len()).collect()
        };
        // select top-k predictions
        web_sys::console::log_1(
            &format!(
                "found {} predictions with indices: {:?} and scores: {:?}",
                confidences_ok.len(),
                &topk_indices,
                &topk_indices
                    .iter()
                    .map(|&i| confidences_ok[i])
                    .collect::<Vec<_>>()
            )
            .into(),
        );

        let scale_x = image.width() as f32 / resized.width() as f32;
        let scale_y = image.height() as f32 / resized.height() as f32;
        web_sys::console::log_1(&format!("scale_x: {}, scale_y: {}", scale_x, scale_y).into());

        // parse predictions
        let preds = topk_indices
            .into_iter()
            .map(|i| {
                raw_preds
                    .index_axis(Axis(1), confidences_ok[i].0)
                    .to_owned()
            })
            .map(|row| HumanPosePrediction::from_slice(row.as_slice().unwrap(), scale_x, scale_y))
            .collect::<Vec<HumanPosePrediction>>();
        web_sys::console::log_1(&"predictions parsed".into());
        Ok(preds)
    }

    fn draw_image_with_keypoints_internals(
        &self,
        image_base64: &str,
        keypoint_visibility_threshold: f32,
    ) -> TractResult<String> {
        let predictions = self.predict_keypoints_internal(image_base64)?;
        let blob = general_purpose::STANDARD.decode(
            image_base64
                .split_once(";")
                .unwrap()
                .1
                .split_once(",")
                .unwrap()
                .1,
        )?;
        let mut image = image::load_from_memory(&blob)?.to_rgb8();
        let label_scale = (image.width().min(image.height()) as f32 / 640.0).max(1.0);
        // set image to grey colors for better visibility of keypoints
        for pixel in image.pixels_mut() {
            let grey =
                (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;
            *pixel = image::Rgb([grey, grey, grey]);
        }
        for pred in predictions.iter() {
            for keypoint in pred.keypoints.iter() {
                if keypoint.visibility > keypoint_visibility_threshold {
                    imageproc::drawing::draw_filled_circle_mut(
                        &mut image,
                        (keypoint.x as i32, keypoint.y as i32),
                        5 * label_scale as i32,
                        image::Rgb([255u8, 0u8, 0u8]),
                    );
                }
                // draw edges to other keypoints
                KEYPOINT_EDGES_TO_COLORS
                    .iter()
                    .for_each(|((kp1, kp2), color)| {
                        if keypoint.keypoint == *kp1 {
                            if let Some(kp2_pred) = pred
                                .keypoints
                                .iter()
                                .find(|kp_pred| kp_pred.keypoint == *kp2)
                            {
                                if kp2_pred.visibility > keypoint_visibility_threshold {
                                    // draw line between keypoint and kp2_pred with thickness
                                    // depending on label_scale similar to:
                                    for i in 0..(5 * label_scale as i32).max(1) {
                                        let i = i as f32;
                                        imageproc::drawing::draw_line_segment_mut(
                                            &mut image,
                                            (keypoint.x - i, keypoint.y - i),
                                            (kp2_pred.x - i, kp2_pred.y - i),
                                            *color,
                                        );
                                    }
                                }
                            }
                        }
                    });
            }
            // draw bounding box
            web_sys::console::log_1(
                &format!(
                    "pre denorm bbox x:{} y:{} w:{} h:{}",
                    pred.x, pred.y, pred.w, pred.h
                )
                .into(),
            );
            let h = (pred.h) as u32;
            let w = (pred.w) as u32;
            let x1 = (pred.x - pred.w / 2.0) as u32;
            let y1 = (pred.y - pred.h / 2.0) as u32;
            web_sys::console::log_1(&format!("bbox x:{} y:{} w:{} h:{}", x1, y1, w, h).into());
            // draw hollow rectangle with multiple pixel border
            // (to ensure visibility on higres res images):
            for i in 0..(3 * label_scale as i32).max(1) {
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut image,
                    imageproc::rect::Rect::at((x1 as i32) - i, (y1 as i32) - i)
                        .of_size(w + (2 * i) as u32, h + (2 * i) as u32),
                    image::Rgb([0u8, 255u8, 0u8]),
                );
            }
        }

        let mut buf = Cursor::new(Vec::new());
        image.write_to(&mut buf, image::ImageFormat::Png)?;
        let bytes = buf.into_inner();
        let encoded_image = format!(
            "data:image/png;base64,{}",
            general_purpose::STANDARD.encode(bytes)
        );
        Ok(encoded_image)
    }

    pub fn predict_keypoints(&self, image_base64: &str) -> Result<JsValue, JsError> {
        let prediction_res = self.predict_keypoints_internal(image_base64);
        let pred = prediction_res.map_err(|err| JsError::new(format!("{:?}", err).as_str()))?;
        Ok(serde_wasm_bindgen::to_value(&pred)?)
    }

    pub fn draw_image_with_keypoints(
        &self,
        image_base64: &str,
        keypoint_visibility_threshold: f32,
    ) -> Result<JsValue, JsError> {
        let res_image_b64 =
            self.draw_image_with_keypoints_internals(image_base64, keypoint_visibility_threshold);
        let image_b64 = res_image_b64.map_err(|err| JsError::new(format!("{:?}", err).as_str()))?;
        Ok(serde_wasm_bindgen::to_value(&image_b64)?)
    }

    pub fn load(min_detection_threshold: f32, n_tops: usize) -> YoloPoser {
        web_sys::console::log_1(&"try loading".into());
        let result = YoloPoser::load_internal(min_detection_threshold, n_tops)
            .map_err(|err| JsError::new(format!("{:?}", err).as_str()))
            .expect("unable to load");
        result
    }
}
