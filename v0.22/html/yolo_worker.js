import init, { YoloPoser } from './yolo.js';

var yolo_poser = null;
const keypoint_visibility_threshold = 0.3;
const human_detection_threshold = 0.3;
const n_humans_detectable = 1;

onmessage = (e) => {
    let msg = e.data;
    console.log("YoloWorker: Message received from main script", msg);
    if (msg.kind == "initYolo") {
        init().then(() => {
            console.log("YoloWorker: start wasm");
            try {
                postMessage({
                    kind: "yoloStatus",
                    value: "loading",
                })
                yolo_poser = YoloPoser.load(human_detection_threshold, n_humans_detectable);
                postMessage({
                    kind: "yoloStatus",
                    value: "ready",
                })
            } catch (error) {
                postMessage({
                    kind: "yoloStatus",
                    value: "fail",
                    message: error
                })
            }
            console.log("YoloWorker: inited Yolo");
        })
    } else if (msg.kind == "drawKeypoints" && yolo_poser !== null) {
        let startTime = performance.now();
        let taggedImageContent = yolo_poser.draw_image_with_keypoints(
            msg.fileTextContent,
            msg.keypoint_visibility_threshold
        );
        let endTime = performance.now();
        console.log("YoloWorker: predicted keypoints and drew on image");
        postMessage({
            kind: "keypointsDrawn",
            value: taggedImageContent,
            duration: endTime - startTime
        });
    } else {
        console.log("YoloWorker: Un-expected request in worker:", e, msg);
    }
};
