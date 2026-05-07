# YOLO11 pose estimation in wasm

[![GitHub](https://img.shields.io/badge/GitHub-ultralytics%2Fultralytics-181717?logo=github)](https://github.com/ultralytics/ultralytics) [![demo](https://img.shields.io/badge/live-demo-brightgreen)](https://sonos.github.io/torch-to-nnef/latest/html/demo_pose_estimation.html)

Exports `yolo11n-pose` (Ultralytics YOLO11, nano pose-estimation variant) to NNEF, then compiles a tract-backed Rust crate to WebAssembly for in-browser pose detection on a webcam feed.

## Run

```bash
cd examples/yolo
./run.sh
```

The `run.sh` script:
1. Sets up `.venv` + Rust toolchain via the bootstrap helpers
2. Runs `export.py` which downloads `yolo11n-pose.pt` via Ultralytics, monkey-patches their exporter to emit NNEF, and produces the archive
3. Builds the Rust crate to wasm with `wasm-pack`
4. Drops the wasm + JS glue into `docs/html/` for the live demo

`export.py` overrides `Exporter.export_nnef` on the Ultralytics side to route through `torch_to_nnef.export_model_to_nnef`. The NMS post-processing is wrapped via `NMSModel` and exported alongside the detection backbone.

Live demo: [https://sonos.github.io/torch-to-nnef/latest/html/demo_pose_estimation.html](https://sonos.github.io/torch-to-nnef/latest/html/demo_pose_estimation.html).
