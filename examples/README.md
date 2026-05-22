# Examples

Working code that accompanies the [torch-to-nnef
documentation](https://sonos.github.io/torch-to-nnef/latest/). Each subdir is
self-contained: clone the repo, `cd examples/<name>/`, follow the per-example
README.

## Examples by tutorial

| Example | What it does | Tutorial | Live demo |
| --- | --- | --- | --- |
| [`getting_started_py/`](getting_started_py/) | First export to NNEF (Python) | [1. Getting started](https://sonos.github.io/torch-to-nnef/latest/tutos/1_getting_started/) |: |
| [`getting_started_rs/`](getting_started_rs/) | First inference with tract (Rust) | [1. Getting started](https://sonos.github.io/torch-to-nnef/latest/tutos/1_getting_started/) |: |
| [`multi_io_py/`](multi_io_py/) | Multi-input / multi-output export (ALBERT) | [3. Multiple inputs/outputs](https://sonos.github.io/torch-to-nnef/latest/tutos/3_multi_inputs_outputs/) |: |
| [`dynamic_axes/`](dynamic_axes/) | Dynamic-axes export patterns | [4. Dynamic axes](https://sonos.github.io/torch-to-nnef/latest/tutos/4_dynamic_axes/) |: |
| [`quantization_py/`](quantization_py/) | int8 / mixed-precision quantization | [6. Quantization](https://sonos.github.io/torch-to-nnef/latest/tutos/6_quantization/) |: |
| [`t2n_extra_custom_op/`](t2n_extra_custom_op/) | External custom op via `t2n_extra` (handler auto-load) | [8. Custom operators](https://sonos.github.io/torch-to-nnef/latest/tutos/8_custom_operator/) |: |
| [`mamba/`](mamba/) | Mamba streaming inference (two designs); uses `t2n_extra::ssm_scan{,_y}` handlers and Tract-specific fragments | [8. Custom operators](https://sonos.github.io/torch-to-nnef/latest/tutos/8_custom_operator/), [5. LLM](https://sonos.github.io/torch-to-nnef/latest/tutos/5_llm/) |: |
| [`imageclass-wasm/`](imageclass-wasm/) | Image classifier compiled to wasm | [1. Getting started](https://sonos.github.io/torch-to-nnef/latest/tutos/1_getting_started/) | [demo](https://sonos.github.io/torch-to-nnef/latest/html/demo_image_classifier.html) |
| [`yolo/`](yolo/) | YOLO pose estimation in wasm | [1. Getting started](https://sonos.github.io/torch-to-nnef/latest/tutos/1_getting_started/) | [demo](https://sonos.github.io/torch-to-nnef/latest/html/demo_pose_estimation.html) |
| [`llm_wasm/`](llm_wasm/) | Small LLM running in wasm | [5. Large Language Models](https://sonos.github.io/torch-to-nnef/latest/tutos/5_llm/) | [demo](https://sonos.github.io/torch-to-nnef/latest/html/demo_poem_generator.html) |
| [`vad/FSMN-wasm/`](vad/FSMN-wasm/) | FSMN VAD in wasm | [4. Dynamic axes](https://sonos.github.io/torch-to-nnef/latest/tutos/4_dynamic_axes/) | [demo](https://sonos.github.io/torch-to-nnef/latest/html/demo_vad.html) |
| [`vad/silero-jit/`](vad/silero-jit/) | Silero-VAD JIT artifact straight to NNEF (`harden_jit_for_export`) | [12. JIT-only models](https://sonos.github.io/torch-to-nnef/latest/tutos/12_jit_only_models/) |: |
| [`nemo_asr/`](nemo_asr/) | NeMo ASR (Rust runtime + Python bindings) | [10. NeMo ASR export & eval](https://sonos.github.io/torch-to-nnef/latest/tutos/10_nemo/) |: |
| [`image_gen/`](image_gen/) | SD 1.5 + Flux-Schnell + Sana (DiT, mini configs); SDXL / SD3 placeholders | (exploration, no tutorial yet) |: |
| [`tts/pocket_tts/`](tts/pocket_tts/) | Kyutai Pocket-TTS end-to-end (FlowLM + Mimi) via tract; pulse-mode streaming + fp16 export, RTFx 9.88× on M4 Pro | (no tutorial yet) |: |
| [`speech_enhancement/deepfilternet/`](speech_enhancement/deepfilternet/) | DeepFilterNet 3 per-frame streaming model (matches grazder's ONNX deploy shape) via [grazder's pure-torch port](https://github.com/grazder/DeepFilterNet); 12 state tensors threaded by the caller, tract round-trip passes end-to-end | (no tutorial yet) |: |
| [`speech_enhancement/dpdfnet/`](speech_enhancement/dpdfnet/) | [CEVA's DPDFNet 2](https://github.com/ceva-ip/DPDFNet) (DPRNN-augmented DFN2) at 16 kHz, in-graph STFT/iFFT, **single NNEF artifact**, plus a minimal Rust `wav-cleaner` binary (tract-nnef) that takes WAV in / clean WAV out: 4.44x real-time end-to-end on M4 Pro | (no tutorial yet) |: |

## Bootstrap helpers

Shared shell scripts at the top of `examples/` that the per-example READMEs
call:

| Script | Purpose |
| --- | --- |
| [`bootstrap-uv.sh`](bootstrap-uv.sh) | Install `uv` + Python 3.11, create the example's `.venv` |
| [`bootstrap-rust.sh`](bootstrap-rust.sh) | Install `rustup` + a stable toolchain |
| [`bootstrap-wasm-pack.sh`](bootstrap-wasm-pack.sh) | Install `wasm-pack` for the wasm examples |
| [`clean.sh`](clean.sh) | Remove caches / venvs / build artifacts under `examples/` |

## Live demo HTML

The wasm demos are deployed alongside the docs at
<https://sonos.github.io/torch-to-nnef/latest/html/>. The HTML source lives
in [`docs/html/`](../docs/html/); the `examples/` subdir produces the wasm
module, `docs/html/` hosts the page that loads it.
