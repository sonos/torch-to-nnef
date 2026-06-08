# Live demos

Every model below was exported from PyTorch with `torch_to_nnef`, then runs **entirely in your browser**: no server, no upload, no API call. The inference is powered by [**tract**](https://github.com/sonos/tract/), the neural network engine developed openly by [SONOS](https://sonos.com), compiled to WebAssembly.

!!! tip "Same engine, everywhere"

    The exact NNEF archives driving these demos are the kind you produce by following the [tutorials](tutos/1_getting_started.md). The very same files run on server, desktop, mobile, and embedded targets through tract. WASM here is just one deployment of the portable artifact.

Click any card to launch its demo in a new tab (each one only downloads its model when you open it):

<div class="grid cards" markdown>

- :material-image:{ .lg .middle } __Image classifier__

    ---

    `EfficientNet-B0` labelling an image, fully client-side.

    [:octicons-arrow-right-24: Launch demo](html/demo_image_classifier.html){target=_blank}

- :material-run:{ .lg .middle } __Pose estimator__

    ---

    `YOLO` human pose estimation running on a still image.

    [:octicons-arrow-right-24: Launch demo](html/demo_pose_estimation.html){target=_blank}

- :material-microphone:{ .lg .middle } __Voice activity detection__

    ---

    `FSMN` VAD with dynamic, streamable input over live audio.

    [:octicons-arrow-right-24: Launch demo](html/demo_vad.html){target=_blank}

- :material-text:{ .lg .middle } __Poem generator__

    ---

    A small `LLM` generating text token by token, in-browser.

    [:octicons-arrow-right-24: Launch demo](html/demo_poem_generator.html){target=_blank}

</div>

!!! note "Performance disclaimer"

    These models are not trained by SONOS, so prediction quality is the responsibility of their original authors. The WASM builds are unoptimized (no SIMD WASM, no WebGPU kernels), so speed here understates what tract reaches on native targets. They exist to demonstrate portability, not peak performance.

Curious how they are built? Each demo's source lives under the project's [`examples/` directory](https://github.com/sonos/torch-to-nnef/tree/main/examples), and the step-by-step exports are covered in the [tutorials](tutos/1_getting_started.md). That directory holds many more than the WASM demos above (quantization, text-to-speech, NeMo ASR, Mamba, image generation, multi-input/output models, custom operators, and Rust integration samples), making it the best place to browse end-to-end, runnable usage.
