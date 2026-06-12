<!-- markdownlint-disable-file MD001 MD013 MD014 MD024 MD033 MD041 -->
# <picture><source media="(prefers-color-scheme: dark)" srcset="./docs/img/torch_to_nnef_black.png"><source media="(prefers-color-scheme: light)" srcset="./docs/img/torch_to_nnef.png"><img alt="Torch to NNEF" src="./docs/img/torch_to_nnef.png"></picture>

[![core CI](https://github.com/sonos/torch-to-nnef/actions/workflows/ci-core.yml/badge.svg?branch=main)](https://github.com/sonos/torch-to-nnef/actions/workflows/ci-core.yml) [![LLM CI](https://github.com/sonos/torch-to-nnef/actions/workflows/ci-llm.yml/badge.svg?branch=main)](https://github.com/sonos/torch-to-nnef/actions/workflows/ci-llm.yml) [![NeMo CI](https://github.com/sonos/torch-to-nnef/actions/workflows/ci-nemo.yml/badge.svg?branch=main)](https://github.com/sonos/torch-to-nnef/actions/workflows/ci-nemo.yml) ![python](https://img.shields.io/badge/python-%3E=3.9-green)[![documentation](https://img.shields.io/badge/torch_to_nnef-documentation-blue)]( https://sonos.github.io/torch-to-nnef/) ![MIT/Apache 2](https://img.shields.io/badge/license-MIT_OR_Apache_2.0-blue)

Export any PyTorch model to [NNEF](https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html), a simple and explicit neural network exchange format, in a way that is **auditable, debuggable, and stable across runtimes**.

`torch_to_nnef` traces a vanilla `nn.Module` (any internal tensor type, quantized models included) and produces a portable NNEF archive. Unlike the protobuf used by ONNX, the NNEF graph is human-readable text with weights kept aside as separate files, so you can open it and review exactly which ops got serialized. [tract](https://github.com/sonos/tract/), the inference engine developed openly by [SONOS](https://sonos.com), is the primary supported target: it gets extended operator coverage and an optional `check_io` step that compares tract outputs against PyTorch at export time.

> 🚀 **See it run:** [live browser demos](https://sonos.github.io/torch-to-nnef/latest/demos/) (image classifier, pose estimation, voice activity detection, LLM) exported with `torch_to_nnef` and running through tract compiled to WASM, no install required.

## Quickstart

```bash
pip install torch_to_nnef   # needs pip >= 23.2
```

```python
import torch
from torch_to_nnef import export_model_to_nnef, TractNNEF

model = torch.nn.Linear(8, 4).eval()
example_input = torch.randn(1, 8)

export_model_to_nnef(
    model=model,
    args=example_input,
    file_path_export="my_model.nnef.tgz",
    # version pins the tract opset to target; check_io downloads that tract
    # build once and asserts its output matches PyTorch on example_input.
    inference_target=TractNNEF(version="0.23.0", check_io=True),
    input_names=["input"],
    output_names=["output"],
)
```

This writes a self-contained `my_model.nnef.tgz`; with `check_io=True` the call also raises if tract and PyTorch outputs diverge, so reaching the end means the export is validated. The getting started tutorial (linked below) walks through running the archive in tract from Python, Rust, and the CLI.

<details>
<summary>What's inside <code>my_model.nnef.tgz</code></summary>

The archive keeps the graph and the weights apart:

    graph.nnef   weight.dat   bias.dat

and `graph.nnef` is plain text you can read end to end (metadata header elided):

    graph network(input) -> (output)
    {
        input = tract_core_external(shape = [1, 8], datum_type = 'f32');
        weight = variable<scalar>(label = 'weight', shape = [4, 8]);
        bias = variable<scalar>(label = 'bias', shape = [4]);
        bias_aligned_rank_expanded = unsqueeze(bias, axes = [0]);
        output = linear(input, weight, bias_aligned_rank_expanded);
    }

</details>

## What you can export

Any `nn.Module` whose `forward` takes and returns tensors can be exported out of the box, ordinary CNNs and transformers included. The tutorials below cover the cases that need more care:

- [Dynamic axes](https://sonos.github.io/torch-to-nnef/latest/tutos/4_dynamic_axes/) : variable batch and streamable dimensions
- [Large Language Models](https://sonos.github.io/torch-to-nnef/latest/tutos/5_llm/) and [quantized models](https://sonos.github.io/torch-to-nnef/latest/tutos/6_quantization/)
- [NeMo ASR models](https://sonos.github.io/torch-to-nnef/latest/tutos/10_nemo/)
- [Offloaded tensors](https://sonos.github.io/torch-to-nnef/latest/tutos/7_offloaded_tensor/), [tensor-only / PEFT exports](https://sonos.github.io/torch-to-nnef/latest/tutos/9_peft/), and [custom operators](https://sonos.github.io/torch-to-nnef/latest/tutos/8_custom_operator/)

## Compatibility

PyTorch >= 1.10.0 on Linux and macOS, against the last two major tract releases (>= 0.20.22). Supported Python versions track those that are [not end of life nor pre-release](https://devguide.python.org/versions/). Latest releases give the best operator coverage.

## Learn more

The full documentation has step by step tutorials (from a first image classifier to LLM export), the export API reference, and the list of supported operators:

- [Getting started](https://sonos.github.io/torch-to-nnef/latest/tutos/1_getting_started/) : export, check, and run a real model end to end
- [Live demos](https://sonos.github.io/torch-to-nnef/latest/demos/) : exported models running in your browser via tract WASM
- [Export API](https://sonos.github.io/torch-to-nnef/latest/export_api/)
- [Supported operators](https://sonos.github.io/torch-to-nnef/latest/contributing/supported_operators/)
- [Why NNEF?](https://sonos.github.io/torch-to-nnef/latest/why_nnef/)

## Contributing & support

Contributions are welcome. If you hit a bug, please follow the [Bug report](https://sonos.github.io/torch-to-nnef/latest/contributing/guidelines/) instructions so it can be reproduced quickly.

Licensed under MIT OR Apache-2.0.
