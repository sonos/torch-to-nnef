<!-- markdownlint-disable-file MD001 MD013 MD014 MD024 -->
# ![Torch to NNEF](./docs/img/torch_to_nnef.png)

[![dev workflow](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml/badge.svg?branch=main)](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml) ![python](https://img.shields.io/badge/python-%3E=3.9-green)

Any PyTorch Model or Tensor to NNEF file format.

> [!IMPORTANT]
> This project is still in it's early stage, if you encounter any bug please follow [Bug report](#bug-report)  section instructions

## Goals & Scope

We intend to export any model formulated with vanilla Torch whatever tensor type
(handling quantized model).

Depending on selected NNEF inference target, adjustment are made to enable maximum support.

For example, `TractNNEF` unlock [tract](https://github.com/sonos/tract/) operators and specificities to express:

- recurrent layers (LSTM, GRU, ...)
- dynamic streamable input dimensions
- data type casting (since NNEF spec is too vague in this regard)

This package strives to have minimum dependencies (to allow easy integration in other project).

We aim to support PyTorch > 1.8.0 with tract last 3 major releases (>= 1.19.16 to date) over Linux and MacOS systems.

## Install

For now you can either use internal SONOS artifactory repository:
`https://user:pass@redacted.com/artifactory/api/pypi/pypi-local/simple`
Then use:

```bash
$ pip install torch_to_nnef
$ # or
$ poetry add torch_to_nnef
```

Or reference this GitHub project via your preferred package manager.

## Features

### Python Export

#### Model Export

Export any PyTorch model by providing input and model in a Python script:

```python
import logging
from pathlib import Path

import torch
from torch import nn
from torch_to_nnef import export_model_to_nnef, TractNNEF

test_input = torch.rand(1, 10, 100)
model = nn.Sequential(nn.Conv1d(10, 20, 3))

export_model_to_nnef(
    model=model, # nn.Module
    args=test_input, # list of model arguments
    file_path_export=Path("mybeautifulmodel.nnef.tgz"), # target NNEF filepath
    inference_target=TractNNEF( # inference engine to target
        version="0.21.5", # tract version (to ensure compatible operators)
        check_io=True, # default False need tract installed on machine
        dynamic_axes={"input": {2: "S"}}, # follow onnx export convention with additional constraint
        # that named dimension need to be single letter symbol (due to tract spec)
    ),
    input_names=["input"],
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
)
# More parameters exists,
# you can look at function documentation
# for more informations about each with:
# help(export_model_to_nnef)
```

`export_model_to_nnef` is the main API of this library.

One of it's core features is ability to specify an `inference_target` with spefic options:

- `TractNNEF` is the engine developed by SONOS (see [here](https://github.com/sonos/tract/))
- `KhronosNNEF` is the standard specification as thought by Khronos group (see [here](https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html))

#### Specific Tensors Export

Export any PyTorch Tensor or *InterenceTarget* compatible QTensor in a Python script:

Either directly from disk:

```python
# exemple extracted from torch_to_nnef/peft/cli.py
from pathlib import Path
import re
from torch_to_nnef import export_tensors_from_disk_to_nnef


PATTERN_LORA = "|".join([p for p in [
    f".*.lora_A.default.weight$",
    f".*.lora_B.default.weight$",
]])

def filter_key(key):
    return bool(re.match(PATTERN_LORA, key))

def fn_check_found_tensors(to_export):
    if len(to_export) == 0:
        raise ValueError(
            f"no tensors found in provided file with pattern: {PATTERN_LORA}"
        )
    return True


name_to_tensors = export_tensors_from_disk_to_nnef(
    store_filepath=Path("/my-little-model-file.pt"), #can be .pth, .bin, .safetensors ...
    output_dir=Path("my-nnef-tensor-dump-dir"),
    filter_key=filter_key,
    fn_check_found_tensors=fn_check_found_tensors,
)

```

Or from an already loaded model:

```python
from pathlib import Path
from torch import nn
from torch_to_nnef import export_tensors_from_disk_to_nnef
model = nn.Sequential(nn.Linear(10, 20))

output_dir = Path("/tmp/my-nnef-tensor-dump-dir")
output_dir.mkdir()
export_tensors_to_nnef(
    {
      "linear_0": model[0].weight
    },
    output_dir=output_dir,
)

```

### CLI's export

Export any LLM from HuggingFace transformer library, via cli:

```bash
# need pip install torch_to_nnef[llm_tract]
export_llm_to_tract \
  -s apple/OpenELM-1_1B-Instruct \ # from HuggingFace model hub
  -e ~/model-zoo-data/openelm_f16_q4_0 \
  -f16 \
  -c "min_max_q4_0"
```

There are additional options (with --help), to export from local directory,
load custom compressor libraries, target specific tract version, etc.

Export **PEFT** weights directly:

```bash
export_peft_to_nnef \
    --read-filepath /my-little-model-file.pt \ #can be .pth, .bin, .safetensors ...
    -o /tmp/my-dir
```

By default export **LoRA** weights, if you wish to apply it on different methods look
at additional options (with --help), the core functionality behind this CLI is simple
pattern matching so most of PEFT weight names capturable with regex should work (DoRA, ...).

## Limitation

Torch Model need to be serializable to `torch.jit` for model export feature.

This applies for `nn.Module` with forward outputing complex object or None parameters which
will be filtered out (only torch.Tensor are supported).

Only Quantization supported by tract compatible at export is PyTorch with scheme `torch.per_tensor_affine` (*Static*),
or `Q4_0` with `QTensorTractScaleOnly` (see applied example [here](./torch_to_nnef/llm_tract/compress.py))

## Design choice

We build on top of `torch.jit.trace` Graph representation (API exposed since `1.0`).

Compared to the 2 other possible Graph API for PyTorch we chose it because:

- `torch.fx`: is limited in the shape and type inference it provides. It seems more
  aimed at AST graph manipulation than export. Moreover this API was introduced very
  recently as stable (torch==1.10.0).

- `torch.jit.script`: offer a more flexible graph repr than trace and do not freeze
  Logical structure into the path taken during sample execution (contrary to trace),
  but it seems some tensor_size are not extracted as well.

We may consider also using the new [torch dynamo](https://pytorch.org/docs/stable/onnx_dynamo.html) approach soon.

## Advanced usage

### Control NNEF export fragment for specific nn.Module

In case you want control specific `torch.nn.Module` expansion to NNEF you can
register a new `torch_to_nnef.op.custom_extractors.ModuleInfoExtractor` by
sub-classing it and defining it's `MODULE_CLASS` attribute.

In such scenario you will need to write your own graph expansion logic in
`convert_to_nnef` as follows:

```python
from torch_to_nnef.op.custom_extractors import ModuleInfoExtractor

class MyCustomHandler(ModuleInfoExtractor):
    MODULE_CLASS = MyModuleToCustomConvert

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        **kwargs
    ):
        # here your custom logic to implement NNEF module subgraph
        # you can take inspiration from `torch_to_nnef.op.primitive`
        # or aready written custom extractors such as
        # `torch_to_nnef.op.custom_extractors.LSTMExtractor`
        pass
```

### Support custom compressor library for `export_llm_to_tract` CLI

In order build custom export compressor for llm cli you just need to specify a registry in a module loadable within your environment such as:

```python
# imagine for example this is in a module named experimental_compressors.registery
from torch_to_nnef.llm_tract.models.base import TorchToNNEFWrappedLLM

def quantize_weights_gptq_Q4_0(
    wrapped_model: TorchToNNEFWrappedLLM, **kwargs
):
  # your custom code that replace some torch.Tensor by child class of torch_to_nnef.qtensor.base.QTensor
  return wrapped_model

MY_CUSTOM_COMPRESSIOn_REGISTRY = {
    "gptq_q4_0": quantize_weights_gptq_Q4_0,
}
```

Once done you can directly call your registred function with:

```bash
# need pip install torch_to_nnef[llm_tract]
export_llm_to_tract \
  -s ... \
  -e ... \
  --compression-registry "experimental_compressors.registery.MY_CUSTOM_COMPRESSIOn_REGISTRY" \
  -c "gptq_q4_0"
```

## Bug report & Contributions <a id='bug-report'></a>

Please refer to [this page](./CONTRIBUTING.md)
