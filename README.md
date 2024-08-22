<!-- markdownlint-disable-file MD001 MD013 MD014 MD024 -->
# Torch to NNEF

[![dev workflow](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml/badge.svg?branch=main)](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml)

Any PyTorch Model to NNEF file format.

> warning ! This project is still in it's early stage, if you encounter any bug please follow `Bug report` section instructions

## Goals & Scope

We intend to export any model formulated with vanilla Torch whatever tensor type
(handling quantized model).

Depending on selected NNEF inference target, adjustment are made to enable maximum support.

For example, `TractNNEF` target allow to use [tract](https://github.com/sonos/tract/) operators to express:

- recurrent layers (LSTM, GRU, ...)
- dynamic streamable input dimensions
- data type casting (since NNEF spec is too vague in this regard)

This package strives to have minimum dependencies (to allow easy integration in other project).

We aim to support PyTorch > 1.8.0 with tract last 3 major releases (>= 1.19.16 to date) over Linux and MacOS systems.

## Install

For now you can either use internal Snips Nexus repository:

```bash
$ pip install torch_to_nnef
$ # or
$ poetry add torch_to_nnef
```

Or reference this GitHub project via your preferred package manager.

## Features

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

## Limitation

Torch Model need to be serializable to `torch.jit`.

This applies for `nn.Module` with forward outputing complex object or None parameters which
will be filtered out (only torch.Tensor are supposed).

Only Quantization supported by tract is for now translated only with scheme `torch.per_tensor_affine` (*Static*).

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

## Bug report & Contributions

Please refer to [this page](./CONTRIBUTING.md)
