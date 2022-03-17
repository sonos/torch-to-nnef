# Torch to NNEF
[![dev workflow](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml/badge.svg?branch=feat%2Fci-and-packaging)](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml)

Any Pytorch Model to NNEF file format

> warning ! This project is still in alpha and might break/change api quickly

## Goals & Scope

We intend to export any model formulated with vanilla Torch whatever tensor type
(handling quantized model).

Minimum dependencies in production python package generated (to allow easy
integration in other project).

## Features

Allow to export any pytorch model by providing input and model.
```python3
from pathlib import Path
import torch
from torch import nn
from torch_to_nnef.export import export_model_to_nnef

test_input = torch.rand(1, 10, 100)
model = nn.Sequential(nn.Conv1d(10, 20, 3))

export_model_to_nnef(
    model=model,
    args=test_input,
    file_path_export=Path("mybeautifulmodel.nnef"),
    input_names=["input"],
    output_names=["output"],
)
```

## Limitation

Torch Model need to be serializable to torch.jit (fancy python dict routing
or others might prevent proper tracing of it).

## Design choice

We build on top of `torch.jit.trace` Graph representation (API exposed since `1.0`).

Compared to the 2 other possible Graph API for pytorch we chose it because:
- `torch.fx`: is limited in the shape and type inference it provides. It seems more
  aimed at AST graph manipulation than export. Moreover this API was introduced very
  recently as stable (torch==1.10.0).
- `torch.jit.script`: offer a more flexible graph repr than trace and do not freeze
  Logical structure into the path taken during sample execution (contrary to trace),
  but it seems some tensor_size are not extracted as well.
