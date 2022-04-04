# Torch to NNEF
[![dev workflow](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml/badge.svg?branch=main)](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml)

Any Pytorch Model to NNEF file format

> warning ! This project is still in beta and might break/change api quickly

## Goals & Scope

We intend to export any model formulated with vanilla Torch whatever tensor type
(handling quantized model).

Minimum dependencies in production python package generated (to allow easy
integration in other project).

## Install

For now you can either use internal Snips Nexus repository:
```
$ pip install torch_to_nnef
$ # or
$ poetry add torch_to_nnef
```

or reference this github project via your prefered package manager.

## Features

Allow to export any pytorch model by providing input and model.
```python3
import logging
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
    compression_level: int = 0, # tar.gz compression level
    log_level=logging.WARN, # default being logging.INFO
    check_same_io_as_tract=True, # default False need tract installed on machine
    debug_bundle_path=Path("./debug.tgz"), # if end with tgz will be archived else folder will be created
    # debug_bundle_path is generated only if tract IO is not valid

    renaming_scheme="numeric", # name torch variable in a concise way for readability
    # other possible choice with "natural_verbose" is as close as possible
    # to nn.Module exported variable naming
    # this renaming_scheme is only useful is you intend to read generated
    # NNEF format else do not set it
)
```

As shown in API it is by default not relying on tract inference library but has
optin to do few extra-checks with it.

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

## Advanced usage

In case you want control specific `torch.nn.Module` expansion to NNEF you can
register a new `torch_to_nnef.op.custom_extractors.ModuleInfoExtractor` by
subclassing it and defining it's `MODULE_CLASS` attribute.

In such scenario you will need to write your own graph expansion logic in
`convert_to_nnef` as follows:

```python3

from torch_to_nnef.op.custom_extractors import ModuleInfoExtractor

class MyCustomHandler(ModuleInfoExtractor):
    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
    ):
        # here your custom logic to implement NNEF module subgraph
        # you can take inspiration from `torch_to_nnef.op.primitive`
        # or aready written custom extractors such as
        # `torch_to_nnef.op.custom_extractors.LSTMExtractor`
        pass
```

## Bug report & Contributions

Please refer to [this page](./CONTRIBUTING.md)
