# torch_to_nnef

Any Pytorch Model to NNEF file format

> warning ! This project is still in alpha and might break/change api quickly

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
    base_path=export_path,
    input_names=["input"],
    output_names=["output"],
)
```

## Limitation

Torch Model need to be serializable to torch.jit (fancy python dict routing
or others might prevent proper tracing of it).
