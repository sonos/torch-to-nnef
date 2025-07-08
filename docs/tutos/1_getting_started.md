# 1. Getting Started

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-image: How to export an image model with `torch_to_nnef`
    2. :material-toolbox: The basic commands to check your model is correct within tract
    3. :fontawesome-solid-terminal: How to create a minimal rust binary that perform inference from the cli with tract

!!! example "Prerequisite"

    - [ ] Understanding of what is a neural network
    - [ ] PyTorch and Python basics
    - [ ] Basic rust knowledge
    - [ ] 10 min to read this page

## <span style="color:#2222aa"> :material-step-forward: Step 1.</span> Select an image Model

Let's start by simply selecting a model to export from the well known [torchvision](https://docs.pytorch.org/vision/stable/index.html).
Let write an export python file: `export.py`

```python
import torch
from torchvision import models as vision_mdl
my_image_model = vision_mdl.vit_b_16(pretrained=False)

input_data_sample = torch.rand(1, 3, 224, 224)
```

## <span style="color:#2222aa"> :material-step-forward: Step 2. </span> Export it

Let's now call the main export function from this package:

```python
from pathlib import Path
from torch_to_nnef import export_model_to_nnef, TractNNEF

file_path_export = Path("vit_b_16.nnef.tgz")
export_model_to_nnef(
    model=my_image_model, # any nn.Module
    args=input_data_sample, # list of model arguments (here simply an example of tensor image)
    file_path_export=file_path_export, # filepath to dump NNEF archive
    inference_target=TractNNEF( # inference engine to target
        version="0.21.13", # tract version (to ensure compatible operators)
        check_io=True, # default False (tract binary will be installed on the machine on fly)
    ),
    input_names=["input"],
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"), # create a debug bundle in case model export work
    # but NNEF fail in tract (either due to load error or precision mismatch)
)
print(f"exported {file_path_export.absolute()}")
```

And that's it if we now run our little snippet (full code [here](./examples/getting_started.py)) we should now observe the following output:

```console
.../site-packages/torch/__init__.py:2132: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert condition, message
aten::size replaced by constant traced value (follows NNEF spec).Keeping dynamism would require dynamic_axes specified.
exported .../vit_b_16.nnef.tgz
```

We first observe 2 internal tracing warning:

- The first is inherent to tracing mechanism happening inside `torch_to_nnef` indeed behind the scene
    we use the same tracing API from PyTorch [jit.trace](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html).
    It explains that only torch api control flow will be recorded, all Python manipulation
    that does not happen in PyTorch is 'solidified' into a set of fixed values.
    This is also happen if you export a PyTorch model to ONNX.

- The second is interesting, it highlights a loss of model expressiveness because we did not specify
    that one of the input dimension is in fact a variable that depends on batch size. We will show how
    to solve that in the next tutorial. (spoiler: we use same API as ONNX export to inform `dynamic_axes`)

Finally we observe that model has been correctly exported on disk at: `.../vit_b_16.nnef.tgz`.

## <span style="color:#2222aa"> :material-step-forward: Step 3. </span> Check it run from the tract cli

## <span style="color:#2222aa"> :material-step-forward:  Step 4. </span> Making a minimal rust program

!!! success end "Congratulation"

    Bravo! you made it !
    Your first export with `torch_to_nnef` is now done and
    you ran a successful standalone tract based inference with it.

## :material-archive-search: Archive composition '.nnef.tgz'

## :material-math-log: Display (a glimpse of the internals)
