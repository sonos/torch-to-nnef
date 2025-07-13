# 4. Streaming dimensions

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-toolbox: How to export specify dynamic tensor inputs neural network
    2. :material-clock-time-one-outline: What is tract pulsification and why this is very powerfull

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 10 min to read this page

Numerous neural network act on dimensions aren't known at export time.
Batch size is a common example that is ideally selected at runtime according to user need.
Time dimension is another case were dimension may accumulate over a runtime session.
Also some neural network applied on image support varying resolutions.

In this tutorial we will see how to specify this dynamism inside `NNEF` at export, and
the special case of time dimension for stateful neural networks.

## Simple case: batch dimension only

If we think of our [getting_started](./1_getting_started.md) example earlier,
after export the model generated is having a fixed batch dimensions of 1 sample.
Let's fix this by declaring this dimension as dynamic at export time:

```python title="setting streaming dimensions correctly"
from pathlib import Path
from torch_to_nnef import export_model_to_nnef, TractNNEF

file_path_export = Path("vit_b_16_batchable.nnef.tgz")
export_model_to_nnef(
    model=my_image_model,
    args=input_data_sample,
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        version="0.21.13",
        check_io=True,
        # here we use the first input_names we define
        # and request the first dimension: 0 to have
        # the varying dimensions "B" (for batch)
        dynamic_axes={"input": {0: "B"}},
    ),
    input_names=["input"],
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
)
```
