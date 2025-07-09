# 1. Getting Started

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-image: How to export an image model with `torch_to_nnef`
    2. :material-toolbox: The basic commands to check your model is correct within tract
    3. :fontawesome-solid-terminal: How to create a minimal rust binary that perform inference from the cli with tract

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] Basic rust knowledge
    - [ ] 15 min to read this page

## <span style="color:#6666aa">**:material-step-forward: Step 1.**</span> Select an image classifier and an image

Let's start by simply selecting a model to export from the well known [torchvision](https://docs.pytorch.org/vision/stable/index.html).

- Create a virtual env, install dependencies and assets:

```bash title="Setup"
mkdir t2n_getting_started
cd t2n_getting_started
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.0 torchvision==0.21.0 torch_to_nnef
wget https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg
touch export.py
```

Let write inside export python file: `export.py`

```python title="Get PyTorch model & input data"

import torch
from torchvision import models as vision_mdl
my_image_model = vision_mdl.vit_b_16(pretrained=True)

img = read_image("./Grace_Hopper.jpg")
input_data_sample = vision_mdl.ViT_B_16_Weights.IMAGENET1K_V1.transforms()(
    img.unsqueeze(0)
)

with torch.no_grad():
    best_index = my_image_model(input_data_sample).argmax(1).tolist()[0]
    print(
        "class id:",
        best_index,
        "label: ",
        vision_mdl.ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"][
            best_index
        ],
    )

```

Running the file:

```json title="output"
class id: 652 label:  military uniform
```

The class index predicted with PyTorch (`652`) need to be the aligned with tract prediction we will develop.

## <span style="color:#6666aa">**:material-step-forward: Step 2.**</span> Export to NNEF

Let's continue the `export.py` by calling the main export function from this package:

```python title="export API"
from pathlib import Path
from torch_to_nnef import export_model_to_nnef, TractNNEF

file_path_export = Path("vit_b_16.nnef.tgz")
export_model_to_nnef(
    # any nn.Module
    model=my_image_model,
    # list of model arguments (here simply an example of tensor image)
    args=input_data_sample,
    # filepath to dump NNEF archive
    file_path_export=file_path_export,
    # inference engine to target
    inference_target=TractNNEF(
        # tract version (to ensure compatible operators)
        version="0.21.13",
        # default False (tract binary will be installed on the machine on fly)
        # and correctness of output compared to PyTorch for the
        # provided model and input will be performed
        check_io=True,
    ),
    input_names=["input"],
    output_names=["output"],
    # create a debug bundle in case model export work
    # but NNEF fail in tract (either due to load error or precision mismatch)
    debug_bundle_path=Path("./debug.tgz"),
)
print(f"exported {file_path_export.absolute()}")
```

And that's it if we now run our little snippet (full code [here](https://github.com/sonos/torch-to-nnef/blob/feat/mkdocs/docs/examples/getting_started.py))

```bash
source.venv/bin/activate
python export.py
```

We should now observe the following output:

```console
.../site-packages/torch/__init__.py:2132: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert condition, message
aten::size replaced by constant traced value (follows NNEF spec).Keeping dynamism would require dynamic_axes specified.
exported ./vit_b_16.nnef.tgz
```

But wait there is 2 tracing warnings here:

- The first is inherent to tracing mechanism happening inside `torch_to_nnef` indeed behind the scene
    we use PyTorch [jit.trace](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html).
    It is only able to capture torch control flows, so all Python manipulations
    that does not happen in PyTorch internals is 'solidified' into a set of fixed values.
    This also happen if you export a PyTorch model to ONNX with their internal tool.

- The second is interesting, it highlights a loss of model expressiveness because we did not specify
    that one of the input dimension is in fact the batch size, a parameter that may vary. We will show how
    to solve that in the next tutorial. (spoiler: we use same API as ONNX export to inform `dynamic_axes`)

Finally last line indicates that the model has been correctly exported on disk at: `.../vit_b_16.nnef.tgz`.

## <span style="color:#6666aa">**:material-step-forward: Step 3.**</span> tract cli checks

We will now check with the tract cli that everything is working as expected.

Let's first display the help of the command line we downloaded when we checked io between tract and PyTorch (step 2.)

```bash title="Setup"
alias tract=$HOME/.cache/svc/tract/0.21.13/tract
tract --help
```

If you did skip this steps you can always download manually the cli from the [tract release page](https://github.com/sonos/tract/releases),
or run `cargo install tract` (which will compile it for you system).

This command line is pretty dense so we will only use part of it today.

Let's first load and dump our model properties:

```bash title="Dump model properties with tract"
tract ./vit_b_16.nnef.tgz --nnef-tract-core -O dump --allow-random-input --profile
```

Here a lot is happening:

- tract load the nnef registry relative to core operators
- it then load the model
- it declutter and optimize it  (thanks to the `-O`)
- the `--allow-random-input` avoid us to provide a concrete input example
- the `--profile` informs the cli that we want to observe the speed of it

Output in stdout is composed of following sections in order:

The graph of computation (after decluttering and optimization) with each operation speed:

```json title="Graph display"
  0.000 ms/i  0.0%  ┏ 0 Source input
                    ┃   ━━━ 1,3,224,224,F32
....
  0.000 ms/i  0.0%  ┣ 686 OptMatMulPack output_linear_output.pack_b
                    ┃   ━━━ 1,Opaque 🔍 DynPackedOpaqueFact { k: Val(768), mn: Val(1), packers: [PackedF32[1]@128+1] }
  0.046 ms/i  0.0%  ┣┻ 688 OptMatMul output_linear_output
                        ━━━ 1,1000,F32
```

This already tell us about how network is composed and which specialized operators kernels were select.
(In this display we are on an ARM CPU.)
Then we have the list of custom properties that have been exported by `torch_to_nnef`:

```json title="Exported properties"
* export_cmd: ,String docs/examples/getting_started.py
* export_date: ,String 2025-07-08 ...
* exported_py_class: ,String VisionTransformer
* hostname: ,String ...
* os: ,String ...arm64 Darwin
* py_version: ,String 3.12.9 ...
* torch_to_nnef_version: ,String 0.18.6
* torch_version: ,String 2.6.0
* tract_stage: ,String optimized
* tract_target_version: ,String 0.21.13
* transformers_version: ,String 4.49.0
* user: ,String tuto
```

These are metadata that are automatically set when exporting models.
This often come handy during debugging sessions.
You can set custom ones with the `specific_properties` parameter in `TractNNEF` init.

Finally the aggregated per operator kind performance is shown:

```json title="Performance per operator kind"
 * OptMatMul               74 nodes:  90.859 ms/i 65.6%
 * OptMatMulPack           97 nodes:  12.779 ms/i  9.2%
 * Softmax                 12 nodes:  10.345 ms/i  7.5%
...

```

With percentage of time spent (again aggregated per operator kind).

Finally you get the total time spent to run the network:

```json title="Total performance"
Entire network performance: 138.525 ms/i
```

!!! info

    This command only include time to run the inference (model load and optimization is not accounted).

!!! tip "GPU usage"

    If you have a recent Apple Silicon device try the same command adding `--metal` before the dump
    and observe the speed difference.

## <span style="color:#6666aa">**:material-step-forward:  Step 4.**</span> Making a minimal rust program

Ok, we have our model asset we confirmed it run well from tract cli, now let's integrate it in a rust program.

We will build it step by step, but note that the code is very similar to this [tract example](https://github.com/sonos/tract/tree/main/examples/nnef-dump-mobilenet-v2).

```bash title="minimal setup rust binary and download a dummy image"
cd ..
cargo init t2n_getting_started_tract
cd t2n_getting_started_tract
cargo add tract-core tract-nnef image
cp ../t2n_getting_started/vit_b_16.nnef.tgz ./
cp ../t2n_getting_started/Grace_Hopper.jpg ./
```

Let's now compile the project it should output hello world:

```bash title="check project compile"
cargo run --release
```

We should observe `Hello, world!`
Now let's write the core interesting parts in `src/main.rs`:

Add the [prelude](https://doc.rust-lang.org/reference/names/preludes.html) from tract_nnef

```rust
use tract_nnef::prelude::*;
```

and replace the type signature of the main (for simpicity of this example):

```rust
fn main() -> TractResult<()> {
    println!("Hello, world!");
    Ok(())
}
```

Inside the main we replace println with:

```rust
    let model = tract_nnef::nnef()
        .with_tract_core()
        .model_for_path("./vit_b_16.nnef.tgz")?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;
```

This code is responsible to load, declutter and optimize the model.
We can now prepare image to be ingested by the neural network:

```rust
    // open image, resize it and make a Tensor out of it
    let image = image::open("Grace_Hopper.jpg")?.to_rgb8();
    // scale to model input dimension
    let resized = image::imageops::resize(
        &image,
        224,
        224,
        ::image::imageops::FilterType::Triangle
    );
    // normalization step
    let image = tract_ndarray::Array4::from_shape_fn(
        (1, 3, 224, 224),
        |(_, c, x, y)| {
            let mean = [0.485, 0.456, 0.406][c];
            let std = [0.229, 0.224, 0.225][c];
            (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
        }
    )
    .into_tensor();

```

This tensor is now ready to be run with tract:

```rust
    // run the model on the input
    let result = model.run(tvec!(image.into()))?;
```

Let's now get the index of classified class for the image and print it:

```rust
    // find and display the max value with its index
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(0..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    println!("result: {best:?}");
```

That's it our code is complete, (your code should now [look like this](https://github.com/sonos/torch-to-nnef/tree/feat/mkdocs/docs/examples/getting_started_tract))

You can now rebuild and run the code with cargo:

```bash title="compile & run completed project"
cargo run --release
```

!!! success end "Congratulation"

    Bravo! you made it !
    Your first export with `torch_to_nnef` is now done and
    you ran a successful standalone tract based inference with it.
