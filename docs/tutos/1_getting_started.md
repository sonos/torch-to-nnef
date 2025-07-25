# 1. Getting Started

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-image: How to export an image model with `torch_to_nnef`
    2. :material-toolbox: The basic commands to check your model is correct within tract
    3. :fontawesome-brands-python: How to create a minimal Python program that perform inference with tract
    4. :fontawesome-brands-rust: *(Bonus)* How to create a minimal rust binary that perform inference from the cli with tract

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] Basic rust knowledge (for the Bonus)
    - [ ] 15 min to read this page

## <span style="color:#6666aa">**:material-step-forward: Step 1.**</span> :material-image: Select an image classifier and an image

Let's start by simply selecting a model to export from the well known [torchvision](https://docs.pytorch.org/vision/stable/index.html).

- Create a virtual env, install dependencies and assets:

```bash title="Setup"
mkdir getting_started_py
cd getting_started_py
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch==2.7.0 \
    torchvision==0.22.0 \
    torch_to_nnef
wget https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg
touch export.py
```

Let write inside export python file: `export.py`
to get PyTorch model & input data and perform inference
with the given image.

```python title="export.py (part 1)"

import torch
from torchvision import models as vision_mdl
from torchvision.io import read_image

my_image_model = vision_mdl.vit_b_16(pretrained=True) # (1)!

img = read_image("./Grace_Hopper.jpg")
classification_task = vision_mdl.ViT_B_16_Weights.IMAGENET1K_V1 # (2)!
input_data_sample = classification_task.transforms()(
    img.unsqueeze(0)
)

with torch.no_grad():
    predicted_index = my_image_model(
        input_data_sample
    ).argmax(1).tolist()[0]
    print(
        "class id:",
        predicted_index,
        "label: ",
        classification_task.meta["categories"][
            predicted_index
        ],
    )

```

1. Selected model is [documented here](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html)
2. The classification task is [documented here](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.EfficientNet_B0_Weights)

Running the file:

```json title="output"
class id: 652 label:  military uniform
```

The class index predicted with PyTorch (`652`) need to be the aligned with tract prediction we will develop.

## <span style="color:#6666aa">**:material-step-forward: Step 2.**</span> Export to NNEF

Let's continue the `export.py` by calling the main export function from this package:

```python title="export.py (part 2)"
from pathlib import Path
from torch_to_nnef import export_model_to_nnef, TractNNEF

file_path_export = Path("vit_b_16.nnef.tgz")
export_model_to_nnef( # (1)!
    # any nn.Module
    model=my_image_model,
    # list of model arguments
    # (here simply an example of tensor image)
    args=input_data_sample,
    # filepath to dump NNEF archive
    file_path_export=file_path_export,
    # inference engine to target
    inference_target=TractNNEF( # (2)!
        # tract version (to ensure compatible operators)
        version="0.21.13",
        # default False
        # (tract cli will be installed on the machine on fly)
        # and correctness of output compared to PyTorch for the
        # provided model and input will be performed
        check_io=True,
    ),
    input_names=["input"],
    output_names=["output"],
    # create a debug bundle in case model export work
    # but NNEF fail in tract
    # (either due to load error or precision mismatch)
    debug_bundle_path=Path("./debug.tgz"),
)
print(f"exported {file_path_export.absolute()}")
```

1. Full function documentation available [here](/reference/torch_to_nnef/export/#torch_to_nnef.export.export_model_to_nnef)
2. Full Class documentation available [here](/reference/torch_to_nnef/inference_target/tract/#torch_to_nnef.inference_target.tract.TractNNEF)

And that's it if we now run our little snippet (full code [here](https://github.com/sonos/torch-to-nnef/blob/feat/mkdocs/docs/examples/getting_started_py/export.py))

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
    to solve that in the [next tutorial](./4_dynamic_axes.md). (spoiler: we use same API as ONNX export to inform `dynamic_axes`)

Finally last line indicates that the model has been correctly exported on disk at: `.../vit_b_16.nnef.tgz`.

## <span style="color:#6666aa">**:material-step-forward: Step 3.**</span> :material-toolbox: tract cli checks

We will now check with the tract cli that everything is working as expected.

Let's first display the help of the command line we downloaded when we checked io between tract and PyTorch (in step 2.)

```bash title="Setup"
alias tract=$HOME/.cache/svc/tract/0.21.13/tract
tract --help
```

If you did skip this steps you can always download manually the cli from the [tract release page](https://github.com/sonos/tract/releases),
or run `cargo install tract` (which will compile it for your system).

This command line is pretty dense so we will only use part of it today.

Let's first load and dump a profile of our model:

```bash title="Dump model properties with tract"
tract ./vit_b_16.nnef.tgz \
    --nnef-tract-core \
    -O \
    dump \
    --allow-random-input \
    --profile
```

Here a lot is happening:

- tract loads the NNEF registry relative to core operators
- its then load the model
- it declutters and optimize it  (thanks to the `-O`)
- the `--allow-random-input` avoid us to provide a concrete input example
- the `--profile` informs the command-line that we want to observe the speed of it

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
(this display is from an ARM CPU)
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

With percentage of time spent (again aggregated per operator kind) and you get the total time spent to run the network:

```json title="Total performance"
Entire network performance: 138.525 ms/i
```

On classical networks, matrix multiplication operations should dominate the compute time.

!!! info

    This command only display time to run the inference (model load and optimization is not accounted).

!!! tip "GPU usage"

    If you have a recent Apple Silicon device try the same command adding `--metal` before the `dump`
    and observe the speed difference.

## <span style="color:#6666aa">**:material-step-forward:  Step 5.**</span> :material-language-python: tract inference with Python

We just created a great NNEF model, and it has been checked during export to get same output for same input
between PyTorch and tract (thanks to the `check_io=True` option). That said you may now wish to
interact with it to perform a fully fledged evaluation of the model (to ensure this new inference engine
do not get imprecise results on some specific samples).

For this purpose we need to install a new package in our activated `venv`:

```bash title="add tract python package"
pip install "tract<0.22,>=0.21"
```

Let's now create a new python file called `run.py`:

Let's read our example image again with torch vision and transform it
in `numpy` feature matrix, this part is specific to the image classifaction,
and could be done with any tool you wish (this is not `tract` or `torch_to_nnef` related).

```python title="run.py (part 1)"
import tract
import numpy as np
from torchvision import models as vision_mdl
from torchvision.io import read_image

img = read_image("./Grace_Hopper.jpg")
classification_task = vision_mdl.ViT_B_16_Weights.IMAGENET1K_V1
input_data_sample = classification_task.transforms()(
    img.unsqueeze(0)
).numpy()
```

Now we can load the `NNEF` model with tract, declutter and optimize it:

```python title="run.py (part 2)"
model = (
    tract.nnef() #(1)!
    .with_tract_core()
    .model_for_path("./vit_b_16.nnef.tgz")
    .into_optimized()
    .into_runnable()
)
```

1. documentation [available here](https://sonos.github.io/tract/dev/nnef/)

Finally we can run the inference for the provided input and extract predicted result:

```python title="run.py (part 3)"
result = model.run([input_data_sample])
confidences = result[0].to_numpy()
prediced_index = np.argmax(confidences)
print(
    "class id:",
    predicted_index,
    "label: ",
    classification_task.meta["categories"][
        predicted_index
    ],
)
```

And that's it, we can now run our little snippet (full code [here](https://github.com/sonos/torch-to-nnef/blob/feat/mkdocs/docs/examples/getting_started_py/run.py)).

!!! success end "Congratulation"

    Your first export with `torch_to_nnef` is now done and
    you ran a successful standalone tract based inference with it.
    This is sufficent if you intend to use python only.

## <span style="color:#6666aa">**:material-step-forward:  Step 6.**</span> :fontawesome-brands-rust: Making a minimal rust program

Ok, we have our model asset we confirmed it run well from tract cli and Python, now let's integrate it in a rust program.

We will build it step by step, but note that the code is very similar to this [tract example](https://github.com/sonos/tract/tree/main/examples/nnef-dump-mobilenet-v2).

```bash title="minimal setup rust binary and download a dummy image"
cd ..
cargo init getting_started_rs
cd getting_started_rs
cargo add tract-core tract-nnef image
cp ../getting_started_py/vit_b_16.nnef.tgz ./
cp ../getting_started_py/Grace_Hopper.jpg ./
```

Now compile the project:

```bash title="check project compile"
cargo run --release
```

We should observe `Hello, world!` in stdout.
Let's write the core interesting parts in `src/main.rs`:

Add the [prelude](https://doc.rust-lang.org/reference/names/preludes.html) from tract_nnef

```rust title="main.rs (part 1)"
use tract_nnef::prelude::*;
```

Replace the type signature of the main (for simplicity of this example):

```rust title="main.rs (part 2)"
fn main() -> TractResult<()> {
    println!("Hello, world!");
    Ok(())
}
```

Inside the main replace *println* with:

```rust title="main.rs (part 3)"
    let model = tract_nnef::nnef()
        .with_tract_core()
        .model_for_path("./vit_b_16.nnef.tgz")?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;
```

This code is responsible to load, declutter and optimize the model.
Prepare image to be ingested by the neural network:

```rust title="main.rs (part 4)"
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

Notice that tract use [ndarray](https://docs.rs/ndarray/latest/ndarray/) to manipulate tensors with it's user facing API.

This tensor is now ready to be run with our tract model:

```rust title="main.rs (part 5)"
    // run the model on the input
    let result = model.run(tvec!(image.into()))?;
```

Let's now get the index of classified class for the image and print it:

```rust title="main.rs (part 6)"
    // find and display the max value with its index
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(0..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    println!("result: {best:?}");
```

That's it our code is complete, (your code should now [look like this](https://github.com/sonos/torch-to-nnef/tree/feat/mkdocs/docs/examples/getting_started_rs))

You can now rebuild and run the code with cargo:

```bash title="compile & run completed project"
cargo run --release
```

This should display to you

```json
result: Some((9.439479, 652))
```

!!! success end "Congratulation"

    :tada: you made it !
    You first exported the network with `torch_to_nnef` and
    ran a successful standalone rust cli command with tract based inference in it.

## <span style="color:#6666aa">**:material-step-forward:  Live Demo**</span> :fontawesome-brands-rust: Image classifier

 Using the knowledge you acquired during this tutorial and a bit of extra for [WASM in rust](https://rustwasm.github.io/book/introduction.html)
 we demo a small [`Efficient NET B0`](https://arxiv.org/pdf/1905.11946) neural network running in your browser (smaller than [ViT](https://arxiv.org/pdf/2010.11929) to ensure fast download of the asset for you - 22Mo for the model).
We let this model predict from image class from the [ImageNET 1K challenge](https://www.image-net.org/update-mar-11-2021.php).

Since this model fully run on your browser there is no server needed beyond serving the initial
asset, this is private by design (You can send whatever photo), no data is collected,
if you are in doubt just turn off your network and try the playground (without reloading).

### Image classifier based on tract running in browser with WASM

<div class="grid cards">
    <div class="card">
        <label for="avatar">Select a picture:</label>
        <input type="file" id="img" name="img" accept="image/png, image/jpeg" />
    </div>
    <div id="image-preview" class="card"></div>
    <script type="module">
import init, { ImageClassifier } from '/js/imageclass_wasm.js';
const image_preview = document.getElementById('image-preview');
let img_classifier = null;
function file_predict(file) {
    if (!file || !file.type.startsWith("image/")) {
        return;
    }
    image_preview.innerHTML = "";
    const img = document.createElement("img");
    img.classList.add("obj");
    img.file = file;
    image_preview.appendChild(img);
    const reader = new FileReader();
    reader.onload = (e) => {
        let fileTextContent = e.target.result;
        img.src = fileTextContent;
        console.log("requested file process" + file.name);
        let startTime = performance.now();
        let res = img_classifier.predict_class(fileTextContent);
        let endTime = performance.now();
        let timeDiff = endTime - startTime; //in ms
        console.log(res);
        console.log(res.label);
        const p = document.createElement("p");
        p.innerHTML = "predicted: '" + res.label + "'' - id:" + res.class_id + " (score='" + res.score.toFixed(2) + "'')<br/> image resized & predicted in: " + timeDiff + "ms";
        image_preview.appendChild(p);
    };
    reader.readAsDataURL(file);
}
init().then(() => {
    img_classifier = ImageClassifier.load();
    console.log("inited wasm");
    document.getElementById('img').addEventListener('change', function(e) {
        const file = e.target.files[0];
        file_predict(file);
    });
    fetch("/img/cat.jpg")
        .then(response => response.blob())
        .then(myBlob => {
            file_predict(myBlob);
        })
})
    </script>
</div>

!!! note
    Performance are descent, but little to no effort was made to make tract WASM efficient (no SIMD wasm, no WebGPU kernels),
    this demo is for demonstration purpose.

Curious to read the code behind it ? Just look at our [example directory here](https://github.com/sonos/torch-to-nnef/tree/main/docs/examples/imageclass-wasm) and this [raw page content](https://github.com/sonos/torch-to-nnef/blob/main/docs/tutos/1_getting_started.md).
