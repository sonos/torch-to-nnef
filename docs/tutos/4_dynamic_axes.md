# 4. Dynamic axes

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-toolbox: How to export specify dynamic tensor inputs neural network
    2. :material-clock-time-one-outline: What is tract pulsification and why this is very powerfull

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 15 min to read this page

Numerous neural network act on dimensions that aren't known quantity at export time.
Batch size is a common example that is ideally selected at runtime according to the
user need.
Time dimension is another case were dimension may accumulate over a runtime session,
and change between sessions.
Also some neural network applied on image support varying resolutions.

In this tutorial we will see how to specify this dynamism inside `NNEF` at export, and
the special case of time dimension for stateful neural networks.

## Simple case: batch dimension only

If we think of our [getting_started](./1_getting_started.md) example earlier,
after export: the model generated is having a fixed batch dimensions of 1 sample.
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

After running this script you should have a new asset: `vit_b_16_batchable.nnef.tgz`.
Looking at the `graph.nnef` there is 2 new obvious things:

- A new extension at the beginning of the file with the introduced symbol

```nnef
extension tract_symbol B;

```

- The input external introduce this symbol in the requested dimensions:

```nnef

input = tract_core_external(shape = [B, 3, 224, 224], datum_type = 'f32');
```

More subtly a lot of operations now doesn't assume shape is static but instead build from this variable shape:

```nnef
input_shape = tract_core_shape_of(input);
input_shape_sliced0 = slice(input_shape, axes = [0], begin = [0], end = [1], stride = [1]);
input_dim0 = squeeze(input_shape_sliced0, axes = [0]);
# ...
x_reshape0 = reshape(conv_proj__x__convolution0, shape = [input_dim0, 768, mul0]);

```

If you run tract with the exported model:

```bash
tract ./vit_b_16_batchable.nnef.tgz --nnef-tract-core -O dump
```

You should observe that the batch dimension flow from the input to the last output of the graph:

```bash
    ━━━ B,1000,F32
```

That's great but how can you do profiling ?

```bash
tract ./vit_b_16_batchable.nnef.tgz --nnef-tract-core -O dump --allow-random-input --profile
```

leads to the following stderr:

```json
[... ERROR tract] Expected concrete shape, found: B,3,224,224,F32
```

That's expected as you now need to concretize this symbol before profiling anything.
You can do that before or after the 'dump' keyword but be careful this has a different meaning:

- if before this means the compiled graph by tract in memory will be of concretized dimensions you provided
- if after this means the compiled graph by tract in memory will be offer dynamic dimensions to be defined at runtime per session:
The way to specify it is with the `--set B=3` where 3 can be whatever whole number upper or equal to 1.

So running:

```bash
tract ./vit_b_16_batchable.nnef.tgz --nnef-tract-core -O \
    dump --set B=3 --allow-random-input --profile
```

You should be able to observe as previously a nice evaluation of network speed and it's breakdown.

!!! success "Congratulation"

    you made your first dynamic network export with `torch_to_nnef` :tada: !

## Streaming Audio with stateful model

Imagine that you want to add one of these symbol to the time dimension.

We will for this purpose build a very simple audio network that will have to predict that
events occured every 4 frames in an infinite stream of frames.

We already did 2 examples with transformers like architecture (ViT & BERT),
while a Conformer would work fine, let's instead go old school with a
RNN stack from the [DeepSpeech paper from 2014](https://arxiv.org/pdf/1412.5567) with some convolution on top.

```python title="a custom audio model with streaming"
from pathlib import Path
import torch
import torchaudio

class CustomDeepSpeech(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre = torch.nn.Sequential(
            torch.nn.BatchNorm1d(64),
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, kernel_size=5),
            torch.nn.ReLU(),
        )
        self.maxpool = torch.nn.MaxPool1d(2)
        self.deepspeech = torchaudio.models.DeepSpeech(128, n_hidden=256)

    def forward(self, x):
        x = x.permute([0, 2, 1])
        x = self.pre(x)
        x = self.maxpool(x)
        x = x.permute([0, 2, 1])
        x = x.unsqueeze(1)
        return self.deepspeech(x)
```

We can instantiate a non trained model with it and export it with following command:

```python title="export audio model with streaming dimension"
file_path_export = Path("custom_deepspeech.nnef.tgz")
custom_deepspeech = CustomDeepSpeech()
input = torch.rand(7, 100, 64)
export_model_to_nnef(
    model=custom_deepspeech,
    args=input,
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        dynamic_axes={
            "melbank": {0: "B", 1: "S"},
        },
        version="0.21.13",
        check_io=True,
    ),
    input_names=["melbank"],
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
    custom_extensions=[
        "tract_assert S >= 1",
    ],
)
```

After running the script we look at it from a tract perspective
by dumping it with the classical command:

```bash
tract ./custom_deepspeech.nnef.tgz --nnef-tract-core -O dump
```

We observe a peculiar output dimension:

```bash
    ━━━ B,-3+(S+3)/4,40,F32
```

While batch dimensions is fine, the temporal one is different.

### What did just happen ?

In reality this is quite normal, since some operations in the neural network
happen on the time dimension the streaming dimensions outputed is an expression
based on S:

Since the first convolution has a kernel of 3 we need at least 3 frames
to fill our [receptive field](https://en.wikipedia.org/wiki/Convolutional_neural_network), hence the -3.
Since there is a stride of 2 and a max-pooling of 2: we divide original S by 4 .

tract is able to manage this state of receptive field and the caching of RNN state
for you transparently. To achieve that we need to pulse the network:
Pulsing is a concept specific to tract. It's choosing the 'time' step at which you wish your network to operate.
 By example for this neural network you can select any pulse value that would be a multiple of 4.
Due to it's internal structure we discussed upper.
 As an example we select 8:

```bash
tract custom_deepspeech.nnef.tgz \
    --nnef-tract-core \
    --pulse S=8 \
    dump \
    --nnef custom_deepspeech_pulse8.nnef.tgz
```

By calling this command you create a new NNEF asset that just replaced your
streaming dimensions S by 8. If you look at this newly generated *graph.nnef*, you will also observe several novelties:

- a new extension is added:

```nnef
extension tract_registry tract_pulse;
```

- The introduction of new tract properties:

```nnef
  ("pulse.delay", tract_core_cast([3], to = "i64")),
  ("pulse.input_axes", tract_core_cast([1], to = "i64")),
  ("pulse.output_axes", tract_core_cast([1], to = "i64")),
```

- And some novel operators are set:

```nnef
tract_pulse_delay(
    pre___0__batch_norm0_batch_normalization_output_14,
    axis = 2,
    delay = 1,
    overlap = 1
);
```

They explicitly state the delay expected after the operation at this point
of the graph.
Now each time you will call this loaded model within the same state: it will expect to receive the next 8 frames of melbanks.
And as explained earlier the state caching is managed internally by tract :magic_wand: .

!!! info

    There is only 1 possible pulse dimensions within a tract model

## NLP: stateless model with dynamic batch and token dimension

In a prior example in tutorial [on multiple input outputs](./3_multi_inputs_outputs.md) we recommended to avoid using the provided code as such. Let's remedy to the snippet
to make a better Albert in NNEF:

```python title="fixed Albert export with dynamic dimensions"
from pathlib import Path
from torch_to_nnef import export_model_to_nnef, TractNNEF

file_path_export = Path("albert_v2_dyn.nnef.tgz")
input_names = ['input_ids', 'attention_mask', 'token_type_ids']
export_model_to_nnef(
    model=albert_model,
    args=[inputs[k] for k in input_names],
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        # here we have to specify the symbols for all inputs
        # and all dimensions
        # same symbol is applied several time because:
        # it's the same dimension.
        dynamic_axes={
            "input_ids": {0: "B", 1: "S"},
            "attention_mask": {0: "B", 1: "S"},
            "token_type_ids": {0: "B", 1: "S"}
        },
        version="0.21.13",
        check_io=True,
    ),
    input_names=input_names,
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
    # here we are adding some constraints to our introduced
    # S symbol to help tract reason about what it can do
    # with this symbol
    custom_extensions=[
        "tract_assert S >= 1",
        # we constrain arbitrary our model to be at max 32k tokens
        # of context length
        "tract_assert S <= 32000",
    ]
)
```

Great, now at least we have a bit of dynamism, our newly exported model:

- can handle multiple queries at once (with single batch)
- can ingest varying number of tokens

But is it enough to make it complete ?

Likely not, because we would like by example to cache previously computed tokens
to speed-up inference.

To do that we need to introduce a new set of input for KV cache and a new set of
output for the updated KV cache. This is not managed as an internal state of tract
because we use the `transformers` that design states to be held aside a stateless model
that receive the case and update it at each forward pass.

The past KV-Cache tensors in graph inputs will need
a new symbol that we can call `P` for past and that will lead to following
set of additional constraints:

```python
...
    custom_extensions=[
        "tract_assert P >= 0",
        "tract_assert S >= 1",
        "tract_assert S+P < "
        f"{self.model_infos.max_position_embeddings}",
        # information about processing modes
        "tract_assert tg: S==1",  # text generation
        "tract_assert pp: P==0",  # prompt processing
    ],
...
```

Here again we introduce a new notation the modes:

- Each mode may have a different set of constraints (hence be optimized differently by tract).

To avoid each new user of this library to define these cumbersome settings we provide
a dedicated set of helpers for Languages models as we will see in the [next section](./5_llm.md)

## <span style="color:#6666aa">**:material-step-forward:  Live Demo**</span> :fontawesome-brands-rust: VAD with tract running in browser

As an example of what we just learned we propose a simple VAD running live in this web-page.

<iframe src="/html/demo_vad.html" style="width:100%; height:230px; border: 0 solid #fff;">
</iframe>

!!! note
    This model is not trained by SONOS so prediction accuracy is responsibility of original [nemo](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speech_classification/models.html) authors. Inference performance is descent, but little to no effort was made to make tract WASM efficient (no SIMD WASM, no WebGPU kernels),
    this demo is for demonstration purpose.

Curious to read the code behind it ? Just look at our [example directory here](https://github.com/sonos/torch-to-nnef/tree/main/docs/examples/vad) and this [raw page content](https://github.com/sonos/torch-to-nnef/tree/main/docs/html/demo_vad.html).
