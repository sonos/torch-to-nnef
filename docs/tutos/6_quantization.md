# 6. Quantization

!!! abstract "Goals"

    At the end of this tutorial you will be able to:

    1. :material-toolbox: Use quantization interfaces in `torch_to_nnef`
    2. :material-book-cog: Define your own quantization library on top of `torch_to_nnef`

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] Understanding of what is [quantization](https://arxiv.org/pdf/2106.08295) for Neural network
    - [ ] 10 min to read this page

<figure markdown="span">
    ![quant ilu](/img/quant_ilu.png)
    <figcaption>*Illustration by Maarten Grootendorst*</figcaption>
</figure>

Quantization is a set of techniques that allow to reduce significantly model
size, and in case of memory-bound computation for model inference:
speed up model as well. These techniques reduce the 'size' needed
to store the numerical values representing the parameters of the neural network.

In order to make those techniques efficient, the inference engine that run the
neural network need in most cases have specific kernels to support the
quantization scheme selected.

`torch_to_nnef` primary support today being [`tract`](github.com/sonos/tract), the quantization
presented here are all targeting this inference engine.

Today tract support 2 kind of quantization:

- Q40: almost identical to [GGUF Q40](https://huggingface.co/docs/hub/en/gguf), it target weights only where matmul and embedding gathering transform those into float activations.
- 8 bit asymmetric per tensor quantization built-in in PyTorch that can target weights and activations and allow integer only arithmetic

Let's take a look at each in turn starting by Q40.

## Custom Tensor quantization support

### Q40 Export example

For LLM as we explained in prior [tutorial](./5_llm.md) quantization is as simple as
adding the `-c` (or `--compression-method`) option with `min_max_q4_0_all`.

```bash
t2n_export_llm_to_tract \
    -s "meta-llama/Llama-3.2-1B-Instruct" \
    -dt f16 \
    -f32-attn \
    -e $HOME/llama32_1B_q40 \
    --dump-with-tokenizer-and-conf \
    --tract-check-io-tolerance ultra \
    -c "min_max_q4_0_all"
```

It should take around same time to export (quantization time being compensated by less content to dump on disk).

Ok that's nice, but where does this registry come from ?

The registry location is defined with the `--compression-registry` which by default
point to:

<div class="grid cards" markdown>
- ::: torch_to_nnef.compress.DEFAULT_COMPRESSION
    handler: python
</div>

### Defining your own LLM quantization registry

Anyone can create a new registry as long as it follows those rule

- accessible as a global variable dict
- with as key a string that reference the compression to apply
- as value a function that has the following signature:

```python
def my_quantization_function(
    model, # your torch.nn.Module / full model to be quantized
    # huggingface tokenizer or equivalent
    tokenizer,
    # may be usefull to dump compression evaluations results
    # or some specific metrics
    export_dirpath,
    # original trained model location
    # may be usefull to perform internal evaluations of reference
    # when more data than just llm torch is available
    local_dir,
):
    pass
```

A typical function will transform some model tensors (parameters, buffers, ...)
into [`torch_to_nnef.tensor.QTensor`](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/tensor/quant/base.py#L188) a concrete QTensor that support NNEF export today being:
<div class="grid cards" markdown>
- ::: torch_to_nnef.tensor.quant.qtract.QTensorTractScaleOnly
    handler: python
</div>
which has of now only support which is identical to [`Q40`](https://huggingface.co/docs/hub/en/gguf) (that means: 4bit symmetric quantization with a granularity per group of 32 elements, totaling 4.5bpw).

A `QTensor` is a Python object that behave and should be used as a
classical `torch.Tensor` with few exceptions: it can not hold any gradient, it can not be modified, it contains internals objects necessary to it's definition like:

- A blob of binary data (the compressed information) named `u8_blob`
- A [`torch_to_nnef.tensor.quant.Qscheme`](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/tensor/quant/base.py#L20) which define how to quantize/ dequantize the blob from u8 (like [QScalePerGroupF16](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/tensor/quant/base.py#L46))
- A list of [U8Compressor](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/tensor/quant/base.py#L144) that can act on the u8 blob and compress it further by for
example applying bit-packing to it. Say each represented element is specified in 4 bit (16 value represented) without compressor we waste 4 bit per element because each element take 8bit (here we ignore the attached quantization information that add up to the size). Also Compressor are not necessary just bit-packing that can be any kind of classical compression algorithm (Huffman, Lzma, ...) as long as the compression is lossless.

Each access to the QTensor for torch operations will make it be decompressed on-fly saving RAM allocation when unused. This QTensor will also be identified by `torch_to_nnef` at export time and translated to requested `.dat` based on the specific method:

```python
def write_in_file(
        self,
        dirpath: T.Union[str, Path],
        label: str,
        inference_target: InferenceTarget = TractNNEF.latest(),
 ):
    pass
```

Each subclass will define how to dump it (by example [for tract Q40](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/tensor/quant/qtract.py#L178)).

The transformation from a float tensor to a Q40 QTensor can be done through
a step we call tensor quantization which may be as simple as
a min and max calibration as shown in the function [fp_to_tract_q4_0_with_min_max_calibration](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/tensor/quant/qtract.py#L211C5-L211C46), but all compatible techniques can be applied like GPTQ, AWQ, ...
(those are just not part of `torch_to_nnef` package which intend to just provide common primitive to be easily exportable).

A concrete example of `my_quantization_function` can be found [compress module here](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/compress.py#L20).

Today Q40 is supported by tract on matmul, convolutions and embeddings operations.
The `min_max_q4_0_all` will try to apply it to all those encountered modules within a
model.

### Q40 for non-llm network

By reading the previous section you should now understand that beyond specific calibration
which is not part of this library all of what was explained apply to all neural network parameters used in matmul (nn.Linear, F.linear, ...), conv (Conv1d, Conv2d), and embeddings (gather operator).
In fact you can just reuse as is the [compress method](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/compress.py#L20) we referenced upper on any neural network defined in PyTorch it should just work.

### Q40 Use specific quantization method

Ok min-max is cool, but quality it provides in Q40 is bad, how do I implement my own quantization (even with prior sections, I feel confused) ?

Let's take an example step by step:

1. We will use Q40 so no need to redefine the QTensor nor the QScheme, we will just have to create a new tensor quantization function register

2. Let's create a custom register on our own module `super_quant.py` were we will implement a scale grid search based on [Mean Square Error calibration](https://en.wikipedia.org/wiki/Mean_square_quantization_error) for the demo.

3. we first copy almost same function as `quantize_weights_min_max_Q4_0` and rename it `quantize_weights_grid_mse_Q40` and adapt it slightly

```python title="super_quant.py"
from functools import partial
import logging
import typing as T
import torch
from torch import nn
from torch_to_nnef.compress import offloaded_tensor_qtensor
from torch_to_nnef.exceptions import TorchToNNEFImpossibleQuantization
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.quant import QTensor
from torch_to_nnef.tensor.updater import ModTensorUpdater

LOGGER = logging.getLogger(__name__)

def fp_to_tract_q4_0_with_grid_mse_calibration(weight, grid_size=100, maxshrink=0.8):
    # TODO: implementation
    pass


def quantize_weights_grid_mse_Q40(model: nn.Module, **kwargs):
    to_quantize_module_classes = kwargs.get(
        "to_quantize_module_classes", (nn.Linear,)
    )
    assert isinstance(to_quantize_module_classes, tuple), (
        to_quantize_module_classes
    )
    assert all(issubclass(_, nn.Module) for _ in to_quantize_module_classes), (
        to_quantize_module_classes
    )
    with torch.no_grad():
        ids_to_qtensor: T.Dict[int, T.Tuple[QTensor, OffloadedTensor]] = {}
        """ try to avoid quant if used in other operators like mix of embedding/linear if linear only quant """
        mod_tensor_updater = ModTensorUpdater(model)

        for name, mod in model.named_modules():
            if isinstance(mod, to_quantize_module_classes):
                LOGGER.info(f"quantize layer: {name}")
                weight_id = id(getattr(mod, "weight"))
                if weight_id in ids_to_qtensor:
                    LOGGER.info(
                        f"detected shared weight between: '{ids_to_qtensor[weight_id].nnef_name}' and '{name}.weight'"
                    )
                    continue
                if not all(
                    isinstance(m, to_quantize_module_classes)
                    for m in mod_tensor_updater.id_to_modules[weight_id]
                ):
                    clss = [
                        m.__class__
                        for m in mod_tensor_updater.id_to_modules[weight_id]
                    ]
                    LOGGER.warning(
                        f"detected shared weight: '{name}' candidate has incompatible layer usage: {clss}, "
                        f" but requested {to_quantize_module_classes}"
                    )
                    continue
                try:

                    def q_fn(weight):
                        q_weight = fp_to_tract_q4_0_with_grid_mse_calibration(
                            weight,
                            **{
                                k: v
                                for k, v in kwargs.items()
                                if k in ["grid_size"]
                            },
                        )
                        q_weight.nnef_name = f"{name}.weight"
                        return q_weight

                    q_weight = offloaded_tensor_qtensor(
                        q_fn, mod.weight, "q40_mse"
                    )
                except TorchToNNEFImpossibleQuantization as exp:
                    LOGGER.error(f"quant layer: {name} error: {exp}")
                    continue
                # => needs assignation next cause update_by_ref may create new Parameter object
                q_weight = mod_tensor_updater.update_by_ref(
                    getattr(mod, "weight"), q_weight
                )
                ids_to_qtensor[id(q_weight)] = q_weight
    return model

EXAMPLE_REGISTRY = {
    "grid_mse_q4_0_all": partial(
        quantize_weights_grid_mse_Q40,
        grid_size=100,
        to_quantize_module_classes=(
            nn.Linear,
            nn.Embedding,
            nn.Conv1d,
            nn.Conv2d,
        ),
    ),
}
```

Note here the use of `ModTensorUpdater` this module updater allow to avoid breaking shared reference to a common tensor among your network (by example embedding layer shared between input and output of a LLM) while updating the weights.

We now just need to fill the `fp_to_tract_q4_0_with_grid_mse_calibration` function and we are done. Also note that I could have done a calibration stage with external data before end at the beginning (some quantization method need to minimize quantization error for activations). In this case we opt for simplicity:

```python
from torch_to_nnef.tensor.quant import fp_to_tract_q4_0_with_min_max_calibration

def fp_to_tract_q4_0_with_grid_mse_calibration(
    fp_weight, grid_size=50, maxshrink=0.8
):
    qtensor = fp_to_tract_q4_0_with_min_max_calibration(fp_weight)
    qscheme_min_max = qtensor.qscheme
    lower_bound_search_vals = qscheme_min_max.scale * maxshrink
    step_size = (qscheme_min_max.scale - lower_bound_search_vals) / grid_size
    current_vals = qscheme_min_max.scale.clone()
    best_vals = current_vals

    def get_current_error():
        return (
            ((fp_weight - qtensor.decompress()).abs() ** 2)
            .view(-1, qscheme_min_max.group_size)
            .mean(1)
        )

    best_val_error = get_current_error()
    orignal_val_error = best_val_error.clone()
    for _ in range(grid_size):
        current_vals -= step_size
        qtensor.qscheme.scale = current_vals.clone()
        current_val_error = get_current_error()
        better_error = current_val_error < best_val_error
        best_val_error = torch.where(
            better_error, current_val_error, best_val_error
        )
        best_vals = torch.where(
            better_error.unsqueeze(1), current_vals, best_vals
        )
    gain_over_min_max = (orignal_val_error - best_val_error).mean()
    LOGGER.info(
        f"[{fp_weight.name}] quant grid search gained mse error from min/max: {gain_over_min_max}"
    )
    qtensor.qscheme.scale = best_vals
    return qtensor
```

Ok now we can simply run the register we just created by pointing it out with arguments we stated in upper sections and observe the magic. Of course gain using this mse technique are modest, but you now have the full knowledge to make your
own super quant :tada:.

## 8bit Post Training Quantization example

Quantization in 8bit including activation is something that is built-in PyTorch
since a while. This is great because it means this is as well represented in the
Intermediate representation after graph is traced, hence easily exportable with
`torch_to_nnef`. Still today tract only support 8bit asymmetric linear quantization
per tensor (no per channel).

We will still demonstrate this ability on a simple usecase:
Let's do a CNN + ReLU example and apply a [classical PTQ](https://docs.pytorch.org/docs/stable/quantization.html) from there:

```python title="simple PTQ export example"
from pathlib import Path
import torch
from torch import nn
from torch_to_nnef import TractNNEF, export_model_to_nnef


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.cnn1 = nn.Conv1d(10, 10, 3)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv1d(10, 1, 3)
        self.relu2 = nn.ReLU()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.dequant(x)
        return x


torch.backends.quantized.engine = "qnnpack"
m = Model()
m.eval()
m.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
mf = torch.ao.quantization.fuse_modules(
    m, [["cnn1", "relu1"], ["cnn2", "relu2"]]
)
mp = torch.ao.quantization.prepare(mf)
input_fp32 = torch.randn(1, 10, 15)
mp(input_fp32)
model_int8 = torch.ao.quantization.convert(mp)
res = model_int8(input_fp32)
file_path_export = Path("model_q8_ptq.nnef.tgz")
export_model_to_nnef(
    model=model_int8,
    args=input_fp32,
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        version="0.21.13",
        check_io=True,
    ),
    input_names=["input"],
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
)
```

if you look at the model **graph.nnef**, You will obvserve no
difference with a classical NNEF model but .dat exported are uint8 and
a new textual file is set called  `graph.quant`.

This new file contains quantization information for each tensors as follows:

```title="graph.quant (with scale truncated for clarity)"

"quant__input_quantize_per_tensor0": zero_point_linear_quantize(zero_point = 119, scale = 0.021, bits = 8, signed = false, symmetric = false);
"cnn1__input_weight": zero_point_linear_quantize(scale = 0.0014, zero_point = 0, bits = 8, signed = true, symmetric = false);
"cnn1__input_conv": zero_point_linear_quantize(scale = 0.0046, zero_point = 0, bits = 8, signed = false, symmetric = false);
"cnn1__input": zero_point_linear_quantize(scale = 0.0046, zero_point = 0, bits = 8, signed = false, symmetric = false);
"cnn2__Xq_weight": zero_point_linear_quantize(scale = 0.0013, zero_point = 0, bits = 8, signed = true, symmetric = false);
"cnn2__Xq_conv": zero_point_linear_quantize(scale = 0.0017, zero_point = 0, bits = 8, signed = false, symmetric = false);
"cnn2__Xq": zero_point_linear_quantize(scale = 0.00172, zero_point = 0, bits = 8, signed = false, symmetric = false);
```

Finally running tract cli on this model:

```bash
tract ./model_q8_ptq.nnef.tgz --nnef-tract-core dump
```

You should observe that operators are correctly understood as Quantized with QU8 notation:

```
 input
┃   ━━━ 1,10,15,F32
┣ 1 Cast quant__input_quantize_per_tensor0
┃   ━━━ 1,10,15,QU8(Z:119 S:0.021361168)
┣┻┻┻┻┻┻┻┻ 9 Conv cnn1__input_conv_4
┃   ━━━ 1,10,13,QU8(Z:0 S:0.004661602)
┣┻ 11 Max cnn1__input_relu_y_5
┣┻┻┻┻┻┻┻┻ 16 Conv cnn2__Xq_conv_1
┃   ━━━ 1,1,11,QU8(Z:0 S:0.0017290331)
┣┻ 18 Max cnn2__Xq_relu_y_4
┣ 19 Cast output
    ━━━ 1,1,11,F32
```

!!! success end "Congratulation"

    You first exported the network with `torch_to_nnef` quantized in 8bit
    and learned how to create and manage own quantization registry !
