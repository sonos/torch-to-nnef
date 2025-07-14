# 6. Quantization

!!! abstract "Goals"

    At the end of this tutorial you will be able to:

    1. :material-toolbox: Use quantization interfaces in `torch_to_nnef`
    2. :material-book-cog: Define your own quantization library on top of `torch_to_nnef`

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] Understanding of what is [quantization](https://arxiv.org/pdf/2106.08295) for Neural network
    - [ ] 10 min to read this page

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
point to [`torch_to_nnef.compress.DEFAULT_COMPRESSION`](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/compress.py#L112-L131).

### defining your own LLM quantization registry

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
into [`torch_to_nnef.tensor.QTensor`](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/tensor/quant/base.py#L188) a concrete QTensor that support NNEF export today being [`torch_to_nnef.tensor.quant.tract.QTensorTractScaleOnly`](https://github.com/sonos/torch-to-nnef/blob/main/torch_to_nnef/tensor/quant/qtract.py#L93) which has of now only support which is identical to [`Q40`](https://huggingface.co/docs/hub/en/gguf) (that means: 4bit symmetric quantization with a granularity per group of 32 elements, totaling 4.5bpw).

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

Ok min-max is cool, but quality it provide in Q40 is bad, how do I implement my own quantization even with prior section, I feel confused.

Let's take an example step by step:

## 8bit Post Training Quantization example
