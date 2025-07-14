# 5. Large Language Models Support

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-toolbox: How to export causal Large Language Models
    2. :octicons-cross-reference-24: Current status of this library

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 10 min to read this page

Since 2020, Large Language Models have gathered significant attention in the industry
to the point where every product start to integrate them. **tract** have been polishing
for this special networks since late 2023, and the inference engine is now competitive
with state of the art on Apple Silicon and soon likely on Nvidia GPU's.
In the industry most players use the `transformers` library and a lot of the HuggingFace
ecosystem to specify their models in PyTorch. This make this library the most up to
date source of Model architecture and pretrained weights.
To ease the export and experiments with such models `torch_to_nnef` (this library),
has added a dedicated set of modules that we will now present to you.

## Exporting a transformers pre-trained model

If you only want to export a model already trained, available [on huggingface hub](https://huggingface.co/) and
compatible with the `transformers` library like for example: [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) for chat or text generation purpose.
There is no need for you to learn the api's of `torch_to_nnef`, we have a nice
easy to use command line for you (once `torch_to_nnef` is installed):

```bash title="torch_to_nnef LLM cli"
t2n_export_llm_to_tract -e . --help
```

Which should output something like

```bash
usage: t2n_export_llm_to_tract [-h] -e EXPORT_DIRPATH [-s MODEL_SLUG] [-dt {f32,f16,bf16}] [-idt {f32,f16,bf16}] [-mp] [--compression-registry COMPRESSION_REGISTRY] [-d LOCAL_DIR]
                               [-f32-attn] [-f32-lin-acc] [-f32-norm] [--num-logits-to-keep NUM_LOGITS_TO_KEEP] [--device-map DEVICE_MAP] [-tt {exact,approximate,close,very,super,ultra}]
                               [-n {raw,natural_verbose,natural_verbose_camel,numeric}] [--tract-specific-path TRACT_SPECIFIC_PATH] [--tract-specific-version TRACT_SPECIFIC_VERSION] [-td]
                               [-dwtac] [-sgts SAMPLE_GENERATION_TOTAL_SIZE] [-iaed] [-nv] [-v]
                               [-c {min_max_q4_0,min_max_q4_0_with_embeddings,min_max_q4_0_with_embeddings_99,min_max_q4_0_all}]
...
```

Ok, there is a lot of options here, instead let's do a concrete export of the [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
we mentioned earlier:

```bash
t2n_export_llm_to_tract \
    -s "meta-llama/Llama-3.2-1B-Instruct" \
    -dt f16 \
    -e $HOME/llama32_1B_f16 \
    --dump-with-tokenizer-and-conf \
    --tract-check-io-tolerance ultra
```

On a modern laptop with HuggingFace model already cached locally it should take around 50 seconds to export to NNEF.
Tips: if you have [`rich`](https://github.com/Textualize/rich) installed as dependency, logs will be displayed in color and more elegantly.

Here we export the llama 3.2 referenced from PyTorch where the model is mostly stored
in `float16` temporary activations in `bfloat16` to tract where almost all will be in `float16` (given our `-dt` request, excepted for normalization kept in `f32`), we also check conformance between tract and PyTorch
on a generic text (in english) and observe in last log line that it match:

```
IO bit match between tract and PyTorch for ...
```

Looking at what we just exported we see in the folder just created `$HOME/llama32_1B_f16`:

```
[2.3G]  $HOME/llama32_1B_f16
├── [2.3G]  model
│   ├── [2.2K]  config.json
│   └── [2.3G]  model.nnef.tgz
├── [  78]  modes.json
├── [4.0M]  tests
│   ├── [838K]  export_io.npz
│   ├── [902K]  prompt_io.npz
│   ├── [1.1M]  prompt_with_past_io.npz
│   └── [1.2M]  text_generation_io.npz
└── [ 16M]  tokenizer
    ├── [3.7K]  chat_template.jinja
    ├── [ 296]  special_tokens_map.json
    ├── [ 49K]  tokenizer_config.json
    └── [ 16M]  tokenizer.json
```

The most important file being the NNEF dump of the model of 2.3Go.

If we look at the signature of generated model we should see something like this:

```nnef
graph network(
    input_ids,
    in_cache_key_0, in_cache_value_0,
    ...,
    in_cache_key_15, in_cache_value_15)

-> (
    outputs,
    out_cache_key_0, out_cache_value_0,
    ...,
    out_cache_key_15, out_cache_value_15
)
```

To run such model you can for example use [this crate of tract](https://github.com/sonos/tract/tree/causal_llm_runner/transformers/causal_llm).

!!! tip "work in progress"

    This cli is still early stage, we intends to support
    embedding & classifaction in a near future, as well as
    other modalities model like Visual and Audio LM.

This same cli allow you to export a model that you would have fine-tuned yourself
and saved with [`.save_pretrained`](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)
by replacing the `-s {HUGGING_FACE_SLUG}` by a `-d {MY_DIR_PATH_ON_TRANSFORMERS_MODEL_WEIGHTS}`,
if you did your finetuning with PEFT you can just add `-mp` to merge the PEFT
weights before export (in-case this is your wish: this will allow faster inference
but remove ability to have multiple 'PEFT finetuning' sharing same base exported model).

## Quantize your model

Quantization of models is essential to get the best model on limited resource devices.
It is also very simple to apply opt-in at export time with this command line:

- `--compression-registry` that control the registry that contains the quantization method available it can be any dict from installed modules
    including modules unrelated to `torch_to_nnef` package.
- `--compression-method` that select the quantization method to apply, as a toy example you can
export models linear layers in Q40 (that means: 4bit symmetric quantization with a granularity per group of 32 elements, totaling 4.5bpw)
with simple `min_max_q4_0`. If you wish to leverage best quantization techniques we recommend you to
read our [tutorial on Quantization and export](./6_quantization.md) to implement your own.

## Export a model that does not fit in RAM

You want to go big, but you find that renting an instance will hundreds of Go
of RAM just to export a model is ridiculous ? We agree ! The CLI described upper
provide a convenient solution if you have a *descent SSD* disk just add:

```
--device-map t2n_offload_disk
```

to your prior command like for example:

```
t2n_export_llm_to_tract \
    --device-map t2n_offload_disk \
    -s "Qwen/Qwen3-8B" \
    -dt f16 \
    -f32-attn \
    -e $HOME/qwen3_8B \
    --dump-with-tokenizer-and-conf \
    --tract-check-io-tolerance ultra
```

And pouf done. It will be a bit slower because SSD are slower than RAM but hey
exporting Qwen3 8B in f16 takes around 4min for a 16Go stored model (this trade
is fine for most big models). See [our offloaded tensor tutorial](./7_offloaded_tensor.md)
to learn more about how to leverage this further.

## Export a model from different library

As long as your model can be serialized into the `torch.jit` internal
intermediate representation (which is the case of almost all
neural-networks, whole or parts). This library should be able to do
the heavy lifting of the translation to NNEF for you.

Here is few key considerations to take before starting to support a non
[transformers](https://github.com/huggingface/transformers) (here we are speaking of the package, not the other architectures like Mamba, RWKV, ...)
language model:

- How past states of your neural network is managed inside your library, is it like
transformers an ad-hoc system that is passed as input and output of your neural network
main module ?
- If not is it easy to transform your modeling to approach this architecture ?

If you can answer yes to one of those 2 questions congratulation, you should be able
to easily adapt [these transformers specific torch_to_nnef modules](https://github.com/sonos/torch-to-nnef/tree/main/torch_to_nnef/llm_tract).

Else if state management is internal to specific modules you will likely need to write
[custom operator exporter](./8_custom_operator.md) to express those IO at export time
or add specific operators in tract to manage it.
