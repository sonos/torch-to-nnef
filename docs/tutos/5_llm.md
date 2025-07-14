# 5. Large Language Models Support

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-toolbox: How to export causal Large Language Models
    2. :octicons-cross-reference-24: Current status of this library

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 20 min to read this page

Since 2020, Large Language Models have gathered significant attention in the industry
to the point where every product start to integrate them. **tract** have been polishing
for this special networks since late 2023, and the inference engine is now competitive
with state of the art on Apple Silicon and soon likely on Nvidia GPU's.
In the industry most players use the `transformers` library and a lot of the HuggingFace
ecosystem to specify their models in PyTorch. This make this library the most up to
date source of Model architecture and pretrained weights.
To ease the export and experiments with such models `torch_to_nnef` (this library),
has added a dedicated set of module that we will now present to you.

## Exporting a pre-trained model

If you only want to export a model already trained, available at [https://huggingface.co/] and
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

On a modern laptop with HuggingFace model already cached locally it should take around 50 seconds. Tips: if you have [`rich`](https://github.com/Textualize/rich) installed as dependency, logs will be displayed in color and more elegantly.

Here we export the llama 3.2 referenced from PyTorch where the model is mostly stored
in `float16` with activations are in `bfloat16` to tract where most will be in `float16` (excepted for normalization in f32), we also check conformance between tract and PyTorch
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
by replacing the `-s {HUGGING_FACE_SLUG}` by a `-d {MY_DIR_PATH_ON_TRANSFORMERS_MODEL_WEIGHTS}`.

## Export a model that does not fit in RAM

You want to go big, but you find that renting an instance will hundreds of Go of RAM just to
export a model is ridiculous, we agree ! The cli described upper provide a convenient solution
if you have a *descent SSD* disk just add:

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
exporting Qwen3 8B in f16 takes around 4min for a 16Go stored model (this trade is fine for most big models).
see [our offloaded tensor tutorial](./7_offloaded_tensor.md) to learn more about how to leverage this further.
