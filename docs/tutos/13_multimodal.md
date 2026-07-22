# 13. Multimodal models (vision & audio)

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-toolbox: How to export a multimodal model (vision-language or
       audio-language) as coordinated NNEF graphs
    2. :octicons-cross-reference-24: The layout and `multimodal.json` manifest a
       runtime needs to wire the encoder(s) to the decoder

!!! example "Prerequisite"

    - [ ] The [Large Language Models](./5_llm.md) tutorial
    - [ ] 10 min to read this page

Most popular open multimodal models (SmolVLM/Idefics3, Gemma 3, Qwen2.5-VL,
Qwen3-VL, Voxtral, …) share one export-friendly pattern: a modality **encoder**
(vision tower or audio tower) turns pixels/mel features into embeddings, and
those embeddings are **spliced into the token embedding sequence** of a normal
causal **decoder** at the positions of a placeholder token (`<image>`,
`<audio>`, …). `torch_to_nnef` exports the two parts as **two separate NNEF
graphs** tied together by a small `multimodal.json` manifest, rather than one
monolithic graph. This keeps each graph streamable/optimizable on its own and
lets a runtime run the encoder only when media is present.

Cross-attention models (Llama-3.2-Vision/Mllama, Whisper) use a different
injection mechanism and are out of scope.

## Exporting

As with the LLM CLI, a pre-trained `transformers` multimodal checkpoint can be
exported without touching the API:

```bash title="torch_to_nnef multimodal cli"
t2n_export_multimodal_to_tract -e . --help
```

It shares every flag with `t2n_export_llm_to_tract` (dtype, compression,
tract version/path, tolerance, `--no-verify`, …); only the model needs to be a
multimodal checkpoint whose `model_type` has a registered encoder handler.

```bash title="export SmolVLM-256M"
t2n_export_multimodal_to_tract \
    --model-slug HuggingFaceTB/SmolVLM-256M-Instruct \
    --export-dirpath ./smolvlm_nnef
```

Equivalently from Python:

```python
from torch_to_nnef_llm.exporter import dump_multimodal

dump_multimodal(
    model_slug="HuggingFaceTB/SmolVLM-256M-Instruct",
    export_dirpath="./smolvlm_nnef",
)
```

!!! warning "Memory"

    The loader defaults to `bf16` on purpose. Forcing `f32` on a multi-billion
    parameter model, while the tract `check_io` subprocess loads a second copy
    of the graph, can double peak RAM. For a large model prefer `bf16`/`f16`,
    ensure enough free RAM, or pass `--no-verify` to skip the tract check.

## Output layout

```
smolvlm_nnef/
├── multimodal.json          # manifest tying the graphs together
├── decoder/
│   ├── model.nnef.tgz        # the causal decoder graph
│   └── tests/…               # reference IO bundles
└── vision/                   # one directory per encoder modality
    └── model.nnef.tgz        # the vision (or audio) encoder graph
```

## The `multimodal.json` manifest

The manifest is the contract a runtime needs to connect the graphs:

```json
{
  "decoder": { "path": "decoder/model.nnef.tgz" },
  "encoders": [
    {
      "modality": "image",
      "path": "vision/model.nnef.tgz",
      "placeholder_token_id": 49190,
      "outputs": [
        {
          "name": "out_image_embeddings",
          "feeds": "in_image_embeddings",
          "shape": ["IMG", 576],
          "dtype": "f32"
        }
      ]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `decoder.path` | Relative path to the decoder NNEF archive. |
| `encoders[]` | One entry per modality encoder. |
| `encoders[].modality` | `"image"` or `"audio"`. |
| `encoders[].path` | Relative path to the encoder NNEF archive. |
| `encoders[].placeholder_token_id` | Decoder input-id whose positions receive the encoder embeddings. |
| `encoders[].outputs[].name` | Output tensor of the **encoder** graph. |
| `encoders[].outputs[].feeds` | Input tensor of the **decoder** graph it must be fed into. |
| `encoders[].outputs[].shape` | `[dynamic_axis_symbol, hidden_size]`. |
| `encoders[].outputs[].dtype` | Element type of the embeddings exchanged. |
| `injection_layers` *(optional)* | `{modality: [layer_idx, …]}` for models (e.g. Qwen3-VL DeepStack) that inject features at several decoder layers rather than only at the embedding input. |

## Runtime integration

For a request that contains media:

1. Run the **encoder** graph on the preprocessed media (pixel values, or log-mel
   `input_features` produced by the HuggingFace feature extractor, which stays
   host-side). It yields the `outputs[].name` embeddings.
2. Feed those embeddings into the **decoder** graph input named by
   `outputs[].feeds`, and mark the sequence positions where `input_ids ==
   placeholder_token_id` so the decoder splices them in place of the placeholder
   token embeddings.
3. When `injection_layers` is present, the embeddings are additionally injected
   at the listed decoder layers.

For a text-only request the encoder is not run at all: the decoder graph behaves
exactly like the plain LLM export from [tutorial 5](./5_llm.md).
