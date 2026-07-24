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

Most popular open multimodal models (Qwen3-VL, Gemma 4, Qwen2.5-VL, Gemma 3,
SmolVLM/Idefics3, Voxtral, …) share one export-friendly pattern: a modality **encoder**
(vision tower or audio tower) turns pixels/mel features into embeddings, and
those embeddings are **spliced into the token embedding sequence** of a normal
causal **decoder** at the positions of a placeholder token (`<image>`,
`<audio>`, …). `torch_to_nnef` exports the two parts as **two separate NNEF
graphs** tied together by a small `multimodal.json` manifest, rather than one
monolithic graph. This keeps each graph streamable/optimizable on its own and
lets a runtime run the encoder only when media is present.

Cross-attention models (Llama-3.2-Vision/Mllama, Whisper) condition the decoder
through cross-attention rather than the embedding-injection pattern above, so
they fall outside *this* joint-export abstraction. That is a scope choice of the
abstraction, not a `torch_to_nnef` limitation: such encoders export on their own
(the Voxtral handler here already exports a Whisper-style audio tower).

## Exporting

As with the LLM CLI, a pre-trained `transformers` multimodal checkpoint can be
exported without touching the API:

```bash title="torch_to_nnef multimodal cli"
t2n_export_multimodal_to_tract -e . --help
```

It shares every flag with `t2n_export_llm_to_tract` (dtype, compression,
tract version/path, tolerance, `--no-verify`, …); only the model needs to be a
multimodal checkpoint whose `model_type` has a registered encoder handler.

Use this command (not `t2n_export_llm_to_tract`) for multimodal checkpoints:
the plain LLM export produces the decoder graph only, so pointing it at a
multimodal checkpoint would drop the encoder tower(s) and the manifest. To
prevent that silent under-export, `t2n_export_llm_to_tract` / `dump_llm`
refuses a checkpoint with registered encoder handlers and points here.

```bash title="export SmolVLM-256M"
t2n_export_multimodal_to_tract \
    --model-slug HuggingFaceTB/SmolVLM-256M-Instruct \
    --export-dirpath ./smolvlm_nnef
```

Equivalently from Python:

```python
from torch_to_nnef_llm.multimodal_exporter import dump_multimodal

dump_multimodal(
    model_slug="HuggingFaceTB/SmolVLM-256M-Instruct",
    export_dirpath="./smolvlm_nnef",
)
```

!!! tip "Choosing a dtype"

    Export defaults to `f32`, which is exact and works for every supported
    model. Pass `-dt f16` (`float16`) to halve the weight footprint.

    In `f16` the exporter automatically routes every attention to SDPA and keeps
    normalization, attention and matmul accumulation in `f32` (fp16 towers and
    decoders otherwise overflow in their eager attention), then checks against
    `tract` at its loosest tolerance. Both the torch reference (CPU SDPA upcasts
    internally) and the exported graph then agree. This makes `f16` a verified,
    no-extra-flags path for SigLIP-based towers (e.g. SmolVLM). A few encoders
    still hit an architecture-specific fp16 gap in their tower (tracked by the
    export test suite); for those, export the encoder in `f32` or pass
    `--no-verify`.

    `bfloat16` is not exportable yet (the numpy-backed NNEF tensor layer has no
    `bfloat16`); use `f16` or `f32`.

    Memory: `check_io` roughly doubles peak RAM (the torch model stays resident
    while the `tract` subprocess loads the graph), so a multi-billion parameter
    `f32` model is heavy. Prefer `-dt f16`, pass `--no-verify`, or export on a
    larger-RAM host.

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
| `encoders[].input` *(optional)* | How to shape (and, if needed, pad) the processor output before feeding the encoder graph. Absent means feed the processor tensor as-is. Fields: `name` (encoder input tensor), `layout` (its axes; integers are fixed sizes, strings are dynamic symbols), `host_prep` (human-readable steps). Dynamic-resolution towers set this. Qwen2.5-VL additionally sets `requires_window_multiple: true` and `window_size`: the grid **must** be zero-padded to a whole number of `window_size`-wide merger-windows and the extra output tokens discarded, otherwise the result is wrong. |
| `encoders[].outputs[].name` | Output tensor of the **encoder** graph. |
| `encoders[].outputs[].feeds` | Input tensor of the **decoder** graph it must be fed into. |
| `encoders[].outputs[].shape` | `[dynamic_axis_symbol, hidden_size]`. |
| `encoders[].outputs[].dtype` | Element type of the embeddings exchanged. |
| `encoders[].deepstack[]` *(optional)* | Per-layer residual streams for multi-layer schemes (e.g. Qwen3-VL DeepStack). Each entry has `name` (encoder output), `feeds` (decoder input), `layer` (decoder layer index it is injected at), `shape` and `dtype`. |
| `injection_layers` *(optional)* | Summary `{modality: [layer_idx, …]}` of the decoder layers targeted by `deepstack` (same indices as `deepstack[].layer`). |

## Runtime integration

For a request that contains media:

1. Run the **encoder** graph on the preprocessed media (pixel values, or log-mel
   `input_features` produced by the HuggingFace feature extractor, which stays
   host-side). When `encoders[].input` is present, first shape (and pad) the
   processor output as its `host_prep` describes; for a tower with
   `requires_window_multiple`, honoring the padding and discarding the extra
   output tokens is mandatory for correctness. It yields the `outputs[].name`
   embeddings.
2. Feed those embeddings into the **decoder** graph input named by
   `outputs[].feeds`, and mark the sequence positions where `input_ids ==
   placeholder_token_id` so the decoder splices them in place of the placeholder
   token embeddings.
3. When `deepstack` is present, also feed each `deepstack[].name` encoder output
   into the matching `deepstack[].feeds` decoder input; those residual streams
   are injected at the decoder layers the decoder graph was exported to use
   (summarized by `deepstack[].layer` / `injection_layers`).

For a text-only request the encoder is not run at all: the decoder graph behaves
exactly like the plain LLM export from [tutorial 5](./5_llm.md).
