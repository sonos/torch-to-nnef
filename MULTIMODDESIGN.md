# Multimodal Joint Export Design (VLM + Voice)

Status: draft for review
Scope: `packages/llm/` (`torch_to_nnef_llm`)
Related: issue #68 (VLM export plan), PR #67 (handler layer), PR #70 (Qwen3-VL decoder)

## 1. Goal

Export transformer multimodal models (vision-language first, audio-language next)
to NNEF for `tract`, as **two coordinated graphs**:

1. an **encoder graph** (vision tower or audio tower) that turns raw modality
   input (`pixel_values`, `input_features`, ...) into modality embeddings, and
2. the existing **decoder graph** (a causal LM) that consumes those embeddings
   as inputs and splices them into its token-embedding stream at
   placeholder-token positions.

The two graphs are exported separately but share one contract, so a runtime can
chain them: `encoder(pixels) -> embeddings -> decoder(input_ids, embeddings)`.

This is the **embedding-injection** pattern. It already underpins the landed
Qwen3-VL *decoder* handler, which takes `in_image_embeddings` as an input. The
missing half is the encoder, plus a generic, modality-neutral abstraction so the
same machinery serves audio.

## 2. Non-goals

- **Cross-attention multimodal models** (Llama 3.2 Vision / Mllama, Whisper,
  SeamlessM4T). They inject modality signal through gated cross-attention layers
  inside the decoder, not through `inputs_embeds`. That is a different decoder
  graph shape and is explicitly out of scope here.
- **Speech / audio output** (Talker heads, RVQ codec token streams: Moshi,
  Qwen-Omni Talker, Voxtral-TTS). We target modality **input** only.
- **Host-side preprocessing** (image patchification, Pan and Scan cropping,
  log-mel feature extraction). These stay outside the graph, matching how
  HuggingFace processors already split responsibilities. The encoder graph
  begins at `pixel_values` / `input_features`.
- Replacing the `nemo-asr` package. Classic ASR (CTC / RNN-T / attention
  decoder, in-graph mel front-end) keeps its own paradigm.

## 3. Landscape summary (why the design is shaped this way)

The single axis that decides fit is embedding-injection vs cross-attention.
Almost every popular open model is embedding-injection, across both modalities.

Clean fit (standard RoPE, single splice, native in `transformers`):
- Vision: Gemma 3, SmolVLM / Idefics3, LLaVA-OneVision, Pixtral / Mistral-Small.
- Audio: Qwen2-Audio, Ultravox, Voxtral-base.

Fits with known risk:
- Vision: Qwen2.5-VL / Qwen3-VL (mRoPE + `rope_deltas`, conv3d patch embed,
  window attention, and Qwen3-VL DeepStack), GLM-4V (3D RoPE), InternVL
  (pixel-unshuffle, dynamic tiling), Phi-4 vision (decoder LoRA to bake),
  MiniCPM-V / Molmo (`remote_code`).
- Audio: Phi-4 audio (conformer + LoRA), Granite Speech (conformer + q-former),
  Qwen2.5/3-Omni Thinker path (must cut the Talker / speech-out head).

Does not fit (cross-attention / codec tokens / classic ASR):
- Mllama, Whisper, SeamlessM4T, Moshi/Mimi, NeMo Canary/Parakeet.

Two landscape facts drive the design:
- **Audio generalizes cleanly.** On the HuggingFace path the STFT and mel
  filterbank are done by a `FeatureExtractor` in preprocessing, so audio
  encoders trace from log-mel `input_features`. We avoid the in-graph STFT/mel
  problem that dominates `nemo-asr`, and "voice" becomes "just another encoder".
- **Variable token count is the same problem in both modalities.** Dynamic
  image resolution (`grid_thw`) and dynamic audio length both yield a
  data-dependent number of embeddings, spliced at a matching number of
  placeholder tokens. Solving it once serves both.

## 4. Current architecture (what exists today)

Decoder-only export lives in `torch_to_nnef_llm`:

- `models/handlers/base.py`: `ArchitectureHandler` ABC with `IOSpec` and
  `StateContext` dataclasses. Hooks: `build_input_spec`, `build_forward_inputs`,
  `call_model`, `build_forward_outputs`, plus `get_auto_model_class` and
  `prepare_model_for_export`.
- `models/handlers/registry.py`: `register_handler` decorator, `get_handler`
  resolving `config.model_type` to a handler class (falling back to `default`).
- `models/handlers/default.py`: `DefaultArchitectureHandler` for plain causal
  decoders (builds input_ids + KV cache, causal mask, position ids).
- `models/handlers/qwen3_vl.py`: `Qwen3VLArchitectureHandler`. The decoder-side
  half of VLM support. It declares `STATE_INPUT_NAMES`
  (`in_image_embeddings`, `in_video_embeddings`, grids, `in_rope_deltas`),
  injects features into `inputs_embeds` via `_inject_token_features`, and manages
  `rope_deltas` / `get_rope_index`.
- `config.py`: `HFConfigHelper` wraps the HF config, resolves the handler,
  exposes `decoder_conf` (handles `text_config`) and KV-cache shape helpers.
- `models/base.py`: `BaseCausal` wraps the HF model, rewrites `forward`'s
  signature from the `IOSpec` (`update_forward_signature`), and drives
  `handler.build_forward_inputs -> call_model -> build_forward_outputs`.
- `exporter.py`: `LLMExporter` orchestrates load, `prepare`
  (`check_wrapper_io`), and `export_model` (calls `export_model_to_nnef` with
  `input_names`, `output_names`, `dynamic_axes`, and `tract_assert` custom
  extensions).

Gaps relative to the goal:
- No encoder export path. The decoder consumes `in_image_embeddings` but nothing
  produces them from `pixel_values`.
- The Qwen3-VL decoder handler is only tested against `FakeQwen3VLModel`
  (`tests/test_qwen3_vl_handler.py`). It has never run against a real checkpoint,
  so the DeepStack gap below is currently invisible.
- No cross-graph manifest tying an encoder output to a decoder input.

`nemo-asr` (`torch_to_nnef_nemo`) is the other precedent: a subnet-decomposition
+ glue + batch-collapse machinery for NeMo typed modules. It is informative for
audio front-end concerns but is not the abstraction reused here.

## 5. Core abstraction

Introduce a modality-neutral encoder handler that mirrors `ArchitectureHandler`,
plus a small contract object that ties an encoder output to a decoder input.

### 5.1 EmbeddingContract

One contract per modality stream the decoder can consume.

```python
@dataclass
class EmbeddingContract:
    modality: str                 # "image", "video", "audio"
    hidden_size: int              # must equal decoder embedding dim
    placeholder_token_id_attr: str  # e.g. "image_token_id" on the HF config
    dynamic_axis: str             # NNEF symbol for the variable token count
    # DeepStack and similar multi-layer injection. Empty for the common case
    # (single splice at the embedding layer). Non-empty lists decoder layer
    # indices that receive an extra residual embedding.
    injection_layers: tuple[int, ...] = ()
```

The contract is the shared truth: the encoder's output tensor named
`out_<modality>_embeddings` has shape `[dynamic_axis, hidden_size]` and feeds the
decoder input `in_<modality>_embeddings` of the same shape and dynamic symbol.

### 5.2 EncoderHandler

```python
class EncoderHandler(ABC):
    MODALITY: str                 # "vision", "audio"
    ARCH_NAMES: tuple[str, ...]   # config.model_type values it handles

    @staticmethod
    def get_auto_model_class(transformers): ...
    def prepare_model_for_export(self, model) -> None: ...

    def get_encoder_module(self, hf_model) -> nn.Module:
        """Return the submodule to trace (vision tower / audio tower + projector)."""

    def build_input_spec(self, *, config_helper, inputs_dtype) -> IOSpec:
        """Raw-modality inputs: pixel_values (+ grid_thw) or input_features."""

    def build_forward_inputs(self, *, inputs, wrapper) -> StateContext: ...
    def build_forward_outputs(self, *, model_outputs, state_context) -> list[Tensor]:
        """Return embeddings (+ DeepStack tensors) matching output_names."""

    def contracts(self, config_helper) -> list[EmbeddingContract]: ...
```

The encoder graph is wrapped by a new `BaseEncoder(TorchToNNEFWrappedLLM)`
(sibling of `BaseCausal`), reusing `update_forward_signature` so traced input
names come from the `IOSpec`.

### 5.3 MultiModalArchitectureHandler

Pairs a decoder handler with one or more encoder handlers and owns the contracts.
This is what the registry resolves for a multimodal `model_type`.

```python
class MultiModalArchitectureHandler:
    decoder_handler: ArchitectureHandler
    encoder_handlers: dict[str, EncoderHandler]   # modality -> handler
    def contracts(self, config_helper) -> list[EmbeddingContract]: ...
```

The existing decoder handlers stay valid: for a text-only model the multimodal
handler is absent and nothing changes.

## 6. Cross-graph coordination

### 6.1 Naming and shapes

The decoder already names its inputs `in_image_embeddings` etc. The encoder names
its outputs `out_image_embeddings` etc. Coordination is by convention plus an
explicit manifest so nothing relies on string matching at runtime.

### 6.2 Manifest

`MultiModalExporter` writes `multimodal.json` next to the two model dirs:

```json
{
  "decoder": {"path": "decoder/model.nnef.tgz"},
  "encoders": [
    {"modality": "image", "path": "vision/model.nnef.tgz",
     "outputs": [{"name": "out_image_embeddings",
                  "feeds": "in_image_embeddings",
                  "shape": ["IMG", 2048], "dtype": "f16"}],
     "placeholder_token_id": 151655}
  ],
  "injection_layers": {"image": [8, 16, 24]}
}
```

A runtime reads this to wire the graphs and to know how many placeholder tokens
each embedding block fills.

### 6.3 Export orchestration

`MultiModalExporter` (thin layer over the existing pieces):

1. Load the HF `*ForConditionalGeneration` model once.
2. Export the decoder via the current `LLMExporter` path (unchanged).
3. For each encoder handler: wrap `get_encoder_module(model)` in `BaseEncoder`,
   build its `IOSpec`, run `check_wrapper_io`, and call `export_model_to_nnef`
   with per-encoder `dynamic_axes` and `tract_assert` extensions.
4. Emit `multimodal.json`.

Reuse, not reinvention: encoder export goes through the same
`export_model_to_nnef` / `TractNNEF` / `build_io` path as the decoder.

## 7. Hard cases and how each is handled

### 7.1 DeepStack (Qwen3-VL): multi-layer injection

Qwen3-VL injects multi-level ViT features into several decoder layers
(`deepstack_visual_indexes`, default `[8, 16, 24]`) by residual add, not only at
the embedding layer. It is still embedding-injection (no attention over image
K/V), but it breaks the "single splice at layer 0" assumption of the landed
handler.

Design:
- The vision encoder emits `out_image_embeddings` (the merger output for the
  layer-0 splice) plus `out_image_deepstack_0..k` (one per injection layer).
- The decoder handler declares matching inputs `in_image_deepstack_0..k` and, in
  `build_forward_inputs`, installs forward-pre-hooks on
  `model.language_model.layers[idx]` that scatter-add the passed embedding into
  the hidden state at placeholder positions. The hooks read tensors carried on
  the `StateContext`.
- `injection_layers` on the contract records the indices for the manifest.

This keeps the decoder a pure embedding-injection graph with extra inputs, and no
cross-attention. Correctness is validated against a real Qwen3-VL checkpoint
(the fake-model test cannot see this).

### 7.2 mRoPE / rope_deltas (Qwen2.5-VL, Qwen3-VL, GLM-4V)

Already handled decoder-side in `qwen3_vl.py` (`get_rope_index`,
`_build_cached_position_ids`, `in_rope_deltas` carried across decode steps). The
encoder does not need rope_deltas. The design keeps position handling entirely in
the decoder handler; the encoder is position-agnostic beyond its own internal ViT
2D-RoPE.

### 7.3 LoRA baking (Phi-4 vision and audio)

`prepare_model_for_export` on the relevant handler merges modality LoRA adapters
into base weights before tracing (peft `merge_and_unload` or equivalent), so the
traced decoder is a plain dense graph. Recorded in tract properties.

### 7.4 Variable token count (all dynamic-resolution / dynamic-length models)

The encoder output uses a dynamic axis (`IMG`, `AUD`) for the token dimension.
The decoder's placeholder count must match at runtime. `tract_assert` extensions
bound the relationship (for example the number of image tokens equals
`prod(grid_thw) / merge_size**2`). Sample export uses a small fixed grid
(`SAMPLE_IMAGE_GRID_THW`, as today) but the axis stays symbolic.

### 7.5 Bidirectional attention over the image span (Gemma 3)

Gemma 3 attends bidirectionally within the image-token span. The decoder handler
builds the attention mask accordingly for the placeholder region. This is a mask
construction detail inside `build_forward_inputs`, not a structural change.

## 8. tract op-coverage risks

Encoder graphs exercise ops the decoder path does not. Verify coverage early and
open tract issues where needed:

- conv2d and conv3d patch embedding (conv3d temporal patch in Qwen).
- pixel-shuffle / pixel-unshuffle reshapes (SmolVLM, InternVL, Qwen merger).
- `F.interpolate` / bicubic position-embedding resize (fixed-grid ViTs).
- average-pool token reduction (Gemma 4096 to 256).
- data-dependent shapes (variable token counts, `cu_seqlens`, window attention).
- conv1d subsampling and relative-position attention (audio conformers).
- flash-attention-only paths: force `attn_implementation="eager"` before trace,
  as the decoder handler already does.

Each first-of-kind model exercises a subset; the milestone plan sequences them so
tract gaps surface on the smallest possible model.

## 9. Module layout

```
packages/llm/torch_to_nnef_llm/
  models/
    base.py                     # + BaseEncoder
    handlers/
      base.py                   # + EncoderHandler, EmbeddingContract,
                                #   MultiModalArchitectureHandler
      registry.py               # register_encoder_handler, resolve multimodal
      encoders/
        __init__.py
        vision_siglip.py        # SmolVLM / Gemma / LLaVA / Pixtral vision towers
        vision_qwen.py          # Qwen2.5-VL / Qwen3-VL vision tower (+ DeepStack)
        audio_whisper.py        # Qwen2-Audio / Ultravox / Voxtral audio towers
  multimodal_exporter.py        # MultiModalExporter + manifest
  cli.py                        # + `dump-multimodal` subcommand
```

Naming note: the package is called `torch_to_nnef_llm` but is really the
transformers export path. Keep the name for now to avoid churn; the encoder work
lives inside it.

## 10. Public API and CLI

- `MultiModalExporter.load(model_slug, ...)` mirrors `LLMExporter.load`.
- `dump_multimodal(model_slug, export_dirpath, ...)` mirrors `dump_llm`, produces
  `decoder/`, one dir per encoder, and `multimodal.json`.
- CLI: `dump-multimodal --model Qwen/Qwen3-VL-2B-Instruct --out ...`, honoring the
  same dtype / tract-version / compression flags as `dump-llm`.
- Text-only models keep using `dump_llm` unchanged.

## 11. Testing strategy

Two layers, matching the existing pattern:

Unit (fast, no network, fake modules):
- Encoder handler IOSpec shape and name assertions.
- Contract wiring: encoder `out_*` names and shapes match decoder `in_*`.
- DeepStack hook scatter-add correctness on a fake decoder with known layers.

Integration (gated by tox env + marker, tiny real checkpoints):
- `check_wrapper_io`: `BaseEncoder` output matches the raw HF vision/audio tower.
- `check_io` through tract: encoder graph parity within `LM_CHECK_TOLERANCE`.
- End-to-end chain: encoder embeddings fed into the decoder graph reproduce the
  reference multimodal logits within tolerance.
- Qwen3-VL DeepStack specifically validated against a real checkpoint, since the
  existing fake-model test cannot exercise it.

Smallest usable checkpoints for CI: SmolVLM-256M / 500M (vision), Gemma-3-4B
(vision, fixed res), Qwen3-VL-2B (mRoPE + DeepStack), Qwen2-Audio-7B or an
Ultravox small variant (audio). Prefer the smallest per family; mark the larger
ones as opt-in.

## 12. Milestones

M0. Abstraction skeleton.
- `EncoderHandler`, `EmbeddingContract`, `MultiModalArchitectureHandler`,
  `BaseEncoder`, `MultiModalExporter`, manifest writer, encoder registry.
- Unit tests with fakes. No real model yet.
- Acceptance: a fake encoder + the existing decoder export two graphs and a valid
  `multimodal.json`.

M1. SmolVLM vision encoder end to end (prove the contract on the cleanest case).
- Vision tower + projector (pixel-shuffle) export, real `check_io`.
- Acceptance: encoder embeddings chained into the decoder reproduce reference
  logits within tolerance on SmolVLM-256M.

M2. Gemma 3 vision.
- Fixed 896 resolution, SigLIP tower, 4096 to 256 pooling, bidirectional image
  mask in the decoder handler. Pan and Scan crop count stays host-side.
- Acceptance: end-to-end parity on Gemma-3-4B.

M3. Qwen2.5-VL vision encoder.
- Native dynamic resolution (`grid_thw`), conv3d patch embed, window attention,
  `cu_seqlens`; coordinate `rope_deltas` with the existing decoder handler.
- Acceptance: end-to-end parity on Qwen2.5-VL-3B; tract gaps filed if any.

M4. Qwen3-VL DeepStack.
- Multi-layer injection: encoder emits DeepStack tensors, decoder hooks scatter
  them into layers `[8, 16, 24]`. Fix the landed decoder handler's single-splice
  gap. Validate against a real Qwen3-VL-2B checkpoint.
- Acceptance: end-to-end parity on real Qwen3-VL-2B, DeepStack active.

M5. Audio generalization (prove modality-neutrality).
- Qwen2-Audio or Ultravox: log-mel `input_features` in, Whisper-style encoder +
  projector, splice at the audio placeholder token. Reuse the same
  `EncoderHandler` and contract with `modality="audio"`.
- Acceptance: end-to-end parity on the chosen audio model; the abstraction needed
  no vision-specific escape hatch.

M6. CLI, docs, example.
- `dump-multimodal` subcommand, an `examples/` entry, README, and a short design
  note on the manifest for downstream runtimes.

## 13. Open decisions (need maintainer input)

1. First proving target: SmolVLM (smallest, fastest to iterate) vs Gemma 3 vs
   going straight to Qwen3-VL. The plan above leads with SmolVLM then Gemma; the
   Qwen path is M3 and M4. Confirm or reorder.
2. Audio inclusion depth for the first pass: land M5 now (Qwen2-Audio / Ultravox)
   to validate the abstraction against a second modality, or ship vision-only and
   keep the `EncoderHandler` contract audio-ready but implement audio later.
3. Manifest format: the JSON above is a proposal. Confirm the fields a downstream
   `tract`-based runtime actually needs (placeholder token id, dynamic axis
   symbols, injection layers).
4. Package naming: keep `torch_to_nnef_llm` or rename to something modality-
   neutral. Recommendation: keep for now, revisit after M5.
5. `remote_code` families (MiniCPM-V, Molmo): support opportunistically or defer.
   Recommendation: defer until a native-transformers baseline is solid.

## 14. Risks

- Qwen3-VL is the stated high-value target but carries the most trace hazards
  (mRoPE + DeepStack + conv3d + window attention). The plan de-risks by proving
  the abstraction on SmolVLM and Gemma first.
- tract may lack coverage for pixel-unshuffle, conv3d, or bicubic interpolate.
  Surfaced early by milestone ordering; may require upstream tract work.
- The landed Qwen3-VL decoder handler, validated only against a fake model, may
  need correction once a real checkpoint runs (DeepStack, and any real
  `get_rope_index` behavior the fake does not reproduce).
