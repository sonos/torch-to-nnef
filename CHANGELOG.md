<!-- markdownlint-disable-file MD001 MD013 MD024 -->
# Changelog

## [Unreleased]

### Added
- LLM export: opt-in **`--upcast-quant`** (loader `upcast_quant=`) to dequantize a natively-quantized model (mxfp4, fp8, bitsandbytes, ...) to dense float before export, since tract cannot ingest those formats. Selectable per quant method (names from transformers' `QuantizationMethod`, e.g. `mxfp4`, `mxfp4,fp8`) or `any`, validated up-front (requires transformers >= 4.38). Routes by mechanism: load-time `dequantize=True` for mxfp4/fp8/metal, post-load `model.dequantize()` for bnb/higgs, then verifies the result is fully dense (a format transformers cannot dequantize fails with a clear error). Compose with `-dt` (float target) and `-c` (re-quantize to a tract scheme).
- **LLM: dynamic `logits_to_keep`** as a runtime input (`--num-logits-to-keep dynamic`). The model emits all positions and the wrapper gathers the last `k` rows at runtime, so a single export serves both cheap prefill and speculative decode without re-exporting. The integer form is unchanged.
- **`resize` / `grid_sample` family lowering** to `tract_core_resize` / `tract_core_grid_sample` on tract releases that expose them (`upsample_nearest{,_exact}`, `upsample_{linear1d,bilinear2d,trilinear3d}`, `upsample_bicubic2d`, `grid_sampler{,_2d,_3d}`), replacing the deconv/reshape-tile decompositions; `scale_factor`/`output_size` become constant vectors (dynamic-input safe).

### Fixed
- **Advanced indexing (`aten::index`) with multiple index tensors now broadcasts them to their common shape** before building the `tract_core_gather_nd` coordinates. The un-broadcast concat previously produced a shape-inconsistent NNEF that tract refused to build (`Inconsistent concat`); this affected any broadcasted multi-index gather even under static shapes, and was newly triggered by transformers 5.x mask indexing.
- **LLM causal mask + `position_ids` for transformers > 4.52.4**: a degenerate `[1, 1]` batch-dim mask made generation non-causal; the mask and positions are now built correctly.
- **Parameter views are no longer materialized as constants**: IR shape inference keeps view-like `select`/`slice` outputs as graph values instead of recording each as a fresh constant, so packed MoE expert weights (e.g. Granite MoE) serialize once. One MoE block dropped from ~6.3 GB to ~96 MB of logical constants.
- **NeMo shape-config `renamed_symbols` now rewrites custom/derived `tract_assert` extensions too**: renaming a symbol (e.g. `AUDIO_SIGNAL__TIME` -> `S`) previously left extension asserts referencing the old, now-undeclared symbol.
- Exporter/IR robustness: tolerate HF configs with `head_dim = None`, accept zero-length `split_with_sizes` sections (negative still rejected), safe `TensorVariable` equality for tensor subclasses / meta / numpy tensors, and shape-inference rules for `aten::_grouped_mm` and `aten::gather`.
- Correct payload update when setting an offloaded tensor's data.

### Security
- Bumped **transformers to 5.3.0** (clears HIGH `GHSA-29pf-2h5f-8g72`, no 4.x fix exists) and **pygments to 2.20.0** (`CVE-2026-4539`).

## [0.24.0] - 2026-06-11

### Added
- **ATen operator coverage expanded from 636 to 762 supported operators (+126)** across ~40 batched PRs. Highlights by family:
  - Elementwise math: `frac`, `signbit`, `erfc`, `tanhshrink`, `ldexp`, `addcdiv`, `sgn`, `logaddexp`/`logaddexp2`, `copysign`, `sinc`, `isclose`, `deg2rad`/`rad2deg`, `float_power`, `positive`, plus a `logsumexp` fragment.
  - FFT: `fft_rfft`, `fft_fftn`, `fft_ifftn`, `fft_irfft` (Hermitian reconstruction), `fftfreq`, and Hamming/Blackman/Kaiser/Hann/Bartlett windows.
  - Fused matmul: `addbmm`, `addmv`, `addr` (with `bias_addmm` alias), `chain_matmul`, `bilinear`.
  - Scans: `cummax`, `cummin`, `cumprod`, `logcumsumexp`.
  - Scatter family: `scatter_add`, `scatter_reduce`, `select_scatter`, `slice_scatter`.
  - Patch ops: `im2col` (`F.unfold`), `col2im` (`F.fold`), and `Tensor.unfold`.
  - Special functions: `i0`, `special_i0e`, `i1`, `lgamma`, `mvlgamma`, `digamma`, `special_entr`, `special_xlog1py`.
  - Distances and dot products: `cdist`, `pdist`, `pairwise_distance`, `cross`, `tensordot`, `inner`, `vdot`, plus an l1/huber/smooth_l1/hinge/triplet-margin loss family and `numpy_T`.
  - Complex: `complex` constructor, `polar`, `angle`, `conj`/`conj_physical`, `real`, `imag`.
  - Shape utilities: `meshgrid`, `tensor_split`, `broadcast_tensors`, `column_stack`, `dstack`, `v|h|dsplit`, `movedim`, `channel_shuffle`, `pixel_*`, `count_nonzero`, `diag`, `diag_embed`, `diagflat`, `linalg_diagonal`, `block_diag`, `kron`, `cartesian_prod`, `ravel`, matrix-transpose and `flip` aliases.
  - Misc: `bitwise_left_shift`/`bitwise_right_shift`, `instance_norm`, `max_pool1d_with_indices`, `max_pool2d_with_indices`, `renorm`, `take`/`take_along_dim`, `index_*`, `aminmax`, `var`/`std`/`var_mean`/`std_mean`, `diagonal` (offset support), `vander`, `logspace`, `linspace`, `trilu_indices`, `gru_cell`/`rnn_tanh_cell`/`rnn_relu_cell`, `diff`, `trapezoid`/`trapz`.
- **Python 3.14 support** added; **Python 3.9 dropped** (EOL).
- New end-to-end export examples with zoo tests: Sana mini DiT, Flux-Schnell transformer (MM-DiT), Pocket-TTS (Mimi decoder + flow_net), Mamba selective-scan (external-state and pulse-mode), and a FunASR FSMN-VAD demo (pulsed==batch parity).
- **`force_norm_in_f32`** for `layer_norm` (fp16 export stability, used by Pocket-TTS).
- LLM loader robustness: local-first model loading (use the HF cache without a hub call) and configurable retries on transient HF download failures (`hf_download_n_retries`).
- `map_location` for `iter_torch_tensors_from_disk` (skips non-tensors).
- Hypothesis-driven proptest coverage for primitive ops.
- External custom op support via `t2n_extra` handlers (extensible):
  - `torch_to_nnef.op.extras.register("<name>")` handler API.
  - Auto-loading of handler modules in `export_model_to_nnef` via:
    - `load_extra_op_modules=[...]` parameter,
    - `TORCH_TO_NNEF_EXTRA_MODULES` environment variable,
    - Python entry points under the `torch_to_nnef.extras` group.
  - Meta-forward fallback in export: if eager forward fails, exporter retries
    with meta tensors to infer output structures (requires op meta kernels).
  - New example `examples/t2n_extra_custom_op/` demonstrating end-to-end usage.
  - Docs: expanded tutorial (8. Custom operators) and a new reference page
    (Contributing → Extras API) covering handler signature and helpers.
- **JIT-only model export support** (`torch.jit.ScriptModule` with non-importable inner classes, e.g. Silero-VAD's `silero_vad.jit`).
  - `torch_to_nnef.harden_jit_for_export(model, args, *, freeze=True, diagnostics=None)`: high-level helper that bundles freeze + selective inline + size folds + scalar arithmetic + constant Ifs + tuple round-trip folds + `prim::data` strip + assertion-If strip + data-dependent If fold.
  - `export_model_to_nnef` auto-applies the helper when `model` is a `torch.jit.ScriptModule`. Opt out via `auto_harden_jit=False`.
  - Individual passes exposed at `torch_to_nnef.torch_graph` for fine-grained control: `inline_unresolvable_submodules`, `replace_size_calls_with_constants`, `fold_constant_scalar_arithmetic`, `fold_constant_ifs`, `fold_tuple_index_through_tuple_construct`, `fold_tuple_unpack_through_tuple_construct`, `strip_prim_data`, `strip_assertion_ifs`, `fold_data_dependent_ifs`.
- **Aten RNN op handlers**: `aten::lstm`, `aten::lstm_cell`, `aten::gru`, `aten::rnn_tanh`, `aten::rnn_relu` now have direct handlers, sharing the same fragment-emission code as the module-level extractors.
- **`lstm_cell` NNEF fragment** (single-call, grouped-matmul) replaces the per-gate sigmoid/tanh/mul decomposition.
- **`examples/vad/silero-jit/`**: end-to-end demo of the JIT-only export path on a real artifact, gated in CI via the `silero_vad_demo` tox env.
- **Tutorial**: `docs/tutos/12_jit_only_models.md` covers both the auto-detection path and the manual chain.
- **`aten::istft` handler** for the common case (`onesided=True`, `normalized=False`, `return_complex=False`, `length=None`). Supports rank-3 `(freq, T_frames, 2)` and rank-4 `(B, freq, T_frames, 2)` inputs; raises on rank > 4. Decomposes into Hermitian-symmetric spectrum build, `tract_core_fft` inverse, window multiply, OLA via `deconv` with an identity kernel, divide by the window² OLA, and optional `center=True` crop. Under dynamic axes the window² divisor falls back to a scalar central-region COLA constant; the export verifies that constant actually holds across the central region and raises with a clear message for non-COLA `(window, hop)` pairs.
- **`t2n_extra::exp_unit_norm` / `t2n_extra::exp_mean_norm` handlers** lowering to the matching `tract_extra_exp_*_norm` ops (tract's `OpPulsifier` is already wired, so a streaming-axis trace pulses end-to-end). Powers DPDFNet's `ErbNorm` / `SpecNorm` per-frame EMA replacements in `examples/speech_enhancement/dpdfnet/export_pulse.py`; round-tripped under both static and dynamic axes in `tests/test_t2n_extra_exp_norm.py`.
- **Streaming-friendly `aten::unfold` handler.** Static unfold axis keeps the existing slice-stack lowering (cheap, no extra compute). When the unfold axis is flagged as a streaming dim (via `change_dynamic_axes`), the handler emits a new `unfold` NNEF fragment that lowers to a 1D convolution with an identity kernel of shape `(size, 1, size)`. Tract's existing `Conv` pulsifier handles the streaming case natively, so unfold over a `STREAM` axis works without any tract-side changes. Round-tripped in `tests/test_dynamic_axes.py` for rank-2 (last axis) and rank-3 (middle axis) inputs.
- **`examples/speech_enhancement/dpdfnet/export_pulse.py`** and companion `wav-cleaner-pulse/` Rust wrapper: streaming-friendly DPDFNet export that folds the rolling STFT, NN, iSTFT, OLA, and GRU state into a single streaming-axis NNEF artifact, with tract pulse mode handling buffering and state downstream.

### Changed
- **Breaking: artifact output is suffix-steered.** `export_model_to_nnef` now selects the output form from the path suffix (`None` -> directory, `0` -> `.tar`, `1..9` -> `.tgz` at the matching compression level; default `0`) and returns the path to the produced artifact. Callers relying on the previous fixed output shape must update accordingly.
- **`trust_remote_code` is now opt-out** (LLM): the exporters default to trusting remote code (matching prior behaviour) but emit a warning, and a new `--no-trust-remote-code` CLI flag disables it for untrusted Hugging Face repositories.
- **Python 3.14 supported, Python 3.9 dropped** (EOL).
- **View-tagged complex IR is now an enforced invariant.** `TorchToNGraphExtractor.build_nnef_graph` promotes every complex IR tensor to rank N+1 with a trailing axis of size 2 carrying `(real, imag)`, unconditionally and once up-front. Producer handlers (`_fft`, `stft`, `fft_rfft`, `complex`, `polar`, …) no longer re-promote `node.outputs[0].shape`; consumer handlers (`fft_irfft`, `transpose`, `pick_axis`, …) rely on the uniform invariant `complex_dtype => IR rank == storage rank == logical_rank + 1`. The previous `shape[-1] == 2` heuristic misfired on logical complex tensors whose last axis happened to be 2 (e.g. `fft.fft` of `float32[2,2]`, `fft.rfft(x, dim=0)` on a length-2 signal). Side effect: `torch.complex(r, i)` and `torch.polar(a, p)` outputs now correctly survive chaining into `torch.fft.fft`/`fft.ifft`.
- Officially supported tract versions bumped to 0.23.0 and 0.22.1 (0.21.15 dropped).

### Fixed
- Reified `scaled_dot_product_attention` (`tract_transformers_sdpa`) now tiles a size-1 query axis on the attention mask up to the query length. tract 0.23.0's `FlashSDPA` does not broadcast that axis at eval time, so key-padding masks (e.g. torchaudio Conformer) raised an incompatible-shape error.
- Nearest 2-D upsample with static shapes now lowers via the rank-generic reshape/tile path instead of `deconv`. The deconv lowering emits a broadcast `Mul` that tract 0.23.0's `OptMatMul` fuse pass mis-substitutes when an adjacent conv consumes the upsample output (e.g. VAE-style decoder). The deconv path is kept for the dynamic-axes case.
- Parser bugs surfaced by JIT-only export (each was latent in main):
  - `slice_` recognizes `prim::Constant[NoneType]` for begin/end (Python's `t[:]`).
  - `div` wraps scalar division result in a 0-d tensor before `.to(dtype)`.
  - `_convolution_mode` accepts list-padding from `aten::conv1d/2d/3d`.
  - `prim::TupleUnpack` output names are normalised through `cleanup_data_name`, fixing dotted SSA-name lookups (e.g. `h0.1`).
- `constant_pad_nd` with a negative entry on a dynamic axis: the decomposed `slice` now uses `dyn_slice_begin` (open-ended) instead of a concrete-`end` `slice`, so the streaming-axis symbolic dim survives the crop. Without this, the cropped axis collapsed to a fixed size and any downstream broadcast against a streaming sibling failed under pulse mode (surfaced by DPDFNet's causal `pad_feat`).
- `flatten` of a view-tagged complex tensor emitted the reshape with the complex `dtype`, which has no NNEF datum type and raised `KeyError` at serialization; it now emits the real component dtype (the storage is real `(..., 2)`). `permute` of a view-tagged complex tensor now builds the full storage-rank axes list with the trailing `(re, imag)` axis kept fixed, matching the `transpose` handler instead of relying on NNEF's append-remaining-axes behaviour.

### Security
- **Checkpoints load `weights_only`-first** (core): `torch_safe_load` attempts a `weights_only=True` load before any fallback, avoiding arbitrary code execution from pickled checkpoints.
- **Archive extraction is path-traversal safe**: extraction now refuses members that would write outside the target directory.
- **Coordinated disclosure policy**: added `SECURITY.md` (private vulnerability reporting via GitHub).
- **CI security gates**: CodeQL SAST (security-extended), TruffleHog secret scanning (PRs + weekly history sweep), Trivy, `cargo-deny`, dependency review, and CODEOWNERS on security-critical config.
- **Dependency CVE patches**: bumped `onnx` (1.21.0), `sentencepiece` (0.2.1), `urllib3` (2.7.0), `pillow` (12.2.0), `nemo-toolkit` (2.7.3), `GitPython` (3.1.50), and `cryptography`/`pyarrow`/`protobuf` to address high-severity CVEs (RCE, OOB read/write, TOCTOU, path traversal, decompression bomb). Example lockfiles bumped `tar`/`time`/`mako`/`nltk`/`pyarrow` for the same reasons.

## [0.23.2] - 2026-04-21

### Added
- **Auto slug resolution** (NeMo ASR): resolve pretrained slug from encoder architecture fingerprint, so local `.nemo` finetunes (which don't store the slug) inherit the pretrained's tract extensions.
  - `slug_fingerprints.json` registry mapping known slugs to `EncoderFingerprint`.
  - `python -m torch_to_nnef_nemo.tools.refresh_slug_fingerprints` tool to regenerate the JSON from `ASRModel.from_pretrained`.
- **Derived tract extensions** (NeMo ASR): compute `tract_assert` bounds from encoder architecture (`pos_emb_max_len`, subsampling factor, attention variant) instead of hand-maintaining per-slug strings. Manual `SLUG_EXTENSIONS` registry remains as an overrides path; both are merged at export time with deduplication.

### Fixed
- NeMo fingerprint/deriver now guard against models without an `encoder` submodule.
- `uv.lock` restored in `.bumpversion.cfg` (dropped during the 0.23.0 monorepo split).

## [0.23.1] - 2026-04-13

### Added
- **transformers 5.x support** (LLM): relax dependency from `<5` to `<6`, tested with 5.0.0 and 5.5.0.
- **NeMo 2.7.2 / Python 3.13 support**: new tox env, fix optional input handling.
- Parakeet V3 and MarbleNet VAD export tests for NeMo.
- Runtime warning when STFT is used with tract 0.21.14/0.21.15 (known slice-fusion bug).

### Changed
- Officially supported tract versions bumped to 0.22.1 and 0.21.15.
- LLM CI matrix now tests transformers 5.0.0 and 5.5.0 on Python 3.13 (replaces 4.55.0).
- SDPA test updated for tract 0.22.1 `reify_sdpa_operator` (`tract_transformers_sdpa` op).
- `isnan`/`isinf` tests gated to tract > 0.22.1 (ops landed in tract 0.23).

### Fixed
- `DynamicCache.from_legacy_cache()` / `to_legacy_cache()` removed in transformers 5.x -- version-aware helpers added.
- `Parameter.__new__()` rejecting HF `_is_hf_initialized` kwarg in transformers 5.x.
- `strict=True` added to `zip` in `expand_input_names` (ruff B905).
- Handle v-prefixed tract release tags.

### Known tract issues (upstream)
- tract 0.21.14/0.21.15: slice-fusion optimization corrupts STFT results (`8b8f4537c`).
- tract 0.22.1 on linux x86_64: `OptMatMulPack` tries to pack F32 tensors as PackedF16 when `force_f32_attention=True`.

## [0.23.0] - 2026-04-10

### Changed
- Split into 3 independent packages in a uv workspace monorepo:
  - `torch_to_nnef` (core) -- base export library, Python >=3.9
  - `torch_to_nnef_llm` (LLM + PEFT) -- transformers-based export, Python >=3.10
  - `torch_to_nnef_nemo_asr` (NeMo ASR) -- NeMo ASR export, Python >=3.10
- `requires-python = ">=3.9"` now enabled on core (was disabled due to nemo constraints).
- `pyyaml` added to core dependencies (was transitive via nemo).
- Decouple `remodeler` from nemo: use `T.Any` for registry types instead of importing `AxisSymbolRegistry`.
- Each sub-package manages its own `_optional_types.py` for dependency-injection type stubs.
- Test dependencies moved from extras to dependency-groups (PEP 735) -- `pip install torch_to_nnef[test]` no longer works; use `uv sync --group test` instead.
- Core test suite split into lightweight core envs and heavy zoo envs (torchvision/torchaudio/librosa) with matched version pins per torch version.
- Release workflow builds and publishes all 3 packages (core first, then LLM and NeMo).

### Migration
- CLI commands unchanged: `t2n_export_llm_to_tract`, `t2n_export_peft_to_nnef`, `t2n_export_nemo` work as before.
- Install: `pip install torch_to_nnef[llm-tract]` still works (backward-compat redirect extras).
- Python imports changed:
  - `from torch_to_nnef.llm_tract.X` becomes `from torch_to_nnef_llm.X`
  - `from torch_to_nnef.nemo_tract.X` becomes `from torch_to_nnef_nemo.X`
  - `from torch_to_nnef.peft.X` becomes `from torch_to_nnef_llm.peft.X`

## [0.22.0] - 2026-04-01

### Added
- remodeler (`torch_to_nnef.remodeler`): generic shape-remodeling pipeline extracted from NeMo-specific logic, reusable for any export workflow.
  - structured `shapes.yaml` per subnet (`original_shape`, `collapse_dims`, `bind_scalar_to_dim_size`, `renamed_symbols`, `eval_symbols`) to control boundary-only transforms.
  - per-output `collapse_dims` support in shape config.
  - export-time `BoundaryAdapter` applying tuple flattening, alias-aware collapse (batch and other dynamic dims), dynamic scalar binding from `shape(source)[axis]`, and dynamic-axes recomputation; only triggered for structural transforms, symbol-only renames applied via lightweight `_apply_symbol_renames_to_dyn`.
  - generic `BoundaryAdapter`, `RenameOutputs` in `torch_to_nnef.remodeler.adapter`; generic `prepare_subnet_export` helper.
  - dedicated `dyn_axes` module (`torch_to_nnef.remodeler.dyn_axes`) for dynamic-axes evaluation logic.
  - IO collision checker for `model_wrapper`; assert extensions for NeMo models.
  - binding keeps dynamism by tracing `aten::size` + cast (no baked constants), and reinserts target-collapsed axes for correct internal ranks.
- inspector: `--inspect-signatures`, `--inspect-stage`, `--inspect-format`, `--inspect-output`, `--inspect-diff`; human/human-rich/JSON output; model header; tuple input expansion; config overlay; per-stage diffs; stricter config validation (qualified/bare name resolution, rank mismatches); symbol overlay/substitution.
- template dump: nested YAML with header and inline lists; auto-suggest `renamed_symbols` for decoder/decoder_joint when multiple batch-like symbols are seen across inputs.
- symbol generation: batch dims are now namespaced as `<INPUT>__BATCH` (e.g., `ENCODER_OUTPUTS__BATCH`) for clarity and consistency across inputs.
- NeMo export: Tract-facing dynamic axes honor subnet `renamed_symbols`; assertions consolidated to alias targets; dtype preparation (`.half()`, `WrapPreprocessorCast`) handled internally based on `cfg.data_type`; subnets with unused traced inputs handled gracefully (`check_io_names_qte_match=False`).

### Removed
- legacy `--collapse-batch-dim` flag and its wrapper; use `shapes.yaml` (`collapse_dims`) instead.

### Fixed
- torchvision compatibility check: correctly map torch 2.x to torchvision 0.(15+minor).x (e.g., torch 2.9.x ↔ torchvision 0.24.x).
- circular import: moved `InspectFormat` enum to `config.py` to break `config→inspect→export→config` cycle; `NamingPrecisionConfig.naming_scheme` and `InspectionConfig.inspect_format` now typed as enums.
- VAD model loading: fallback to `EncDecClassificationModel.from_pretrained` when `ASRModel.from_pretrained` fails.

### Tests
- extended nemo export test suite: config variants (skip-preprocessor, float16, only-subnets, quantization, naming), shape config from YAML (parakeet full, VAD collapsed), programmatic batch-collapse with bind+strip, dry-run dump round-trip.
- per-model tract IO tolerance (QuartzNet uses `VERY`).
- NeMo log silencing fixture in conftest.
- `eval_symbols` test cases.

### Docs
- NeMo ASR guide updated with a Shapes config section: dump → edit → inspect → export, with examples for `collapse_dims`, `bind_scalar_to_dim_size`, and `renamed_symbols`.



## [0.21.0] - 2026-02-15

### Added

- NeMo ASR export via new `torch_to_nnef.nemo_tract` package: CLI, wrappers, dynamic-axes utilities, and model loader helpers.
- Example scripts for docs/examples: `bootstrap-uv.sh`, `bootstrap-wasm-pack.sh`, `bootstrap-rust.sh`, and `clean.sh` to streamline local setup and cleanup.
- Tests: artifact packaging behavior (`tests/test_artifacts.py`) to validate `.nnef`/`.tar`/`.tgz` outputs.
- Tests: expanded coverage around new features and edge cases, including cumsum (`tests/test_cumsum.py`), MaxPool2d with indices (`tests/test_pool_with_indices.py`), output renaming safeguards (`tests/test_rename_outputs.py`), and NeMo subnet iteration/splitting (`tests/test_nemo_iter_subnets.py`).
- API: `export_model_to_nnef` now returns the exported artifact path for easier downstream use.
- Ops: added support for `cumsum` (exported as `tract_cumsum`) and MaxPool2d with indices.
- Export helpers: `iter_torch_tensors_from_disk(map_location=...)` to control device mapping; skips non‑tensor entries in state dicts.
- Writer: pass‑through identity (input→output) renders as an assignment when targeting Tract; prevents silent aliasing.
- Tract option: `--tract-reify-sdpa` to reify SDP attention for improved tract optimizations.
- Versioning: new SemVer 2.0 utilities (`SemanticVersion`) and helpers integrated; enables correct prerelease/build handling and consistent version comparisons across the codebase.

### Changed

- NNEF export artifacts: honoring `.nnef.tgz` as an intent for the final archive while always writing to the base `.nnef` path internally; consistent selection of directory (`compression=None`), `.tar` (`0`), or `.tgz` (`1..9`).
- Writer: explicit `archive_format` for predictable `.tar` vs `.tgz` emission; avoids unnecessary fragment extensions for Tract targets.
- Export robustness: normalized mapping-like/single outputs via `ensure_tuple_io`; stricter IO name collision checks with opt-in `allow_same_io_names` (kept False by default).
- NeMo export pipeline: improved subnet iteration (optional split of decoder/joint), batch-dimension collapse option, safer preprocessor export, and automatic output renaming to avoid input/output name collisions.
- RNN utils monkeypatching is now narrowly scoped and only applied for Tract targets; clearer error messages when unsupported.
- Dynamic‑axes inference refined for unflatten/argsort/sort using `get_tract_dyn_axis_size_soc`.
- Cumsum: initial simple implementation introduced and later renamed from `t2n_cumsum` to `tract_cumsum`; improved tract ONNX↔NNEF compare utility.
- Versioning: `torch_version()` now yields a semantic `SemanticVersion`; feature toggles (e.g., default SDPA reification) are gated semantically and auto‑enable reify for Tract > 0.22.0.
- Docs/examples: added `docs/examples/nemo_asr/requirements.txt`, hardened NeMo ASR `run.sh`, and removed a stale YOLO TorchScript asset.
- Tooling/build: clarified pip 23.2 dependency constraints; pinned/adjusted setuptools for older torch toolchains (e.g., 1.10); minor doc clarifications.

### Fixed

- `.nnef.tgz` target path now produces a `.nnef` directory when `compression_level=None` (base-path semantics), matching tests and documentation.
- Dynamic-axes propagation and rank filtering across NeMo subnets to better reflect actual exposed IO.
- Minor linting and config tweaks (prospector rule, gitignore patterns).

## [0.20.4] -  2026-02-06

### Added

- Completed NeMo ASR export and evaluation system CLI tooling (t2n_export_nemo) with novel options
- Shape inference for conv and test suite validating IR operation shape/dtype correctness with approximate and exact tracing modes
- Comprehensive NeMo model subnet with improved batch alignment checks for encoder/decoder/joint export
- Analysis tooling for debugging batch-mode encoder issues and SDPA operator behavior
- Support for runtime config in nemo tract rust example to better control the runner.

### Changed

- Export performance significantly improved via shape inference speculation system - avoiding redundant model executions during export by guessing shapes for common operations
- NeMo evaluation framework restructured with modular dataset handling, manifest comparison tools, and batch alignment verification utilities
- Model wrapper now handles complex input/output structures (nested tuples, dicts, custom objects) more robustly with better constantization support

### Fixed

- Nemo export - removed useless parameters from model exports via wrappers
- Named tensor operations now avoid unnecessary clones and properly handle dtype/shape access during graph parsing (improved speedups)
- Reducer operations (sum, mean, etc.) now better handle boolean dtype inputs (improved support in tract)
- Model zoo tests refactored for better isolation and legacy quantization compatibility

## [0.20.3] - 2026-01-26

### Added

- Initial support for nemo-toolkit ASR export (including example & evaluation system).

### Fixes

- to matmul/linear operation tracing with more than 2d inputs
- masked_fill operation was casted to bool incorrectly in some edge cases
- renaming_scheme was failing with `natural_verbose` on some models edge cases
- edge case with lstm states being reused in subsequent layers
- numerous deadlinks in documentation fixed


## [0.20.2] - 2026-01-15

### Fixes

- Fix issue with empty tensor concatenation in tract
- logo in mkdocs is now displayed only once on mobile
- some broken links in doc fixed
- fix related unitest using librosa


## [0.20.1] - 2025-09-13

### Added

- Open-Sourced the project under MIT|Apache2 license
- Official support for tract `v0.22.0`
- test coverage of LLM export with various `transformers` lib version (trying to support last 10ish minor versions with CI/CD)
- Add context manager to force loading with offloaded tensor
- Added opt-in support for reification of `spda` operator when targeting tract export (thanks to @emricksinisonos contribution) this should help further optimization in tract of attention blocks
- Added support for `upsample` operator via `deconv` or `debox` depending on tract version
- Added Licenses file
- `ModTensorUpdater` is now useful with legacy torch version (bellow 2.0)
- Add `aten::new`
- New logo (thanks to @lizavetaobadava-commits)

#### Formatting & style

- All exception now inherit from `T2NError` (allow easier catch)
- Stricter line length (even in doc)
- Stricter doc formatting with `ruff`
- Improved `prospector` strictness
- `isort` retired in favor of `ruff`

#### Documentation

- Documentation versioning with `mike` (allowing to get older version doc)
- Documentation: fixed typos, rewording (thanks to @thomasnigoghossiansonos for the review)
- WASM LLM poetry generator example expose the prompt for clarity
- Nicer WASM example with more loading state infos
- Fix WASM VAD example handling more audio context (more robust)
- Added WASM Yolo example with pose-estimation

### Fixed

- transformers regression since 4.56 around cache handling
- better support for OffloadedTensor with assignations and some in-place operations
- pylint tweaks

### Change

- Following open-sourcing of the project, packaging is now targeting PyPI.

## [0.19.1] - 2025-08-06

### Added

- CI/CD for torch version bellow 2.7: 2.2, 1.13 and 1.10
- specific checks around dtype for qtensors tests generated assets

### Fixes

- make this package work again for torch version between 1.10 and 2.3.

## [0.20.0] - 2025-09-13

Failed release in CI/CD

## [0.19.0] - 2025-07-25

### Added

- mkdoc documentation revamp
- no more approximation of `logsofmax` for TractNNEF
- added support for operators: `fmod`, `expm1`, `atan2`, `addmm`, `maximum`, `minimum`, `logical_or`, `logical_and`, `logical_not`, `fill_`, `var`, `avg_adaptive_pool_nd`, `max_adaptive_pool_nd`, `amin`, `amax`, `nn.LocalResponseNorm`

## [0.18.6] - 2025-07-03

### Added

- base param updater

## [0.18.5] - 2025-06-13

### Added

- `update_values` in OffloadedTensor

### Fix

- `Parameter` addition in OffloadedTensor
- `to_json_file` use in config dump in LLM

## [0.18.4] - 2025-06-11

### Change

- bunch of cache, ordered conditioning, torch.compile to make export faster

## [0.18.3] - 2025-06-05

### Fix

- `dtype` change (via .to) of OffloadedTensor handled
- `numel` call avoid OffloadedTensor reload
- `dtype` getter aligned
- dissociated getter aligned

### Added

- support for python 3.13 landed (some issue still with Quant+Offloaded tensor mem alloc)
- `aten::bitwise_or` operator

## [0.18.2] - 2025-06-04

### Fix

- `safetensors` import, only called when needed

## [0.18.1] - 2025-06-03

### Added

- Official tract support version: `0.21.13`

## [0.18.0] - 2025-06-03

### Added

- addition of an `OffloadTensor` that allow to write on disk the tensor and reload it each time from there (trading memory space for disk usage/reloading speed -> this is not intended to be used beyond compression and export of neural net stage).
- Plug of a load step by step into `OffloadTensor` method for `tract_llm` (as an opt-in via `--device-map`=`t2n_offload_disk` option). This option is also compatible with accelerate if installed to spread model partitions load across available hardware devices in an instance.

### Change

- refactor of all custom PyTorch tensors used on torch to NNEF into a unified module
- [OPTIM] removal of part of redundant inference tracing computation for shape and type

### Fix

- avoid duplicate weights in **Numpy** data within `nnef.Graph until` serialization (write) step

## [0.17.4] - 2025-05-15

### Fix

- Add eq in TensorVariable to build proper dict keys and in queries from it (without traced data accounted)
- all tract_core_gather add attrs datum_type
- Q4 compression_method tag compat with internal llm lib
- Skip check_io between wrapper_model vs hf_model if wrapped_model

## [0.17.3] - 2025-05-09

### Added

- aten::`full_like`, `_softmax`, `mm`, `logical_not`, `scalar_tensor`, `to_copy`
- forward signature of wrapped llm models is updated live based on model KV cache quantity to help `torch.export` understand all parameters (*args, **kwargs does not work)

### Change

- `HFConfigHelper` now only need HF conf (no more slug name)

## [0.17.2] - 2025-04-10

### Added

- bump tract `0.21.12`
- avoid compress weight if shared with 1 module that is not requested to compress (by example: request `nn.Linear` only while shared with `nn.Embedding`)

### Fix

- some `ignore-already-exist-dir` missing case in `llm_tract`

## [0.17.1] - 2025-04-02

### Fix

- Avoid duplicating weights in case they are shared with assignation post `nn.Module` load

## [0.17.0] - 2025-03-31

### Change

- All parameters variable in graph are be named the same their label if `NamedTensor`

### Fix

- RNN expansion with multiple call within same graph now refer to same set of weight instead of duplicating them

## [0.16.11] - 2025-03-27

### Fix

- `set_priority` in `with sdpa_kernel` only appear in torch 2.6

## [0.16.10] - 2025-03-24

### Fix

- `aten::flatten` with partial dimensions
- `aten::remainder` force f32 (if other implicit dtype support like ints)
- `aten::pad...` now support dynamic dimensions
- `aten::zeros`, ... now default to f32 in cases where unspecified in jit graph
- Merge of subgraph in ir_graph is now done with preserving `subgraph` output names (needed since some output may be repeated while main graph unaware of it)

### Added

- Conv are now supported for Q40 exports (tract `v0.21.12`)
- compress registry `min_max_q4_0_all` export all supported tensors in Q40 (including Conv1d, Conv2d)

## [0.16.9] - 2025-03-20

### Fix

- regression on `uint32`, `uint64` support (pre torch 2.4)

## [0.16.8] - 2025-03-20

### Fix

- regression on `uint16` support (pre torch 2.4)

## [0.16.7] - 2025-03-20

### Fix

- complex slice index gather nd fix

## [0.16.6] - 2025-03-20

### Added

- official tract support is now `0.21.11` (new default target)
- support `to` device like `cuda`,`mps` for our internal QTensor  ...
- support for new operators: `aten::empty_like`, `aten::prod`, `aten::index_select`, `aten::scatter`, `aten::numel`

### Change

- additional tracing cues for whole number values that may be used in tensors shaping/construction.
- disabled support for Python >=3.13 as of now as it leads to unexpected hash/set issues to be investigated

### Fix

- `aten::baddbmm` extra args handled during tracing
- better alignment of arity for rnn inputs
- equality operators (`ne`, `ge`, `le`, `gt`, `eq`) now implicit cast to common dtype if heterogeneous
- `to` operators with from float to unsigned with negative values was found to have an arch dependant behavior (code now align to the arch used at export with warning for non arm)
- tolerate export pad operators with dynamic values

## [0.16.5] - 2025-03-11

### Change

- test by default on 2.6

### Fix

- SPDA regression if pytorch > 2.3 and usage of specific scale

## [0.16.4] - 2025-03-11

### Added

- support new `Q40` tract format starting with target tract>=0.21.11

### Fix

- remove useless hard dependencies (regression since 0.15.10 about) and relaxing numpy version

## [0.16.3] - 2025-03-07

### Fix

- edge-case in `tract_llm` export forward_kwargs

## [0.16.2] - 2025-03-07

### Added

- better debug dump with shell script to reproduce failing case

### Fix

- export RNN with 2nd or 3rd outputs used only
- export support `tract_llm` architecture without `num-logits-to-keep`
- explicit peft dependency referenced in pyproject

## [0.16.1] - 2025-03-06

### Added

- export with `tract_llm` merge PEFT option is set
- CI now fail-fast
- VERSION is set at project root to help compare with str
- better test_suite naming for dump and debug

### Change

- export with `tract_llm` will use `num-logits-to-keep` avoiding useless compute at inference

## [0.16.0] - 2025-03-03

### Change

- Breaking change `-f16`,`--as-float16` removed and replaced by `--force-module-dtype`, `--force-inputs-dtype` that re-express this

## [0.15.18] - 2025-02-28

### Fix

- PEFT loading from tract llm cli regression
- using embedding gather with 1d tensor indices input

## [0.15.17] - 2025-02-24

### Fix

- correct branching in tract selection cmd llm export

## [0.15.16] - 2025-02-24

### Fix

- Avoid auto log settings except in cli's

### Added

- f32 norm options in llm cli

## [0.15.15] - 2025-02-19

### Fix

- Format safety in tract_properties (avoid caret return escape and other closing quote)

## [0.15.14] - 2025-02-19

### Fix

- another compress import issue

## [0.15.13] - 2025-02-19

### Fix

- wrong default for compress registry llm_tract cli

## [0.15.12] - 2025-02-19

### Change

- move `torch_to_nnef.llm_tract.compress` to `torch_to_nnef.compress` as it is generic

### Fix

- test suite pass again on Darwin OS
- some remaining trace of `flake8`,`black` to `ruff`

## [0.15.11] - 2025-02-17

### Added

- support p norm with p != 1 or 2 (including inf and -inf norms)
- upcast to f32 norm operations if f16 inputs such as `BatchNorm`, `norm_p2`, `group_norm`, `weight_norm`
- more tract default properties among which export command, python version, (and opt-out) username, hostname, OS info (uname -a)

## [0.15.10] - 2025-02-14

### Change

- packaging/building project with `uv` (`poetry` deprecated since latest uv version are better)

## [0.15.9] - 2025-02-10

### Added

- ready to support tract 0.21.9 (once regression tract side solved)

## [0.15.8] - 2025-02-07

### Added

- TractNNEF now dump: `tract_properties` in graph.nnef with metadata infos and possible additional custom infos can be passed with `specific_properties`
- TractNNEF: control over check io precision with `check_io_tolerance` parameters (exposed in llm cli)
- TractNNEF: has now `force_attention_inner_in_f32` that force f32 compute for SDPA in tract
- TractNNEF: has now `force_linear_accumulation_in_f32` that should be active after tract release `0.21.10` and allow accumulation in f32 for linears (opt-in)
- cli llm: export of specific model like qwen force f32 parameters defined upper by default (for others architectures those are exposed in cli directly)

## [0.15.7] - 2025-01-29

### Fix

- LLM cli export: `PEFT` better support
- LLM cli export: multiple `.safetensors` support
- `LLMExporter` decoupled and better supported

## [0.15.6] - 2025-01-10

### Fix

- `unsqueeze` on dim -1
- `sum` without arguments

### Added

- `uint16` support (since PyTorch 2.4)
- `gather`, `sort`, `argsort`, `topk` PyTorch operators support

## [0.15.5] - 2024-12-13

### Change

- `erf`, `hardswish` use tract NNEF core component if inference targeted.

## [0.15.4] - 2024-11-04

### Fix

- test suite working again for KhronosNNEF (full test suite green)
- hide some warning

### Change

- `export_tensors_to_nnef`, `export_tensors_from_disk_to_nnef` as root module access
- allow compression method to use gradients if needed
- expose ability to manage device in QTensor mechanism with `.to_device` in `QScheme` & `U8Compressor`
- better collision handling of tensor with different dtype in `QTensorTractScaleOnly`

### Added

- dump debug bundle with `KhronosNNEF` inference_target
- new option in cli `--no-verify` skip all correctness checks of exported LLM model
- new option in cli `--sample-generation-total-size` Number of tokens to generate in total for reference 'modes' samples npz dumped
- new option in compress quantization `min_max_q4_0_with_embeddings`

## [0.15.3] - 2024-10-16

### Fix

- implicit casting of dtype in mixed tensor math ops (better strategy)

### Change

- API of `llm_tract` compress registry functions

## [0.15.2] - 2024-10-14

### Fix

- bugs with weight_and_biases operators (linear, conv, ...) with new introduced NamedTensor

### Added

- API to export only specific tensors
- PEFT export cli support
- maintain order in NNEF `custom_extensions` (as some tract extensions are order sensitive)

## [0.15.1] - 2024-10-10

### Fix

- edge case of interaction between QTensor and NamedTensor
- f16 mix and allclose check

## [0.15.0] - 2024-10-09

### Change

- NNEF `variable` *label* values are now same as PyTorch module attributes naming, if Tensor are holded in any (sub-)modules

## [0.14.0] - 2024-10-08

### Added

- refactor of `llm_tract` into sub-modules
- added support for `modes` IO dump and checks

### Fix

- `intel` based `mac` tract export download correct CLI
- expand more robust
- align correctly all dimensional 'int' value as Int64
- force implicit mixed inputs dtype in PyTorch math operator to add explicit casting in exported graph
- `Phi3` export correctly

## [0.13.16] - 2024-10-01

### Fix

- `dynamic_axes` working for `Llama` model family

## [0.13.15] - 2024-09-24

### Fix

- slice with dyn axis edge case

## [0.13.14] - 2024-09-23

### Added

- Official support tract `0.21.7`

## [0.13.13] - 2024-09-20

### Fix

- Support QTensor for legacy (bellow 2.0), up to 1.12.0 <= torch

## [0.13.12] - 2024-09-18

### Fix

- flexible checks

## [0.13.11] - 2024-09-18

### Fix

- Split further functionalities & add some arguments as opt-in in LLM cli to add more reusable code

## [0.13.10] - 2024-09-18

### Fix

- (missfire) mkdir parents dir if needed while cache tract binary

## [0.13.9] - 2024-09-17

### Fix

- mkdir parents dir if needed while cache tract binary

## [0.13.8] - 2024-09-17

### Fix

- filter more possible *stdout* tract (avoid to land in *stderr*)
- tract inference target more robust with no subprocess shell=True and no *wget* needed
- in case of potential collision while merging graph and sub-graph during torch graph parsing, auto incrementation of variable name is performed

### Added

- `aten::linalg_norm` basic support for p=1 and p=2 added

## [0.13.7] - 2024-09-16

### Fix

- `export_llm_to_tract` API underlying no more need hugging face slug if only local dir.
- `export_llm_to_tract` log error if IO check wrong.

## [0.13.6] - 2024-09-11

### Fix

- `export_llm_to_tract` export cli more modular and reusable fn's

## [0.13.5] - 2024-09-11

### Fix

- `f16` export of LLM more stable (LayerNorm handling)
- more robust `export_llm_to_tract` export cli (+ full tokenizer, config export)

## [0.13.4] - 2024-09-09

### Fix

- `f16` export of LLM export correctly
- `Q4_0` accurately serialize to tract

## [0.13.3] - 2024-09-05

### Change

- QTensor inherit now from torch.Tensor and support any weight sharing

## [0.13.2] - 2024-08-27

### Fix

- add missing `arm64` in arch64 for tract downloader

## [0.13.1] - 2024-08-26

### Added

- `tract_llm` with various tract target support

### Change

- refactor `renaming_scheme` -> `nnef_variable_naming_scheme`

### Fix

- few remaining `nnef_spec_strict` replaced
- logger.warning for unofficially supported inference target fixed

## [0.13.0] - 2024-08-22

### Added

- Support for explicit `InferenceTarget` in core function `export_model_to_nnef` (so far 2 variants: `KhronosNNEF` and `TractNNEF`)
- Added `KhronosNNEF` test suite based on nnef-tool interpreter
- In case of `TractNNEF` binary management is handled internally (no more system wide `tract` reference)

### Change

- refactor tract within inference_target
- refactor module "primitives" as "aten"
- refactor class "NamedItemOrderedSet" as "ReactiveNamedItemDict"
- updated README in accordance with new exposed API

## [0.12.3] - 2024-08-21

### Added

- support for all variants of `torch.nn.functional.scaled_dot_product_attention`
- add GELU with `tanh` approximation option
- slice with out of bound reformulation, to allow tract to work (ie. [-100:] on a 50 size dim)
- new LLM pass: `Mistral` is passing, `Gemma2` pass but some IO diff

## [0.12.2] - 2024-08-19

### Added

- refactor NNEF variable naming in a ir_naming in module aside
- new NNEF variable naming scheme `natural_verbose_camel`
- added export IO support for dict/list/tuple of torch.Tensor via flattening
- added export IO support for other object via constantization (not part of graph `external`)

## [0.12.1] - 2024-08-09

### Added

- tract `Q4_0` support
- new `llm_tract` extension installable with `pip install torch_to_nnef[llm_tract]`
  - hold cli `export_llm_to_tract` for direct LLM export from any huggingface model with optional quant
  - replace `scripts` dir at root of the project
- added support for Python 3.12

### Removed

- dropped support for Python 3.8
- dropped support for unused QTensor formats

## [0.11.3] - 2024-07-26

### Added

- Tested export for `Llama`,`openELM`, `Phi`  LLM family works
- Added support aten::ops : `tril`, `repeat_interleave`, `type_as`
- Variable naming scheme: old `natural_verbose` option renamed `raw`,  new option `natural_verbose` means 'as close as possible' to torch Python code
- Protection against variable naming collision with `input_names`, `output_names`
- Updated NNEF `extensions` to comply to tract expectations

### Fix

- Improved support aten::ops : `index_` multi index gathering, `masked_fill`, `ones_like`
- added naming for models unit-tests, 'useful' in case of failures
- Compliance with tract>0.21.3 (introduced more restrictive definition within NNEF with different notation of scalar between float and TDim/long/int )
- Substantial performance improvement for internals graph IR (via by example new data-structures:  `NamedItemOrderedSet`)

## [0.10.2] - 2024-06-21

### Fix

- squeeze after getting shape slice to get scalar (specific to tract to get rank 0)

## [0.10.1] - 2024-04-19

### Fix

- better dynamic shape handling: remove realized shape from IR and adapt translation of slice accordingly

## [0.10.0] - 2024-04-17

### Removed

- drop python 3.7.0 support

### Added

- added `triu` export support
- script to export Llama2

### Fix

- Support aten::ops : `ones_like`, `zeros_like`, `arange` with dynamic shape
- Support aten::ops: `expand`, `masked_fill` with dynamic shape (but no tract support)
- more unit test of primitive
- fix poetry source pypi

## [0.9.1] - 2024-04-04

### Removed

- drop python 3.7.0 support
- updated tract version tested against: 0.19.16, 0.20.22, 0.21.3

### Added

- (*alpha*) `scripts/generate_qtensor_gguf_matmul.py` to generate unit tests with [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format for tract
- (*alpha*) `[gguf]` feature gate to support export to **GGUF** format and quantization
- (*alpha*) Support 2 new quantization tensor type (implemented as module for now):
  - `QTensorGGUF` support almost all **GGUF** data types -> with export prototype working
  - `QTensorSepParamsWithPack` more flexible than **GGUF** format, with support of classical per group with different sizes, per channel, per weight quantisation scheme at different bit-width 1, 2, 3, 4, 8 (useful for experimentation/accuracy simulation)
- move `[dev]` dependencies as a poetry **group**, to avoid exposition as packaged optional feature
- new `torch_version()` and `tract_version()` utility functions now allows for direct comparison to string version "`X.Y.Z`"
- Updated all tests packages torch/torch_audio/..., to torch `2.2` compatible 🎉
- added `weight_norm` export support

### Fix

- support for latest scaled_dot_product_attention aten version (last PyTorch version)
- quantization of bias as i32 at export for better support in tract (checked accuracy no-regression on bigger model)
- additional test for quantization with PyTorch different inputs q params activated (since last tract version merged related PR)
- custom_extractors have been refactored into sub-modules

## [0.8.11] - 2024-03-04

### Fix

- `linear`, `conv`, quantized operators accurately export bias to tract
- `activations`, quantized operators export output tensor scale / offset

## [0.8.10] - 2024-02-23

### Added

- `add`, `mul`, `div` element wise operators for quantized elements

### Fix

- `deconv` with group now export correctly to tract

## [0.8.9] - 2024-01-16

### Added

- `tract_core_external` in case of graph input being not i64, nor f32

## [0.8.8] - 2023-11-29

### Fix

- `rnn` states can now be manipulated in graph even in `multi-layers`

## [0.8.7] - 2023-11-29

### Fix

- `rnn` states can now be manipulated in graph
- `dynamic_axes` with tensor construction such as `zeros`, `ones` (and all related variants) now produce correct dynamic matrix

## [0.8.6] - 2023-10-27

### Fix

- `tract_core` NNEF extension added when using slice with dynamic_axes (to use `tract_core_shape_of`)
- `python 3.7` is now authorized again for the package even if no more supported

### Added

- `tract_extra` is added parameters to tract when running `check_same_io_as_tract` starting at tract 0.20.20

## [0.8.5] - 2023-09-12

### Added

- `PyTorch` v2 support
- Python `3.7` no more tested/supported as it is deprecated
- Support Python `3.8` to `3.11` tested/supported

## [0.8.4] - 2023-08-28

### Fix

- In case of `RNN`,`GRU`,`LSTM` we expand explicitly state initializers to batch dimensions (helping tract in case of some `dynamic_axes` graph formulation)
- Refactor of `torch_graph` module in sub-modules

### Added

- `hstack` and `vstack` support
- `unflatten` support
- `einsum` support

## [0.8.2] - 2023-08-02

### Fix

- slice with end being dynamic (akka max dimension size) given tract export target and dynamic_axes enabled

## [0.8.1] - 2023-08-01

### Fix

- fail if `tract` binary not found but `check_same_io_as_tract=True`
- better tract handling when `check_same_io_as_tract`
- disable fft's tests for now

### Added

- `Llama` partial export
- `_convolution_mode` aten operator (padding same and valid)

## [0.8.0] - 2023-05-01

### Added

- Refactored internals in primitive/quantized with submodule and registries
- `relu6`, `hardswish` activations

### Fix

- Support tract 0.19.15
- Support tract 0.20.4

### Removed

- deprecated support tract 0.17 (we support only last 3 majors)
- deprecated support of fft's ops prior to tract 0.20.0

## [0.7.7] - 2023-02-20

### Added

- add `narrow` support
- fix `copy` should not be used for tract
- `tile` akka expand allow dynamic dimension as repeat

## [0.7.6] - 2023-01-25

### Added

- complex support for `abs`
- `log10` ops supported
- `torchaudio.transform.MelSpectrogram` supported out of the box

## [0.7.5] - 2023-01-23

### Added

- `stft`, `fft`, `ifft` and basic complex number manipulations, torch now export to nnef with tract core experimental
  implementation in 0.19.0

## [0.7.4] - 2023-01-18

### Fix

- Avoid global log config setting in export module (restrict it to test)

## [0.7.3] - 2023-01-12

### Fix

- `aten:Int` catched even if not part of a list
- In case a float or an int is too big it use exponential notation and may trunk
  part of the number at serialization by example: `torch.finfo(self.dtype).min`
  (from huggingface transformers lib).

### Added

- `embedding` operator
- `Albert` model is passing

## [0.7.2] - 2023-01-11

### Fix

- dynamic_axes generated stream variables should be better casted to NNEF tensor ref

## [0.7.1] - 2023-01-10

### Added

- `roll`, `new_zeros`, `zeros` operators
- `pow` operator now support negative and scalars as exponent

### Fix

- `rsub` & `remainder` operator with constant should be precomputed output constants
- `avg_pool1d`, `avg_pool2d` operators now work as expected

## [0.6.10] - 2022-11-07

### Fix

- `aten:floor_divide` new op from torch 1.13 (torch 1.13 is passing)

## [0.6.9] - 2022-11-04

### Fix

- `aten:size` fix lost context for dyn shapes

## [0.6.8] - 2022-10-31

### Fix

- `aten:size` expand is now consistant in nameing pattern and should be more
  robust

## [0.6.7] - 2022-10-31

### Fix

- `aten:size` case with negative index is now translated correctly
- `...-pre` tract version are now handled correctly

## [0.6.6] - 2022-10-21

### Fix

- Handle case with no tract binary found ( thanks to Theo :tada: )

## [0.6.5] - 2022-10-20

### Fix

- Missing use of SONOS infra

## [0.6.4] - 2022-10-20

### Fix

- Push to SONOS repo as well

## [0.6.3] - 2022-10-19

### Fix

- `round` operator is now following tract core IEE implementation and warn if vanilla NNEF version is used
- `ipdb` is no more a dependency of this package
- bump to black formatter v22 (to avoid click raising errors)
- support tract > v0.18.0 (changed Conv1d bias expected shapes)

## [0.6.1] - 2022-09-27

- `baddbmm` operator is supported

### Fix

- all small fixes to have torch_to_nnef works with torch 1.12.0 and beyond (keeping backward compatibility)

## [0.6.0] - 2022-09-27

### Added

- `nnef_spec_strict` option in `export` allows to export strict the NNEF spec compliant model.
- `select`, `group_norm`, `erf` operators are supported.
- `gelu` was rewritten with `erf` fragment for precision.
- `ConvTasNet` is supported.
- `Wav2Vec2` encoder is supported.
- `VisionTransformer` (ViT) is supported.

### Fix

- negative index in `slice` are now handled for fixed dimensions

### Change

- `Exceptions` are now unified under T2NError

## [0.5.3] - 2022-09-08

### Change

- naming exported file with `.nnef` is no more required

## [0.5.2] - 2022-09-06

### Change

- update `nnef` deps with real original dir since poetry now support subdirectory
- tract v0.17.7 should make the CI tests pass again

## [0.5.1] - 2022-08-17

### Change

- update `nnef` deps

## [0.5.0] - 2022-08-16

### Change

- `aten:size` is now transformed in `tract_core_shape_of` which is against NNEF
  protocol specification but allow 'more' dynamic network to be expressed
- `aten:reshape` allow symbolic dims as parameters

## [0.4.0] - 2022-07-20

### Added

- `tensor.norm` with p 1 or 2
- `tensor.clamp_min(float)` and `tensor.clamp_max(float)`

### Fix

- fix nn.MultiHeadAttention case (not self attention) allow to export [Transpose](https://github.com/yangsenius/TransPose)

### Change

- torch quantize op lead to explicit `tract_core_cast` now

## [0.3.4] - 2022-05-06

### Fix

- expand can be expressed with negative values and repeat within rank dim
- Conformer Architecture now export correctly regardless the number of Attention Head

## [0.3.3] - 2022-05-02

### Fix

- Quantization info are passed correctly in case of type neutral information
  like ((un)squeeze, transpose, split).
- Dequantize is applied as a forced cast

## [0.3.2] - 2022-04-29

### Fix

- Arity was not properly tracked in some Subgraph expansion when parameter where
  flipped during torch optimization process (that modified ordering), this lead
  to wrong matching between io of graph and subgraph during recursive process.

- Div with an int type was not possible to cast implicitly to float by tract, to
  avoid rounding behavior missmatch we did had casting wrapper to handle such
  usecase properly.

### Added

- Better collected environment with OS, GCC, python and more package info
- Export Q8 Conv{1,2}d and Linear
- In Quantized network use scale/zero point of weight & **input** for bias export

## [0.3.1] - 2022-04-22

### Fix

- LogSofmax with negative value [#9](https://github.com/sonos/torch-to-nnef/issues/)
- switch-on cast test

### Added

- `dynamic_axes` in export API allowing to handle streaming dimensions
- Added aten::ops : `stack`, `unbind`,
- Filter `slice` if applied without effect (slice on full range)

## [0.3.0] - 2022-04-13

### Fix

- Rank expansion done right (`TRUnet` normalisations options works)
- TorchTrace optimization may from time to time change signature of `nn.Module` so we needed to take it into account in `torch_to_nnef.torch_graph` module.
- NNEF fragments file now express with their own extensions, this allows for finer
  grain export notation
- macos-latest OS removed from matrix test in CI since we have limited use (
 we will re-add it once tract latest version will be out
)

### Added

- Added aten::ops : `zeros_like`, `ones`, `expand`, `GLU`, `split`, `arange`, `chunk`, `layer_norm`, `trunc`, `masked_fill`, `clamp`, `to`
- Ability to export and unit-tested: `Deepspeech`, `Conformer`
- Ability to export `Wavenet`, `TDNN-ECAPA`
- Added LSTM with `projection`

## [0.2.2] - 2022-04-04

### Fix

- Fix base TRUNet
- Expose renaming scheme
- Add id to unittest for easier debug

## [0.2.1] - 2022-03-31

### Fix

- try correct parse with release workflow

## [0.2.0] - 2022-03-30

### Added

- Hook system on modules (allowing to avoid jit.trace expansion replaced by custom code )
- py.test Tract IO complaints added to errors
- better test representation
- LSTM/GRU/RNN handled (excepted LSTM with projection)
- Hard tanh
- ISO with tract check
- Logging with log level exposed
- TRUNet export
- debug bundling opt-in
- Numerous operators
- Q8 errors explorations

## [0.1.4] - 2022-03-17

### Fixed

- CI calibration finished

## [0.1.1] - 2022-03-17

### Added

- Support basic models conversion (if there is not quantized layers nor LSTM in it)
- CI is working with appropriate test suite (PyTorch->Tract ISO IO checked for ~80 cases)
- variable renaming scheme to keep NNEF generated files short

## [0.1.0] - 2022-02-28

- First release on Sonos Org.
