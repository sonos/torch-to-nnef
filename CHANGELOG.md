<!-- markdownlint-disable-file MD001 MD013 MD024 -->
# Changelog

## Unreleased

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
