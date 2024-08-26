<!-- markdownlint-disable-file MD001 MD013 MD024 -->
# Changelog

## Unreleased

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

- (_alpha_) `scripts/generate_qtensor_gguf_matmul.py` to generate unit tests with [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format for tract
- (_alpha_) `[gguf]` feature gate to support export to **GGUF** format and quantization
- (_alpha_) Support 2 new quantization tensor type (implemented as module for now):
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

- `Exceptions` are now unified under TorchToNNEFError

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

- LogSofmax with negative value [#9](https://github.com/sonos/torch-to-nnef/issues/9)
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
