# Changelog

## Unreleased


## [0.8.0] - 2023-05-01

### Added
- Refactored internals in primitive/quantized with submodule and registries

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
- `round` operator is now following tract core IEE implem and warn if vanilla NNEF version is used
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
- try corect parse with release worflow

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
- debug bundling optin
- Numerous operators
- Q8 errors explorations

## [0.1.4] - 2022-03-17
### Fixed
- CI calibration finished

## [0.1.1] - 2022-03-17
### Added
- Support basic models conversion (if there is not quantized layers nor LSTM in it)
- CI is working with appropriate test suite (Pytorch->Tract ISO IO checked for ~80 cases)
- variable renaming scheme to keep nnef generated files short

## [0.1.0] - 2022-02-28

- First release on Sonos Org.
