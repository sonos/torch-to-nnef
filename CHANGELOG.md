# Changelog

## Unreleased

### Added
- `tensor.norm` with p 1 or 2
- `tensor.clamp_min(float)` and `tensor.clamp_max(float)`
- torch quantize op lead to explicit `tract_core_cast` now

### Fix
- fix nn.MultiHeadAttention case (not self attention) allow to export [Transpose](https://github.com/yangsenius/TransPose)

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
