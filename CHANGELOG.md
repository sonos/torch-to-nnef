# Changelog

## [0.3.0] - 2022-04-13

### Fix
- Rank expansion done right (`TRUnet` normalisations options works)
- TorchTrace optimization may from time to time change signature of `nn.Module` so we needed to take it into account in `torch_to_nnef.torch_graph` module.

### Added
- Added aten::ops : `zeros_like`, `ones`, `expand`, `GLU`, `split`, `arange`, `chunk`, `layer_norm`, `trunc`, `masked_fill`
- Ability to export and unit-tested: `Deepspeech`, `Conformer`
- Ability to export `Wavenet`, `TDNN-ECAPA`

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
