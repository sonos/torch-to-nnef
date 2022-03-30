# Changelog

## [0.2.0] (2022-03-30)

### Added

* Hook system on modules (allowing to avoid jit.trace expansion replaced by custom code )
* py.test Tract IO complaints added to errors
* better test representation
* LSTM/GRU/RNN handled (excepted LSTM with projection)
* Hard tanh
* ISO with tract check
* Logging with log level exposed
* TRUNet export
* debug bundling optin
* Numerous operators
* Q8 errors explorations

## [0.1.4] (2022-03-17)

### Fixed

* CI calibration finished

## [0.1.1] (2022-03-17)

### Added

* Support basic models conversion (if there is not quantized layers nor LSTM in it)
* CI is working with appropriate test suite (Pytorch->Tract ISO IO checked for ~80 cases)
* variable renaming scheme to keep nnef generated files short

## [0.1.0] (2022-02-28)

* First release on Sonos Org.
