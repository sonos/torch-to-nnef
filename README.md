<!-- markdownlint-disable-file MD001 MD013 MD014 MD024 -->
# ![Torch to NNEF](./docs/img/torch_to_nnef.png)

[![dev workflow](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml/badge.svg?branch=main)](https://github.com/sonos/torch-to-nnef/actions/workflows/dev.yml) ![python](https://img.shields.io/badge/python-%3E=3.9-green)[![documentation](https://img.shields.io/badge/torch_to_nnef-documentation-blue)]( https://sonos.github.io/torch-to-nnef/)

Any PyTorch Model or Tensor to NNEF file format.

> [!IMPORTANT]
> If you encounter any bug please follow [Bug report](#bug-report)  section instructions

## Goals & Scope

We intend to export any model formulated with vanilla Torch whatever tensor type
(handling quantized model).

Depending on selected NNEF inference target, adjustment are made to enable maximum support.

For example, `TractNNEF` unlock [tract](https://github.com/sonos/tract/) operators and specificities to express:

- recurrent layers (LSTM, GRU, ...)
- dynamic streamable input dimensions
- data type casting (since NNEF spec is too vague in this regard)

This package strives to have minimum dependencies (to allow easy integration in other project).

We aim to support PyTorch > 1.8.0 with tract last 2 major releases (>= 0.20.22 to date) over Linux and MacOS systems.

## Documentation

All documentation is available here:

[![documentation](https://img.shields.io/badge/torch_to_nnef-documentation-blue)]( https://sonos.github.io/torch-to-nnef/)
