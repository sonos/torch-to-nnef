# Welcome to ![Torch to NNEF](./docs/img/torch_to_nnef.png) documentation

<figure markdown="span">
    ![torch_to_nnef](./img/torch_to_nnef.png){ align=center }
</figure>
## Goals & Scope

This python package allow to export any model formulated with vanilla
PyTorch whatever tensor type (handling quantized model) into [NNEF format](https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html). Depending on selected NNEF inference target, adjustment are made to enable maximum support.

For example, `TractNNEF` unlock [tract](https://github.com/sonos/tract/) operators and specificities to express:

- Transformers blocks
- recurrent layers (LSTM, GRU, ...)
- dynamic streamable input dimensions
- data type casting (since NNEF spec is too vague in this regard)

This package strives to have minimum dependencies (to allow easy integration in other project).

We support PyTorch > 1.8.0 with tract last 2 major releases (>= 0.20.22 to date) over Linux and MacOS systems.

!!! IMPORTANT
    This project is still in it's early stage, if you encounter any bug please follow [Bug report](#bug-report)  section instructions
