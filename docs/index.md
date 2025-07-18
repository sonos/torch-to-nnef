# Welcome to **Torch to NNEF** documentation

<figure markdown="span">
    ![torch_to_nnef](./img/torch_to_nnef.png){ align=center }
</figure>

## Goals & Scope

This python package allows to export any model formulated with vanilla
PyTorch whatever tensor type (handling quantized model) into [NNEF format](https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html).

[![tract](./img/tract.png){: style="width: 120px;margin:0;"}](https://github.com/sonos/tract/), the neural network inference engine
developed openly by [![SONOS](./img/sonos.png){: style="width: 80px;margin:0;"}](https://sonos.com) is our primary supported target,
and we strive best compatibility with it. To use it you need to specify `TractNNEF` inference_target
, this unlock extended NNEF operators and specificities to express:

- Transformers blocks
- recurrent layers (LSTM, GRU, ...)
- dynamic streamable input dimensions
- data type casting (since NNEF spec is too vague in this regard)

This package strives to have minimum dependencies (to allow easy integration in other project).

We support PyTorch > 1.8.0 with tract last 2 major releases (>= 0.20.22 to date) over Linux and MacOS systems.

## Install

To install it you can run depending on your package manager:

=== "pip"

    ```bash
    pip install torch_to_nnef
    ```

=== "uv"

    ```bash

    uv add torch_to_nnef
    ```

=== "poetry"

    ```bash
    poetry add torch_to_nnef
    ```

!!! note
    This project is still in it's early stage, if you encounter any bug please follow [Bug report](./CONTRIBUTING.md)  section instructions
