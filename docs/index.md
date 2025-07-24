# Welcome to **Torch to NNEF** documentation

<figure markdown="span">
    ![torch_to_nnef](./img/torch_to_nnef.png){ align=center }
</figure>

## Goals & Scope

This Python package allows to export any model formulated with vanilla
PyTorch whatever the internal tensor types (handling quantized model) into [NNEF format](https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html).

[![tract](./img/tract.png){: style="width: 120px;margin:0;"}](https://github.com/sonos/tract/), the neural network inference engine
developed openly by [![SONOS](./img/sonos.png){: style="width: 80px;margin:0;"}](https://sonos.com) is our primary supported target,
and we strive best compatibility with it. To use it you need to specify [`TractNNEF`](/reference/torch_to_nnef/inference_target/tract/) inference_target.
This unlock extended NNEF operators and specificities to express:

- Transformers blocks
- recurrent layers (LSTM, GRU, ...)
- dynamic streamable input dimensions
- data type casting (since NNEF spec is too vague in this regard)

This package keeps minimal dependencies (to allow easy integration in other project).

We support officially PyTorch > 1.8.0 with tract last 2 major releases patched (>= 0.20.22 to date) over Linux and MacOS systems.

## Install

Today, the project is packaged in internal SONOS [Artifactory](https://jfrog.com/artifactory/),
please be sure to have it configured (replace `user` and `pass` accordingly):

```ini title="$HOME/.pip/pip.conf"
extra-index-url=https://{user}:{pass}@redacted.com/artifactory/api/pypi/pypi-local/simple
```

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
    The project scope is broad and [contributions are welcome](./contributing/guidelines.md), if you encounter any bug please follow the [Bug report](./contributing/guidelines.md) instructions
