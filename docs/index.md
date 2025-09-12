<figure markdown="span">
    ![torch_to_nnef](./img/torch_to_nnef.png){ align=center }
</figure>

# Welcome to the documentation


## Goals & Scope

`torch_to_nnef` Python package is used to export any model formulated with vanilla
PyTorch, whatever the internal tensor types (including quantized models), into [NNEF format](https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html).

[![tract](./img/tract.png){: style="width: 120px;margin:0;"}](https://github.com/sonos/tract/), the neural network inference engine
developed openly by [![SONOS](./img/sonos.png){: style="width: 80px;margin:0;"}](https://sonos.com), is the primary supported target,
and best compatibility with it is ensured. To use it, the [`TractNNEF`](/reference/torch_to_nnef/inference_target/tract/) inference_target must be specified.
This allows extended NNEF operators and specificities to be expressed:

- Transformer blocks
- recurrent layers (LSTM, GRU, …)
- dynamic streamable input dimensions
- data type casting (since NNEF spec is too vague in this regard)

Minimal dependencies are kept in this package (to allow easy integration in other projects).

### Support

PyTorch >= 1.10.0 with the last 2 major releases of tract (>= 0.20.22 to date) over Linux and MacOS systems is officially supported, and the package is maintained/tested for Python versions that are [not end of life, nor pre-release](https://devguide.python.org/versions/).
Only pre-compiled PyTorch wheels and dependencies available on [pypi](https://pypi.org/project/torch/) are used in CI, so this support evolves over time. Latest package versions ensure better opset coverage and unlock all features.


## Install

Today, the project is packaged in [PyPi](https://pypi.org/project/torch-to-nnef/).
Installation can be performed depending on the package manager:

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
    The project scope is broad and [contributions are welcome](./contributing/guidelines.md), if any bug is encountered, the [Bug report](./contributing/guidelines.md) instructions should be followed.
