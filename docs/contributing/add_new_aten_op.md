# :+1: Add new aten / prim / quantized operator

!!! abstract "Goals"

    At the end of this tutorial you will be able to:

    1. :material-toolbox: Contribute a new operator support

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 5 min to read this page

There is a lot of operators in the PyTorch internal representation.
[Aten](https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html) is the underling cpp namespace in which most of the PyTorch computational operators are specified.
[Looking at the list](https://docs.pytorch.org/docs/main/torch.compiler_ir.html) in the PyTorch IR it may seems at first there is only: 191 ops available, (not accounting quantized & prims namespace) but digging  with few scripts we maintain a  [generated compatibility list](./supported_operators.md) that seems to show something different (~500 ops).

While the most common are handled correctly, this list is ever expanding or an operator receive new arguments, so we alway need to catch-up when a new model end up adopting one of those.

In the development of this library we did thing on a per need basis (aka we need to export a new model,
ow it miss this and that operator let's implement it).

In this tutorial we will share with you how to contribute a new operator:
