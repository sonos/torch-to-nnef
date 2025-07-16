# :+1: Add new aten / prim / quantized operator

!!! abstract "Goals"

    At the end of this tutorial you will be able to:

    1. :material-toolbox: Contribute a new operator support

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 5 min to read this page

There is a lot of operators in the PyTorch internal representation,
and while the most common are handled correctly, some
are still missing in `torch_to_nnef` due to the development of this library which
is done on a per need basis (aka we need to exort a new model, ow it
miss this and that operator let's implement it).

In this tutorial we will share with you how to contribute a new operator:
