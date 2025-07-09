# 3. Model with multiple inputs or outputs

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-toolbox: How to export multi io neural network
    2. :octicons-cross-reference-24: limitations of the NNEF

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 5 min to read this page

Some neural network model requires more than 1 input or output.
In that case we support no less that classical ONNX export.

### How to do ?

### Limitations

#### Dynamic number of input or output is not supported out of the box

Given your network have a variable number of input or outputs
with the same shape you can envision to wrap your network inside
a `torch.nn.Module` that will concatenate those into a single tensor
of variable size.

If the shapes are of varying size preventing the direct concatenation,
you can pad your tensors before and add another tensor responsible to keep
track of your padding values.

While suboptimal in RAM and compute, it should allow to express any possible
network in that respect.

#### Python Object and primitives

All python object and primitive are ignored during the export step that trace the
graph. That means that if you wish to keep part of those parameters dynamic at runtime you need
to wrap your network inside a `torch.nn.Module`, that: expose each tensor primitive to be assigned
inside those object in the `forward` function.
