# :jigsaw: Why use NNEF ?

Wait but **what** is NNEF ?

It's stand for Neural Network Exchange Format.

<figure markdown="span">
    ![NNEF Idea](/img/nnef_idea.jpg){ align=center }
</figure>

Landed in 2018 (1 year after ![ONNX](/img/onnx.png){: style="width: 100px;margin:0;"}), it solves the same problem as ONNX & the specification is developed by the

<figure markdown="span">
    ![Khronos group](/img/khronos.png){: style="width: 360px;margin:0;" align=center}
</figure>

An open, non-profit member driven consortium of ~170 companies.

They are better known for standardizing the Graphic Rendering
(WebGL, OpenCL, Vulkan, …).

Beyond the specification, they provides few [code tools](https://github.com/KhronosGroup/NNEF-Tools)
, that allow to do some partial conversions (by example from Tensorflow or ONNX).
But none directly from PyTorch and none has complete as we propose in this
package. NOTE that we use those package to apply final serialization in `torch_to_nnef`.
Also to the best of our knowledge the only inference engine supporting NNEF (that is not a full blown training framework) as first class citizen is [tract](github.com/sonos/tract) the SONOS open source neural inference engine.

## NNEF Specification the good

- Stop reinvent the wheel, just use container system that are used everywhere, this is as efficient and more supported (Think video container/formats decoupling):
tar is fine, and you want you can compress it.
If another container solution is better for you, just use it.

- Split each tensor in a binary blob .dat
    (npy would have felt more standard but less extensible …)
    Format specified allows to add custom data types:

> 4-byte code indicating the item-type of the tensor
>
> (4,2G possible custom types)

- A .nnef graph representation in textual readable format
like  any declarative language, simple no control flow, still
Flexible to extend that’s just TEXT (read it inside your favorite editor jump to definitions ...).

- Quantization can be expressed as another textual file aside .quant
With variable name and quantization function used and parameters listed
In case of advanced quantization scheme like Q40 (per group) we can use custom data type
.

- The textual representation allow for composition of pure functions avoiding anoying
repeat and better formalism.

## NNEF Specification the bad

✗   No reference implementation or test-suite
 (only converters for basic models from TF, and ONNX, and an ’interpreter’ in PyTorch ?)

✗   Designed mostly for image predictions

✗  Static dimension tensors

✗  No recurrent layers provided

✗  Undefined data-types

✗  A spec last updated v1.0.5 the 2022-02

## NNEF Tract extension

- Unlock text & signals neural networks thanks to extended operator set
- Allow dynamic shapes thanks to symbol introductions.
- Fine-grained data-types are correctly handled
- Multiple subgraph assembly possible

Those alterations/extensions of the specification are encapsulated inside the 'inference target'
notion in `torch_to_nnef` allowing each inference engine to define it's own NNEF flavor,
while maintaining broad base of operators and syntax/global format the same.

## Why not ONNX or any other protocol buffer spec ?

!!! abstract
    First off let's be clear ONNX is good enough for a wide variety
    of Neural Network usecase.
    It's the defacto standard in the industry we are **not** denying it.
    Also the tooling around it is mature and more optimized than for NNEF (for general purpose).

That said.

ONNX is a protocol buffer specification and protocol buffer suffer from problems when creating neural networks asset as stated in their [own official documentation](https://protobuf.dev/overview/):

1.

> Protocol buffers  assume entire messages can be loaded into memory at once and are not larger than an object graph. {==For data that exceeds a few megabytes, consider a different solution;==} when working with larger data, you may effectively end up with several copies of the data due to serialized copies, which can cause surprising spikes in memory usage.

2.

> Protocol buffer messages are {==less than maximally efficient in both size and speed for many scientific and engineering uses that involve large, multi-dimensional arrays of floating point numbers.==} For these applications, FITS and similar formats have less overhead.

3.

> Messages are not compressed

There is more argument on their own page but the gist is here.

### Opinionated grief

Additional grief against it for the NN use case:

1. Too tight coupling between tensors & graph
    (what if I just want to patch a model with my latest PEFT, or just change few finetuned parameters or ...)

2. Used as intermediate representation but unreadable without dedicated tools
([tensorboard](https://www.tensorflow.org/tensorboard?hl=fr) or [Netron](https://netron.app/) viewers become unreadable as soon as you have more than 10 variables in or out an operator, not even speaking about residual connections).

3. Can not access specific tensors without parsing the full graph and doing multiple hop inside it (seek).

4. Doing quantization with it is painful especially as you go bellow Q4 or have custom
quantization operations/methods.

5. Extensibility is still lagging behind an will always be harder to do than on a plain text file.

## vs .safetensors

Safetensor is a gloried list of tensor saved in a binary file that can be reloaded directly to the right device without pickle risks (+ a bit of metadata).
Most of the benefit is this direct device addressing unrelated to the format itself
It could as well as been tar archive, but well ...

The big issue is that it does not hold any information about the graph of computation that is the NN itself,
this lead to rewriting every single architecture on inference side (error prone & uselessly time consuming),
while also leaving the responsibility to optimize/fuse operators to the 'implementer' on a per model basis (no consolidation).

## vs GGUF

GGUF has the same limitation as *.safetensors* with the additional benefit that they define a lot of custom formats.
To be clear we like what they propose in term of quantization format and even borrow Q40 in tract/torch_to_nnef from theirs,
but no graph to the horizon.
