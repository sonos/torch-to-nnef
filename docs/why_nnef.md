# :jigsaw: Why Use NNEF?

## :thinking: Wait, What Is NNEF?

NNEF stands for Neural Network Exchange Format.

<figure markdown="span">
    ![NNEF Idea](/img/nnef_idea.jpg){ align=center }
</figure>

Introduced in 2018—just a year after ![ONNX](/img/onnx.png){: style="width: 100px;margin:0;"}.

NNEF addresses the same core challenge as ONNX: providing a standardized way to exchange neural network models across different tools and frameworks.

<figure markdown="span">
    ![Khronos group](/img/khronos.png){: style="width: 360px;margin:0;" align=center}
</figure>

It is specified by the Khronos Group, an open, non-profit consortium of around 170 member organizations, better known for defining major graphics and compute standards such as WebGL, OpenCL, and Vulkan.

## :tools: Tools and Ecosystem

Beyond the specification itself, Khronos also provides several reference tools to enable partial model conversion (e.g., from TensorFlow or ONNX). However, these tools:

- Do not support PyTorch directly,

- And none offer the extensive support provided by this package.

!!! note "Note"
    We leverage these Khronos tools for final serialization within `torch_to_nnef` (thanks **Viktor Gyenes & al** for their continued support on [`NNEF-tools`](https://github.com/KhronosGroup/NNEF-Tools)).

## :brain: NNEF Inference Support

As of today, the only inference engine (excluding full training frameworks) that natively supports NNEF as a first-class format is [tract](github.com/sonos/tract) — the open-source neural inference engine developed by Sonos.

---

## :white_check_mark: The Good: What Makes the NNEF Specification Appealing

1. **Leverages Existing, Widely-Supported Containers**

    Stop reinventing the wheel—NNEF embraces common container systems.
    It's efficient, well-supported, and decouples data storage
    from model structure (think of video formats vs. codecs).

    - Example: `tar` is totally fine—and if you want compression, just apply it.
    - Prefer another container format? You're **free to use it**.

2. **Efficient Tensor Storage**

    Each tensor is stored as a binary `.dat` blob.

    - While `.npy` might seem more standard, `.dat` offers **better extensibility**.
    - The format supports **custom data types** with a:

        > 4-byte code indicating the tensor's item-type
        > (*Up to 4.2 billion possible custom types!*)

3. **Readable Graph Structure**

    The main `.nnef` file represents the model graph in a **simple, declarative, text-based format**:

    - No control flow complexity
    - Easy to read and edit (e.g., jump to definitions in your favorite editor)
    - **Flexible and extensible**—it's just **text**.

4. **Separation of Quantization Logic**

    Quantization metadata lives in a **separate `.quant` file**:

    - Defines **variables**, **quantization functions**, and **parameters**
    - Supports **advanced schemes** (e.g., Q40 per-group) via **custom data types**

5. **Textual Composition with Pure Functions**

    Neural-network are made of repetition of blocks (group of layers), the text format promotes **reusability**, avoids repetition, and enables a **clean functional structure**.


## :material-close: The Bad: Limitations of the NNEF Specification

1. **No Reference Implementation or Test Suite**

    Only basic converters exist (TensorFlow/ONNX), and a rudimentary interpreter in PyTorch—**nothing production-grade**.

2. **Image-Centric Design**

    The spec was initially tailored for **image inference tasks**, limiting its general applicability.

3. **Static Tensor Shapes**

    No support for **dynamic dimensions**.

4. **No Built-In Support for Recurrent Layers**

5. **Undefined or Poorly-Specified Data Types** for activations

6. **Stagnant Development**

    Last official update: **`v1.0.5` on 2022-02**


## :rocket: NNEF Extensions in Tract

1. **Supports Text and Signal Models**

    Through an **extended operator set**.

2. **Dynamic Shape Support**

    Enabled by **symbolic dimensions**.

3. **Advanced Data Type Handling**

    **Fine-grained, low-level types** are natively supported.

4. **Modular Subgraph Assembly**

    Enables **flexible architecture composition**.

> These extensions are encapsulated under the concept of **inference targets** in `torch_to_nnef`, allowing inference engines to define their own "NNEF flavor"—while retaining a shared **syntax and graph structure and common set of 'specified' operators**.

---

## :thinking: Why Not ONNX or Other Protocol Buffer-Based Formats?

!!! abstract

    Let's be clear: **ONNX is a great standard.**
    It's **mature**, **widely adopted**, and works well for many neural network applications.

However, ONNX is based on **Protocol Buffers**, which introduce real limitations—**even acknowledged in [their own docs](https://protobuf.dev/overview/)**:

1. **Not Suitable for Large Data Assets**

    > ... assume that entire messages can be loaded into memory at once and are not larger than an object graph. For data that exceeds a few megabytes, consider a different solution; when working with larger data, you may effectively end up with several copies of the data due to serialized copies, which can cause surprising spikes in memory usage.

2. **Inefficient for Large Float Arrays**

    > Protocol buffer messages are less than maximally efficient in both size and speed for many scientific and engineering uses that involve large, multi-dimensional arrays of floating point numbers ...

3. **No Built-In Compression**


### :material-robot-angry: Opinionated Grievances (Specific to NN Use Cases)

1. **Tightly Coupled Graph & Tensors**
   Want to patch a model with new PEFT weights or tweak a few parameters? **Good luck**—everything’s entangled.

2. **Unreadable Without Specialized Tools**
   Tools like [TensorBoard](https://www.tensorflow.org/tensorboard) or [Netron](https://netron.app/) are needed for visualization but **difficult to read** when more than 10 I/O tensors are linked to an operator (e.g having long residual connection deforms the graph visuals).

3. **No Direct Tensor Access**
   Requires **full graph parsing** and **multi-hop traversal**.

4. **Quantization definition is not very flexible**
   Especially for **custom formats** or precision below Q4.

5. **Extensibility is Harder**
   To add new data formats, you need change of protocol buffer spec, features like `symbols` definition in tract need to be defined ad-hoc. Adding plain text extensions is easier to do and read (at the cost of loosing code-gen ser/deser from protobuf). Prior PyTorch 2.0, adding custom ops (when it has no-equivalent chain of supported ops) is also tedious and partly unspecified.

---

## :vs: Safetensors

**Safetensors** is essentially a secure, structured list of tensors stored in binary—plus minimal metadata.

1. **Directly Loadable to Devices**

2. **Avoids Pickle Security Issues**

> 🔍 But: Its benefits are tied to loading efficiency—not the format itself.
> It could just as well have been implemented using `tar`.

### :material-close: Major Drawback

- **No Computation Graph**

    Every model architecture must be **re-implemented manually** on top of the inference engine—**error-prone** and **wasteful**.

- **No Operator Fusion or Optimization Guidance**

    That burden falls entirely on the implementer, **per model**.

---

## :vs: GGUF

**GGUF** is similar to `.safetensors`, but includes a lot of **quantization format definitions**.

1. **Vast choices of Quantization formats**

    Especially the **Q40 format**, which we've borrowed in `tract/torch_to_nnef`.

2. **Still No Graph Structure**

    **Just like `.safetensors`**, GGUF lacks a way to express **model computation graphs**.
