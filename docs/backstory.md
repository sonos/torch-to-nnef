# :popcorn: Backstory

At **Sonos**, we’ve long been committed to building machine learning solutions that span the full pipeline—from training all the way to inference—delivering smart experiences to millions of users. A core part of this journey has been enabling **on-device neural network computation**, a challenging but critical step in making our products more responsive, private, and reliable.

To support this goal, we developed and open-sourced [**Tract**](https://github.com/sonos/tract), a neural network inference engine written entirely in Rust. From the very beginning, Tract has used the **NNEF (Neural Network Exchange Format)** as its intermediate model representation. NNEF offered the right mix of practical benefits: it loads quickly, is human-readable, and is easy to extend—ideal for fast iteration and debugging. (We go into more detail about our choice in [Why NNEF](./why_nnef.md).)

Fast forward to early 2022. Like many others working on edge ML, we started experimenting with compression techniques—especially **quantization**. As models ballooned in size and complexity, the need to optimize for limited on-device resources became urgent. We knew that exporting quantized models was going to be essential to keep up with the scale and performance constraints of modern deployments.

However, when we looked at the state of ONNX at the time—especially its approach to quantization via QDQ (Quantize-Dequantize)—we found it lacking. It felt rigid and tacked on, not something we could easily extend or trust to support our long-term needs. So we took a shot at doing it differently.

In just two weeks, we built a prototype that exported **quantization-aware trained (QAT)** models from PyTorch directly into NNEF, fully readable by Tract. The result was eye-opening. We could now debug and extend graph operators without compiling protobuf or navigating through opaque binary formats. Everything was right there in a clean, readable NNEF file. It unlocked a new level of agility and transparency in our workflow, and we haven’t looked back since.

What started as a small side experiment turned into a powerful internal capability—one that continues to shape how we build and deploy machine learning models at Sonos.

