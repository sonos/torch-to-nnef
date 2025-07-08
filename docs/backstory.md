# :popcorn: Backstory

At SONOS we build machine learning solutions from training to inference while serving millions of customers.
We have been pushing hard to allow neural network computation to happen on device,
 and as part of this journey we open-sourced and maintain [tract](https://github.com/sonos/tract) our full rust neural network inference.

Tract use NNEF format as intermediate storage since it's early days since this format allow to
have faster model load time, good human readability, is easy extensible ... (see [Why NNEF](./why_nnef.md) for more informations).

In early 2022 we started to investigate use of compression techniques and in particular quantization.
This ability to export quantized models is critical because of limited on-device resources and
the advent of multi-billions parameters models.

Unfortunatly the state of [ONNX](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) in that regard, is unsatisfactory (and was worse back then),
QDQ in particular lack flexibility and feel like and ad-hoc design. We took a shot tried to build a prototype to allow exporting QAT
models into tract, it took us 2 weeks and unlocked ability to debug and extend graph operators with such ease (no more compiling protobuf :tada:, readable NNEF format)
that it leads us continuing this work until today.
