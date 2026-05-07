# Quantization

Companion to the [Quantization tutorial](https://sonos.github.io/torch-to-nnef/latest/tutos/6_quantization/). Two scripts:

- **`export_toy_cnn_8bit.py`**: minimal int8 PTQ flow on a toy `nn.Conv1d` stack via `torch.ao.quantization` (eager-mode), then exported to NNEF.
- **`super_quant.py`**: utility for grid-search MSE calibration of weight-only `q4_0` quantization (tract's 4-bit format), built on `torch_to_nnef.tensor.quant`.

No specific upstream model: the toy CNN is constructed in-script. `super_quant.py` is a calibration helper meant to be applied to your own LLM / encoder weights as part of an export pipeline.

## Run

```bash
cd examples/quantization_py
pip install -r ../getting_started_py/requirements.txt   # any t2n env works
python export_toy_cnn_8bit.py
```

For `super_quant.py`, see the tutorial for usage in context.
