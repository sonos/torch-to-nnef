# Dynamic axes patterns

Companion to the [Dynamic axes tutorial](https://sonos.github.io/torch-to-nnef/latest/tutos/4_dynamic_axes/). Three small models exercising the three common dynamic-axes flavors: a fixed-arity transformer, a batchable image model, and a streaming audio model.

## What's exported

| Script | Model | What's dynamic | References |
| --- | --- | --- | --- |
| `export_albert_fixed.py` | `albert-base-v2` | sequence length | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-albert--base--v2-yellow)](https://huggingface.co/albert-base-v2) [![arXiv](https://img.shields.io/badge/arXiv-1909.11942-b31b1b.svg)](https://arxiv.org/abs/1909.11942) |
| `export_with_batchable.py` | `torchvision.models.vit_b_16` | batch dimension | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-google%2Fvit--base--patch16--224-yellow)](https://huggingface.co/google/vit-base-patch16-224) [![arXiv](https://img.shields.io/badge/arXiv-2010.11929-b31b1b.svg)](https://arxiv.org/abs/2010.11929) |
| `cnn_deepspeech_stream.py` | `torchaudio.models.DeepSpeech` + custom CNN front-end | streaming time axis | [![arXiv](https://img.shields.io/badge/arXiv-1412.5567-b31b1b.svg)](https://arxiv.org/abs/1412.5567) |

## Run

```bash
cd examples/dynamic_axes
pip install -r requirements.txt
python export_albert_fixed.py
python export_with_batchable.py
python cnn_deepspeech_stream.py
```

Each script passes `dynamic_axes={...}` to `export_model_to_nnef` and verifies the resulting NNEF round-trips against tract.
