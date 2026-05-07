# Getting started: first NNEF export from Python

[![arXiv](https://img.shields.io/badge/arXiv-2010.11929-b31b1b.svg)](https://arxiv.org/abs/2010.11929) [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fvit--base--patch16--224-yellow)](https://huggingface.co/google/vit-base-patch16-224)

Minimal end-to-end example used by the [Getting started tutorial](https://sonos.github.io/torch-to-nnef/latest/tutos/1_getting_started/). Loads `torchvision.models.vit_b_16` (ViT-B/16, "An Image is Worth 16x16 Words"), exports to NNEF, and runs the standard tract round-trip check.

## Run

```bash
cd examples/getting_started_py
pip install -r requirements.txt
python export.py    # produces vit_b_16.nnef.tgz + a debug bundle
python run.py       # loads the archive and runs an inference
```

`export.py` calls `export_model_to_nnef(check_io=True)` so the export will fail loudly if the NNEF doesn't match PyTorch numerically. Companion Rust example: [`getting_started_rs/`](../getting_started_rs/).
