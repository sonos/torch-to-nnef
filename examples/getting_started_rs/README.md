# Getting started: first NNEF inference in Rust with tract

[![arXiv](https://img.shields.io/badge/arXiv-2010.11929-b31b1b.svg)](https://arxiv.org/abs/2010.11929) [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fvit--base--patch16--224-yellow)](https://huggingface.co/google/vit-base-patch16-224)

Rust counterpart to [`getting_started_py/`](../getting_started_py/). Loads the NNEF archive produced there (ViT-B/16 ImageNet classifier) and runs an inference through [tract](https://github.com/sonos/tract).

## Run

First export the NNEF artifact via the Python example, then:

```bash
cd examples/getting_started_rs
cargo run --release -- ../getting_started_py/vit_b_16.nnef.tgz Grace_Hopper.jpg
```

Prints the predicted ImageNet class id and label.
