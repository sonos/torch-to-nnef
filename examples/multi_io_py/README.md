# Multi-input / multi-output export (ALBERT)

[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-albert--base--v2-yellow)](https://huggingface.co/albert-base-v2) [![arXiv](https://img.shields.io/badge/arXiv-1909.11942-b31b1b.svg)](https://arxiv.org/abs/1909.11942)

Companion to the [Multiple inputs / outputs tutorial](https://sonos.github.io/torch-to-nnef/latest/tutos/3_multi_inputs_outputs/). Exports `albert-base-v2` (ALBERT, "A Lite BERT for Self-supervised Learning of Language Representations") with multiple input tensors (`input_ids`, `attention_mask`, `token_type_ids`) and the full last hidden state output.

## Run

```bash
cd examples/multi_io_py
pip install -r requirements.txt
python export_albert.py    # produces albert.nnef.tgz
```

The script wires the tokenizer, runs `export_model_to_nnef` with `input_names` and `output_names` matching ALBERT's forward signature, and verifies tract IO with `check_io=True`.
