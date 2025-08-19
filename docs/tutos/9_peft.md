# 9. Export selected tensors & PEFT

!!! abstract "Goals"

    At the end of this tutorial you will be able to:

    1. :material-toolbox: Export specific set of weight tensors without the graph.

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 5 min to read this page

Sometime you wish to export only some weights, this happen for example if you
fine-tuned a LLM with a PEFT technique like LoRA. In this case only a very limited
set of weights are modified compared to the pre-trained neural network.
So this can make sense to ship to the customer the base model only once (with the graph)
at the beginning (for example app download),
and then to send the LoRA weights separately as you iterate through your product (for example app update).

This patching logic is also very interesting in case you wish to allow multiple PEFT
to be shipped and switched/selected at model load time.

To do so `torch_to_nnef` provide few convenient methods.

### Command line

First if you happen to exactly want to export **PEFT** weights, we have a CLI for you:

```bash
# filepath can be .pth, .bin, .safetensors ...
t2n_export_peft_to_nnef \
    --read-filepath /my-little-model-file.pt \
    -o /tmp/my-outputdir
```

By default it exports **LoRA** weights, if you wish to apply it on different methods look
at additional options (with `--help`), the core functionality behind this CLI is simple
pattern matching so most of PEFT weight names matching with regex should work (DoRA, ...).

### Basic API

If you wish to have programmatic control you can also use for on disk tensors:

<div class="grid cards" markdown>
- ::: torch_to_nnef.export_tensors_from_disk_to_nnef
    handler: python
</div>

Or from loaded tensors:
<div class="grid cards" markdown>
- ::: torch_to_nnef.export_tensors_to_nnef
    handler: python
</div>
