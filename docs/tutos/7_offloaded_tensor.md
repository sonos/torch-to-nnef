# 7. Offloaded Tensors

!!! abstract "Goals"

    At the end of this tutorial you will be able to:

    1. :material-toolbox: Use offloaded tensors when wished

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 5 min to read this page

Offload tensors have been developed to allow to manipulate and export more
easily large neural network models.

Recall that if you only want to export a LLM model offloaded you can look at [our
related LLM tutorial](./5_llm.md) and do not need to look at what happen behind.

This class is defined as such:

<div class="grid cards" markdown>
- ::: torch_to_nnef.tensor.offload.OffloadedTensor
    handler: python
</div>

You can directly load any *.safetensor* or *.pt* into this object that will mimic classical
`torch.Tensor` except that each access will load the Tensor from disk and remove it from RAM as
soon as those are not needed, allowing to manipulate very large model bit by bit.
It is composable with other `torch_to_nnef.tensor.opaque.OpaqueTensor` such as `QTensor`.

To load from disk without overhead,
you can call the `t2n_load_checkpoint_and_dispatch` with appropriate options like in the following example:

```python title="example of offload usage from disk (extracted from LLM exporter)"
import tempfile
from pathlib import Path
from torch_to_nnef.tensor.offload import (
    ON_DISK_DEVICE_MAP_KEY,
    t2n_load_checkpoint_and_dispatch,
)
from torch_to_nnef.utils import init_empty_weights

from transformers import AutoModelForCausalLM
import huggingface_hub

slug = "meta-llama/Llama-3.2-1B-Instruct"
with init_empty_weights():
    # model instantiation with empty tensors
    # this can be come from any library (here transformers)
    model = AutoModelForCausalLM.from_pretrained(slug, **kwargs)
hf_repo_files = huggingface_hub.list_repo_files(slug)
weights_location = Path(
    huggingface_hub.hf_hub_download(
        slug, hf_repo_files[-1]
    )  # assume at least 1 file is in targeted repo
).parent

# here model tensors are properly loaded into
t2n_load_checkpoint_and_dispatch(
    model,
    weights_location,
    device_map=ON_DISK_DEVICE_MAP_KEY,
    offload_dir=Path(tempfile.mkdtemp(suffix="offload_t2n")),
)
```

These `OffloadedTensor` are also very useful to implement into quantization techniques to
support very large model quantization with a calibration based on observed values like Hessian from activation.
Indeed if we think of the Hessian example: these square matrices can be pretty large especially
when multiplied by the number of activations on a big neural network.

If you only wish to maintain QTensor into OffloadedTensor if original float
tensor was offloaded you can just use the helper:

<div class="grid cards" markdown>
- ::: torch_to_nnef.compress.offloaded_tensor_qtensor
    handler: python

</div>

If this is a new tensor just use the `OffloadedTensor.from_original_tensor` defined upper.

## Scoped residency and prefetch

Repeated operations on offloaded values can retain them in memory with
`TensorResidencyPool`. Activating the pool for an execution scope makes normal
tensor operations use resident values transparently, while `prefetch` starts
loading a later value on a background worker.

```python
from torch_to_nnef.tensor import TensorResidencyPool

with TensorResidencyPool(max_resident_bytes=2 * 1024**3) as pool:
    with pool.scope(device="cuda"):
        pool.prefetch(next_weight, device="cuda")
        output = input @ current_weight
        output = output @ next_weight
```

This allows a model or module hook to schedule the next tensor without changing
the operations that consume the current one. Use an explicit `read_write`
lease when an operation changes the resident tensor. The pool writes dirty
values back before eviction, during `flush`, or when the pool closes.

```python
with pool.acquire(statistic, mode="read_write") as value:
    value.add_(update)
```

The optional byte budget uses least-recently-used eviction. Active leases are
never evicted, so an individual leased value may temporarily exceed the
budget. Scheduling decisions, such as which model block to prefetch next,
remain with the caller.
