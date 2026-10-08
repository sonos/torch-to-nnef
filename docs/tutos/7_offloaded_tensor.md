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
from torch_to_nnef.tensor import ResidencyStrategy, TensorResidencyPool

with TensorResidencyPool(max_cached_bytes=2 * 1024**3) as pool:
    with pool.scope(device="cuda"):
        pool.prefetch(next_weight, device="cuda")
        output = input @ current_weight
        output = output @ next_weight
```

This allows a model or module hook to schedule the next tensor without changing
the operations that consume the current one.

The first access to an `OffloadedTensor` through a pool that does not yet
contain it is a cache miss: the pool materializes it from disk on the selected
device. With the default `ResidencyStrategy.LRU`, a value that fits within
`max_cached_bytes` becomes cached, and later accesses through the same pool
reuse it without another disk load. It remains cached until explicit eviction,
least-recently-used eviction to make room for another value, or pool closure.
`prefetch` starts this first load early but otherwise follows the same retention
rules. This caching is specific to access routed through the pool; other access
to an `OffloadedTensor` keeps its normal transient materialization behavior.

LRU performs poorly for a cyclic scan whose working set is larger than the
cache: values can be evicted shortly before the next forward pass needs them.
`ResidencyStrategy.FIXED` instead admits the first completed values that fit in
the budget and does not replace them during later accesses. Values that do not
fit are still delivered to their callers but are not retained. This provides a
stable resident subset while the rest of each forward pass streams through the
pool.

```python
with TensorResidencyPool(
    max_cached_bytes=2 * 1024**3,
    strategy=ResidencyStrategy.FIXED,
) as pool:
    with pool.scope(device="cuda"):
        for batch in calibration_batches:
            calibrate(model, batch)
```

With the default single worker, fixed admission follows request order. With
multiple workers, concurrently prefetched values are admitted in completion
order. Explicit `evict` can remove an admitted value and make capacity
available for a later materialization. Pool closure releases the full set.

Manual `pin` and `unpin` apply only to `ResidencyStrategy.LRU`; the FIXED
strategy rejects them with `T2NErrorMisuse` because it already controls fixed
admission. Under LRU, pinning starts or reuses a load and is idempotent. A
pinned value is excluded from LRU eviction until `unpin` is called or the pool
closes, counts toward `resident_bytes` and the cache budget, and can make
residency exceed the soft budget.

A lease is a scoped claim that a materialized tensor is currently in use. It
lasts for the duration of the `with pool.acquire(...)` block. While any lease
is active, the pool keeps that value resident and does not evict it. Multiple
callers may lease the same resident value. Use a `read_write` lease when an
operation changes the value; the pool writes dirty values back before
eviction, during `flush`, or when the pool closes.

```python
with pool.acquire(statistic, mode="read_write") as value:
    value.add_(update)
```

`max_cached_bytes` is a soft limit on payloads retained for reuse, not a hard
limit on process or device memory. LRU enforces it by evicting unleased,
unpinned values in least-recently-used order. FIXED preserves admitted values
and removes non-admitted values after delivery. If an active caller leases a
value larger than the cache budget, the pool still materializes it so the
operation can proceed, keeps it resident until the lease ends, and then evicts
it instead of caching it. A prefetched oversized value is similarly delivered
to its waiting caller without being retained. Consequently, active values and
temporary framework allocations can exceed `max_cached_bytes`; callers that
require a hard allocation limit must validate their working-set sizes
separately.
The `torch_to_nnef.tensor.residency` logger emits these decisions at `DEBUG`
level, including whether the value is not admitted by FIXED, delivered without
cache retention, kept until its final lease ends, or retained by an LRU pin.

Scheduling decisions, such as which model block to prefetch next, remain with
the caller. The pool only manages `OffloadedTensor` values. Passing a regular
`torch.Tensor` to `acquire`, `prefetch`, `resolve`, `pin`, `unpin`, `flush`, or
`evict` raises `T2NErrorMisuse` immediately; ordinary tensors are already
materialized and do not need the residency layer.
