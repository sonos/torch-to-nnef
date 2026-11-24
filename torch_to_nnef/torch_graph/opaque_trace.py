"""Retry unsafe opaque-weight JIT boundaries with concrete expansion.

Quantized and offloaded weights normally use fake tensors during JIT tracing
so their floating-point values do not need to be materialized. A fake
activation reaching an ordinary named weight or buffer interrupts the trace
before PyTorch's real-to-fake conversion can trigger a JIT internal assertion.
The checkpoint restores tensor values, registered buffer bindings, and
CPU/initialized-CUDA RNG state before one concrete-expansion retry. Unrelated
tracing errors are not retried.

The retry keeps custom-op identities and metadata-only parameter references,
so exported weights retain their packed format. Unlike legacy tracing, it
does not eagerly build dense references for every opaque parameter. Temporary
decompression, input/buffer snapshots, and real computation still increase
tracing memory and work; a warning reports this tradeoff. Supported fake
traces continue without materializing weights.

T2N_LEGACY_OPAQUE_TRACING=1 is not required for this named-operand boundary,
but remains a compatibility option for other unsupported tracing paths.
Inference forwards must not mutate parameters or rely on arbitrary Python
side effects: the tensor-state checkpoint cannot roll those effects back.
"""

import logging

import torch

from torch_to_nnef.exceptions import T2NErrorOpaqueTraceBoundary
from torch_to_nnef.tensor import opaque
from torch_to_nnef.utils import select_ctx_disable_torch_fn

LOGGER = logging.getLogger(__name__)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _plain(tensor):
    with select_ctx_disable_torch_fn():
        return tensor.as_subclass(torch.Tensor)


class _TraceCheckpoint:
    """Restore tensor inputs, registered buffers and RNG before a retry.

    As for JIT tracing generally, forwards must not rely on arbitrary Python
    side effects. Parameter mutation during inference is not supported.
    """

    def __init__(self, module, args):
        self.buffers = [(m, dict(m._buffers)) for m in module.modules()]
        tensors = list(_tensors(args)) + [
            value
            for _, buffers in self.buffers
            for value in buffers.values()
            if isinstance(value, torch.Tensor)
        ]
        self.values = {}
        with torch.no_grad():
            for tensor in tensors:
                if id(tensor) not in self.values and not isinstance(
                    tensor, (opaque.OpaqueTensor, opaque.OpaqueTensorRef)
                ):
                    self.values[id(tensor)] = (tensor, _plain(tensor).clone())
        self.cpu_rng = torch.random.get_rng_state()
        self.cuda_rng = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else None
        )

    def restore(self):
        with torch.no_grad():
            for tensor, saved in self.values.values():
                plain = _plain(tensor)
                if torch.equal(plain, saved):
                    continue
                if plain.shape != saved.shape:
                    plain.resize_(saved.shape)
                plain.copy_(saved)
        for module, buffers in self.buffers:
            module._buffers.clear()
            module._buffers.update(buffers)
        torch.random.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)


def trace_with_opaque_retry(module, args, **kwargs):
    """Keep fake tracing when it works; retry only its known boundary bug."""
    enabled = opaque.NEW_OPAQUE_TRACING_STRATEGY and any(
        isinstance(p, opaque.OpaqueTensorRef) for p in module.parameters()
    )
    if not enabled:
        return torch.jit.trace(module, args, **kwargs)
    return _trace_with_checkpoint(module, args, **kwargs)


def _trace_with_checkpoint(module, args, **kwargs):
    checkpoint = _TraceCheckpoint(module, args)
    try:
        return torch.jit.trace(module, args, **kwargs)
    except T2NErrorOpaqueTraceBoundary:
        checkpoint.restore()
        LOGGER.warning(
            "FakeTensor/JIT boundary failure: retrying with concrete opaque "
            "weight expansion at use sites. Packed export storage is retained; "
            "temporary tracing memory and computation may increase."
        )
        with opaque.materialized_opaque_trace():
            try:
                return torch.jit.trace(module, args, **kwargs)
            except Exception:
                checkpoint.restore()
                raise
