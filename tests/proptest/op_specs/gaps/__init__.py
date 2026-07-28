"""Specs for operators torch_to_nnef does not translate.

Every other spec module covers something we ship. This package covers the
opposite: operators with no emitter, so the ONNX sweep can measure them.

Why that matters. Specs previously existed only where t2n succeeds, so
the measured ONNX population was a strict subset of our own supported
set. An operator ONNX handles and we do not could never be graded, only
inherited as an unverified claim from the retired torch listing, or left
blank. The comparison was structurally blind exactly where we are
weakest, and the support page's `Gap vs ONNX` filter was capped at the
handful of rows that listing happened to name.

Each spec declares an `NnefGap`, which the tract driver *asserts* rather
than trusts (see `tests/proptest/nnef_gap.py`), so a gap that gets closed
turns into a failing test rather than a stale page.

Modules are grouped by *why* the operator is out of reach, because that
is what someone planning to close one needs: `linalg` is one missing
factorization repeated thirty times, `rng` is a format question rather
than a kernel question, `dataflow` splits into extents a static graph
cannot declare and layouts NNEF has no word for.
"""

import typing as T

from .._common import OpSpec
from . import dataflow, linalg, nn, rng, special, stats

__all__ = ["EXCLUDED", "SPECS"]

_MODULES = (dataflow, linalg, stats, special, rng, nn)

SPECS: T.Tuple[OpSpec, ...] = tuple(
    spec for module in _MODULES for spec in module.SPECS
)

#: Rows the support page calls unsupported that deliberately get no spec,
#: and why. Recorded here rather than left silent: "we chose not to
#: measure this" and "nobody got to it" are different states, and only
#: the second is a backlog.
EXCLUDED: T.Dict[str, str] = {
    # -- not reachable from a traced graph --
    "as_tensor": (
        "torch decomposes it at trace time into `aten::to`, which we do "
        "translate, so a graph never contains it. The page calls the row "
        "unsupported only because it reads the emitter registry, which "
        "cannot hold a key for a name that never appears. A generator "
        "fix, not a gap."
    ),
    "allclose": (
        "returns a Python bool, so tracing folds it to a constant and "
        "the operator leaves no node behind"
    ),
    "broadcast_shapes": (
        "returns a shape, not a tensor: same constant-folding as `allclose`"
    ),
    "upsample": (
        "a dispatcher name; the concrete `upsample_*` variants have "
        "their own rows and are translated"
    ),
    "native_multi_head_self_attention": (
        "torch dispatches `nn.MultiheadAttention` to "
        "`_native_multi_head_attention`, which the page's source grep "
        "drops for being `_`-prefixed, so this row has no reachable "
        "spelling"
    ),
    # -- internal or non-operator rows --
    "first": "not a torch operator; an artefact of the source grep",
    "second": "not a torch operator; an artefact of the source grep",
    "hash_tensor": "debug helper, never part of an exported graph",
    "convrelu": "a fusion pattern name rather than a torch-level op",
    "linalg__powsum": "an internal helper of the `linalg` namespace",
    "copy_to": "an internal copy helper, not reachable from Python",
    "flatten_dense_tensors": (
        "a distributed-training buffer utility, not part of inference"
    ),
    "unflatten_dense_tensors": "the inverse of `flatten_dense_tensors`",
    "embedding_renorm_": "a training-time mutation of the weight table",
    "index_put_impl_": "the internal spelling of `index_put_`",
    # -- mutating buffer management --
    "resize_": (
        "mutates the tensor's storage, which a value-semantics graph "
        "has no way to express"
    ),
    "set_": "rebinds a tensor to another's storage; same reason as `resize_`",
    # -- sparse --
    "copy_sparse_to_sparse_": "t2n has no sparse tensor support at all",
    "resize_as_sparse_": "t2n has no sparse tensor support at all",
    # -- quantized --
    # These need a quantized module to produce them, and t2n reaches
    # quantization through its own path rather than through these aten
    # rows, so a spec here would measure the harness, not the exporter.
    "empty_quantized": "needs a quantized tensor to exist first",
    "quantize": "quantization goes through t2n's own path",
    "quantize_per_channel": "quantization goes through t2n's own path",
    "quantize_per_tensor_dynamic": "quantization goes through t2n's own path",
    "quantized_batch_norm": "needs a prepared quantized module",
    "quantized_gru": "needs a prepared quantized module",
    "quantized_lstm": "needs a prepared quantized module",
    "quantized_max_pool1d": "needs a prepared quantized module",
    "quantized_max_pool2d": "needs a prepared quantized module",
    "quantized_max_pool3d": "needs a prepared quantized module",
    "wrapped_linear_prepack": "a packing helper of the quantized path",
    "wrapped_quantized_linear_prepacked": (
        "a packing helper of the quantized path"
    ),
}
