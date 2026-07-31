"""Operators we do not translate and cannot measure as page rows.

Everything else we cannot translate carries an `nnef_gap` spec in the
themed module where it belongs (see `_gap_common.gap_spec`), so the ONNX
sweep can measure it and the tract driver can assert the gap is real.
This module is the other half: the rows that get no spec because the
catalog cannot reliably attribute a drawn sample to that support-page
row, and why.

Recorded rather than left silent, because "we chose not to measure this"
and "nobody got to it" are different states and only the second is a
backlog. `tests/test_proptest_nnef_gap.py` checks both directions: an
unsupported row in neither list fails, and an entry here that stops
being unsupported fails too.
"""

import typing as T

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
    "hash_tensor": (
        "the row exists in some torch builds, but not in the CI torch "
        "operator packet, so no portable proptest sample can target it"
    ),
    "convrelu": "a fusion pattern name rather than a torch-level op",
    "linalg__powsum": (
        "internal linalg helper present in some torch builds but absent "
        "from the CI torch operator packet"
    ),
    "copy_to": "an internal copy helper, not reachable from Python",
    # -- mutating buffer management --
    "resize_": (
        "eager can call it, but JIT tracing warns that `resize_` cannot "
        "be represented and emits no `aten::resize_` node"
    ),
    # -- quantized --
    "quantize": "no public torch API and no exact `aten::quantize` schema",
    "quantized_gru": (
        "the prepared dynamic module exists and eager-runs, but this "
        "environment's JIT path cannot trace its `forward`, so the "
        "attribution guard cannot prove `aten::quantized_gru`"
    ),
    "quantized_lstm": (
        "same prepared dynamic-module tracing issue as `quantized_gru`"
    ),
    "wrapped_linear_prepack": (
        "private `_wrapped_linear_prepack` backend op; the local torch "
        "build has no FBGEMM kernel for a portable attribution sample"
    ),
    "wrapped_quantized_linear_prepacked": (
        "private `_wrapped_quantized_linear_prepacked` backend op; same "
        "FBGEMM portability issue as `wrapped_linear_prepack`"
    ),
}
