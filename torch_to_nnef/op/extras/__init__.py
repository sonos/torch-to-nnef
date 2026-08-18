"""Dispatch for custom PyTorch ops registered under the `t2n_extra::` namespace.

Users can ship their own opaque ops by declaring them via
`torch.library.custom_op("t2n_extra::<name>", ...)` and providing a
matching handler here. The handler emits a NNEF op (typically a custom
fragment), so the trace stays a single opaque node end-to-end instead
of being decomposed by `torch.jit.trace`.

Today this powers Mamba's selective scan: see
`torch_to_nnef.op.extras.scan_ops:ssm_scan` for the wiring pattern.
"""

from __future__ import annotations

import typing as T

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.op.helper import OpHelper, OpRegistry

_T2N_EXTRA_REGISTRY = OpRegistry(torch_mod_id="t2n_extra")


def register(op_name: str):
    """Decorator: register a handler for `t2n_extra::<op_name>`."""
    return _T2N_EXTRA_REGISTRY.register([op_name])


def t2n_extra_to_nnef_tensor_and_ops(
    g, node, name_to_tensor, null_ref, *, torch_graph, inference_target
) -> T.List[str]:
    """Dispatch a `t2n_extra::*` op node to its registered handler."""
    ops_family, op_name = node.kind.split("::")
    assert ops_family == "t2n_extra"
    try:
        handler = _T2N_EXTRA_REGISTRY.get(op_name)
    except T2NErrorNotImplemented as err:
        raise T2NErrorNotImplemented(
            f"no t2n_extra handler registered for '{node.kind}'. "
            "Register via `@torch_to_nnef.op.extras.register('<name>')` "
            "and import the module before export (use "
            "load_extra_op_modules, TORCH_TO_NNEF_EXTRA_MODULES, or an "
            "entry point under 'torch_to_nnef.extras')."
        ) from err
    return handler(
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
        torch_graph=torch_graph,
        inference_target=inference_target,
        op_helper=OpHelper(g, node, name_to_tensor, null_ref, inference_target),
    )


# Import side-effect: register the bundled handlers.
from torch_to_nnef.op.extras import exp_norm  # noqa: E402, F401, I001
from torch_to_nnef.op.extras import scan_ops  # noqa: E402, F401, I001
