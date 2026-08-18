"""Shared surface for the gated-delta-net (GDN) linear-attention recurrence.

Two export paths reach the SAME tract operator and must agree on its
contract, so the contract lives here once:

- `t2n_extra::gated_delta_scan` (see `op/extras/scan_ops.py`), a torch custom
  op whose portable lowering is a `tract_core_scan`, and
- a reified GDN module captured by a `ModuleInfoExtractor`, which has no
  portable lowering and only ever emits the native operator.

What the two paths do NOT share is how they obtain the six operands (a
traced custom-op node against a module boundary) and the layout those
operands arrive in, so `emit_native_gdn_recurrent` takes a `head_major`
flag instead of assuming one.
"""

from __future__ import annotations

import logging
import typing as T

import numpy as np
import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.inference_target.tract import (
    NATIVE_GDN_RECURRENT_MIN_VERSION,
    TractNNEF,
)

LOGGER = logging.getLogger(__name__)

NATIVE_GDN_OP = "tract_transformers_gdn_recurrent"

# The released operator only accepts this key/value head dim, and only these
# dtypes (it upcasts the recurrence to f32 inside). A later tract generalizes
# both, at which point these become version-dependent rather than fixed.
NATIVE_GDN_HEAD_DIM = 128
NATIVE_GDN_DTYPES = {
    "query": np.float16,
    "key": np.float16,
    "value": np.float16,
    "log_decay": np.float32,
    "beta": np.float16,
    "state": np.float32,
}
OPERAND_NAMES = tuple(NATIVE_GDN_DTYPES)
#: the five per-step operands (everything but the recurrent state)
PER_STEP_NAMES = OPERAND_NAMES[:5]


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize the last axis, with HF's `use_qk_l2norm_in_kernel` eps.

    Shared so the pre-normalization applied OUTSIDE the op cannot drift from
    the eps tract's fused operator uses internally.
    """
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gated_delta_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    s0: torch.Tensor,
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch gated-delta recurrence over the time axis (axis 2).

    The single reference for the whole repo: the `t2n_extra` custom op's eager
    body, the reified module's eager forward and the tests all call this, so a
    divergence cannot hide in one of the copies. Matches HF's
    `torch_recurrent_gated_delta_rule` for pre-normalized q/k.
    """
    state = s0
    outs = []
    for step in range(q.shape[2]):
        q_t, k_t, v_t = q[:, :, step], k[:, :, step], v[:, :, step]
        state = state * g[:, :, step].exp()[..., None, None]
        kv = (state * k_t.unsqueeze(-1)).sum(-2)
        delta = (v_t - kv) * beta[:, :, step][..., None]
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-1)).sum(-2))
    return torch.stack(outs, dim=2), state


def native_gdn_reject_reason(
    inference_target, operands: T.Sequence["NTensor"], head_major: bool
) -> T.Optional[str]:
    """Why these operands can NOT become tract's fused operator.

    `tract_transformers_gdn_recurrent` is a fused SINGLE decode step (it also
    folds the q/k l2-norm and the `1 / sqrt(head_dim)` output scale), with
    hard constraints checked by tract at load time. Returns `None` when the
    traced tensors satisfy all of them, else a short reason for logging
    before falling back.

    `operands` is (query, key, value, log_decay, beta, state); `head_major`
    says whether the per-step tensors carry `(B, H, S, ...)` rather than
    tract's own `(B, S, H, ...)`.
    """
    query, key, value, log_decay, beta, state = operands
    if not isinstance(inference_target, TractNNEF):
        return "not a tract inference target"
    if not inference_target.native_gated_delta_op:
        return "disabled on this inference target"
    seq_axis = 2 if head_major else 1
    seq = query.shape[seq_axis]
    if seq != 1:
        return f"time axis is {seq} (fused op decodes one step)"
    # NNEF tensor shapes are lists or tuples depending on the producer, so
    # normalize before comparing (a type mismatch would silently reject).
    per_step = {
        name: tuple(tensor.shape[:3])
        for name, tensor in zip(
            PER_STEP_NAMES,
            (query, key, value, log_decay, beta),
            strict=True,
        )
    }
    if len(set(per_step.values())) != 1:
        # tract would read mismatched head counts as GQA and repeat q/k
        # itself, while these paths hand it the repeat already applied.
        return f"operands disagree on batch/head/time: {per_step}"
    head_dims = [
        query.shape[-1],
        key.shape[-1],
        value.shape[-1],
        *state.shape[-2:],
    ]
    if any(dim != NATIVE_GDN_HEAD_DIM for dim in head_dims):
        return (
            f"head dims {head_dims} are not all {NATIVE_GDN_HEAD_DIM} "
            "(fused op is specialized)"
        )
    bad = {
        name: tensor.dtype
        for name, tensor in zip(OPERAND_NAMES, operands, strict=True)
        if tensor.dtype != NATIVE_GDN_DTYPES[name]
    }
    if bad:
        return f"dtypes {bad} do not match the fused op signature"
    return None


def native_gdn_min_version() -> str:
    """First tract release carrying the fused operator."""
    return NATIVE_GDN_RECURRENT_MIN_VERSION


def emit_native_gdn_recurrent(
    g,
    op_helper,
    operands: T.Sequence["NTensor"],
    outputs: T.Sequence["NTensor"],
    head_major: bool,
) -> T.List[str]:
    """Emit `tract_transformers_gdn_recurrent(q, k, v, log_decay, beta, s0)`.

    tract's operand layout puts the sequence axis at 1 and the head axis at 2
    (`[B, S, H, W]` for q/k/v, `[B, S, H]` for log-decay and beta), while the
    state keeps `[B, H, W, W]`. With `head_major` the per-step operands and
    the first output arrive as `[B, H, S, ...]` instead, so transpose them on
    the way in and take one back on the way out. At `S == 1` those transposes
    are pure shape changes, but they are what makes the emitted graph match
    the operator's documented layout rather than its flat-index reading of a
    single step.

    `outputs` is (output, final_state), already created by the caller in ITS
    own layout, since the two paths build output tensors differently.
    """
    query, key, value, log_decay, beta, state = operands
    out, final_state = outputs

    if not head_major:
        NOperation(
            g,
            type=NATIVE_GDN_OP,
            inputs=tuple(operands),
            outputs=(out, final_state),
        )
        return ["tract_transformers"]

    def _seq_second(src, axes, shape):
        return op_helper.add_intermediate_op(
            src=src,
            op_type="transpose",
            attrs={"axes": axes},
            new_shape=shape,
            suffix="gdn_seq_second",
        )

    def _4d(src):  # (B, H, S, D) -> (B, S, H, D)
        return _seq_second(
            src,
            [0, 2, 1, 3],
            [src.shape[0], src.shape[2], src.shape[1], src.shape[3]],
        )

    def _3d(src):  # (B, H, S) -> (B, S, H)
        return _seq_second(
            src, [0, 2, 1], [src.shape[0], src.shape[2], src.shape[1]]
        )

    b, h, seq = query.shape[:3]
    out_native = NTensor(
        g,
        name=f"{out.name}_gdn_seq_second",
        dtype=out.dtype,
        shape=(b, seq, h, value.shape[-1]),
    )
    NOperation(
        g,
        type=NATIVE_GDN_OP,
        inputs=(
            _4d(query),
            _4d(key),
            _4d(value),
            _3d(log_decay),
            _3d(beta),
            state,
        ),
        outputs=(out_native, final_state),
    )
    NOperation(
        g,
        type="transpose",
        attribs={"axes": [0, 2, 1, 3]},
        inputs=(out_native,),
        outputs=(out,),
    )
    return ["tract_transformers"]
