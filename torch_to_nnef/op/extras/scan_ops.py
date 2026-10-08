"""Handlers for `t2n_extra::*` scan-shaped ops.

Currently provides `ssm_scan` for Mamba's selective state-space scan.
The handler emits a `mamba_ssm_scan` NNEF fragment call which wraps a
`tract_core_scan` over a per-step `mamba_ssm_step` body. Tract's pulse
declutter compiles the scan into a streaming graph, so the prefill
cost is one tract call instead of one per token.
"""

from __future__ import annotations

import logging
import typing as T

from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import (
    T2NErrorInvalidArgument,
    T2NErrorNotImplemented,
)
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.extras import register
from torch_to_nnef.op.gated_delta import (
    emit_native_gdn_recurrent,
    native_gdn_reject_reason,
)

LOGGER = logging.getLogger(__name__)


def _emit_ssm_scan_common(
    *,
    g,
    node,
    op_helper,
    inference_target,
    name_to_tensor,
    y_only: bool,
):
    """Common emit for `ssm_scan` and `ssm_scan_y`.

    Expects:
      - discrete_A, deltaB_u: (B, D, T, N)
      - C: (B, T, N)
      - h_init: (B, D, N)
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "t2n_extra::ssm_scan{,_y} requires a TractNNEF target"
        )

    discrete_a_in, delta_b_u_in, c_in, h_init_in = node.inputs
    discrete_a = op_helper.get_or_add_tensor_variable_in_nnef(discrete_a_in)
    delta_b_u = op_helper.get_or_add_tensor_variable_in_nnef(delta_b_u_in)
    c = op_helper.get_or_add_tensor_variable_in_nnef(c_in)
    h_init = op_helper.get_or_add_tensor_variable_in_nnef(h_init_in)

    # Validate ranks/shapes.
    def _rank(t):
        return len(t.shape)

    if _rank(discrete_a) != 4 or _rank(delta_b_u) != 4:
        raise T2NErrorInvalidArgument(
            "'discrete_A' and 'deltaB_u' must be rank-4 (B, D, T, N)"
        )
    if _rank(c) != 3:
        raise T2NErrorInvalidArgument("'C' must be rank-3 (B, T, N)")
    if _rank(h_init) != 3:
        raise T2NErrorInvalidArgument("'h_init' must be rank-3 (B, D, N)")

    # Pre-transpose so the scan axis (time) lands at position 0.
    # discrete_A / deltaB_u: (B, D, T, N) -> (T, B, D, N) via (2, 0, 1, 3).
    b, d, t, n = discrete_a.shape
    a_perm = op_helper.add_intermediate_op(
        src=discrete_a,
        op_type="transpose",
        attrs={"axes": [2, 0, 1, 3]},
        new_shape=[t, b, d, n],
        suffix="scan_time_first",
    )
    bu_perm = op_helper.add_intermediate_op(
        src=delta_b_u,
        op_type="transpose",
        attrs={"axes": [2, 0, 1, 3]},
        new_shape=[t, b, d, n],
        suffix="scan_time_first",
    )
    # C: (B, T, N) -> (T, B, N) via (1, 0, 2).
    bc, tc, nc = c.shape
    c_perm = op_helper.add_intermediate_op(
        src=c,
        op_type="transpose",
        attrs={"axes": [1, 0, 2]},
        new_shape=[tc, bc, nc],
        suffix="scan_time_first",
    )

    # Outputs: y_final is (B, D, T). h_final mirrors h_init.
    y_final = NTensor(
        g,
        name=node.outputs[0].export_name,
        dtype=discrete_a.dtype,
        shape=(b, d, t),
    )
    name_to_tensor[node.outputs[0].export_name] = y_final

    if y_only:
        NOperation(
            g,
            type="mamba_ssm_scan_pulse",
            attribs={"scan_pace": 1},
            inputs=(a_perm, bu_perm, c_perm, h_init),
            outputs=(y_final,),
        )
        return ["mamba_ssm_scan_pulse"]

    h_final = NTensor(
        g,
        name=node.outputs[1].export_name,
        dtype=h_init.dtype,
        shape=tuple(h_init.shape),
    )
    name_to_tensor[node.outputs[1].export_name] = h_final
    NOperation(
        g,
        type="mamba_ssm_scan",
        attribs={"scan_pace": 1},
        inputs=(a_perm, bu_perm, c_perm, h_init),
        outputs=(y_final, h_final),
    )
    return ["mamba_ssm_scan"]


@register("ssm_scan")
def ssm_scan(
    g, node, name_to_tensor, op_helper, inference_target, **kwargs
) -> T.List[str]:
    """Emit a `mamba_ssm_scan` fragment call.

    Signature on the torch side:
        t2n_extra::ssm_scan(discrete_A, deltaB_u, C, h_init)
            -> (scan_outputs, h_final)

    The fragment scans along axis 0 of its inputs (matches the
    GRU/LSTM convention). The handler pre-transposes the SSM tensors
    so the time axis lands at position 0 before the scan:

        discrete_A  (B, D, T, N) -> (T, B, D, N)
        deltaB_u    (B, D, T, N) -> (T, B, D, N)
        C           (B, T, N)    -> (T, B, N)
        h_init      (B, D, N)    -- unchanged (state)

    After the scan:
        scan_y      (T, B, D)    -> (B, D, T) to match `scan_outputs`'s
                                    PyTorch shape (stack on last axis).
        h_final     (B, D, N)
    """
    return _emit_ssm_scan_common(
        g=g,
        node=node,
        op_helper=op_helper,
        inference_target=inference_target,
        name_to_tensor=name_to_tensor,
        y_only=False,
    )


@register("ssm_scan_y")
def ssm_scan_y(
    g, node, name_to_tensor, op_helper, inference_target, **kwargs
) -> T.List[str]:
    """Pulse-friendly variant of `ssm_scan`: emits only y_t (no h_final).

    The Scan pulsifier in tract rejects `"last"` outputs (h_final).
    Dropping it makes the scan body compatible with `into_pulse`.
    """
    return _emit_ssm_scan_common(
        g=g,
        node=node,
        op_helper=op_helper,
        inference_target=inference_target,
        name_to_tensor=name_to_tensor,
        y_only=True,
    )


@register("gated_delta_scan")
def gated_delta_scan(
    g, node, name_to_tensor, op_helper, inference_target, **kwargs
) -> T.List[str]:
    """Emit a `gated_delta_scan` fragment call (Qwen3.5 gated-delta-net).

    Torch side:
        t2n_extra::gated_delta_scan(q, k, v, g, beta, s0) -> (y, s_final)
    with (time axis at 2):
        q, k    (B, H, T, hk)   v  (B, H, T, hv)
        g, beta (B, H, T)       s0 (B, H, hk, hv)
    The scan iterates axis 0, so pre-transpose the per-step inputs to put
    time first (s0 is the state, left as-is), then the fragment wraps
    `tract_core_scan`. Outputs: y (B, H, T, hv), s_final (B, H, hk, hv).

    From tract 0.23.5 a single-step graph instead emits tract's fused
    `tract_transformers_gdn_recurrent` (CPU/CUDA/Metal kernels) when it fits
    the operator's constraints, see `_native_gdn_reject_reason`. That operator
    additionally folds the q/k l2-norm and a `1 / sqrt(head_dim)` output
    scale, so it assumes the Qwen3.5 convention this op is written for: q
    passed as `l2norm(q) / sqrt(head_k_dim)` and k as `l2norm(k)`.
    Re-normalizing an already-normalized q/k is a no-op, and the internal
    scale (tract reads `head_dim` off the same axis, so it always matches
    ours) then restores the one normalizing q stripped, so both lowerings agree
    (to ~6e-5 relative, from the operator's `1e-6` norm epsilon; under f16
    resolution, and covered by `check_io`). Feeding raw, un-normalized q/k
    would NOT be equivalent.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "t2n_extra::gated_delta_scan requires a TractNNEF target"
        )
    q_in, k_in, v_in, g_in, beta_in, s0_in = node.inputs
    q = op_helper.get_or_add_tensor_variable_in_nnef(q_in)
    k = op_helper.get_or_add_tensor_variable_in_nnef(k_in)
    v = op_helper.get_or_add_tensor_variable_in_nnef(v_in)
    gg = op_helper.get_or_add_tensor_variable_in_nnef(g_in)
    beta = op_helper.get_or_add_tensor_variable_in_nnef(beta_in)
    s0 = op_helper.get_or_add_tensor_variable_in_nnef(s0_in)

    def _rank(t):
        return len(t.shape)

    if _rank(q) != 4 or _rank(k) != 4 or _rank(v) != 4:
        raise T2NErrorInvalidArgument(
            "'q'/'k'/'v' must be rank-4 (B, H, T, head_dim)"
        )
    if _rank(gg) != 3 or _rank(beta) != 3:
        raise T2NErrorInvalidArgument("'g'/'beta' must be rank-3 (B, H, T)")
    if _rank(s0) != 4:
        raise T2NErrorInvalidArgument(
            "'s0' must be rank-4 (B, H, key_head_dim, value_head_dim)"
        )

    operands = (q, k, v, gg, beta, s0)
    reject = native_gdn_reject_reason(
        inference_target, operands, head_major=True
    )
    if reject is None:
        b, h, t = q.shape[:3]
        y_final = NTensor(
            g,
            name=node.outputs[0].export_name,
            dtype=q.dtype,
            shape=(b, h, t, v.shape[-1]),
        )
        name_to_tensor[node.outputs[0].export_name] = y_final
        s_final = NTensor(
            g,
            name=node.outputs[1].export_name,
            dtype=s0.dtype,
            shape=tuple(s0.shape),
        )
        name_to_tensor[node.outputs[1].export_name] = s_final
        return emit_native_gdn_recurrent(
            g,
            op_helper,
            operands,
            (y_final, s_final),
            head_major=True,
        )
    LOGGER.debug(
        "gated_delta_scan keeps the tract_core_scan lowering: %s", reject
    )

    b, h, t = q.shape[:3]
    hv = v.shape[-1]

    def _time_first_4d(src):  # (B, H, T, D) -> (T, B, H, D)
        return op_helper.add_intermediate_op(
            src=src,
            op_type="transpose",
            attrs={"axes": [2, 0, 1, 3]},
            new_shape=[src.shape[2], src.shape[0], src.shape[1], src.shape[3]],
            suffix="scan_time_first",
        )

    def _time_first_3d(src):  # (B, H, T) -> (T, B, H)
        return op_helper.add_intermediate_op(
            src=src,
            op_type="transpose",
            attrs={"axes": [2, 0, 1]},
            new_shape=[src.shape[2], src.shape[0], src.shape[1]],
            suffix="scan_time_first",
        )

    q_p, k_p, v_p = _time_first_4d(q), _time_first_4d(k), _time_first_4d(v)
    g_p, beta_p = _time_first_3d(gg), _time_first_3d(beta)

    y_final = NTensor(
        g, name=node.outputs[0].export_name, dtype=q.dtype, shape=(b, h, t, hv)
    )
    name_to_tensor[node.outputs[0].export_name] = y_final
    s_final = NTensor(
        g,
        name=node.outputs[1].export_name,
        dtype=s0.dtype,
        shape=tuple(s0.shape),
    )
    name_to_tensor[node.outputs[1].export_name] = s_final
    NOperation(
        g,
        type="gated_delta_scan",
        attribs={"scan_pace": 1},
        inputs=(q_p, k_p, v_p, g_p, beta_p, s0),
        outputs=(y_final, s_final),
    )
    return ["gated_delta_scan"]
