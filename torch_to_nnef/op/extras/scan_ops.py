"""Handlers for `t2n_extra::*` scan-shaped ops.

Currently provides `ssm_scan` for Mamba's selective state-space scan.
The handler emits a `mamba_ssm_scan` NNEF fragment call which wraps a
`tract_core_scan` over a per-step `mamba_ssm_step` body. Tract's pulse
declutter compiles the scan into a streaming graph, so the prefill
cost is one tract call instead of one per token.
"""

from __future__ import annotations

import typing as T

from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.extras import register


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
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "t2n_extra::ssm_scan requires a TractNNEF target "
            "(uses `tract_core_scan` under the hood)."
        )

    discrete_a_in, delta_b_u_in, c_in, h_init_in = node.inputs
    discrete_a = op_helper.get_or_add_tensor_variable_in_nnef(discrete_a_in)
    delta_b_u = op_helper.get_or_add_tensor_variable_in_nnef(delta_b_u_in)
    c = op_helper.get_or_add_tensor_variable_in_nnef(c_in)
    h_init = op_helper.get_or_add_tensor_variable_in_nnef(h_init_in)

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
    # C: (B, T, N) -> (T, B, N) by permute(1, 0, 2).
    bc, tc, nc = c.shape
    c_perm = op_helper.add_intermediate_op(
        src=c,
        op_type="transpose",
        attrs={"axes": [1, 0, 2]},
        new_shape=[tc, bc, nc],
        suffix="scan_time_first",
    )

    # Author the two scan outputs by hand so we can plug in custom
    # post-shape ops (the per-step `y_t` stacks to (T, B, D) and we
    # transpose back to (B, D, T) to match the torch trace's logical
    # output shape).
    # The fragment's `output` spec inserts T at axis 2 of the body's
    # y_t shape, giving (B, D, T) directly: no post-transpose needed.
    y_final = NTensor(
        g,
        name=node.outputs[0].export_name,
        dtype=discrete_a.dtype,
        shape=(b, d, t),
    )
    h_final = NTensor(
        g,
        name=node.outputs[1].export_name,
        dtype=h_init.dtype,
        shape=tuple(h_init.shape),
    )
    name_to_tensor[node.outputs[0].export_name] = y_final
    name_to_tensor[node.outputs[1].export_name] = h_final
    NOperation(
        g,
        type="mamba_ssm_scan",
        attribs={"scan_pace": 1},
        inputs=(a_perm, bu_perm, c_perm, h_init),
        outputs=(y_final, h_final),
    )
    return ["mamba_ssm_scan"]


@register("ssm_scan_y")
def ssm_scan_y(
    g, node, name_to_tensor, op_helper, inference_target, **kwargs
) -> T.List[str]:
    """Pulse-friendly variant of `ssm_scan`: emits only y_t (no h_final).

    The Scan pulsifier in tract rejects `"last"` outputs (h_final).
    Dropping it makes the scan body compatible with `into_pulse`.
    """
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "t2n_extra::ssm_scan_y requires a TractNNEF target."
        )

    discrete_a_in, delta_b_u_in, c_in, h_init_in = node.inputs
    discrete_a = op_helper.get_or_add_tensor_variable_in_nnef(discrete_a_in)
    delta_b_u = op_helper.get_or_add_tensor_variable_in_nnef(delta_b_u_in)
    c = op_helper.get_or_add_tensor_variable_in_nnef(c_in)
    h_init = op_helper.get_or_add_tensor_variable_in_nnef(h_init_in)

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
    bc, tc, nc = c.shape
    c_perm = op_helper.add_intermediate_op(
        src=c,
        op_type="transpose",
        attrs={"axes": [1, 0, 2]},
        new_shape=[tc, bc, nc],
        suffix="scan_time_first",
    )

    y_final = NTensor(
        g,
        name=node.outputs[0].export_name,
        dtype=discrete_a.dtype,
        shape=(b, d, t),
    )
    name_to_tensor[node.outputs[0].export_name] = y_final
    NOperation(
        g,
        type="mamba_ssm_scan_pulse",
        attribs={"scan_pace": 1},
        inputs=(a_perm, bu_perm, c_perm, h_init),
        outputs=(y_final,),
    )
    return ["mamba_ssm_scan_pulse"]
