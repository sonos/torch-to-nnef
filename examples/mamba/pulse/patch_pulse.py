"""Patch HF MambaMixer.slow_forward to call `t2n_extra::ssm_scan_y`.

The y-only scan variant drops `h_final`: tract's Scan pulsifier
rejects `"last"` outputs, so for the pulse export path we cannot
expose the final state. Internal state still loops back inside the
scan body, which is exactly what we want for pulsed streaming.

Only fires under `torch.jit.is_tracing()` and when `cache_params` is
None (the prefill path).  Eager / training paths keep the upstream
`for t in range(seq_len)` loop unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.mamba.modeling_mamba import MambaMixer


@torch.library.custom_op(
    "t2n_extra::ssm_scan_y",
    mutates_args=(),
    schema=(
        "(Tensor discrete_A, Tensor deltaB_u, Tensor C, Tensor h_init) "
        "-> Tensor"
    ),
)
def _ssm_scan_y(
    discrete_a: torch.Tensor,
    delta_b_u: torch.Tensor,
    c: torch.Tensor,
    h_init: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for the y-only selective scan."""
    h = h_init
    outs = []
    seq_len = discrete_a.shape[2]
    for t in range(seq_len):
        h = discrete_a[:, :, t, :] * h + delta_b_u[:, :, t, :]
        y = torch.matmul(h, c[:, t, :].unsqueeze(-1)).squeeze(-1)
        outs.append(y)
    return torch.stack(outs, dim=-1)


@_ssm_scan_y.register_fake
def _meta(discrete_a, delta_b_u, c, h_init):
    b, d, t, _ = discrete_a.shape
    return discrete_a.new_empty((b, d, t))


_ORIG_SLOW_FORWARD = MambaMixer.slow_forward


def _patched(self, input_states, cache_params=None, attention_mask=None):
    # Intercept under tracing whenever the input length is > 1 (the
    # prefill / whole-sequence path). We deliberately ignore
    # `cache_params` here: `MambaModel.forward` auto-creates a
    # `DynamicCache` when `use_cache=True` (HF default), so the
    # cache_params check is not a reliable prefill signal.
    if not torch.jit.is_tracing() or input_states.shape[1] == 1:
        return _ORIG_SLOW_FORWARD(
            self, input_states, cache_params, attention_mask
        )

    batch_size, _, _ = input_states.shape
    dtype = input_states.dtype

    projected = self.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected.chunk(2, dim=1)
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)
    # Pulse-friendly conv1d: hand-roll the causal pad on the left and
    # call `F.conv1d` with padding=0 so the output length equals the
    # input length without any trailing slice. The upstream HF code
    # uses `Conv1d(padding=K-1)` followed by `[..., :seq_len]` which
    # introduces a slice whose end depends on the streaming symbol;
    # that form is rejected by tract's pulse pipeline.
    pad = self.conv1d.kernel_size[0] - 1
    padded = torch.nn.functional.pad(hidden_states, (pad, 0))
    conv_out = torch.nn.functional.conv1d(
        padded,
        self.conv1d.weight,
        self.conv1d.bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=self.conv1d.groups,
    )
    hidden_states = self.act(conv_out)
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, b_param, c_param = torch.split(
        ssm_parameters,
        [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
        dim=-1,
    )
    discrete_time_step = self.dt_proj(time_step)
    discrete_time_step = nn.functional.softplus(discrete_time_step)
    discrete_time_step = discrete_time_step.transpose(1, 2)

    a_param = -torch.exp(self.A_log.float())
    discrete_a = torch.exp(
        a_param[None, :, None, :] * discrete_time_step[:, :, :, None]
    )
    discrete_b = (
        discrete_time_step[:, :, :, None] * b_param[:, None, :, :].float()
    )
    delta_b_u = discrete_b * hidden_states[:, :, :, None].float()

    ssm_state = torch.zeros(
        (batch_size, self.intermediate_size, self.ssm_state_size),
        device=hidden_states.device,
        dtype=dtype,
    )

    scan_output = torch.ops.t2n_extra.ssm_scan_y(
        discrete_a, delta_b_u, c_param, ssm_state
    )

    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)
    return self.out_proj(scan_output.transpose(1, 2))


def install() -> None:
    MambaMixer.slow_forward = _patched
