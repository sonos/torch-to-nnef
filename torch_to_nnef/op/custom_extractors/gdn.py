"""Gated delta net recurrence export to tract_qwen35_gdn_recurrent.

HF's Qwen3.5 linear attention computes its core with a Python loop over
the sequence axis (`torch_recurrent_gated_delta_rule` /
`torch_chunk_gated_delta_rule`). Traced, that loop unrolls at the traced
sequence length and the exported graph is frozen to it (S=1 for the
decode-path trace). Reifying the rule as ONE custom op keeps the graph
S-generic: tract's `GatedDeltaNetRecurrent` runs the recurrence
sequentially for any S.

The exporter swaps the module-bound rule functions with
:class:`GatedDeltaNetRecurrentReified` (see the Qwen3.5 architecture
handler's `prepare_model_for_export`); the extractor below then emits
the op instead of tracing the shim's internals.

Op contract (must match tract `transformers/src/ops/gdn_recurrent.rs`):
    query/key/value  [b, S, h, w]  (heads already repeated to the
                                    value-head count, key width == value
                                    width)
    log_decay/beta   [b, S, h]
    initial_state    [b, h, w, w]
    -> output [b, S, h, w] (query dtype), final_state [b, h, w, w]
The op L2-normalizes q/k (eps 1e-6) and scales q by 1/sqrt(w) itself,
i.e. HF `use_qk_l2norm_in_kernel=True` semantics.
"""

import logging
import typing as T

import torch
from torch import nn

from torch_to_nnef.exceptions import (
    T2NErrorNotImplemented,
    T2NErrorStrictNNEFSpec,
)
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor

LOGGER = logging.getLogger(__name__)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class GatedDeltaNetRecurrentReified(nn.Module):
    """Drop-in for HF's (chunk|recurrent)_gated_delta_rule functions.

    Eager forward reproduces `torch_recurrent_gated_delta_rule`
    semantics exactly (the chunked variant computes the same recurrence
    with different blocking, so substituting it is numerically
    equivalent up to fp reordering). At export the whole call is
    reified as one `tract_qwen35_gdn_recurrent` op.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: T.Optional[torch.Tensor] = None,
        beta: T.Optional[torch.Tensor] = None,
        initial_state: T.Optional[torch.Tensor] = None,
        output_final_state: bool = True,
        use_qk_l2norm_in_kernel: bool = True,
        **_kwargs,
    ):
        # The tract op bakes the L2 norm in (Qwen3.5 always passes True);
        # the extractor rejects a traced False (see convert_to_nnef).
        #
        # ORDERING CONTRACT: the traced module-boundary input order is the
        # FIRST-USE order of each tensor below, and q/k/v (and g/beta)
        # share shapes so the extractor cannot re-identify them. Keep the
        # first use of query, key, value, g, beta, initial_state in exactly
        # this (signature) order.
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            q = _l2norm(query)
            k = _l2norm(key)
        else:
            q, k = query, key
        # [b, S, h, w] -> [b, h, S, w], f32 compute
        q, k, v = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (q, k, value)
        ]
        g_t = g.transpose(1, 2).contiguous().to(torch.float32)
        beta_t = beta.transpose(1, 2).contiguous().to(torch.float32)

        batch, heads, s_len, k_width = k.shape
        v_width = v.shape[-1]
        q = q * (1.0 / (k_width**0.5))

        # copy=True: a same-dtype `.to` ALIASES its input; the parent then
        # mutates the cache buffer this state comes from, and the tracer
        # would expose the alias as an extra module-boundary output,
        # scrambling the traced output tuple.
        state = (
            torch.zeros(
                batch, heads, k_width, v_width, dtype=v.dtype, device=v.device
            )
            if initial_state is None
            else initial_state.to(dtype=v.dtype, copy=True)
        )
        outs = []
        for i in range(s_len):
            state = state * g_t[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            kv_mem = (state * k[:, :, i].unsqueeze(-1)).sum(dim=-2)
            delta = (v[:, :, i] - kv_mem) * beta_t[:, :, i].unsqueeze(-1)
            state = state + k[:, :, i].unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((state * q[:, :, i].unsqueeze(-1)).sum(dim=-2))
        core = torch.stack(outs, dim=2)
        core = core.transpose(1, 2).contiguous().to(initial_dtype)
        return core, (state if output_final_state else None)


class CausalConvUpdateReified(nn.Module):
    """Drop-in core for HF's `torch_causal_conv1d_update`.

    Depthwise causal conv1d + SiLU with a k-sample carry state. Eager
    forward matches HF exactly but avoids `F.conv1d` (whose fake-tensor
    meta kernel breaks under the offload tracing strategy): the conv is
    an unfold + mul + sum. At export the call is reified as one
    `tract_qwen35_causal_conv1d_update` op.

    Layout: hidden_states `[b, C, S]`, conv_state `[b, C, k]`,
    weight `[C, k]` -> (out `[b, C, S]`, final_state `[b, C, k]`).
    The caller keeps responsibility for writing final_state back into
    the HF cache (a plain `copy_`, handled by the aten copy handler).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
    ):
        k = conv_state.shape[-1]
        s_len = hidden_states.shape[-1]
        # copy=True: a same-dtype `.to` ALIASES its input; the caller then
        # mutates the cache buffer this state comes from (copy_), and the
        # tracer would expose the alias as an extra module-boundary output.
        full = torch.cat(
            [
                conv_state.to(dtype=hidden_states.dtype, copy=True),
                hidden_states,
            ],
            dim=-1,
        ).to(weight.dtype)
        final_state = full[:, :, -k:]
        # windows[..., t, :] = full[..., t : t + k]; output keeps the last
        # S positions (windows ending at each new sample).
        windows = full.unfold(-1, k, 1)
        out = (windows * weight.unsqueeze(0).unsqueeze(2)).sum(-1)
        out = torch.nn.functional.silu(out[:, :, -s_len:])
        return out.to(hidden_states.dtype), final_state


class CausalConvUpdateExtractor(ModuleInfoExtractor):
    """Emit `tract_qwen35_causal_conv1d_update` for the reified conv."""

    MODULE_CLASS = CausalConvUpdateReified

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        inference_target,
        **kw,
    ):
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorStrictNNEFSpec(
                "causal conv export requires tract inference target"
            )
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef import torch_graph as tg

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

        tensor_inputs = [
            inp for inp in node.inputs if isinstance(inp, tg.TensorVariable)
        ]
        if len(tensor_inputs) != 3:
            raise T2NErrorNotImplemented(
                "reified causal conv expects (hidden_states, conv_state, "
                f"weight) tensor inputs, got {len(tensor_inputs)}"
            )
        # The traced module-boundary input order is the FIRST-USE order
        # inside the shim, not the signature order: identify the inputs
        # semantically. weight is the only rank-2 tensor [C, k]; the state
        # is the rank-3 tensor whose time axis equals the kernel width.
        weights = [t for t in tensor_inputs if len(t.shape) == 2]
        rank3 = [t for t in tensor_inputs if len(t.shape) == 3]
        if len(weights) != 1 or len(rank3) != 2:
            raise T2NErrorNotImplemented(
                "cannot identify causal conv inputs by rank: "
                f"{[list(t.shape) for t in tensor_inputs]}"
            )
        weight = weights[0]
        kernel_width = list(weight.shape)[-1]
        states = [t for t in rank3 if list(t.shape)[-1] == kernel_width]
        if len(states) != 1:
            raise T2NErrorNotImplemented(
                "cannot disambiguate causal conv state from hidden states "
                f"(kernel width {kernel_width}): "
                f"{[list(t.shape) for t in rank3]}"
            )
        conv_state = states[0]
        hidden = next(t for t in rank3 if t is not conv_state)
        input_shape = list(hidden.shape)
        state_shape = list(conv_state.shape)
        if input_shape == state_shape:
            raise T2NErrorNotImplemented(
                "cannot disambiguate causal conv outputs: input and state "
                f"shapes coincide ({input_shape})"
            )
        outs = list(node.outputs)
        if len(outs) != 2:
            raise T2NErrorNotImplemented(
                f"reified causal conv expects 2 outputs, got {len(outs)}"
            )
        # node.outputs order follows the tracer's module-boundary escape
        # order: identify (out, final_state) by shape.
        if list(outs[0].shape) == state_shape and list(
            outs[1].shape
        ) == input_shape:
            outs = [outs[1], outs[0]]
        elif not (
            list(outs[0].shape) == input_shape
            and list(outs[1].shape) == state_shape
        ):
            raise T2NErrorNotImplemented(
                "reified causal conv outputs match neither (out, state) nor "
                f"(state, out): {[list(o.shape) for o in outs]}"
            )
        # NNEF op argument order is (input, weight, initial_state).
        nnef_inputs = tuple(
            helper.get_or_add_tensor_variable_in_nnef(g, inp, name_to_tensor)
            for inp in (hidden, weight, conv_state)
        )
        nnef_outputs = tuple(
            helper.add_tensor_variable_node_as_nnef_tensor(
                g, out, name_to_tensor, prevent_variable=True
            )
            for out in outs
        )
        helper.cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="tract_qwen35_causal_conv1d_update",
            inputs=nnef_inputs,
            outputs=nnef_outputs,
            attribs={},
            force_consistent_inputs_shapes=False,
        )
        return ["tract_transformers"]


class GatedDeltaNetRecurrentExtractor(ModuleInfoExtractor):
    """Emit `tract_qwen35_gdn_recurrent` for the reified GDN shim."""

    MODULE_CLASS = GatedDeltaNetRecurrentReified

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        inference_target,
        **kw,
    ):
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorStrictNNEFSpec(
                "GDN export requires tract inference target "
                "(tract_qwen35_gdn_recurrent is a tract extension)"
            )
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef import torch_graph as tg

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

        # The two trailing python bools are output_final_state and
        # use_qk_l2norm_in_kernel; the op's semantics require the L2 norm,
        # and Qwen3.5 always enables it. Reject a traced False loudly.
        bool_consts = [
            inp.data
            for inp in node.inputs
            if isinstance(inp, tg.ir_data.PythonConstant)
            and isinstance(inp.data, bool)
        ]
        if any(flag is False for flag in bool_consts):
            raise T2NErrorNotImplemented(
                "the reified GDN op bakes in q/k L2 normalization and a "
                "returned final state; a traced False flag is unsupported: "
                f"{bool_consts}"
            )
        tensor_inputs = [
            inp for inp in node.inputs if isinstance(inp, tg.TensorVariable)
        ]
        if len(tensor_inputs) != 6:
            raise T2NErrorNotImplemented(
                "reified GDN expects exactly (query, key, value, g, beta, "
                f"initial_state) tensor inputs, got {len(tensor_inputs)}: "
                f"{[i.name for i in tensor_inputs]}. A None initial_state "
                "means the caller ran the cache-less chunked branch; feed "
                "the recurrent state through the cache instead."
            )
        nnef_inputs = tuple(
            helper.get_or_add_tensor_variable_in_nnef(g, inp, name_to_tensor)
            for inp in tensor_inputs
        )
        if len(node.outputs) != 2:
            raise T2NErrorNotImplemented(
                f"reified GDN expects 2 outputs, got {len(node.outputs)}"
            )
        # node.outputs order follows the tracer's module-boundary escape
        # order, not the (core, final_state) python return order: identify
        # them by shape (core matches query, final_state matches
        # initial_state).
        query_shape = list(tensor_inputs[0].shape)
        state_shape = list(tensor_inputs[5].shape)
        outs = list(node.outputs)
        if query_shape == state_shape:
            raise T2NErrorNotImplemented(
                "cannot disambiguate GDN outputs: query and state shapes "
                f"coincide ({query_shape})"
            )
        if list(outs[0].shape) == state_shape and list(
            outs[1].shape
        ) == query_shape:
            outs = [outs[1], outs[0]]
        elif not (
            list(outs[0].shape) == query_shape
            and list(outs[1].shape) == state_shape
        ):
            raise T2NErrorNotImplemented(
                "reified GDN outputs match neither (core, state) nor "
                f"(state, core): {[list(o.shape) for o in outs]} vs "
                f"query {query_shape} / state {state_shape}"
            )
        nnef_outputs = tuple(
            helper.add_tensor_variable_node_as_nnef_tensor(
                g, out, name_to_tensor, prevent_variable=True
            )
            for out in outs
        )
        helper.cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type="tract_qwen35_gdn_recurrent",
            inputs=nnef_inputs,
            outputs=nnef_outputs,
            attribs={},
            force_consistent_inputs_shapes=False,
        )
        return ["tract_transformers"]
