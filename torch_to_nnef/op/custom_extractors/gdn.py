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
