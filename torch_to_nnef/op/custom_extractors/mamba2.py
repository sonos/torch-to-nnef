"""Mamba-2 selective state-space recurrence export.

HF's Mamba-2 mixers compute the SSD core either per token or by chunks of the
sequence, with Python loops that unroll at the traced length. Calling
:class:`Mamba2RecurrentReified` instead keeps the graph S-generic: the
extractor emits one `tract_transformers_mamba2_recurrent` op, which runs the
recurrence sequentially for any S.

Op contract (must match tract `transformers/src/ops/mamba2_recurrent.rs`):
    x            [b, S, H, P]  already scaled by dt
    b / c        [b, S, G, N]  head h reads group h // (H / G)
    log_decay    [b, S, H]     dt * A
    initial_state [b, H, P, N]
    -> y [b, S, H, P] (x dtype), final_state (initial_state dtype)
The D skip connection and the z gate stay outside the op.
"""

import torch
from torch import nn

from torch_to_nnef import torch_graph as tg
from torch_to_nnef.exceptions import (
    T2NErrorNotImplemented,
    T2NErrorStrictNNEFSpec,
)
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op import helper
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor

MAMBA2_MIN_TRACT_VERSION = "0.23.11"


class Mamba2RecurrentReified(nn.Module):
    """Mamba-2 recurrence, exported as one tract op (see module doc).

    The eager forward computes in f32 like the op. Its first statement uses
    the five inputs in signature order: b and c share a shape, so the
    extractor tells them apart by that order only.
    """

    def forward(self, x, b, c, log_decay, state):
        y_dtype = x.dtype
        x, b, c, log_decay = x.float(), b.float(), c.float(), log_decay.float()
        new_state = state.float()
        heads, groups = x.shape[2], b.shape[2]
        b = b.repeat_interleave(heads // groups, dim=2)
        c = c.repeat_interleave(heads // groups, dim=2)
        ys = []
        for t in range(x.shape[1]):
            decay = log_decay[:, t].exp()[:, :, None, None]
            new_state = new_state * decay + (
                x[:, t, :, :, None] * b[:, t, :, None, :]
            )
            ys.append((new_state * c[:, t, :, None, :]).sum(-1))
        y = torch.stack(ys, dim=1).to(y_dtype)
        return y, new_state.to(state.dtype)


class Mamba2RecurrentExtractor(ModuleInfoExtractor):
    """Emit `tract_transformers_mamba2_recurrent` for the reified core."""

    MODULE_CLASS = Mamba2RecurrentReified

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
                "Mamba-2 export requires tract inference target"
            )
        if inference_target.version < MAMBA2_MIN_TRACT_VERSION:
            raise T2NErrorNotImplemented(
                f"Mamba-2 recurrence requires tract >= "
                f"{MAMBA2_MIN_TRACT_VERSION}; targeted version is "
                f"{inference_target.version}."
            )
        inputs = [
            inp for inp in node.inputs if isinstance(inp, tg.TensorVariable)
        ]
        if [len(t.shape) for t in inputs] != [4, 4, 4, 3, 4]:
            raise T2NErrorNotImplemented(
                "reified Mamba-2 expects (x, b, c, log_decay, state) of ranks "
                f"4, 4, 4, 3, 4, got {[list(t.shape) for t in inputs]}"
            )
        x_shape, state_shape = list(inputs[0].shape), list(inputs[4].shape)
        if x_shape == state_shape:
            raise T2NErrorNotImplemented(
                f"cannot tell Mamba-2 outputs apart: shapes are {x_shape}"
            )
        outs = list(node.outputs)
        if len(outs) != 2:
            raise T2NErrorNotImplemented(
                f"reified Mamba-2 expects 2 outputs, got {len(outs)}"
            )
        if list(outs[0].shape) != x_shape:
            outs = outs[::-1]
        nnef_inputs = tuple(
            helper.get_or_add_tensor_variable_in_nnef(g, inp, name_to_tensor)
            for inp in inputs
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
            type="tract_transformers_mamba2_recurrent",
            inputs=nnef_inputs,
            outputs=nnef_outputs,
            attribs={},
            force_consistent_inputs_shapes=False,
        )
        return ["tract_transformers"]
