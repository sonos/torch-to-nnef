"""Handlers for `t2n_extra::exp_unit_norm` / `exp_mean_norm` ops.

Both lower to the matching `tract_extra_exp_unit_norm` /
`tract_extra_exp_mean_norm` NNEF ops. Tract's `OpPulsifier` is already
registered for `ExpUnitNorm` (see `tract/extra/src/exp_unit_norm.rs`),
so a streaming-axis trace pulses end-to-end.

DPDFNet's `ErbNorm` (centring per-frame EMA followed by a fixed-std
scale) maps to `exp_mean_norm` with `scaling_factor=40.0`. `SpecNorm`
(per-frame magnitude EMA divided into the complex spectrum) maps to
`exp_unit_norm` with `complex=True`.

Signatures (matching the eager torch.library declarations in
example/test code):

    t2n_extra::exp_unit_norm(input, state_init, axis, alpha, epsilon, complex)
    t2n_extra::exp_mean_norm(input, state_init, axis, alpha, scaling_factor)

`state_init` has the shape of `input` with `axis` removed (and, for
`complex=True`, the trailing-2 axis also removed); pass zeros at
trace time -- the pulsifier overrides `skip` with the runtime delay.
"""

from __future__ import annotations

import typing as T

from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import (
    T2NErrorInvalidArgument,
    T2NErrorNotImplemented,
)
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.extras import register


def _read_attr(node, idx, expected_kind):
    inp = node.inputs[idx]
    if inp.data is None:
        raise T2NErrorInvalidArgument(
            f"t2n_extra::exp_*_norm: argument #{idx} must be a constant "
            f"({expected_kind}); got a dynamic input"
        )
    return inp.data


def _emit_exp_norm(
    *,
    g,
    node,
    op_helper,
    inference_target,
    name_to_tensor,
    mean: bool,
) -> T.List[str]:
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            "t2n_extra::exp_{unit,mean}_norm requires a TractNNEF target"
        )

    input_node = node.inputs[0]
    state_node = node.inputs[1]
    axis = int(_read_attr(node, 2, "int"))
    alpha = float(_read_attr(node, 3, "float"))

    if mean:
        scaling_factor = float(_read_attr(node, 4, "float"))
        op_type = "tract_extra_exp_mean_norm"
        attribs = {
            "axis": axis,
            "alpha": alpha,
            "stateless": False,
            "skip": 0,
            "scaling_factor": scaling_factor,
        }
    else:
        epsilon = float(_read_attr(node, 4, "float"))
        complex_ = bool(_read_attr(node, 5, "bool"))
        op_type = "tract_extra_exp_unit_norm"
        attribs = {
            "axis": axis,
            "alpha": alpha,
            "stateless": False,
            "skip": 0,
            "complex": complex_,
            "epsilon": epsilon,
        }

    input_nnef = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    state_nnef = op_helper.get_or_add_tensor_variable_in_nnef(state_node)

    out_node = node.outputs[0]
    out = NTensor(
        g,
        name=out_node.export_name,
        dtype=input_nnef.dtype,
        shape=tuple(input_nnef.shape),
    )
    name_to_tensor[out_node.export_name] = out

    NOperation(
        g,
        type=op_type,
        attribs=attribs,
        inputs=(input_nnef, state_nnef),
        outputs=(out,),
    )
    return ["tract_extra"]


@register("exp_unit_norm")
def exp_unit_norm(
    g, node, name_to_tensor, op_helper, inference_target, **kwargs
) -> T.List[str]:
    """Lower `t2n_extra::exp_unit_norm` to `tract_extra_exp_unit_norm`.

    Eager signature (matches the example / test declarations):

        t2n_extra::exp_unit_norm(input, state_init, axis: int,
                                 alpha: float, epsilon: float,
                                 complex: bool) -> Tensor

    Computes a per-time-step EMA of the input magnitude, then divides
    `input` by `sqrt(state)` along `axis`. `state_init` is the initial
    hidden state (zeros at trace time); its shape equals `input` with
    `axis` removed (and the trailing-2 axis removed for `complex=True`).
    """
    return _emit_exp_norm(
        g=g,
        node=node,
        op_helper=op_helper,
        inference_target=inference_target,
        name_to_tensor=name_to_tensor,
        mean=False,
    )


@register("exp_mean_norm")
def exp_mean_norm(
    g, node, name_to_tensor, op_helper, inference_target, **kwargs
) -> T.List[str]:
    """Lower `t2n_extra::exp_mean_norm` to `tract_extra_exp_mean_norm`.

    Eager signature:

        t2n_extra::exp_mean_norm(input, state_init, axis: int,
                                 alpha: float, scaling_factor: float)
                                 -> Tensor

    Centres `input` with a per-time-step EMA mean, then divides by
    `scaling_factor`. `state_init`'s shape is `input` with `axis`
    removed.
    """
    return _emit_exp_norm(
        g=g,
        node=node,
        op_helper=op_helper,
        inference_target=inference_target,
        name_to_tensor=name_to_tensor,
        mean=True,
    )
