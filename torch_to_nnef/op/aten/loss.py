"""ATen loss-family op emitters (mse_loss, nll_loss, cross_entropy_loss, ...).

Each loss is decomposed via a pointwise NNEF fragment (where pointwise
makes sense -- mse, bce-with-logits, kl_div) plus a full-tensor
`mean_reduce` / `sum_reduce` + `squeeze` chain governed by torch's
reduction enum (`0 = none`, `1 = mean`, `2 = sum`).
"""

import typing as T

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.helper import AtenOpRegistry

OP_REGISTRY = AtenOpRegistry()

_REDUCTION_NONE = 0
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2


def _reduction_value(node_value, op_name: str) -> int:
    """Resolve torch's `reduction` enum (int constant 0 / 1 / 2)."""
    if hasattr(node_value, "data"):
        node_value = node_value.data
    if not isinstance(node_value, int) or node_value not in (
        _REDUCTION_NONE,
        _REDUCTION_MEAN,
        _REDUCTION_SUM,
    ):
        raise T2NErrorNotImplemented(
            f"{op_name}: unsupported reduction value {node_value!r}; "
            "expected 0 (none), 1 (mean) or 2 (sum)"
        )
    return node_value


def _apply_reduction(
    op_helper,
    node,
    pointwise_ref,
    reduction: int,
    rank: int,
    op_label: str,
) -> None:
    """Wire `pointwise_ref` into `node.outputs[0]` with full reduction.

    `reduction == 0` -> the fragment's pointwise tensor IS the output
    (emitted without a name suffix above).  Otherwise reduce across
    every axis (`mean_reduce` / `sum_reduce`) then `squeeze` the
    resulting `(1, ..., 1)` shape down to a 0-D scalar -- matching
    torch's loss-output rank for the reduced cases.
    """
    if reduction == _REDUCTION_NONE:
        return  # already wired by the caller
    axes = list(range(rank))
    reduce_op = "mean_reduce" if reduction == _REDUCTION_MEAN else "sum_reduce"
    reduced = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        reduce_op,
        inputs=pointwise_ref,
        attrs={"axes": axes},
        output_tensor_name_suffix=f"_{op_label}_reduce",
    )
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "squeeze",
        inputs=reduced,
        attrs={"axes": axes},
    )


def _emit_pointwise_loss(
    op_helper,
    node,
    fragment_name: str,
    inputs,
    reduction: int,
    rank: int,
    attrs: T.Optional[T.Dict[str, T.Any]] = None,
) -> T.List[str]:
    """Common path: call `fragment_name` then optionally reduce.

    When `reduction == none` the fragment writes directly into the node
    output. When reduced, the fragment writes to an intermediate and
    `_apply_reduction` wires the squeeze into the final tensor.
    Scalar fragment parameters (e.g. `delta` for huber, `beta` for
    smooth-l1) ride through `attrs`.
    """
    attrs = attrs or {}
    if reduction == _REDUCTION_NONE:
        op_helper.add_single_output_op_from_nnef_tensors(
            node,
            fragment_name,
            inputs=inputs,
            attrs=attrs,
            force_consistent_inputs_shapes=False,
        )
        return [fragment_name]
    pointwise = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        fragment_name,
        inputs=inputs,
        attrs=attrs,
        force_consistent_inputs_shapes=False,
        output_tensor_name_suffix=f"_{fragment_name}_pw",
    )
    _apply_reduction(op_helper, node, pointwise, reduction, rank, fragment_name)
    return [fragment_name]


@OP_REGISTRY.register()
def mse_loss(node, op_helper, **kwargs):
    """Map PyTorch `aten::mse_loss(input, target, reduction)` to NNEF.

    Pointwise `(input - target) ** 2` is delegated to the `mse_loss`
    fragment, then reduced if `reduction != none`. Torch broadcasts
    `input` / `target` upstream of the aten op (we see a separate
    `aten::broadcast_tensors` in the trace), so the fragment can assume
    matching shapes.
    """
    input_node, target_node, reduction_node = node.inputs
    reduction = _reduction_value(reduction_node, "mse_loss")
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    tgt = op_helper.get_or_add_tensor_variable_in_nnef(target_node)
    return _emit_pointwise_loss(
        op_helper, node, "mse_loss", [inp, tgt], reduction, input_node.rank
    )


@OP_REGISTRY.register()
def l1_loss(node, op_helper, **kwargs):
    """Map PyTorch `aten::l1_loss(input, target, reduction)` to NNEF.

    Pointwise `|input - target|` via the `l1_loss` fragment, then
    reduced. Like `mse_loss`, torch broadcasts upstream via
    `aten::broadcast_tensors`, so the fragment assumes matching shapes.
    """
    input_node, target_node, reduction_node = node.inputs
    reduction = _reduction_value(reduction_node, "l1_loss")
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    tgt = op_helper.get_or_add_tensor_variable_in_nnef(target_node)
    return _emit_pointwise_loss(
        op_helper, node, "l1_loss", [inp, tgt], reduction, input_node.rank
    )


@OP_REGISTRY.register()
def huber_loss(node, op_helper, **kwargs):
    """Map PyTorch `aten::huber_loss(input, target, reduction, delta)`.

    Pointwise piecewise: quadratic when `|input - target| < delta`,
    linear otherwise. Reduction applied by the emitter.
    """
    input_node, target_node, reduction_node, delta_node = node.inputs
    reduction = _reduction_value(reduction_node, "huber_loss")
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    tgt = op_helper.get_or_add_tensor_variable_in_nnef(target_node)
    delta = float(delta_node.data) if delta_node.data is not None else 1.0
    return _emit_pointwise_loss(
        op_helper,
        node,
        "huber_loss",
        [inp, tgt],
        reduction,
        input_node.rank,
        attrs={"delta": delta},
    )


@OP_REGISTRY.register()
def smooth_l1_loss(node, op_helper, **kwargs):
    """Map `aten::smooth_l1_loss(input, target, reduction, beta)`.

    Same piecewise shape as `huber_loss` with a different scaling: the
    quadratic branch is `0.5 * diff^2 / beta` and the linear branch is
    `|diff| - 0.5 * beta` (vs huber's `delta * (|diff| - 0.5 * delta)`).
    """
    input_node, target_node, reduction_node, beta_node = node.inputs
    reduction = _reduction_value(reduction_node, "smooth_l1_loss")
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    tgt = op_helper.get_or_add_tensor_variable_in_nnef(target_node)
    beta = float(beta_node.data) if beta_node.data is not None else 1.0
    return _emit_pointwise_loss(
        op_helper,
        node,
        "smooth_l1_loss",
        [inp, tgt],
        reduction,
        input_node.rank,
        attrs={"beta": beta},
    )


@OP_REGISTRY.register()
def hinge_embedding_loss(node, op_helper, **kwargs):
    """Map `aten::hinge_embedding_loss(input, target, margin, reduction)`.

    Pointwise `input if target==1 else max(0, margin - input)`.
    """
    input_node, target_node, margin_node, reduction_node = node.inputs
    margin = float(margin_node.data) if margin_node.data is not None else 1.0
    reduction = _reduction_value(reduction_node, "hinge_embedding_loss")
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    tgt = op_helper.get_or_add_tensor_variable_in_nnef(target_node)
    return _emit_pointwise_loss(
        op_helper,
        node,
        "hinge_embedding_loss",
        [inp, tgt],
        reduction,
        input_node.rank,
        attrs={"margin": margin},
    )


@OP_REGISTRY.register()
def triplet_margin_loss(node, op_helper, **kwargs):
    """Map `aten::triplet_margin_loss` to NNEF.

    Signature: `(anchor, positive, negative, margin, p, eps, swap,
    reduction)`. Per-sample distance along the trailing feature axis;
    swap=True picks `min(||a-n||, ||p-n||)` and is emitted via a
    second fragment call + `min` (lets the main fragment stay focused).
    """
    (
        anchor_node,
        positive_node,
        negative_node,
        margin_node,
        p_node,
        eps_node,
        swap_node,
        reduction_node,
    ) = node.inputs
    margin = float(margin_node.data) if margin_node.data is not None else 1.0
    p_val = float(p_node.data) if p_node.data is not None else 2.0
    eps = float(eps_node.data) if eps_node.data is not None else 1e-6
    swap = bool(getattr(swap_node, "data", False))
    reduction = _reduction_value(reduction_node, "triplet_margin_loss")
    if p_val <= 0:
        raise T2NErrorNotImplemented(
            f"triplet_margin_loss: p={p_val} not supported (require p > 0)"
        )
    if anchor_node.rank < 2:
        raise T2NErrorNotImplemented(
            "triplet_margin_loss expects rank-2 inputs (B, D); got "
            f"rank={anchor_node.rank}"
        )
    feature_axis = anchor_node.rank - 1
    anchor = op_helper.get_or_add_tensor_variable_in_nnef(anchor_node)
    positive = op_helper.get_or_add_tensor_variable_in_nnef(positive_node)
    negative = op_helper.get_or_add_tensor_variable_in_nnef(negative_node)
    if swap:
        raise T2NErrorNotImplemented(
            "triplet_margin_loss: swap=True not yet supported "
            "(needs a second fragment call + min reduction)"
        )
    return _emit_pointwise_loss(
        op_helper,
        node,
        "triplet_margin_loss",
        [anchor, positive, negative],
        reduction,
        anchor_node.rank - 1,
        attrs={
            "feature_axis": feature_axis,
            "margin": margin,
            "p": p_val,
            "eps": eps,
        },
    )


@OP_REGISTRY.register()
def binary_cross_entropy_with_logits(node, op_helper, **kwargs):
    """Map `aten::binary_cross_entropy_with_logits` to NNEF.

    Signature: `(input, target, weight, pos_weight, reduction)`.
    Pointwise BCE via the numerically-stable softplus formulation lives
    in the `binary_cross_entropy_with_logits` fragment; `weight` /
    `pos_weight` modulators are not currently supported.
    """
    (
        input_node,
        target_node,
        weight_node,
        pos_weight_node,
        reduction_node,
    ) = node.inputs
    for opt, label in [
        (weight_node, "weight"),
        (pos_weight_node, "pos_weight"),
    ]:
        if hasattr(opt, "data") and opt.data is not None:
            raise T2NErrorNotImplemented(
                f"binary_cross_entropy_with_logits: {label} != None "
                "not supported"
            )
    reduction = _reduction_value(
        reduction_node, "binary_cross_entropy_with_logits"
    )
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    tgt = op_helper.get_or_add_tensor_variable_in_nnef(target_node)
    return _emit_pointwise_loss(
        op_helper,
        node,
        "binary_cross_entropy_with_logits",
        [inp, tgt],
        reduction,
        input_node.rank,
    )


@OP_REGISTRY.register()
def kl_div(node, op_helper, **kwargs):
    """Map `aten::kl_div(input, target, reduction, log_target)` to NNEF.

    Two pointwise fragments, picked by `log_target`:
    - `kl_div` (default):    target * (log(target) - input)
    - `kl_div_log_target`:   exp(target) * (target - input)

    `input` is assumed to be log-probabilities (caller normally feeds
    `log_softmax(...)`). Torch's `reduction='batchmean'` is lowered to
    `sum` plus an external division upstream of the aten op, so the
    aten reduction enum here is only 0 / 1 / 2.
    """
    input_node, target_node, reduction_node, log_target_node = node.inputs
    log_target = bool(getattr(log_target_node, "data", False))
    reduction = _reduction_value(reduction_node, "kl_div")
    fragment_name = "kl_div_log_target" if log_target else "kl_div"
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    tgt = op_helper.get_or_add_tensor_variable_in_nnef(target_node)
    return _emit_pointwise_loss(
        op_helper,
        node,
        fragment_name,
        [inp, tgt],
        reduction,
        input_node.rank,
    )


def _emit_nll_per_sample(
    op_helper,
    node,
    input_ref,
    target_ref,
    op_label: str,
    final_op_suffix: str,
):
    """Emit `-input[target]` along the class axis (=1).

    Decomposed via `tract_core_gather_elements`: target is unsqueezed
    at the class axis so it has the same rank as input, the gather
    reads one entry per `(N, *spatial)` position, then we squeeze the
    now-singleton class axis and negate.

    Set `final_op_suffix=""` to land the final `neg` directly in
    `node.outputs[0]` (no-reduction path); pass a non-empty suffix to
    park it in an intermediate that the reduction chain then consumes.
    """
    if not isinstance(op_helper.inference_target, TractNNEF):
        raise T2NErrorNotImplemented(
            f"{op_label}: requires `tract_core_gather_elements` "
            "(TractNNEF target)"
        )
    # target rank == input rank - 1; unsqueeze at the class axis (=1).
    tgt_unsq = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "unsqueeze",
        inputs=target_ref,
        attrs={"axes": [1]},
        output_tensor_name_suffix=f"_{op_label}_tgt_unsq",
    )
    gathered = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "tract_core_gather_elements",
        inputs=[input_ref, tgt_unsq],
        attrs={"axis": 1},
        output_tensor_name_suffix=f"_{op_label}_ge",
        force_consistent_inputs_shapes=False,
    )
    squeezed = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "squeeze",
        inputs=gathered,
        attrs={"axes": [1]},
        output_tensor_name_suffix=f"_{op_label}_sq",
    )
    return op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "neg",
        inputs=squeezed,
        output_tensor_name_suffix=final_op_suffix,
    )


@OP_REGISTRY.register(torch_op_ids=["nll_loss", "nll_loss2d", "nll_loss_nd"])
def nll_loss(node, op_helper, **kwargs):
    """Map PyTorch's nll_loss family to NNEF.

    Signature (all three variants):
    `nll_loss(input, target, weight, reduction, ignore_index)`.

    The per-sample loss is `-input[n, target[n], ...]` along the class
    axis (=1). Class-weighting and ignore-index masking are common
    training-side knobs; we raise `T2NErrorNotImplemented` for both
    until a real need shows up.
    """
    (
        input_node,
        target_node,
        weight_node,
        reduction_node,
        ignore_index_node,
    ) = node.inputs
    if hasattr(weight_node, "data") and weight_node.data is not None:
        raise T2NErrorNotImplemented("nll_loss: weight != None not supported")
    ignore_index = getattr(ignore_index_node, "data", -100)
    if ignore_index != -100:
        raise T2NErrorNotImplemented(
            f"nll_loss: ignore_index={ignore_index} not supported"
        )
    if input_node.rank < 2:
        raise T2NErrorNotImplemented(
            f"nll_loss expects input rank >= 2; got {input_node.rank}"
        )
    reduction = _reduction_value(reduction_node, "nll_loss")
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    tgt = op_helper.get_or_add_tensor_variable_in_nnef(target_node)
    is_terminal = reduction == _REDUCTION_NONE
    per_sample = _emit_nll_per_sample(
        op_helper,
        node,
        inp,
        tgt,
        "nll",
        final_op_suffix="" if is_terminal else "_nll_pw",
    )
    _apply_reduction(
        op_helper, node, per_sample, reduction, target_node.rank, "nll"
    )
    return ["tract_core"]


@OP_REGISTRY.register()
def cross_entropy_loss(node, op_helper, **kwargs):
    """Map `aten::cross_entropy_loss` to NNEF.

    Lowers to `nll_loss(log_softmax(input, dim=1), target, ...)`.
    `weight` / `ignore_index` / `label_smoothing` are not currently
    supported (raise on non-default values).
    """
    (
        input_node,
        target_node,
        weight_node,
        reduction_node,
        ignore_index_node,
        label_smoothing_node,
    ) = node.inputs
    if hasattr(weight_node, "data") and weight_node.data is not None:
        raise T2NErrorNotImplemented(
            "cross_entropy_loss: weight != None not supported"
        )
    ignore_index = getattr(ignore_index_node, "data", -100)
    if ignore_index != -100:
        raise T2NErrorNotImplemented(
            f"cross_entropy_loss: ignore_index={ignore_index} not supported"
        )
    label_smoothing = getattr(label_smoothing_node, "data", 0.0)
    if label_smoothing != 0.0:
        raise T2NErrorNotImplemented(
            f"cross_entropy_loss: label_smoothing={label_smoothing} "
            "not supported"
        )
    if input_node.rank < 2:
        raise T2NErrorNotImplemented(
            f"cross_entropy_loss expects input rank >= 2; got {input_node.rank}"
        )
    reduction = _reduction_value(reduction_node, "cross_entropy_loss")
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    tgt = op_helper.get_or_add_tensor_variable_in_nnef(target_node)
    # log_softmax along the class axis (=1). The standard `log_softmax`
    # fragment is portable across tract / Khronos NNEF.
    log_probs = op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "log_softmax",
        inputs=inp,
        attrs={"axis": 1},
        output_tensor_name_suffix="_ce_logsoftmax",
        force_consistent_inputs_shapes=False,
    )
    is_terminal = reduction == _REDUCTION_NONE
    per_sample = _emit_nll_per_sample(
        op_helper,
        node,
        log_probs,
        tgt,
        "ce",
        final_op_suffix="" if is_terminal else "_ce_pw",
    )
    _apply_reduction(
        op_helper, node, per_sample, reduction, target_node.rank, "ce"
    )
    return ["log_softmax", "tract_core"]
