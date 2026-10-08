"""Shared post-load q4_0 weight-quantization passes for LLM exports.

These helpers promote the export-script pattern "quantize a hand-picked
set of dense projections to tract q4_0 min-max before tracing" into
reusable utilities. The mechanism is pure torch_to_nnef
(``fp_to_tract_q4_0_with_min_max_calibration`` +
``offloaded_tensor_qtensor`` + ``ModTensorUpdater``): each targeted
weight is materialized (possibly from disk offload), quantized,
re-offloaded and the dense copy freed, so peak RAM stays one weight at
a time. Because quantization happens BEFORE tracing, the graph build
never materializes dense float copies of those weights.
"""

import gc
import logging
import typing as T

import torch

from torch_to_nnef.compress import offloaded_tensor_qtensor
from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef.tensor.quant import (
    fp_to_tract_q4_0_with_min_max_calibration,
)
from torch_to_nnef.tensor.updater import ModTensorUpdater

LOGGER = logging.getLogger(__name__)

#: Default element-count floor under which a matched weight is skipped:
#: norms, routers and tiny projections save almost no bandwidth when
#: quantized and carry outsized quality risk.
DEFAULT_MIN_DENSE_QUANT_NUMEL = 1_000_000


def quantize_lm_head_q40(model) -> None:
    """Replace ``lm_head.weight`` with a tract q4_0 min-max QTensor param.

    The lm_head weight is read in full for every generated token, so
    quantizing it cuts decode bandwidth substantially; it also perturbs
    logits directly, hence keeping it a separate pass from the dense
    projections so a quality regression stays attributable.

    Raises when the embedding weights are tied to the lm_head: the
    QTensor would then also serve the input gather path.
    """
    if getattr(model.config, "tie_word_embeddings", False):
        raise T2NErrorMisuse(
            "lm_head.weight is tied to embed_tokens: quantizing it would "
            "also quantize the gather path; keep the lm_head dense instead"
        )

    def _q40(dense: torch.Tensor):
        q_weight = fp_to_tract_q4_0_with_min_max_calibration(dense)
        q_weight.nnef_name = "lm_head.weight"
        return q_weight

    updater = ModTensorUpdater(model)
    weight = model.lm_head.weight
    with torch.no_grad():
        q_weight = offloaded_tensor_qtensor(_q40, weight, "q40_min_max")
    updater.update_by_ref(weight, q_weight)
    del weight
    gc.collect()
    LOGGER.info(
        "lm_head.weight quantized to tract q4_0 min-max (%s)",
        type(model.lm_head.weight).__name__,
    )


def quantize_dense_projections_q40(
    model,
    groups: T.Mapping[str, T.Sequence[str]],
    *,
    group_names: T.Optional[T.Sequence[str]] = None,
    min_numel: int = DEFAULT_MIN_DENSE_QUANT_NUMEL,
) -> T.Dict[str, int]:
    """Quantize dense projection weights to tract q4_0 min-max QTensors.

    ``groups`` maps a weight-class name to parameter-name SUFFIXES
    (matched against ``model.named_parameters()``), so a quality
    regression stays attributable to one class and a subset can be
    re-exported in isolation via ``group_names``.

    Guards: only rank-2 weights with at least ``min_numel`` elements are
    quantized (anything else is skipped and logged); raises if the
    requested groups match nothing at all (a silent no-op export is
    never what the caller wants).

    Returns per-group quantized-weight counts.
    """
    if group_names is None:
        group_names = list(groups)
    unknown = set(group_names) - set(groups)
    if unknown:
        raise T2NErrorMisuse(
            f"unknown dense quant groups: {sorted(unknown)} "
            f"(available: {sorted(groups)})"
        )
    suffix_to_group = {
        suffix: gname for gname in group_names for suffix in groups[gname]
    }

    updater = ModTensorUpdater(model)
    counts: T.Dict[str, int] = {}
    for name, param in list(model.named_parameters()):
        gname = next(
            (
                grp
                for suffix, grp in suffix_to_group.items()
                if name.endswith(suffix)
            ),
            None,
        )
        if gname is None:
            continue
        if param.ndim != 2 or param.numel() < min_numel:
            LOGGER.info(
                "dense-q40[%s] SKIP %s (shape=%s, ndim!=2 or <%d elems)",
                gname,
                name,
                tuple(param.shape),
                min_numel,
            )
            continue

        def _q40(dense: torch.Tensor, _name: str = name):
            q_weight = fp_to_tract_q4_0_with_min_max_calibration(dense)
            q_weight.nnef_name = _name
            return q_weight

        with torch.no_grad():
            q_weight = offloaded_tensor_qtensor(_q40, param, "q40_min_max")
        updater.update_by_ref(param, q_weight)
        del param, q_weight
        gc.collect()
        counts[gname] = counts.get(gname, 0) + 1
        LOGGER.info("dense-q40[%s]: %s", gname, name)
    for gname in group_names:
        LOGGER.info(
            "dense-q40 group %s: %d weights", gname, counts.get(gname, 0)
        )
    if not counts:
        raise T2NErrorMisuse(
            "dense-q40 matched no weight: wrong module names? "
            f"(groups: {sorted(group_names)})"
        )
    return counts
