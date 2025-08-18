from functools import partial
import logging
import typing as T
import torch
from torch import nn
from torch_to_nnef.compress import offloaded_tensor_qtensor
from torch_to_nnef.exceptions import T2NErrorImpossibleQuantization
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.quant import (
    QTensor,
    fp_to_tract_q4_0_with_min_max_calibration,
)
from torch_to_nnef.tensor.updater import ModTensorUpdater

LOGGER = logging.getLogger(__name__)


def fp_to_tract_q4_0_with_grid_mse_calibration(
    fp_weight, grid_size=50, maxshrink=0.8
):
    qtensor = fp_to_tract_q4_0_with_min_max_calibration(fp_weight)
    qscheme_min_max = qtensor.qscheme
    lower_bound_search_vals = qscheme_min_max.scale * maxshrink
    step_size = (qscheme_min_max.scale - lower_bound_search_vals) / grid_size
    current_vals = qscheme_min_max.scale.clone()
    best_vals = current_vals

    def get_current_error():
        return (
            ((fp_weight - qtensor.decompress()).abs() ** 2)
            .view(-1, qscheme_min_max.group_size)
            .mean(1)
        )

    best_val_error = get_current_error()
    orignal_val_error = best_val_error.clone()
    for _ in range(grid_size):
        current_vals += step_size
        qtensor.qscheme.scale = current_vals.clone()
        current_val_error = get_current_error()
        better_error = current_val_error < best_val_error
        best_val_error = torch.where(
            better_error, current_val_error, best_val_error
        )
        best_vals = torch.where(
            better_error.unsqueeze(1), current_vals, best_vals
        )
    gain_over_min_max = (orignal_val_error - best_val_error).mean()
    LOGGER.info(
        "[%s] quant grid search gained mse error from min/max: %s",
        fp_weight.name,
        gain_over_min_max,
    )
    qtensor.qscheme.scale = best_vals
    return qtensor


def quantize_weights_grid_mse_Q40(model: nn.Module, **kwargs):
    to_quantize_module_classes = kwargs.get(
        "to_quantize_module_classes", (nn.Linear,)
    )
    assert isinstance(to_quantize_module_classes, tuple), (
        to_quantize_module_classes
    )
    assert all(issubclass(_, nn.Module) for _ in to_quantize_module_classes), (
        to_quantize_module_classes
    )
    with torch.no_grad():
        ids_to_qtensor: T.Dict[int, T.Tuple[QTensor, OffloadedTensor]] = {}
        """ try to avoid quant if used in other operators like mix of embedding/linear if linear only quant """
        mod_tensor_updater = ModTensorUpdater(model)

        for name, mod in model.named_modules():
            if isinstance(mod, to_quantize_module_classes):
                LOGGER.info("quantize layer: %s", name)
                weight_id = id(mod.weight)
                if weight_id in ids_to_qtensor:
                    LOGGER.info(
                        "detected shared weight between: '%s' and '%s.weight'",
                        ids_to_qtensor[weight_id].nnef_name,
                        name,
                    )
                    continue
                if not all(
                    isinstance(m, to_quantize_module_classes)
                    for m in mod_tensor_updater.id_to_modules[weight_id]
                ):
                    clss = [
                        m.__class__
                        for m in mod_tensor_updater.id_to_modules[weight_id]
                    ]
                    LOGGER.warning(
                        "detected shared weight: '%s' candidate has "
                        "incompatible layer usage: %s,  but requested %s",
                        name,
                        clss,
                        to_quantize_module_classes,
                    )
                    continue
                try:

                    def q_fn(weight, name):
                        q_weight = fp_to_tract_q4_0_with_grid_mse_calibration(
                            weight,
                            **{
                                k: v
                                for k, v in kwargs.items()
                                if k in ["grid_size"]
                            },
                        )
                        q_weight.nnef_name = f"{name}.weight"
                        return q_weight

                    q_weight = offloaded_tensor_qtensor(
                        partial(q_fn, name=name), mod.weight, "q40_mse"
                    )
                except T2NErrorImpossibleQuantization as exp:
                    LOGGER.error("quant layer: %s error: %s", name, exp)
                    continue
                # => needs assignation next cause update_by_ref may create new Parameter object
                q_weight = mod_tensor_updater.update_by_ref(
                    mod.weight, q_weight
                )
                ids_to_qtensor[id(q_weight)] = q_weight
    return model


EXAMPLE_REGISTRY = {
    "grid_mse_q4_0_all": partial(
        quantize_weights_grid_mse_Q40,
        grid_size=50,
        to_quantize_module_classes=(
            nn.Linear,
            nn.Embedding,
            nn.Conv1d,
            nn.Conv2d,
        ),
    ),
}
