from functools import partial
import logging
import typing as T
import torch
from torch import nn
from torch_to_nnef.compress import offloaded_tensor_qtensor
from torch_to_nnef.exceptions import TorchToNNEFImpossibleQuantization
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
        f"[{fp_weight.name}] quant grid search gained mse error from min/max: {gain_over_min_max}"
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
                LOGGER.info(f"quantize layer: {name}")
                weight_id = id(getattr(mod, "weight"))
                if weight_id in ids_to_qtensor:
                    LOGGER.info(
                        f"detected shared weight between: '{ids_to_qtensor[weight_id].nnef_name}' and '{name}.weight'"
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
                        f"detected shared weight: '{name}' candidate has incompatible layer usage: {clss}, "
                        f" but requested {to_quantize_module_classes}"
                    )
                    continue
                try:

                    def q_fn(weight):
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
                        q_fn, mod.weight, "q40_mse"
                    )
                except TorchToNNEFImpossibleQuantization as exp:
                    LOGGER.error(f"quant layer: {name} error: {exp}")
                    continue
                # => needs assignation next cause update_by_ref may create new Parameter object
                q_weight = mod_tensor_updater.update_by_ref(
                    getattr(mod, "weight"), q_weight
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
