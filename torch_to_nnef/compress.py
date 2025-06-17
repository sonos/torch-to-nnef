from collections import defaultdict
import typing as T
from functools import partial
import logging

import torch
from torch import nn

from torch_to_nnef.exceptions import TorchToNNEFImpossibleQuantization
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.quant import (
    QTensor,
    fp_to_tract_q4_0_with_min_max_calibration,
)
from torch_to_nnef.tensor.updater import ModTensorUpdater

LOGGER = logging.getLogger(__name__)


def quantize_weights_min_max_Q4_0(model: nn.Module, **kwargs):
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
                        q_weight = fp_to_tract_q4_0_with_min_max_calibration(
                            weight,
                            **{
                                k: v
                                for k, v in kwargs.items()
                                if k in ["percentile"]
                            },
                        )
                        q_weight.nnef_name = f"{name}.weight"
                        return q_weight

                    q_weight = offloaded_tensor_qtensor(
                        q_fn, mod.weight, "q40_min_max"
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


def offloaded_tensor_qtensor(
    q_fn, tensor: torch.Tensor, suffix_name: str
) -> torch.Tensor:
    original_tensor = tensor
    if isinstance(original_tensor, OffloadedTensor):
        tensor = original_tensor.to_base_tensor()

    q_tensor = q_fn(tensor)

    final_tensor = q_tensor
    if isinstance(original_tensor, OffloadedTensor):
        final_tensor = OffloadedTensor.from_original_tensor(
            q_tensor,
            f"{original_tensor._name}.{suffix_name}",
            offload_dir=original_tensor.offload_dir,
        )
    return final_tensor


def dynamic_load_registry(compression_registry_full_path: str):
    module_str, name = compression_registry_full_path.rsplit(".", maxsplit=1)
    mod = __import__(module_str, fromlist=[""])
    registry = getattr(mod, name)
    assert isinstance(registry, dict)
    return registry


DEFAULT_COMPRESSION = {
    "min_max_q4_0": quantize_weights_min_max_Q4_0,
    "min_max_q4_0_with_embeddings": partial(
        quantize_weights_min_max_Q4_0,
        to_quantize_module_classes=(nn.Linear, nn.Embedding),
    ),
    "min_max_q4_0_with_embeddings_99": partial(
        partial(quantize_weights_min_max_Q4_0, percentile=0.99),
        to_quantize_module_classes=(nn.Linear, nn.Embedding),
    ),
    "min_max_q4_0_all": partial(
        quantize_weights_min_max_Q4_0,
        to_quantize_module_classes=(
            nn.Linear,
            nn.Embedding,
            nn.Conv1d,
            nn.Conv2d,
        ),
    ),
}
