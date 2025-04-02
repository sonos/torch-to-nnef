import typing as T
from functools import partial
import logging

import torch
from torch import nn

from torch_to_nnef.exceptions import TorchToNNEFImpossibleQuantization
from torch_to_nnef.qtensor.base import QTensor
from torch_to_nnef.qtensor.qtract import (
    fp_to_tract_q4_0_with_min_max_calibration,
)

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
        ids_to_qtensor: T.Dict[int, QTensor] = {}
        for name, mod in model.named_modules():
            if isinstance(mod, to_quantize_module_classes):
                LOGGER.info(f"quantize layer: {name}")
                weight_id = id(getattr(mod, "weight"))
                if weight_id in ids_to_qtensor:
                    q_weight = ids_to_qtensor[weight_id]
                    LOGGER.info(
                        f"detected shared weight between: '{q_weight.nnef_name}' and '{name}.weight'"
                    )
                else:
                    try:
                        q_weight = fp_to_tract_q4_0_with_min_max_calibration(
                            mod.weight,
                            **{
                                k: v
                                for k, v in kwargs.items()
                                if k in ["percentile"]
                            },
                        )
                        q_weight.nnef_name = f"{name}.weight"
                        ids_to_qtensor[weight_id] = q_weight
                    except TorchToNNEFImpossibleQuantization as exp:
                        LOGGER.error(f"quant layer: {name} error: {exp}")
                        continue
                setattr(
                    mod,
                    "weight",
                    nn.Parameter(q_weight, requires_grad=False),
                )
    return model


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
