import torch
from torch import nn

from torch_to_nnef.exceptions import TorchToNNEFImpossibleQuantization
from torch_to_nnef.llm_tract.models.base import TorchToNNEFWrappedLLM
from torch_to_nnef.log import log
from torch_to_nnef.qtensor.qtract import (
    fp_to_tract_q4_0_with_min_max_calibration,
)

LOGGER = log.getLogger(__name__)


def quantize_weights_min_max_Q4_0(
    wrapped_model: TorchToNNEFWrappedLLM, **kwargs
):
    with torch.no_grad():
        for name, mod in wrapped_model.named_modules():
            if isinstance(mod, (nn.Linear)):
                # NOTE: nn.Embedding will likely need per channel implem in Tract
                LOGGER.info(f"quantize layer: {name}")
                try:
                    q_weight = fp_to_tract_q4_0_with_min_max_calibration(
                        mod.weight
                    )
                except TorchToNNEFImpossibleQuantization as exp:
                    LOGGER.error(f"quant layer: {name} error: {exp}")
                    continue
                setattr(
                    mod,
                    "weight",
                    nn.Parameter(q_weight, requires_grad=False),
                )

    return wrapped_model


def dynamic_load_registry(compression_registry_full_path: str):
    module_str, name = compression_registry_full_path.rsplit(".", maxsplit=1)
    mod = __import__(module_str, fromlist=[""])
    registry = getattr(mod, name)
    assert isinstance(registry, dict)
    return registry


DEFAULT_COMPRESSION = {"min_max_q4_0": quantize_weights_min_max_Q4_0}
