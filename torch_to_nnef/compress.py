"""Compression module mostly used as demonstration purpose.

Examplify: how to implement quantization
"""

import logging
import typing as T
from collections import defaultdict
from functools import partial

import torch
from torch import nn

from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorMisuse,
)
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.quant import (
    QTensor,
    fp_to_tract_q4_0_with_min_max_calibration,
    fp_to_tract_q8_1_with_min_max_calibration,
)
from torch_to_nnef.tensor.updater import ModTensorUpdater

LOGGER = logging.getLogger(__name__)


def _calib_min_max_fn(weight, name, kwargs, fp_to_qtensor_fn):
    q_weight = fp_to_qtensor_fn(
        weight,
        **{k: v for k, v in kwargs.items() if k in ["percentile"]},
    )
    q_weight.nnef_name = f"{name}.weight"
    return q_weight


def _quantize_weights_min_max(
    model: nn.Module,
    fp_to_qtensor_fn: T.Callable,
    suffix_name: str,
    default_module_classes: T.Tuple[T.Type[nn.Module], ...],
    **kwargs,
):
    to_quantize_module_classes = kwargs.get(
        "to_quantize_module_classes", default_module_classes
    )
    assert isinstance(to_quantize_module_classes, tuple), (
        to_quantize_module_classes
    )
    assert all(issubclass(_, nn.Module) for _ in to_quantize_module_classes), (
        to_quantize_module_classes
    )
    with torch.no_grad():
        ids_to_qtensor: T.Dict[int, T.Union[QTensor, OffloadedTensor]] = {}
        # try to avoid quant if used in other operators like mix of
        # embedding/linear if linear only quant
        mod_tensor_updater = ModTensorUpdater(model)
        # direct scan: torch < 2.0 named_parameters drops tied duplicates
        # so ModTensorUpdater.id_to_modules would miss shared usages
        id_to_modules: T.Dict[int, T.List[nn.Module]] = defaultdict(list)
        for owner in model.modules():
            for param in owner._parameters.values():
                if param is not None:
                    id_to_modules[id(param)].append(owner)

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
                    for m in id_to_modules[weight_id]
                ):
                    clss = [m.__class__ for m in id_to_modules[weight_id]]
                    LOGGER.warning(
                        "detected shared weight: '%s' candidate "
                        "has incompatible layer usage: %s, "
                        "but requested %s",
                        name,
                        clss,
                        to_quantize_module_classes,
                    )
                    continue
                try:
                    q_weight = offloaded_tensor_qtensor(
                        partial(
                            _calib_min_max_fn,
                            name=name,
                            kwargs=kwargs,
                            fp_to_qtensor_fn=fp_to_qtensor_fn,
                        ),
                        mod.weight,
                        suffix_name,
                    )
                except T2NErrorImpossibleQuantization as exp:
                    LOGGER.error("quant layer: %s error: %s", name, exp)
                    continue
                # => needs assignation next cause update_by_ref
                # may create new Parameter object
                q_weight = mod_tensor_updater.update_by_ref(
                    mod.weight, q_weight
                )
                ids_to_qtensor[id(q_weight)] = q_weight
    return model


def quantize_weights_min_max_Q4_0(model: nn.Module, **kwargs):  # noqa: N802
    """Example of quantization function for a model to Q40."""
    return _quantize_weights_min_max(
        model,
        fp_to_tract_q4_0_with_min_max_calibration,
        "q40_min_max",
        (nn.Linear,),
        **kwargs,
    )


def quantize_weights_min_max_Q8_1(model: nn.Module, **kwargs):  # noqa: N802
    """Example of quantization function for a model to Q8_1.

    tract only supports Q8_1 weights for embedding/gathering operations
    so only `nn.Embedding` (or subclasses) can be quantized.
    """
    to_quantize_module_classes = kwargs.get(
        "to_quantize_module_classes", (nn.Embedding,)
    )
    if not all(issubclass(c, nn.Embedding) for c in to_quantize_module_classes):
        raise T2NErrorMisuse(
            "tract only supports Q8_1 on embeddings, "
            f"but requested: {to_quantize_module_classes}"
        )
    return _quantize_weights_min_max(
        model,
        fp_to_tract_q8_1_with_min_max_calibration,
        "q81_min_max",
        (nn.Embedding,),
        **kwargs,
    )


def offloaded_tensor_qtensor(
    q_fn, tensor: torch.Tensor, suffix_name: str
) -> torch.Tensor:
    """Maintains a QTensor offloaded if original tensor is offloaded."""
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
    """Load a registry dynamically based on it's module path + dict name."""
    module_str, name = compression_registry_full_path.rsplit(".", maxsplit=1)
    mod = __import__(module_str, fromlist=[""])
    registry = getattr(mod, name)
    assert isinstance(registry, dict)
    return registry


DEFAULT_COMPRESSION_REGISTRY = "torch_to_nnef.compress.DEFAULT_COMPRESSION"
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
    "min_max_q8_1_embeddings": quantize_weights_min_max_Q8_1,
}
