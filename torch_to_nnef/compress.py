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
    T2NErrorNotImplemented,
)
from torch_to_nnef.op.custom_extractors.moe import (
    SelectedGatedExperts,
    iter_moe_routers,
    mark_moe_experts_for_q40,
)
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.quant import (
    fp_to_tract_iq2_s,
    fp_to_tract_q4_0_with_min_max_calibration,
    fp_to_tract_q4_k,
    fp_to_tract_q5_k,
    fp_to_tract_q6_k,
    fp_to_tract_q8_0_with_min_max_calibration,
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
    # Skip shared weights too: updating another owner would also quantize
    # the excluded module through ModTensorUpdater.update_by_ref.
    skipped_weight_ids = {
        id(param)
        for module in kwargs.get("modules_to_skip", ())
        for param in module.parameters()
    }
    with torch.no_grad():
        quantized_id_to_name: T.Dict[int, str] = {}
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
                if weight_id in skipped_weight_ids:
                    continue
                if weight_id in quantized_id_to_name:
                    LOGGER.info(
                        "detected shared weight between: '%s' and '%s.weight'",
                        quantized_id_to_name[weight_id],
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
                quantized_id_to_name[id(q_weight)] = f"{name}.weight"
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


def _quantize_selected_experts_q40(
    model: nn.Module, *, modules_to_skip=(), percentile=1.0
) -> int:
    """Quantize selected-expert tables in place and return the block count."""
    updater = ModTensorUpdater(model)
    skipped = {
        id(param) for module in modules_to_skip for param in module.parameters()
    }
    selected_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, SelectedGatedExperts):
            continue
        for attr in ("gate_up_proj", "down_proj"):
            weight = getattr(module, attr)
            if id(weight) in skipped:
                continue
            quantized = offloaded_tensor_qtensor(
                partial(
                    fp_to_tract_q4_0_with_min_max_calibration,
                    percentile=percentile,
                ),
                weight,
                "q40_min_max",
            )
            quantized.nnef_name = f"{name}.{attr}"
            updater.update_by_ref(weight, quantized)
        selected_count += 1
    return selected_count


def quantize_weights_min_max_Q4_0_with_experts(  # noqa: N802
    model: nn.Module, **kwargs
):
    """Q4_0 for Linear and Embedding weights and for MoE experts.

    Linear and Embedding weights are quantized as in
    ``min_max_q4_0_with_embeddings``, except MoE routers (including shared
    expert gates), whose weights stay in floating point. Fused expert weights
    are not ``nn.Linear`` modules, so every MoE block handled by the
    ``tract_moe_ffn`` extractor is marked instead: its experts are quantized
    to Q4_0 min-max during extraction, in the linear layout. Q4_0 blocks
    run along the reduction axis as tract's matmul kernels expect.
    SelectedGatedExperts weights are quantized as flattened expert rows
    and dequantized by gather at runtime. A model without experts
    gets the ``min_max_q4_0_with_embeddings`` result.
    """
    kwargs.setdefault("to_quantize_module_classes", (nn.Linear, nn.Embedding))
    kwargs["modules_to_skip"] = (
        *kwargs.get("modules_to_skip", ()),
        *iter_moe_routers(model),
    )
    model = quantize_weights_min_max_Q4_0(model, **kwargs)
    selected_count = _quantize_selected_experts_q40(
        model,
        modules_to_skip=kwargs["modules_to_skip"],
        percentile=kwargs.get("percentile", 1.0),
    )
    try:
        marked = mark_moe_experts_for_q40(model, layout="linear")
    except T2NErrorNotImplemented:
        marked = 0
    LOGGER.info(
        "Q4_0 experts: %d selected-expert blocks, %d fused blocks marked",
        selected_count,
        marked,
    )
    return model


def _quantize_embeddings(
    model: nn.Module, q_fn, suffix_name: str, format_name: str, **kwargs
):
    to_quantize_module_classes = kwargs.get(
        "to_quantize_module_classes", (nn.Embedding,)
    )
    if not all(issubclass(c, nn.Embedding) for c in to_quantize_module_classes):
        raise T2NErrorMisuse(
            f"tract only supports {format_name} on embeddings, "
            f"but requested: {to_quantize_module_classes}"
        )
    return _quantize_weights_min_max(
        model, q_fn, suffix_name, (nn.Embedding,), **kwargs
    )


def quantize_weights_min_max_Q8_1(model: nn.Module, **kwargs):  # noqa: N802
    """Example of quantization function for a model to Q8_1.

    tract only supports Q8_1 weights for embedding/gathering operations
    so only `nn.Embedding` (or subclasses) can be quantized.
    """
    return _quantize_embeddings(
        model,
        fp_to_tract_q8_1_with_min_max_calibration,
        "q81_min_max",
        "Q8_1",
        **kwargs,
    )


def quantize_weights_min_max_Q8_0(model: nn.Module, **kwargs):  # noqa: N802
    """Example of quantization function for a model to Q8_0.

    tract only supports Q8_0 weights for embedding/gathering operations
    so only `nn.Embedding` (or subclasses) can be quantized.
    Requires tract >= 0.23.10.
    """
    return _quantize_embeddings(
        model,
        fp_to_tract_q8_0_with_min_max_calibration,
        "q80_min_max",
        "Q8_0",
        **kwargs,
    )


def _quantize_embeddings_ggml(
    model: nn.Module, q_fn, suffix_name: str, format_name: str, **kwargs
):
    """Quantize embeddings with a ggml reference quantizer (no calibration)."""
    if "percentile" in kwargs:
        raise T2NErrorMisuse(
            f"the ggml {format_name} quantizer has no percentile calibration"
        )
    return _quantize_embeddings(model, q_fn, suffix_name, format_name, **kwargs)


def quantize_weights_ggml_Q4_K(model: nn.Module, **kwargs):  # noqa: N802
    """Example of quantization function for a model to Q4_K.

    Uses the ggml reference quantizer (`make_qkx2_quants` search per
    sub-block, no importance matrix).
    tract only supports Q4_K weights for embedding/gathering operations
    so only `nn.Embedding` (or subclasses) can be quantized.
    """
    return _quantize_embeddings_ggml(
        model, fp_to_tract_q4_k, "q4k_ggml", "Q4_K", **kwargs
    )


def quantize_weights_ggml_Q5_K(model: nn.Module, **kwargs):  # noqa: N802
    """Example of quantization function for a model to Q5_K.

    Uses the ggml reference quantizer (`make_qkx2_quants` search per
    sub-block, no importance matrix).
    tract only supports Q5_K weights for embedding/gathering operations
    so only `nn.Embedding` (or subclasses) can be quantized.
    """
    return _quantize_embeddings_ggml(
        model, fp_to_tract_q5_k, "q5k_ggml", "Q5_K", **kwargs
    )


def quantize_weights_ggml_Q6_K(model: nn.Module, **kwargs):  # noqa: N802
    """Example of quantization function for a model to Q6_K.

    Uses the ggml reference quantizer (`make_qx_quants` search per
    sub-block, no importance matrix).
    tract only supports Q6_K weights for embedding/gathering operations
    so only `nn.Embedding` (or subclasses) can be quantized.
    """
    return _quantize_embeddings_ggml(
        model, fp_to_tract_q6_k, "q6k_ggml", "Q6_K", **kwargs
    )


def quantize_weights_ggml_IQ2_S(model: nn.Module, **kwargs):  # noqa: N802
    """Example of quantization function for a model to IQ2_S.

    Uses the ggml reference quantizer (grid nearest neighbour search with
    a weighted scale fit, no importance matrix).
    tract only supports IQ2_S weights for embedding/gathering operations
    so only `nn.Embedding` (or subclasses) can be quantized.
    """
    return _quantize_embeddings_ggml(
        model, fp_to_tract_iq2_s, "iq2s_ggml", "IQ2_S", **kwargs
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
    "min_max_q4_0_with_embeddings_and_experts": (
        quantize_weights_min_max_Q4_0_with_experts
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
    "min_max_q8_0_embeddings": quantize_weights_min_max_Q8_0,
    "min_max_q8_1_embeddings": quantize_weights_min_max_Q8_1,
    "ggml_q4_k_embeddings": quantize_weights_ggml_Q4_K,
    "ggml_q5_k_embeddings": quantize_weights_ggml_Q5_K,
    "ggml_q6_k_embeddings": quantize_weights_ggml_Q6_K,
    "ggml_iq2_s_embeddings": quantize_weights_ggml_IQ2_S,
}
