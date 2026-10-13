import typing as T

import torch

from .base import StateContext
from .default import DefaultArchitectureHandler
from .registry import register_handler


@register_handler
class GptOssArchitectureHandler(DefaultArchitectureHandler):
    """Handler for GPT-OSS causal decoder models."""

    ARCH_NAMES = ("gpt_oss",)

    def prepare_model_for_export(self, model) -> None:
        # torch grouped_mm currently only has a BF16 fake/meta path. T2N exports
        # GPT-OSS MoE blocks as tract_moe_ffn, but the root torch.jit trace must
        # still run first; use a traceable HF expert implementation for export.
        for module in model.modules():
            config = getattr(module, "config", None)
            if (
                config is not None
                and getattr(config, "_experts_implementation", None)
                == "grouped_mm"
            ):
                config._experts_implementation = "batched_mm"

    @staticmethod
    def _build_mask_mapping(
        *,
        seq_length: int,
        past_length: int,
        sliding_window: int,
        device: torch.device,
    ) -> T.Dict[str, torch.Tensor]:
        """Additive causal and causal-plus-window masks over ``S+P`` keys.

        Built from token positions with arange comparisons rather than a baked
        triangular constant, so the graph stays correct for any ``(S, P)`` at
        inference time.
        """
        # Float arithmetic throughout (comparisons cast straight to the mask
        # dtype, AND via multiply) rather than boolean `&` on tensors: t2n's
        # shape inference re-runs bitwise ops with a float placeholder and
        # fails with "bitwise_and not implemented for Float". Same reason as
        # the note in `gemma3_vl._build_mask_mapping`.
        dtype = torch.float32
        total_length = seq_length + past_length
        q_pos = torch.arange(
            past_length, total_length, device=device
        ).unsqueeze(1)
        k_pos = torch.arange(total_length, device=device).unsqueeze(0)

        visible_full = (k_pos <= q_pos).to(dtype)
        # Query at absolute position q sees keys in (q - window, q], matching
        # `masking_utils.sliding_window_overlay`'s `kv_idx > q_idx - window`.
        within_window = (k_pos > (q_pos - sliding_window)).to(dtype)
        visible_sliding = visible_full * within_window

        neg = torch.finfo(dtype).min

        def to_additive(visible: torch.Tensor) -> torch.Tensor:
            return ((1.0 - visible) * neg).unsqueeze(0).unsqueeze(0)

        return {
            "full_attention": to_additive(visible_full),
            "sliding_attention": to_additive(visible_sliding),
        }

    def build_forward_inputs(
        self,
        *,
        inputs: T.Tuple[torch.Tensor, ...],
        wrapper,
    ) -> StateContext:
        """Pass per-layer masks so sliding layers keep their window.

        GPT-OSS alternates ``sliding_attention`` and ``full_attention`` layers.
        The base handler hands the model a single 4D causal mask, and
        ``masking_utils._preprocess_mask_arguments`` returns any 4D mask as-is,
        so both ``create_causal_mask`` and ``create_sliding_window_causal_mask``
        early-exit with that same tensor. The model's mask mapping then holds
        the unwindowed mask under both keys and every layer attends over the
        whole context.

        That is invisible below the window, where the two masks agree, and
        degrades output as the sequence grows past it.
        """
        ctx = super().build_forward_inputs(inputs=inputs, wrapper=wrapper)
        if not getattr(wrapper, "force_causal_mask", False):
            return ctx

        config = wrapper.model.config
        sliding_window = int(getattr(config, "sliding_window", 0) or 0)
        layer_types = tuple(getattr(config, "layer_types", ()) or ())
        # Nothing to correct when the model is uniformly full-attention: the
        # base handler's single causal mask is already right.
        if sliding_window <= 0 or "sliding_attention" not in layer_types:
            return ctx

        ctx.model_inputs["attention_mask"] = self._build_mask_mapping(
            seq_length=inputs[0].shape[1],
            past_length=inputs[1].shape[2],
            sliding_window=sliding_window,
            device=inputs[0].device,
        )
        return ctx
