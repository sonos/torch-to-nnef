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
