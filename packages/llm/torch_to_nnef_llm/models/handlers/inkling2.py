"""Inkling 2 Mini text export with independent state for every loop pass."""

from types import SimpleNamespace

import torch

from torch_to_nnef.op.custom_extractors.moe import SelectedGatedExperts

from .base import IOSpec, StateContext
from .default import DefaultArchitectureHandler
from .registry import register_handler


def _text_config(config):
    return getattr(config, "text_config", config)


def _cache_layer_ids(config):
    # HF appends subsequent passes after all physical layers, including tail.
    return list(range(config.num_hidden_layers)) + list(
        range(config.loop_layer_index_start, config.loop_layer_index_end)
    ) * (config.loop_num_passes - 1)


class _InklingCache:
    """Functional KV/conv state; full history keeps a common symbolic P axis."""

    def __init__(self, config, states):
        self.layers = []
        self.is_sliding = []
        for i, layer_idx in enumerate(_cache_layer_ids(config)):
            key, value, *conv = states[6 * i : 6 * (i + 1)]
            self.layers.append(
                SimpleNamespace(keys=key, values=value, conv_states=conv)
            )
            self.is_sliding.append(
                config.layer_types[layer_idx] == "hybrid_sliding"
            )

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].keys.shape[2]

    def get_query_offset(self, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx=0):
        return self.get_seq_length(layer_idx) + query_length, 0

    def update(self, key, value, layer_idx, **kwargs):
        layer = self.layers[layer_idx]
        layer.keys = torch.cat((layer.keys, key), dim=2)
        layer.values = torch.cat((layer.values, value), dim=2)
        return layer.keys, layer.values

    def get_conv_state(self, layer_idx, state_idx):
        """Expose no mutable view: upstream must use update_conv_state.

        Returning None selects the general convolution path for every chunk
        size, including single-token decoding. That path receives the saved
        history through update_conv_state and avoids in-place cache updates.
        """
        return None

    def update_conv_state(self, state, layer_idx, state_idx, conv_kernel_size):
        layer = self.layers[layer_idx]
        joined = torch.cat((layer.conv_states[state_idx], state), dim=-1)
        # A zero-width history needs an explicit slice: -0 means all tokens.
        width = conv_kernel_size - 1
        layer.conv_states[state_idx] = (
            joined[..., -width:] if width else joined[..., :0]
        )
        return joined


@register_handler
class Inkling2TextArchitectureHandler(DefaultArchitectureHandler):
    """Text-only Inkling decoder with KV and four conv states per visit."""

    ARCH_NAMES = ("inkling2_loop_text",)
    carries_recurrent_state = True

    def has_dequantized_checkpoint_weights(self, model) -> bool:
        raw = getattr(model.config, "quantization_config", None)
        if (
            isinstance(raw, dict)
            and raw.get("quant_method") == "modelopt"
            and not getattr(model, "hf_quantizer", None)
        ):
            # Upstream Inkling checkpoint conversions dequantize mixed NVFP4 /
            # MXFP8 experts without an HfQuantizer, leaving the source metadata.
            # Verify the resulting expert tensors before treating it as dense.
            experts = [
                module
                for module in model.modules()
                if type(module).__name__ == "Inkling2LoopExperts"
            ]
            if experts and all(
                module.gate_up_proj.dtype
                in (torch.float32, torch.float16, torch.bfloat16, torch.float64)
                and module.down_proj.dtype
                in (torch.float32, torch.float16, torch.bfloat16, torch.float64)
                and tuple(module.gate_up_proj.shape)
                == (
                    module.num_experts,
                    2 * module.intermediate_dim,
                    module.hidden_dim,
                )
                and tuple(module.down_proj.shape)
                == (
                    module.num_experts,
                    module.hidden_dim,
                    module.intermediate_dim,
                )
                for module in experts
            ):
                return True
        return False

    def prepare_model_for_export(self, model):
        if getattr(model, "_t2n_inkling_prepared", False):
            return
        for parent in list(model.modules()):
            for name, module in list(parent.named_children()):
                kind = type(module).__name__
                if kind == "Inkling2LoopExperts":
                    setattr(
                        parent,
                        name,
                        SelectedGatedExperts(
                            module.gate_up_proj, module.down_proj, module.act_fn
                        ),
                    )
        model._t2n_inkling_prepared = True

    def build_input_spec(
        self,
        *,
        tokenizer,
        config_helper,
        inputs_dtype,
        sample_text,
        n_input_tokens,
        n_past_input_tokens,
        real_kv_cache=None,
    ):
        if real_kv_cache is not None:
            raise ValueError(
                "Inkling requires convolution state as well as KV state"
            )
        config = config_helper.decoder_conf
        ids = tokenizer(sample_text, return_tensors="pt").input_ids
        assert ids.shape[1] >= n_input_tokens
        inputs = [ids[:, :n_input_tokens]]
        names, outputs = ["input_ids"], ["outputs"]
        axes = {"input_ids": {1: "S"}}
        for idx, physical in enumerate(_cache_layer_ids(config)):
            sliding = config.layer_types[physical] == "hybrid_sliding"
            heads = (
                config.swa_num_key_value_heads
                if sliding
                else config.num_key_value_heads
            )
            dim = config.swa_head_dim if sliding else config.head_dim
            kv_shape = (1, heads, n_past_input_tokens, dim)
            shapes = [("key", kv_shape), ("value", kv_shape)]
            for conv_idx, channels in enumerate(
                (
                    heads * dim,
                    heads * dim,
                    config.hidden_size,
                    config.hidden_size,
                )
            ):
                shapes.append(
                    (
                        f"conv{conv_idx}",
                        (1, channels, config.conv_kernel_size - 1),
                    )
                )
            for kind, shape in shapes:
                name = f"cache_{kind}_{idx}"
                dtype = (
                    torch.float32 if kind.startswith("conv") else inputs_dtype
                )
                inputs.append(torch.zeros(shape, dtype=dtype))
                names.append(f"in_{name}")
                outputs.append(f"out_{name}")
                if kind in ("key", "value"):
                    axes[f"in_{name}"] = {2: "P"}
        return IOSpec(tuple(inputs), names, outputs, axes)

    def build_forward_inputs(self, *, inputs, wrapper):
        config = _text_config(wrapper.model.config)
        cache = _InklingCache(config, inputs[1:])
        past = cache.get_seq_length()
        seq = inputs[0].shape[1]
        masks = self.build_attention_mask_mapping(
            seq_length=seq,
            past_length=past,
            sliding_window=config.sliding_window_size,
            device=inputs[0].device,
        )
        masks["linear_attention"] = None
        positions = torch.arange(past, past + seq, device=inputs[0].device)
        return StateContext(
            model_inputs={
                "input_ids": inputs[0],
                "cache": cache,
                "use_cache": True,
                "position_ids": positions.unsqueeze(0),
                "attention_mask": masks,
            },
            state={},
        )

    def build_forward_outputs(
        self,
        *,
        model,
        model_outputs,
        state_context,
        num_logits_to_keep,
    ):
        outputs = [model_outputs["logits"]]
        for layer in state_context.model_inputs["cache"].layers:
            outputs.extend((layer.keys, layer.values, *layer.conv_states))
        return outputs


@register_handler
class Inkling2ArchitectureHandler(Inkling2TextArchitectureHandler):
    """Load the released multimodal checkpoint for text generation."""

    ARCH_NAMES = ("inkling2_loop",)

    @staticmethod
    def get_auto_model_class(transformers):
        try:
            return transformers.Inkling2LoopForConditionalGeneration
        except AttributeError as exc:
            raise ImportError(
                "Inkling 2 requires a transformers version with "
                "Inkling2LoopForConditionalGeneration; install transformers "
                "from the upstream main branch."
            ) from exc
