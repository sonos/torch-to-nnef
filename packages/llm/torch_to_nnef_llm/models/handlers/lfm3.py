"""LFM3 streaming export with explicit convolution and attention state."""

from types import SimpleNamespace

import torch

from .base import IOSpec, StateContext
from .default import DefaultArchitectureHandler
from .registry import register_handler


class _ExportCache:
    """Functional cache: retain full KV history, even for sliding attention.

    Keeping a common P axis makes positions and chunked prefill independent
    of cache truncation. The sliding mask still enforces the model's window.
    Convolution history is fixed width and never mutated in place.
    """

    def __init__(self, config, states):
        self.layers = []
        cursor = 0
        self.first_attention = None
        for idx, kind in enumerate(config.layer_types):
            if kind == "conv":
                layer = SimpleNamespace(
                    conv_states={0: states[cursor]}, record_past=True
                )
                cursor += 1
            else:
                layer = SimpleNamespace(
                    keys=states[cursor], values=states[cursor + 1]
                )
                cursor += 2
                if self.first_attention is None:
                    self.first_attention = idx
            self.layers.append(layer)

    def get_seq_length(self, layer_idx=None):
        idx = self.first_attention if layer_idx is None else layer_idx
        return self.layers[idx].keys.shape[2]

    def has_previous_state(self, layer_idx=None):
        return True

    def update(self, key, value, layer_idx, *args, **kwargs):
        layer = self.layers[layer_idx]
        layer.keys = torch.cat((layer.keys, key), dim=2)
        layer.values = torch.cat((layer.values, value), dim=2)
        return layer.keys, layer.values

    def update_conv_state(self, state, layer_idx, conv_kernel_size, **kwargs):
        layer = self.layers[layer_idx]
        joined = torch.cat((layer.conv_states[0], state), dim=-1)
        layer.conv_states = {0: joined[..., -(conv_kernel_size - 1) :]}
        return joined


@register_handler
class Lfm3ArchitectureHandler(DefaultArchitectureHandler):
    """Export unpadded, batch-one LFM3 continuation and prefill."""

    ARCH_NAMES = ("lfm3",)

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
            raise NotImplementedError(
                "LFM3 requires mixed convolution/KV state"
            )
        config = config_helper.decoder_conf
        if not any(t != "conv" for t in config.layer_types):
            raise ValueError("LFM3 export requires an attention layer")
        ids = tokenizer(sample_text, return_tensors="pt").input_ids
        assert ids.shape[1] >= n_input_tokens
        inputs = [ids[:, :n_input_tokens]]
        names = ["input_ids"]
        outputs = ["outputs"]
        axes = {"input_ids": {1: "S"}}
        for idx, kind in enumerate(config.layer_types):
            if kind == "conv":
                shapes = [
                    ("conv", (1, config.hidden_size, config.conv_L_cache - 1))
                ]
            elif kind in ("full_attention", "sliding_attention"):
                shape = (
                    1,
                    config.num_key_value_heads,
                    n_past_input_tokens,
                    config_helper.get_head_dim(),
                )
                shapes = [("key", shape), ("value", shape)]
            else:
                raise ValueError(f"Unsupported LFM3 layer type: {kind}")
            for state_kind, shape in shapes:
                name = f"cache_{state_kind}_{idx}"
                inputs.append(torch.zeros(shape, dtype=inputs_dtype))
                names.append(f"in_{name}")
                outputs.append(f"out_{name}")
                if state_kind != "conv":
                    axes[f"in_{name}"] = {2: "P"}
        return IOSpec(tuple(inputs), names, outputs, axes)

    def build_forward_inputs(self, *, inputs, wrapper):
        config = wrapper.model.config
        cache = _ExportCache(config, inputs[1:])
        past = cache.get_seq_length()
        mask, positions, query = self.build_causal_mask_with_past(
            seq_length=inputs[0].shape[1],
            past_length=past,
            device=inputs[0].device,
        )
        keys = torch.arange(past + inputs[0].shape[1], device=inputs[0].device)
        outside = (
            keys.unsqueeze(0) <= query.unsqueeze(1) - config.sliding_window
        )
        sliding = torch.where(outside, torch.finfo(mask.dtype).min, mask)
        return StateContext(
            model_inputs={
                "input_ids": inputs[0],
                "past_key_values": cache,
                "use_cache": True,
                "position_ids": positions,
                "attention_mask": {
                    "full_attention": mask,
                    "sliding_attention": sliding,
                    "conv": None,
                },
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
        for layer in state_context.model_inputs["past_key_values"].layers:
            if hasattr(layer, "keys"):
                outputs.extend((layer.keys, layer.values))
            else:
                outputs.append(layer.conv_states[0])
        return outputs
