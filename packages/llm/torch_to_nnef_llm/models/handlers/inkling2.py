"""Inkling 2 Mini text export with independent state for every loop pass."""

import math
from types import SimpleNamespace

import torch
from torch.nn import functional as F

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

    def update_conv_state(self, state, layer_idx, state_idx, conv_kernel_size):
        layer = self.layers[layer_idx]
        joined = torch.cat((layer.conv_states[state_idx], state), dim=-1)
        # A zero-width history needs an explicit slice: -0 means all tokens.
        width = conv_kernel_size - 1
        layer.conv_states[state_idx] = (
            joined[..., -width:] if width else joined[..., :0]
        )
        return joined


class _ShortConvolution(torch.nn.Module):
    """Convolve history and current tokens without data-dependent dispatch."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(
        self,
        hidden_states,
        dynamic_input=None,
        cache=None,
        conv_mask=None,
        cache_offset=0,
        **kwargs,
    ):
        if not isinstance(cache, _InklingCache):
            return self.original(
                hidden_states,
                dynamic_input=dynamic_input,
                cache=cache,
                conv_mask=conv_mask,
                cache_offset=cache_offset,
                **kwargs,
            )
        mod = self.original
        residual = hidden_states.float()
        context = cache.update_conv_state(
            residual.transpose(1, 2),
            mod.layer_idx + cache_offset,
            mod.conv_idx,
            mod.conv_kernel_size,
        )
        if hasattr(mod, "static_weight"):
            batch, seq, channels = hidden_states.shape
            weights = (
                mod.weight_proj(dynamic_input)
                .float()
                .reshape(batch, seq, mod.num_heads, 1, mod.conv_kernel_size)
            )
            weights = weights + mod.static_weight.float().T.reshape(
                mod.num_heads, -1, mod.conv_kernel_size
            )
            # Fixed tap loop avoids aten::unfold and preserves symbolic S.
            taps = []
            positions = torch.arange(seq, device=context.device)
            for lag in range(mod.conv_kernel_size):
                start = mod.conv_kernel_size - 1 - lag
                tap = context.index_select(2, positions + start).transpose(1, 2)
                tap = tap.reshape(batch, seq, mod.num_heads, -1)
                taps.append(tap * weights[..., lag])
            out = sum(taps).reshape(batch, seq, channels)
            divisor = math.sqrt(2 * mod.conv_kernel_size + 1)
        else:
            out = F.conv1d(
                context.to(mod.conv1d.weight.dtype),
                mod.conv1d.weight,
                mod.conv1d.bias,
                groups=mod.conv1d.groups,
            ).transpose(1, 2)
            divisor = math.sqrt(mod.conv_kernel_size + 1)
        return ((out + residual) / divisor).to(hidden_states.dtype)


class _Experts(torch.nn.Module):
    """Trace routing dynamically while retaining Inkling's softcapped gates.

    tract_moe_ffn's SwiGLU does not implement Inkling's two tanh softcaps.
    Gather only selected experts and express their computation as matmuls.
    """

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, hidden_states, top_k_index, top_k_weights):
        mod = self.original
        weights = mod.gate_up_proj[top_k_index]
        gate_up = torch.matmul(
            weights, hidden_states[:, None, :, None]
        ).squeeze(-1)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = mod.act_fn(gate, up)
        out = torch.matmul(
            mod.down_proj[top_k_index], activated.unsqueeze(-1)
        ).squeeze(-1)
        return (
            (out.float() * top_k_weights.unsqueeze(-1))
            .sum(1)
            .to(hidden_states.dtype)
        )


class _Router(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, hidden_states):
        mod = self.original
        logits = F.linear(hidden_states.float(), mod.weight)
        scores = mod.score_fn(logits)
        # Expert summation is permutation invariant; tract requires sorted k.
        indices = torch.topk(
            scores + mod.e_score_correction_bias,
            mod.top_k,
            dim=-1,
            sorted=True,
        ).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return logits, weights * mod.routed_scaling_factor, indices


class _RelativePositionBias(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, hidden_states, relative_distance, position_ids):
        mod = self.original
        states = mod.linear(hidden_states).reshape(
            *hidden_states.shape[:-1], mod.num_heads, -1
        )
        states = mod.norm(states, position_ids)
        logits = (states @ mod.proj).transpose(1, 2)
        indices = relative_distance.clamp(0, mod.rel_extent - 1)
        indices = (
            indices.unsqueeze(0).unsqueeze(0).expand(1, mod.num_heads, -1, -1)
        )
        valid = (relative_distance >= 0).to(logits.dtype) * (
            relative_distance < mod.rel_extent
        ).to(logits.dtype)
        return logits.gather(3, indices) * valid


class _AttentionResidual(torch.nn.Module):
    """Broadcast fixed residual queries without expanding symbolic prefixes."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, block_states, partial_block_states, loop_layer_idx=None):
        mod = self.original
        states = mod.norm(
            torch.cat(
                (block_states, partial_block_states.unsqueeze(-2)), dim=-2
            ).float()
        )
        query = (
            mod.query if loop_layer_idx is None else mod.query[loop_layer_idx]
        )
        query = query.float().reshape(1, 1, -1, mod.config.hidden_size)
        scores = (
            query @ states.transpose(-1, -2)
        ) * mod.config.hidden_size**-0.5
        output = scores.softmax(-1) @ states
        return output.unbind(-2) if loop_layer_idx is None else output


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
                if kind in (
                    "Inkling2LoopShortConvolution",
                    "Inkling2LoopDynamicShortConvolution",
                ):
                    setattr(parent, name, _ShortConvolution(module))
                elif kind == "Inkling2LoopExperts":
                    setattr(parent, name, _Experts(module))
                elif kind == "Inkling2LoopTopKRouter":
                    setattr(parent, name, _Router(module))
                elif kind == "Inkling2LoopRelativePositionBias":
                    setattr(parent, name, _RelativePositionBias(module))
                elif kind in (
                    "Inkling2LoopBlockAttentionResidual",
                    "Inkling2LoopMessageAttentionResidual",
                ):
                    setattr(parent, name, _AttentionResidual(module))
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
