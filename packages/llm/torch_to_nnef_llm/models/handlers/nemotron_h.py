"""NemotronH (Mamba-2 / MoE / attention hybrid) decoder export.

The forward is reimplemented over HF's own modules so every layer state is
an explicit graph input and output:
- Mamba-2 layers: `cache_conv` [1, conv_dim, kernel] in the model dtype and
  `cache_ssm` [1, heads, head_dim, state] in f32, through the reified causal
  conv update and Mamba-2 recurrence (one tract op each, any S);
- attention layers (no positional encoding): `cache_key` / `cache_value`
  [1, kv_heads, P, head_dim];
- MoE and MLP layers are stateless; MoE blocks export to tract_moe_ffn.
"""

import typing as T

import torch
import torch.nn.functional as F

from torch_to_nnef.op.custom_extractors import (
    CausalConvUpdateReified,
    Mamba2RecurrentReified,
)
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef_llm.models.handlers.base import (
    ArchitectureHandler,
    IOSpec,
    StateContext,
)
from torch_to_nnef_llm.models.handlers.gdn_reify import _materialize_offloaded
from torch_to_nnef_llm.models.handlers.registry import register_handler

MAMBA = "linear_attention"
ATTENTION = "full_attention"


def _layer_states(config) -> T.List[T.Tuple[int, str, T.Tuple[str, str]]]:
    """(layer index, block type, state names) of the stateful layers."""
    states: T.List[T.Tuple[int, str, T.Tuple[str, str]]] = []
    for idx, kind in enumerate(config.layers_block_type):
        if kind == MAMBA:
            states.append(
                (idx, kind, (f"cache_conv_{idx}", f"cache_ssm_{idx}"))
            )
        elif kind == ATTENTION:
            states.append(
                (idx, kind, (f"cache_key_{idx}", f"cache_value_{idx}"))
            )
    return states


def mamba_forward(mixer, hidden, conv_state, ssm_state):
    """One NemotronH Mamba-2 mixer step over S tokens with explicit states.

    The time step is not clamped to `time_step_min`: HF only clamps in its
    chunked prefill path, not in its per-token path, nor does NVIDIA's own
    modeling code, so a clamp would make prefill and decode disagree.
    """
    if mixer.activation not in ("silu", "swish"):
        raise NotImplementedError(
            f"NemotronH conv activation {mixer.activation!r}"
        )
    seq = hidden.shape[1]
    heads, head_dim = mixer.num_heads, mixer.head_dim
    groups, d_state = mixer.n_groups, mixer.ssm_state_size
    gate, xbc, dt = mixer.in_proj(hidden).split(
        [mixer.intermediate_size, mixer.conv_dim, heads], dim=-1
    )
    xbc, conv_state = mixer.t2n_conv(
        xbc.transpose(1, 2),
        conv_state,
        mixer.conv1d.weight.squeeze(1),
        mixer.conv1d.bias,
    )
    x, b, c = xbc.transpose(1, 2).split(
        [mixer.intermediate_size, groups * d_state, groups * d_state], dim=-1
    )
    dt = F.softplus(dt.float() + mixer.dt_bias.float())
    a = -torch.exp(mixer.A_log.float())
    x = x.reshape(1, seq, heads, head_dim)
    y, ssm_state = mixer.t2n_ssm(
        (x.float() * dt[..., None]).to(x.dtype),
        b.reshape(1, seq, groups, d_state),
        c.reshape(1, seq, groups, d_state),
        dt * a,
        ssm_state,
    )
    y = y + x * mixer.D.to(x.dtype)[:, None]
    y = mixer.norm(y.reshape(1, seq, -1), gate)
    return mixer.out_proj(y.to(hidden.dtype)), conv_state, ssm_state


def attention_forward(attn, hidden, key_in, value_in, mask):
    """NemotronH attention (no positional encoding) over a KV prefix."""
    seq = hidden.shape[1]
    head_dim = attn.head_dim
    query = attn.q_proj(hidden).view(1, seq, -1, head_dim).transpose(1, 2)
    key = attn.k_proj(hidden).view(1, seq, -1, head_dim).transpose(1, 2)
    value = attn.v_proj(hidden).view(1, seq, -1, head_dim).transpose(1, 2)
    key = torch.cat([key_in, key], dim=2)
    value = torch.cat([value_in, value], dim=2)
    rep = attn.num_key_value_groups
    keys = key.repeat_interleave(rep, dim=1)
    values = value.repeat_interleave(rep, dim=1)
    weights = torch.matmul(query, keys.transpose(2, 3)) * attn.scaling
    weights = torch.softmax(weights.float() + mask, dim=-1).to(query.dtype)
    out = torch.matmul(weights, values).transpose(1, 2).reshape(1, seq, -1)
    return attn.o_proj(out), key, value


def decoder_forward(backbone, config, hidden, states, mask):
    """Run every block over `hidden` [1, S, D], threading layer states."""
    states = list(states)
    new_states: T.List[torch.Tensor] = []
    for layer in backbone.layers:
        residual = hidden
        # back to the residual dtype: a disk-offloaded norm may keep f32
        normed = layer.norm(hidden.to(layer.norm.weight.dtype)).to(hidden.dtype)
        if layer.block_type == MAMBA:
            out, conv, ssm = mamba_forward(
                layer.mixer, normed, states.pop(0), states.pop(0)
            )
            new_states += [conv, ssm]
        elif layer.block_type == ATTENTION:
            out, key, value = attention_forward(
                layer.mixer, normed, states.pop(0), states.pop(0), mask
            )
            new_states += [key, value]
        else:
            out = layer.mixer(normed)
        hidden = residual + out
    return backbone.norm_f(hidden), new_states


def install_reified_mamba(model) -> None:
    """Give each Mamba-2 mixer the shims its export is extracted from.

    Also loads back the small disk-offloaded tensors that tracing cannot
    handle: the mixer parameters used in trace-time arithmetic (time step,
    decay, conv) and every buffer of the model (router correction biases,
    norm statistics), a few KB each.
    """
    for module in model.modules():
        kind = type(module).__name__
        if kind == "NemotronHMamba2Mixer":
            for owner, attr in (
                (module.conv1d, "weight"),
                (module.conv1d, "bias"),
                (module, "A_log"),
                (module, "dt_bias"),
                (module, "D"),
            ):
                _materialize_offloaded(owner, attr)
        for name, buffer in list(module.named_buffers(recurse=False)):
            if isinstance(buffer, OffloadedTensor):
                setattr(module, name, buffer.reload().to(buffer.dtype))
    for module in model.modules():
        if type(module).__name__ == "NemotronHMamba2Mixer" and not hasattr(
            module, "t2n_conv"
        ):
            module.add_module("t2n_conv", CausalConvUpdateReified())
            module.add_module("t2n_ssm", Mamba2RecurrentReified())


def state_inputs(config, n_past, inputs_dtype):
    """Zero states, names and dynamic axes of the stateful layers."""
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    conv_dim = (
        config.mamba_num_heads * config.mamba_head_dim
        + 2 * config.n_groups * config.ssm_state_size
    )
    tensors, names, axes = [], [], {}
    for _, kind, layer_names in _layer_states(config):
        if kind == MAMBA:
            tensors.append(
                torch.zeros(
                    (1, conv_dim, config.conv_kernel), dtype=inputs_dtype
                )
            )
            # the recurrence runs in f32 and so does the state it carries
            tensors.append(
                torch.zeros(
                    (
                        1,
                        config.mamba_num_heads,
                        config.mamba_head_dim,
                        config.ssm_state_size,
                    ),
                    dtype=torch.float32,
                )
            )
        else:
            shape = (1, config.num_key_value_heads, n_past, head_dim)
            tensors += [
                torch.zeros(shape, dtype=inputs_dtype),
                torch.zeros(shape, dtype=inputs_dtype),
            ]
            for name in layer_names:
                axes[f"in_{name}"] = {2: "P"}
        names += list(layer_names)
    return tensors, names, axes


def past_length(config, states) -> int:
    for pos, (_, kind, _names) in enumerate(_layer_states(config)):
        if kind == ATTENTION:
            return states[2 * pos].shape[2]
    return 0


@register_handler
class NemotronHArchitectureHandler(ArchitectureHandler):
    """NemotronH causal LM with explicit Mamba-2 and attention states."""

    ARCH_NAMES = ("nemotron_h",)
    with_dyn_cache = False
    carries_recurrent_state = True

    def prepare_model_for_export(self, model) -> None:
        install_reified_mamba(model)

    def build_input_spec(
        self,
        *,
        tokenizer,
        config_helper,
        inputs_dtype: torch.dtype,
        sample_text: str,
        n_input_tokens: int,
        n_past_input_tokens: int,
        real_kv_cache: T.Optional[T.List[torch.Tensor]] = None,
    ) -> IOSpec:
        if real_kv_cache is not None:
            raise NotImplementedError(
                "NemotronH carries Mamba-2 states besides its KV cache"
            )
        config = config_helper.decoder_conf
        if n_input_tokens == config.conv_kernel:
            # the traced conv tells its input from its state by shape only
            n_input_tokens += 1
        ids = tokenizer(sample_text, return_tensors="pt").input_ids
        assert ids.shape[1] >= n_input_tokens
        tensors, names, axes = state_inputs(
            config, n_past_input_tokens, inputs_dtype
        )
        return IOSpec(
            inputs=(ids[:, :n_input_tokens], *tensors),
            input_names=["input_ids"] + [f"in_{n}" for n in names],
            output_names=["outputs"] + [f"out_{n}" for n in names],
            dynamic_axes={"input_ids": {1: "S"}, **axes},
        )

    def build_forward_inputs(self, *, inputs, wrapper) -> StateContext:
        config = wrapper.model.config
        input_ids, states = inputs[0], list(inputs[1:])
        mask, _, _ = self.build_causal_mask_with_past(
            seq_length=input_ids.shape[1],
            past_length=past_length(config, states),
            device=input_ids.device,
        )
        return StateContext(
            model_inputs={},
            state={"input_ids": input_ids, "states": states, "mask": mask},
        )

    def call_model(self, *, model, state_context, wrapper) -> T.Any:
        del wrapper
        st = state_context.state
        backbone = model.model
        hidden = backbone.embeddings(st["input_ids"])
        hidden, new_states = decoder_forward(
            backbone,
            model.config,
            hidden,
            st["states"],
            st["mask"],
        )
        return {"logits": model.lm_head(hidden), "new_states": new_states}

    def build_forward_outputs(
        self,
        *,
        model,
        model_outputs,
        state_context,
        num_logits_to_keep: int,
    ) -> T.List[torch.Tensor]:
        del model, state_context
        logits = model_outputs["logits"]
        if num_logits_to_keep:
            logits = logits[:, -num_logits_to_keep:]
        return [logits] + list(model_outputs["new_states"])
