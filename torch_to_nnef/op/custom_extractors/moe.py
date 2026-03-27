"""MoE FFN export to tract_moe_ffn operator (tract_transformers extension).

Supports:
- MoEFFN (reference wrapper)
- MixtralSparseMoeBlock (transformers)
- GptOssSparseMoeBlock (transformers, GPT-OSS)
- Qwen3_5MoeSparseMoeBlock (transformers, shared expert decomposed outside)

All variants are normalized to the same tract_moe_ffn signature:
    inputs:  x [T,D], wg [E,D], w1 [E,D,H], w2 [E,H,D], w3 [E,D,H]
    attrs:   k (int), activation (str), normalize_gates (bool)
    output:  y [T,D]
"""

import logging
import typing as T

import torch
from torch import nn

from torch_to_nnef.exceptions import (
    T2NErrorNotImplemented,
    T2NErrorStrictNNEFSpec,
)
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight adapters — normalize diverse layouts into unified tensors
# ---------------------------------------------------------------------------


class _MoEWeightAdapter:
    """Base adapter: extract MoE weights from a module into canonical shapes."""

    def gate_weight(self, m: nn.Module) -> torch.Tensor:
        """Router weight [E, D]."""
        raise T2NErrorNotImplemented()

    def expert_w1(self, m: nn.Module) -> torch.Tensor:
        """Gate projection [E, D, H] (SwiGLU gate branch)."""
        raise T2NErrorNotImplemented()

    def expert_w2(self, m: nn.Module) -> torch.Tensor:
        """Down projection [E, H, D]."""
        raise T2NErrorNotImplemented()

    def expert_w3(self, m: nn.Module) -> torch.Tensor:
        """Up projection [E, D, H] (SwiGLU up branch)."""
        raise T2NErrorNotImplemented()

    def top_k(self, m: nn.Module) -> int:
        raise T2NErrorNotImplemented()

    def activation(self, m: nn.Module) -> str:
        return "swiglu"

    def normalize_gates(self, m: nn.Module) -> bool:
        return True


class _MoEFFNAdapter(_MoEWeightAdapter):
    """Adapter for our reference MoEFFN wrapper."""

    def gate_weight(self, m: nn.Module) -> torch.Tensor:
        return m.gate.weight.detach()

    def expert_w1(self, m: nn.Module) -> torch.Tensor:
        return m.w1.detach()

    def expert_w2(self, m: nn.Module) -> torch.Tensor:
        return m.w2.detach()

    def expert_w3(self, m: nn.Module) -> torch.Tensor:
        # MoEFFN has no w3 — duplicate w1 shape as zeros
        # so the SwiGLU gate branch becomes a no-op multiply by 0+silu
        # In practice MoEFFN uses simple activation, not SwiGLU.
        raise T2NErrorNotImplemented(
            "MoEFFN uses simple activation, not SwiGLU. "
            "Use activation attr instead."
        )

    def top_k(self, m: nn.Module) -> int:
        return m.k

    def activation(self, m: nn.Module) -> str:
        return m.activation_name

    def normalize_gates(self, m: nn.Module) -> bool:
        return m.normalize_gates


class _MixtralAdapter(_MoEWeightAdapter):
    """Adapter for transformers MixtralSparseMoeBlock.

    Handles two layouts:
    - Legacy (transformers <4.52): ModuleList of experts with w1/w2/w3
      w1=gate_proj, w2=down_proj, w3=up_proj
    - Modern (transformers >=4.52): fused MixtralExperts with
      gate_up_proj [E, 2*H, D] and down_proj [E, D, H]
    """

    def gate_weight(self, m: nn.Module) -> torch.Tensor:
        return m.gate.weight.detach()

    def _is_fused(self, m: nn.Module) -> bool:
        return hasattr(m.experts, "gate_up_proj")

    def _stack_legacy(self, m: nn.Module, attr: str) -> torch.Tensor:
        return torch.stack(
            [getattr(e, attr).weight.detach() for e in m.experts]
        )

    def expert_w1(self, m: nn.Module) -> torch.Tensor:
        if self._is_fused(m):
            half = m.experts.gate_up_proj.shape[1] // 2
            return m.experts.gate_up_proj.detach()[:, :half, :].transpose(
                -1, -2
            )
        # legacy: w1 = gate_proj [H, D] → [E, D, H]
        return self._stack_legacy(m, "w1").transpose(-1, -2)

    def expert_w2(self, m: nn.Module) -> torch.Tensor:
        if self._is_fused(m):
            return m.experts.down_proj.detach().transpose(-1, -2)
        # legacy: w2 = down_proj [D, H] → [E, H, D]
        return self._stack_legacy(m, "w2").transpose(-1, -2)

    def expert_w3(self, m: nn.Module) -> torch.Tensor:
        if self._is_fused(m):
            half = m.experts.gate_up_proj.shape[1] // 2
            return m.experts.gate_up_proj.detach()[:, half:, :].transpose(
                -1, -2
            )
        # legacy: w3 = up_proj [H, D] → [E, D, H]
        return self._stack_legacy(m, "w3").transpose(-1, -2)

    def top_k(self, m: nn.Module) -> int:
        return m.top_k


class _GptOssAdapter(_MixtralAdapter):
    """Adapter for transformers GPT-OSS MoE block.

    Same ModuleList-of-experts layout as Mixtral.
    Expert attributes may differ; override as needed.
    """

    def top_k(self, m: nn.Module) -> int:
        return m.top_k


class _QwenMoEAdapter(_MoEWeightAdapter):
    """Adapter for transformers Qwen2MoE / Qwen3.5 MoE.

    Experts are fused tensors: gate_up_proj [E, 2*H, D], down_proj [E, D, H].
    Shared expert is NOT handled here — it is decomposed outside the op.
    """

    def gate_weight(self, m: nn.Module) -> torch.Tensor:
        return m.gate.weight.detach()

    def expert_w1(self, m: nn.Module) -> torch.Tensor:
        # gate_up_proj [E, 2*H, D] → split → gate half [E, H, D] → [E, D, H]
        half = m.experts.gate_up_proj.detach().shape[1] // 2
        return m.experts.gate_up_proj.detach()[:, :half, :].transpose(-1, -2)

    def expert_w2(self, m: nn.Module) -> torch.Tensor:
        # down_proj [E, D, H] → [E, H, D]
        return m.experts.down_proj.detach().transpose(-1, -2)

    def expert_w3(self, m: nn.Module) -> torch.Tensor:
        # gate_up_proj [E, 2*H, D] → split → up half [E, H, D] → [E, D, H]
        half = m.experts.gate_up_proj.detach().shape[1] // 2
        return m.experts.gate_up_proj.detach()[:, half:, :].transpose(-1, -2)

    def top_k(self, m: nn.Module) -> int:
        return m.top_k


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------

_ADAPTER_BY_CLASSNAME: T.Dict[str, T.Type[_MoEWeightAdapter]] = {
    "MoEFFN": _MoEFFNAdapter,
    # Mixtral / Mistral
    "MixtralSparseMoeBlock": _MixtralAdapter,
    "MistralSparseMoeBlock": _MixtralAdapter,
    # GPT-OSS
    "GptOssSparseMoeBlock": _GptOssAdapter,
    # Qwen 2 / 3 / 3.5 MoE
    "Qwen2MoeSparseMoeBlock": _QwenMoEAdapter,
    "Qwen3MoeSparseMoeBlock": _QwenMoEAdapter,
    "Qwen3_5MoeSparseMoeBlock": _QwenMoEAdapter,
}


def _get_adapter(module: nn.Module) -> _MoEWeightAdapter:
    cls_name = type(module).__name__
    adapter_cls = _ADAPTER_BY_CLASSNAME.get(cls_name)
    if adapter_cls is None:
        raise T2NErrorNotImplemented(
            f"No MoE weight adapter for '{cls_name}'. "
            f"Supported: {sorted(_ADAPTER_BY_CLASSNAME.keys())}"
        )
    return adapter_cls()


# ---------------------------------------------------------------------------
# Reference MoEFFN module (for testing / wrapping custom MoE blocks)
# ---------------------------------------------------------------------------


class MoEFFN(nn.Module):
    """Reference MoE FFN block for export testing.

    For production models use the transformers extractors directly.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_hidden: int,
        k: int = 2,
        activation: str = "silu",
        normalize_gates: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.activation_name = activation
        self.normalize_gates = normalize_gates

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, d_hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_hidden, d_model))

        if bias:
            self.b1 = nn.Parameter(torch.zeros(num_experts, d_hidden))
            self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        else:
            self.b1 = None
            self.b2 = None

        self._init_weights()

        _activations = {"silu": nn.SiLU(), "gelu": nn.GELU(), "relu": nn.ReLU()}
        if activation not in _activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = _activations[activation]

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.w1)
        nn.init.kaiming_uniform_(self.w2)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: [T, D] -> [T, D]."""
        t_tokens, d = x.shape
        router_logits = self.gate(x)
        top_k_values, top_k_indices = torch.topk(router_logits, self.k, dim=-1)

        if self.normalize_gates:
            gate_weights = torch.softmax(top_k_values, dim=-1)
        else:
            gate_weights = top_k_values

        output = torch.zeros(t_tokens, d, device=x.device, dtype=x.dtype)

        for ki in range(self.k):
            expert_indices = top_k_indices[:, ki]
            weights = gate_weights[:, ki]

            for eid in range(self.num_experts):
                mask = expert_indices == eid
                if not mask.any():
                    continue

                xi = x[mask]
                h = xi @ self.w1[eid]
                if self.b1 is not None:
                    h = h + self.b1[eid]
                h = self.activation(h)
                yo = h @ self.w2[eid]
                if self.b2 is not None:
                    yo = yo + self.b2[eid]

                output[mask] += weights[mask].unsqueeze(-1) * yo

        return output


# ---------------------------------------------------------------------------
# Core NNEF conversion (shared by all extractors)
# ---------------------------------------------------------------------------


def _convert_moe_to_nnef(g, node, name_to_tensor, inference_target):
    """Emit tract_moe_ffn for any supported MoE module."""
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorStrictNNEFSpec(
            "MoE FFN export requires tract inference target "
            "(tract_moe_ffn is a tract extension)"
        )

    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef import torch_graph as tg

    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.op import helper

    moe = node.op_ref
    adapter = _get_adapter(moe)
    is_swiglu = adapter.activation(moe) == "swiglu"

    def _add_weight(name: str, data: torch.Tensor):
        wnode = tg.TensorVariable(
            name=f"{node.outputs[0].name}_{name}",
            data=data,
            shape=list(data.shape),
            dtype=data.dtype,
        )
        return helper.get_or_add_tensor_variable_in_nnef(
            g, wnode, name_to_tensor
        )

    input_tensor = helper.get_or_add_tensor_variable_in_nnef(
        g, node.inputs[0], name_to_tensor
    )

    wg = _add_weight("wg", adapter.gate_weight(moe))
    w1 = _add_weight("w1", adapter.expert_w1(moe))
    w2 = _add_weight("w2", adapter.expert_w2(moe))

    inputs = [input_tensor, wg, w1, w2]

    if is_swiglu:
        w3 = _add_weight("w3", adapter.expert_w3(moe))
        inputs.append(w3)

    attrs = {
        "k": adapter.top_k(moe),
        "activation": adapter.activation(moe),
        "normalize_gates": adapter.normalize_gates(moe),
    }

    if len(node.outputs) == 1:
        helper.add_single_output_op(
            g,
            node,
            name_to_tensor,
            nnef_op_type="tract_moe_ffn",
            inputs=inputs,
            attrs=attrs,
        )
    else:
        helper.add_multi_output_op(
            g,
            node,
            name_to_tensor,
            nnef_op_type="tract_moe_ffn",
            inputs=inputs,
            attrs=attrs,
        )

    return ["tract_transformers"]


# ---------------------------------------------------------------------------
# Extractors (one per MODULE_CLASS, all delegate to _convert_moe_to_nnef)
# ---------------------------------------------------------------------------


class MoEFFNExtractor(ModuleInfoExtractor):
    """Extractor for the reference MoEFFN wrapper."""

    MODULE_CLASS = MoEFFN

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        inference_target,
        **kw,
    ):
        return _convert_moe_to_nnef(g, node, name_to_tensor, inference_target)


# ---------------------------------------------------------------------------
# Lazy registration of transformers MoE classes
# ---------------------------------------------------------------------------


def _try_register(import_path: str, class_name: str):
    """Try to import a transformers MoE class and register an extractor."""
    try:
        # pylint: disable-next=import-outside-toplevel
        import importlib

        mod = importlib.import_module(import_path)
        moe_cls = getattr(mod, class_name)

        def _make_convert():
            def convert_to_nnef(
                self,
                g,
                node,
                name_to_tensor,
                null_ref,
                torch_graph,
                inference_target,
                **kw,
            ):
                return _convert_moe_to_nnef(
                    g,
                    node,
                    name_to_tensor,
                    inference_target,
                )

            return convert_to_nnef

        # dynamically create an extractor subclass
        extractor_cls = type(
            f"{class_name}Extractor",
            (ModuleInfoExtractor,),
            {
                "MODULE_CLASS": moe_cls,
                "convert_to_nnef": _make_convert(),
            },
        )
        # class creation triggers metaclass registration
        LOGGER.debug("Registered %s extractor", class_name)
        return extractor_cls
    except (ImportError, AttributeError):
        return None


def _register_all_transformers_moe():
    _candidates = [
        (
            "transformers.models.mixtral.modeling_mixtral",
            "MixtralSparseMoeBlock",
        ),
        (
            "transformers.models.gpt_oss.modeling_gpt_oss",
            "GptOssSparseMoeBlock",
        ),
        (
            "transformers.models.qwen2_moe.modeling_qwen2_moe",
            "Qwen2MoeSparseMoeBlock",
        ),
        (
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "Qwen3MoeSparseMoeBlock",
        ),
        (
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
            "Qwen3_5MoeSparseMoeBlock",
        ),
    ]
    for import_path, class_name in _candidates:
        _try_register(import_path, class_name)


_register_all_transformers_moe()
