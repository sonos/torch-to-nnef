"""MoE FFN export to tract_moe_ffn operator (tract_transformers extension).

Supports (transformers):
- MoEFFN (reference wrapper)
- MixtralSparseMoeBlock / MistralSparseMoeBlock
- GptOssMLP (router + expert biases, interleaved gate/up, clamped SwiGLU)
- Qwen2/Qwen3/Qwen3.5 MoE (Qwen2 & 3.5 shared expert decomposed outside)
- OlmoeSparseMoeBlock

All variants are normalized to the same tract_moe_ffn signature:
    inputs:  x [T,D], wg [E,D], w1 [E,D,H], w2 [E,H,D], w3 [E,D,H],
             optional biases (wg_bias, w1_bias, w3_bias, w2_bias)
    attrs:   k (int), activation (str), gate (softmax_topk | softmax_all |
             sigmoid | raw), optional act_alpha / act_limit (clamped SwiGLU),
             optional expert_layout ("canonical" | "linear")
    output:  y [T,D]
The default canonical expert layout matches the signature above. The optional
linear layout stores expert weights in their native linear-filter orientation:
    w1/w3 [E,H,D], w2 [E,D,H]
Layout selection is independent from whether those tensors are quantized.
A shared expert (Qwen2 / Qwen3.5) is emitted as a standard NNEF subgraph
added on top of the routed output, not baked into the op.
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
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.opaque import OpaqueTensorRef, opaque_to_final_tensor
from torch_to_nnef.tensor.quant import (
    QTensor,
    fp_to_tract_q4_0_with_min_max_calibration,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight adapters: normalize diverse layouts into unified tensors
# ---------------------------------------------------------------------------


class _MoEWeightAdapter:
    """Base adapter: extract MoE weights from a module into canonical shapes."""

    def __init__(self) -> None:
        self._constant_cache: T.Dict[int, torch.Tensor] = {}

    def _constant(self, tensor: torch.Tensor) -> torch.Tensor:
        """Materialize an export-time constant before taking tensor views."""
        key = (
            id(tensor.opaque_tensor)
            if isinstance(tensor, OpaqueTensorRef)
            else id(tensor)
        )
        if key not in self._constant_cache:
            if isinstance(tensor, OpaqueTensorRef):
                tensor = tensor.opaque_tensor
            self._constant_cache[key] = opaque_to_final_tensor(tensor).detach()
        return self._constant_cache[key]

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

    def expert_sources(self, m: nn.Module) -> T.Sequence[torch.Tensor]:
        """Raw tensors that feed expert_w1/w2/w3.

        This is used only to detect pre-quantized expert sources. Most adapters
        do not need to override it because split-time quantization is normally
        requested explicitly via the module marker.
        """
        return ()

    def top_k(self, m: nn.Module) -> int:
        raise T2NErrorNotImplemented()

    def activation(self, m: nn.Module) -> str:
        return "swiglu"

    def gate(self, m: nn.Module) -> str:
        """How router logits become top-k gate weights (tract_moe_ffn `gate`).

        One of: "softmax_topk" (softmax over the top-k logits), "softmax_all"
        (softmax over all experts, gather top-k, no renormalization),
        "sigmoid" (per-expert sigmoid), "raw" (raw top-k logits).
        """
        return "softmax_topk"

    # Optional biases (None unless the arch has them, e.g. gpt-oss).
    def gate_bias(self, m: nn.Module) -> T.Optional[torch.Tensor]:
        """Router bias [E]."""
        return None

    def expert_w1_bias(self, m: nn.Module) -> T.Optional[torch.Tensor]:
        """Gate projection bias [E, H]."""
        return None

    def expert_w2_bias(self, m: nn.Module) -> T.Optional[torch.Tensor]:
        """Down projection bias [E, D]."""
        return None

    def expert_w3_bias(self, m: nn.Module) -> T.Optional[torch.Tensor]:
        """Up projection bias [E, H]."""
        return None

    # Optional clamped-SwiGLU params (gpt-oss). When act_limit is not None the
    # op uses the clamped activation: gate.clamp(max=limit) /
    # up.clamp(+-limit) / glu = gate*sigmoid(alpha*gate) / out = (up+1)*glu.
    def act_alpha(self, m: nn.Module) -> T.Optional[float]:
        return None

    def act_limit(self, m: nn.Module) -> T.Optional[float]:
        return None

    # Optional always-on shared expert (e.g. Qwen2-MoE / Qwen3.5-MoE):
    #   out = routed_experts(x) + sigmoid(shared_gate(x)) * shared_mlp(x)
    # Returns the raw nn.Linear weights ([out, in], used directly as NNEF
    # linear filters) or None when the arch has no shared expert.
    def shared_expert(
        self, m: nn.Module
    ) -> T.Optional[T.Dict[str, torch.Tensor]]:
        return None


class _MoEFFNAdapter(_MoEWeightAdapter):
    """Adapter for our reference MoEFFN wrapper."""

    def gate_weight(self, m: nn.Module) -> torch.Tensor:
        return m.gate.weight.detach()

    def expert_w1(self, m: nn.Module) -> torch.Tensor:
        return m.w1.detach()

    def expert_w2(self, m: nn.Module) -> torch.Tensor:
        return m.w2.detach()

    def expert_w3(self, m: nn.Module) -> torch.Tensor:
        # MoEFFN has no w3, so duplicate w1 shape as zeros
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

    def gate(self, m: nn.Module) -> str:
        # MoEFFN's normalize_gates softmaxes the top-k logits; otherwise it
        # uses the raw top-k logits.
        return "softmax_topk" if m.normalize_gates else "raw"


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
            [self._constant(getattr(e, attr).weight) for e in m.experts]
        )

    def expert_w1(self, m: nn.Module) -> torch.Tensor:
        if self._is_fused(m):
            gate_up = self._constant(m.experts.gate_up_proj)
            half = gate_up.shape[1] // 2
            return gate_up[:, :half, :].transpose(-1, -2)
        # legacy: w1 = gate_proj [H, D] → [E, D, H]
        return self._stack_legacy(m, "w1").transpose(-1, -2)

    def expert_w2(self, m: nn.Module) -> torch.Tensor:
        if self._is_fused(m):
            return self._constant(m.experts.down_proj).transpose(-1, -2)
        # legacy: w2 = down_proj [D, H] → [E, H, D]
        return self._stack_legacy(m, "w2").transpose(-1, -2)

    def expert_w3(self, m: nn.Module) -> torch.Tensor:
        if self._is_fused(m):
            gate_up = self._constant(m.experts.gate_up_proj)
            half = gate_up.shape[1] // 2
            return gate_up[:, half:, :].transpose(-1, -2)
        # legacy: w3 = up_proj [H, D] → [E, D, H]
        return self._stack_legacy(m, "w3").transpose(-1, -2)

    def top_k(self, m: nn.Module) -> int:
        return m.top_k


class _GptOssAdapter(_MoEWeightAdapter):
    """Adapter for transformers GPT-OSS MoE block.

    Handles the modern GptOssMLP (transformers >=4.55: a `router` plus a fused
    `GptOssExperts`) and the older `GptOssSparseMoeBlock` name. gpt-oss differs
    from the generic op in several ways the adapter normalizes:
    - the router carries a bias and lives on `m.router` (older: `m.gate`)
    - experts fuse the gate and up projections INTERLEAVED inside gate_up_proj
      [E, D, 2H] (gate = [..., 0::2], up = [..., 1::2]), and carry biases
    - the activation is a clamped SwiGLU with alpha / limit and a (up + 1) term
    """

    @staticmethod
    def _router(m: nn.Module) -> nn.Module:
        return m.router if hasattr(m, "router") else m.gate

    def _d_model(self, m: nn.Module) -> int:
        return self.gate_weight(m).shape[1]

    def _gate_up(self, m: nn.Module) -> torch.Tensor:
        """gate_up_proj oriented as [E, D, 2H] (contracted axis = D)."""
        gu = self._constant(m.experts.gate_up_proj)
        d_model = self._d_model(m)
        if gu.shape[1] != d_model and gu.shape[2] == d_model:
            gu = gu.transpose(-1, -2)
        return gu

    def gate_weight(self, m: nn.Module) -> torch.Tensor:
        return self._router(m).weight.detach()

    def gate_bias(self, m: nn.Module) -> T.Optional[torch.Tensor]:
        b = getattr(self._router(m), "bias", None)
        return b.detach() if b is not None else None

    def expert_w1(self, m: nn.Module) -> torch.Tensor:
        # gate (activated) branch: interleaved even columns -> [E, D, H]
        return self._gate_up(m)[:, :, 0::2].contiguous()

    def expert_w3(self, m: nn.Module) -> torch.Tensor:
        # up branch: interleaved odd columns -> [E, D, H]
        return self._gate_up(m)[:, :, 1::2].contiguous()

    def expert_w2(self, m: nn.Module) -> torch.Tensor:
        # down_proj is [E, H, D] already (op's w2 layout); orient defensively
        dp = self._constant(m.experts.down_proj)
        d_model = self._d_model(m)
        if dp.shape[2] != d_model and dp.shape[1] == d_model:
            dp = dp.transpose(-1, -2)
        return dp.contiguous()

    def _gate_up_bias(
        self, m: nn.Module
    ) -> T.Tuple[T.Optional[torch.Tensor], T.Optional[torch.Tensor]]:
        b = getattr(m.experts, "gate_up_proj_bias", None)
        if b is None:
            return None, None
        b = self._constant(b)  # [E, 2H] interleaved
        return b[:, 0::2].contiguous(), b[:, 1::2].contiguous()

    def expert_w1_bias(self, m: nn.Module) -> T.Optional[torch.Tensor]:
        return self._gate_up_bias(m)[0]

    def expert_w3_bias(self, m: nn.Module) -> T.Optional[torch.Tensor]:
        return self._gate_up_bias(m)[1]

    def expert_w2_bias(self, m: nn.Module) -> T.Optional[torch.Tensor]:
        b = getattr(m.experts, "down_proj_bias", None)
        return self._constant(b) if b is not None else None

    def top_k(self, m: nn.Module) -> int:
        r = self._router(m)
        if hasattr(r, "top_k"):
            return r.top_k
        return m.top_k

    def gate(self, m: nn.Module) -> str:
        # gpt-oss softmaxes the top-k router logits.
        return "softmax_topk"

    def act_alpha(self, m: nn.Module) -> T.Optional[float]:
        return float(getattr(m.experts, "alpha", 1.702))

    def act_limit(self, m: nn.Module) -> T.Optional[float]:
        return float(getattr(m.experts, "limit", 7.0))


class _QwenMoEAdapter(_MoEWeightAdapter):
    """Adapter for transformers Qwen2MoE / Qwen3.5 MoE.

    Experts are fused tensors: gate_up_proj [E, 2*H, D], down_proj [E, D, H].
    Shared expert is NOT handled here: it is decomposed outside the op.
    """

    def gate_weight(self, m: nn.Module) -> torch.Tensor:
        return m.gate.weight.detach()

    def expert_w1(self, m: nn.Module) -> torch.Tensor:
        # gate_up_proj [E, 2*H, D] → split → gate half [E, H, D] → [E, D, H]
        gate_up = self._constant(m.experts.gate_up_proj)
        half = gate_up.shape[1] // 2
        return gate_up[:, :half, :].transpose(-1, -2)

    def expert_w2(self, m: nn.Module) -> torch.Tensor:
        # down_proj [E, D, H] → [E, H, D]
        return self._constant(m.experts.down_proj).transpose(-1, -2)

    def expert_w3(self, m: nn.Module) -> torch.Tensor:
        # gate_up_proj [E, 2*H, D] → split → up half [E, H, D] → [E, D, H]
        gate_up = self._constant(m.experts.gate_up_proj)
        half = gate_up.shape[1] // 2
        return gate_up[:, half:, :].transpose(-1, -2)

    def top_k(self, m: nn.Module) -> int:
        # transformers <5.x: m.top_k, >=5.x: m.gate.top_k
        if hasattr(m, "top_k"):
            return m.top_k
        return m.gate.top_k

    def shared_expert(
        self, m: nn.Module
    ) -> T.Optional[T.Dict[str, torch.Tensor]]:
        # Qwen2-MoE / Qwen3.5-MoE add a sigmoid-gated shared expert; Qwen3-MoE
        # has none (no shared_expert attribute).
        if not hasattr(m, "shared_expert"):
            return None
        se = m.shared_expert
        # The converter emits the shared expert as silu-SwiGLU; reject any
        # other activation rather than silently exporting wrong maths.
        act = getattr(se, "act_fn", None)
        if act is not None and "silu" not in type(act).__name__.lower():
            raise T2NErrorNotImplemented(
                "shared expert activation "
                f"{type(act).__name__} is unsupported (only SiLU/SwiGLU)"
            )
        return {
            "gate_proj": se.gate_proj.weight.detach(),
            "up_proj": se.up_proj.weight.detach(),
            "down_proj": se.down_proj.weight.detach(),
            "router": m.shared_expert_gate.weight.detach(),
        }

    def gate(self, m: nn.Module) -> str:
        # Qwen / OLMoE routers softmax over ALL experts then take the top-k.
        # With norm_topk_prob the top-k weights are renormalized, which is
        # identical to softmaxing over the top-k logits ("softmax_topk").
        # Without it, the raw softmax-over-all weights are kept ("softmax_all",
        # e.g. OLMoE-1B-7B).
        router = getattr(m, "gate", None)
        norm = getattr(router, "norm_topk_prob", None)
        if norm is None:
            norm = getattr(m, "norm_topk_prob", True)
        return "softmax_topk" if norm else "softmax_all"


class _GraniteMoEAdapter(_MoEWeightAdapter):
    """Adapter for transformers GraniteMoeMoE (IBM Granite MoE).

    Same maths as Qwen (concatenated fused gate/up, softmax over top-k logits,
    SiLU SwiGLU), only the attribute names differ. Older Granite exports used
    `input_linear` [E, 2H, D] / `output_linear` [E, D, H] directly on the MoE
    module. Current Transformers stores them under `experts.gate_up_proj` and
    `experts.down_proj`, and the router weight directly under `router.weight`.
    """

    def _gate_up_proj(self, m: nn.Module) -> torch.Tensor:
        if hasattr(m, "input_linear"):
            return m.input_linear.weight
        return m.experts.gate_up_proj

    def _down_proj(self, m: nn.Module) -> torch.Tensor:
        if hasattr(m, "output_linear"):
            return m.output_linear.weight
        return m.experts.down_proj

    def gate_weight(self, m: nn.Module) -> torch.Tensor:
        router = m.router
        if hasattr(router, "layer"):
            return router.layer.weight.detach()
        return router.weight.detach()

    def expert_w1(self, m: nn.Module) -> torch.Tensor:
        input_weight = self._constant(self._gate_up_proj(m))
        half = input_weight.shape[1] // 2
        return input_weight[:, :half, :].transpose(-1, -2)

    def expert_w3(self, m: nn.Module) -> torch.Tensor:
        input_weight = self._constant(self._gate_up_proj(m))
        half = input_weight.shape[1] // 2
        return input_weight[:, half:, :].transpose(-1, -2)

    def expert_w2(self, m: nn.Module) -> torch.Tensor:
        return self._constant(self._down_proj(m)).transpose(-1, -2)

    def expert_sources(self, m: nn.Module) -> T.Sequence[torch.Tensor]:
        return (self._gate_up_proj(m), self._down_proj(m))

    def top_k(self, m: nn.Module) -> int:
        return m.router.top_k


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------

_ADAPTER_BY_CLASSNAME: T.Dict[str, T.Type[_MoEWeightAdapter]] = {
    "MoEFFN": _MoEFFNAdapter,
    # Mixtral / Mistral
    "MixtralSparseMoeBlock": _MixtralAdapter,
    "MistralSparseMoeBlock": _MixtralAdapter,
    # GPT-OSS (transformers >=4.55 renamed the block to GptOssMLP)
    "GptOssMLP": _GptOssAdapter,
    "GptOssSparseMoeBlock": _GptOssAdapter,
    # Qwen 2 / 3 / 3.5 MoE
    "Qwen2MoeSparseMoeBlock": _QwenMoEAdapter,
    "Qwen3MoeSparseMoeBlock": _QwenMoEAdapter,
    "Qwen3_5MoeSparseMoeBlock": _QwenMoEAdapter,
    # OLMoE shares the Qwen layout (fused [E, 2H, D] experts, softmax top-k
    # router with norm_topk_prob, no shared expert).
    "OlmoeSparseMoeBlock": _QwenMoEAdapter,
    # IBM Granite MoE: same maths, different attribute names.
    "GraniteMoeMoE": _GraniteMoEAdapter,
    "GraniteMoeSharedMoE": _GraniteMoEAdapter,
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


def _emit_shared_expert(
    g,
    name_to_tensor,
    shared,
    input_tensor,
    routed,
    out0,
    x_shape,
    add_weight,
    mk,
):
    """Graft a shared expert on top of the routed MoE output (Qwen2 / Qwen3.5).

    out = routed + sigmoid(x @ Wr^T) * down(silu(x @ Wg^T) * (x @ Wu^T)).
    All weights are raw nn.Linear [out, in], used as NNEF linear filters
    (linear computes x @ filter^T) so no transpose is needed.
    """
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.op import helper

    def emit(op_type, op_inputs, out_t, attribs=None, as_list=False):
        # Most ops (sigmoid/mul/add) take positional inputs; tract_core_einsum
        # reads its operands from a single list-valued `inputs=[...]` argument.
        helper.cast_and_add_nnef_operation(
            name_to_tensor=name_to_tensor,
            graph=g,
            type=op_type,
            inputs=list(op_inputs) if as_list else tuple(op_inputs),
            outputs=(out_t,),
            attribs=attribs or {},
            force_consistent_inputs_shapes=False,
        )
        return out_t

    def linear(x_t, weight_t, out_t):
        # x @ weight^T with weight [out, in]; NNEF `linear` lowers to a matmul
        # that requires equal input ranks, so use einsum to handle the 3D
        # activation against the 2D weight (and accumulate in f32).
        rank = len(x_shape)
        if rank == 3:
            expr = "bij,oj->bio"
        elif rank == 2:
            expr = "ij,oj->io"
        else:
            raise T2NErrorNotImplemented(
                f"shared expert linear expects rank 2 or 3 input, got {rank}"
            )
        return emit(
            "tract_core_einsum",
            [x_t, weight_t],
            out_t,
            attribs={"expr": expr, "acc": "f32", "output": ""},
            as_list=True,
        )

    d_model = x_shape[-1]
    hs = shared["gate_proj"].shape[0]
    hs_shape = x_shape[:-1] + [hs]
    d_shape = x_shape[:-1] + [d_model]
    one_shape = x_shape[:-1] + [1]

    w_gate = add_weight("se_gate_proj", shared["gate_proj"])
    w_up = add_weight("se_up_proj", shared["up_proj"])
    w_down = add_weight("se_down_proj", shared["down_proj"])
    w_router = add_weight("se_router", shared["router"])

    gate_h = linear(input_tensor, w_gate, mk("se_gate_h", hs_shape))
    up_h = linear(input_tensor, w_up, mk("se_up_h", hs_shape))
    gate_sig = emit("sigmoid", (gate_h,), mk("se_gate_sig", hs_shape))
    silu_g = emit("mul", (gate_h, gate_sig), mk("se_silu", hs_shape))
    inter = emit("mul", (silu_g, up_h), mk("se_inter", hs_shape))
    shared_out = linear(inter, w_down, mk("se_out", d_shape))
    logit = linear(input_tensor, w_router, mk("se_logit", one_shape))
    gate = emit("sigmoid", (logit,), mk("se_gate", one_shape))
    gated = emit("mul", (gate, shared_out), mk("se_gated", d_shape))
    emit("add", (routed, gated), out0)


def _is_qtensor_like(tensor: torch.Tensor) -> bool:
    if isinstance(tensor, OpaqueTensorRef):
        tensor = tensor.opaque_tensor
    if isinstance(tensor, QTensor):
        return True
    if not isinstance(tensor, OffloadedTensor):
        return False
    offloaded_type = getattr(tensor, "offloaded_tensor_type", None)
    return isinstance(offloaded_type, type) and issubclass(
        offloaded_type, QTensor
    )


def _should_quantize_expert_weights(moe, adapter) -> bool:
    return bool(getattr(moe, "_t2n_quantize_moe_experts_q40", False)) or any(
        _is_qtensor_like(src) for src in adapter.expert_sources(moe)
    )


def _maybe_quantize_expert_weight(
    node,
    name: str,
    data: torch.Tensor,
    quantize_experts_q40: bool,
    expert_q40_quantizer,
    expert_q40_quantizer_kwargs,
) -> torch.Tensor:
    if (
        not quantize_experts_q40
        or name not in {"w1", "w2", "w3"}
        or _is_qtensor_like(data)
    ):
        return data
    q_data = expert_q40_quantizer(
        data.contiguous(),
        **expert_q40_quantizer_kwargs,
    )
    q_data.nnef_name = f"{node.outputs[0].name}_{name}"
    return q_data


def _expert_layout(moe) -> str:
    layout = getattr(moe, "_t2n_moe_expert_layout", "canonical")
    if layout == "tract_moe_ffn":
        return "canonical"
    if layout not in {"canonical", "linear"}:
        raise T2NErrorNotImplemented(
            f"unsupported MoE expert layout {layout!r}"
        )
    return layout


def _layout_expert_weight(
    name: str,
    data: torch.Tensor,
    expert_layout: str,
) -> torch.Tensor:
    if expert_layout != "linear" or name not in {"w1", "w2", "w3"}:
        return data
    # Adapters normalize experts to the canonical tract_moe_ffn shapes:
    # w1/w3 [E,D,H], w2 [E,H,D]. The linear layout keeps native nn.Linear
    # storage instead: w1/w3 [E,H,D], w2 [E,D,H]. This is a layout contract,
    # independent from whether the tensors are stored as f32/f16 or quantized.
    return data.transpose(-1, -2)


def _expert_q40_quantizer_config(
    moe,
    quantize_experts_q40_percentile: float,
):
    quantizer = getattr(
        moe,
        "_t2n_quantize_moe_experts_q40_quantizer",
        fp_to_tract_q4_0_with_min_max_calibration,
    )
    kwargs = dict(
        getattr(moe, "_t2n_quantize_moe_experts_q40_kwargs", {}) or {}
    )
    if (
        quantizer is fp_to_tract_q4_0_with_min_max_calibration
        and "percentile" not in kwargs
    ):
        kwargs["percentile"] = quantize_experts_q40_percentile
    return quantizer, kwargs


def _moe_attrs(adapter, moe):
    attrs = {
        "k": adapter.top_k(moe),
        "activation": adapter.activation(moe),
        "gate": adapter.gate(moe),
    }
    act_alpha = adapter.act_alpha(moe)
    act_limit = adapter.act_limit(moe)
    if act_alpha is not None:
        attrs["act_alpha"] = act_alpha
    if act_limit is not None:
        attrs["act_limit"] = act_limit
    layout = _expert_layout(moe)
    if layout != "canonical":
        attrs["expert_layout"] = layout
    return attrs


def _append_optional_moe_biases(moe, adapter, is_swiglu, inputs, add_weight):
    # tract maps positional inputs as:
    # x, wg, w1, w2, w3, wg_bias, w1_bias, w3_bias, w2_bias.
    biases = (
        (adapter.gate_bias(moe), "wg_bias"),
        (adapter.expert_w1_bias(moe), "w1_bias"),
        (adapter.expert_w3_bias(moe), "w3_bias"),
        (adapter.expert_w2_bias(moe), "w2_bias"),
    )
    if not any(bias is not None for bias, _ in biases):
        return
    if not is_swiglu:
        raise T2NErrorNotImplemented(
            "MoE biases are only supported alongside a SwiGLU (w3) gate; "
            f"got biases without w3 for {type(moe).__name__}"
        )
    for bias, label in biases:
        if bias is None:
            raise T2NErrorNotImplemented(
                f"partial MoE bias set: {label} is missing while other "
                "biases are present (positional mapping needs all four)"
            )
    for bias, label in biases:
        inputs.append(add_weight(label, bias))


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

    quantize_experts_q40 = _should_quantize_expert_weights(moe, adapter)
    quantize_experts_q40_percentile = float(
        getattr(moe, "_t2n_quantize_moe_experts_q40_percentile", 1.0)
    )
    expert_q40_quantizer, expert_q40_quantizer_kwargs = (
        _expert_q40_quantizer_config(moe, quantize_experts_q40_percentile)
    )
    expert_layout = _expert_layout(moe)

    def _add_weight(name: str, data: torch.Tensor):
        data = _layout_expert_weight(name, data, expert_layout)
        data = _maybe_quantize_expert_weight(
            node,
            name,
            data,
            quantize_experts_q40,
            expert_q40_quantizer,
            expert_q40_quantizer_kwargs,
        )
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

    attrs = _moe_attrs(adapter, moe)

    _append_optional_moe_biases(moe, adapter, is_swiglu, inputs, _add_weight)

    # tract_moe_ffn is single-output (the routed hidden states). Some modules
    # (e.g. gpt-oss GptOssMLP) also return router_scores as a second output,
    # but transformers discards it at inference (`hidden_states, _ = mlp(...)`)
    # so it has no consumers. Map only node.outputs[0] and ignore any trailing
    # router output.
    if len(node.outputs) > 1:
        LOGGER.debug(
            "%s has %d outputs; mapping output[0] and ignoring the "
            "inference-unused router output(s)",
            type(moe).__name__,
            len(node.outputs),
        )

    # pylint: disable-next=import-outside-toplevel
    from nnef_tools.model import Tensor as NTensor

    shared = adapter.shared_expert(moe)

    out0 = helper.add_tensor_variable_node_as_nnef_tensor(
        g, node.outputs[0], name_to_tensor, prevent_variable=True
    )
    base = node.outputs[0].name
    np_dtype = input_tensor.dtype
    x_shape = list(input_tensor.shape)

    def _mk(suffix, shape):
        return NTensor(
            g, name=f"{base}_{suffix}", dtype=np_dtype, shape=tuple(shape)
        )

    # The routed experts go to the final output directly, unless a shared
    # expert must be added on top (Qwen2 / Qwen3.5), in which case they go to
    # an intermediate tensor.
    routed = _mk("routed", out0.shape) if shared is not None else out0

    # tract_moe_ffn takes intentionally heterogeneous-rank inputs: x is
    # [T, D] (or [B, S, D]), the gate is [E, D], and the expert weights are
    # [E, D, H]. The generic rank-aligner would left-pad the lower-rank
    # operands with a leading 1 to match the 3D weights, turning a 2D x into
    # [1, T, D] and producing a 3D output that no longer matches the 2D
    # PyTorch reference. Disable it so the op sees the ranks it expects.
    helper.cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="tract_moe_ffn",
        inputs=tuple(inputs),
        outputs=(routed,),
        attribs=attrs,
        force_consistent_inputs_shapes=False,
    )

    if shared is not None:
        _emit_shared_expert(
            g,
            name_to_tensor,
            shared,
            input_tensor,
            routed,
            out0,
            x_shape,
            _add_weight,
            _mk,
        )
        return ["tract_transformers", "tract_core"]

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
    except (ImportError, AttributeError, RuntimeError) as err:
        LOGGER.debug(
            "Could not register optional %s.%s MoE extractor: %s",
            import_path,
            class_name,
            err,
        )
        return None


def _register_all_transformers_moe():
    _candidates = [
        (
            "transformers.models.mixtral.modeling_mixtral",
            "MixtralSparseMoeBlock",
        ),
        (
            "transformers.models.gpt_oss.modeling_gpt_oss",
            "GptOssMLP",
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
        (
            "transformers.models.olmoe.modeling_olmoe",
            "OlmoeSparseMoeBlock",
        ),
        (
            "transformers.models.granitemoe.modeling_granitemoe",
            "GraniteMoeMoE",
        ),
        (
            "transformers.models.granitemoeshared.modeling_granitemoeshared",
            "GraniteMoeSharedMoE",
        ),
    ]
    for import_path, class_name in _candidates:
        _try_register(import_path, class_name)


_register_all_transformers_moe()
