"""Shared gated-delta-net (GDN) reification for hybrid architectures.

Several HF transformers families (Qwen3.5, Qwen3.5-MoE, Qwen3-Next,
OLMo-Hybrid, ...) implement linear attention with the SAME fla-style
gated delta rule: module-bound ``recurrent_gated_delta_rule`` /
``chunk_gated_delta_rule`` functions whose torch fallbacks
(``torch_recurrent_gated_delta_rule`` / ``torch_chunk_gated_delta_rule``)
share one signature::

    rule(query, key, value, g=, beta=, initial_state=,
         output_final_state=, use_qk_l2norm_in_kernel=True)
    -> (core_attn_out, last_recurrent_state)

and a matching depthwise causal conv update
(``torch_causal_conv1d_update(hidden_states, conv_state, weight, bias,
activation)``). HF computes the rule with a Python loop over the sequence
axis; traced, that loop unrolls at the traced length and the export is
frozen to it. :func:`reify_gated_delta_net` swaps the module-bound rule
functions with the reified shims from
``torch_to_nnef.op.custom_extractors.gdn`` so the exported graph carries
ONE S-generic ``tract_transformers_gdn_recurrent`` /
``tract_transformers_causal_conv1d_update`` op instead.

Family applicability is duck-typed: any module binding BOTH rule
attributes is reified, and so is a GDN module whose modeling file calls
the rules and ``causal_conv1d_update`` as module-level functions
(transformers >= 5.16). Architecture handlers opt in from their
``prepare_model_for_export`` and choose whether the GQA q/k
repeat_interleave neutralization is safe for their module class (see
``neutralize_gqa_repeat`` below).
"""

import functools
import logging
import sys
import types
import typing as T

import torch

from torch_to_nnef.op.custom_extractors.gdn import (
    CausalConvUpdateReified,
    GatedDeltaNetRecurrentReified,
)
from torch_to_nnef.tensor.offload import OffloadedTensor

LOGGER = logging.getLogger(__name__)


def _materialize_offloaded(module, attr) -> int:
    """Replace a disk-offloaded param with its real tensor.

    The GDN path uses these small params in trace-time arithmetic
    (`F.conv1d` on the conv weight, `A_log.exp()`, `dt_bias` add)
    that the offload meta-tracing strategy does not support; they
    are a few KB per layer, so materializing them is free.
    """
    tensor = getattr(module, attr, None)
    if tensor is None:
        return 0
    inner = tensor.data if isinstance(tensor, torch.nn.Parameter) else tensor
    if not isinstance(inner, OffloadedTensor):
        return 0
    real = inner.reload()
    if real.dtype != inner.dtype:
        real = real.to(inner.dtype)
    param = torch.nn.Parameter(real, requires_grad=False)
    setattr(module, attr, param)
    return 1


def patch_linear_cache_for_export() -> None:
    """Make HF linear-attention cache updates assignment-based.

    `LinearAttentionLayer.update_conv_state` / `update_recurrent_state`
    `copy_` into lazily created buffers (to keep cudagraph static
    addresses). Under the opaque fake-tensor tracing strategy the
    model's activation stream is fake while those buffers are real,
    and an IN-PLACE op cannot absorb a fake source (jit tracer
    internal asserts); plain rebinds trace cleanly and are
    semantically identical for export.
    """
    # Load the HF cache implementation only when preparing a GDN model.
    # Other handlers remain importable without this transformers API.
    # pylint: disable-next=import-outside-toplevel
    from transformers import cache_utils

    layer_cls = cache_utils.LinearAttentionLayer
    if getattr(layer_cls, "_t2n_assignment_cache_patch", False):
        return

    def update_conv_state(
        self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs
    ):
        if not self.is_conv_states_initialized[state_idx]:
            kernel = (
                conv_states.shape[-1]
                if conv_kernel_size is None
                else conv_kernel_size
            )
            self.conv_kernel_size[state_idx] = kernel
            self.is_conv_states_initialized[state_idx] = True
        kernel = self.conv_kernel_size[state_idx]
        if not self.has_previous_state[state_idx]:
            full = conv_states
            self.has_previous_state[state_idx] = True
            if not self.record_past and full.shape[-1] < kernel:
                full = torch.nn.functional.pad(
                    full, (kernel - full.shape[-1], 0), value=0
                )
        else:
            full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
        stored = full[..., -kernel:] if not self.record_past else full
        # Backref so the reified conv update can REBIND the cache slot
        # instead of mutating the stored tensor in place (which would
        # both corrupt the fed example input through the view above
        # and reintroduce the in-place real/fake mixing).
        stored._t2n_cache_slot = (self, state_idx)
        self.conv_states[state_idx] = stored
        return full

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        self.is_recurrent_states_initialized[state_idx] = True
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states

    layer_cls.update_conv_state = update_conv_state
    layer_cls.update_recurrent_state = update_recurrent_state
    layer_cls._t2n_assignment_cache_patch = True
    LOGGER.info("patched LinearAttentionLayer cache updates to assignment form")


def _make_conv_update(conv_shim):
    def causal_conv1d_update(
        hidden_states, conv_state, weight, bias=None, activation=None
    ):
        if bias is not None:
            raise NotImplementedError(
                "reified causal conv does not support a bias"
            )
        if activation not in (None, "silu"):
            raise NotImplementedError(
                f"reified causal conv is silu-only, got {activation}"
            )
        out, final_state = conv_shim(hidden_states, conv_state, weight)
        # The state write stays OUTSIDE the reified boundary.
        # Preferred: REBIND the cache slot (no in-place op, no
        # real/fake mixing under the opaque tracing strategy).
        # Fallback for an unpatched cache: plain copy_ (the aten
        # copy handler resolves it).
        slot = getattr(conv_state, "_t2n_cache_slot", None)
        if slot is not None:
            layer, state_idx = slot
            final_state._t2n_cache_slot = slot
            layer.conv_states[state_idx] = final_state
        else:
            conv_state.copy_(final_state)
        return out

    return causal_conv1d_update


# HF module-level function name -> GDN module attribute holding its shim.
_MODULE_LEVEL_RULES = {
    "torch_recurrent_gated_delta_rule": "recurrent_gated_delta_rule",
    "torch_chunk_gated_delta_rule": "chunk_gated_delta_rule",
    "causal_conv1d_update": "causal_conv1d_update",
}

_ACTIVE_GDN_MODULES: T.List[torch.nn.Module] = []


def _binds_rules_as_attributes(module: torch.nn.Module) -> bool:
    return hasattr(module, "recurrent_gated_delta_rule") and hasattr(
        module, "chunk_gated_delta_rule"
    )


def _modeling_module_with_rules(
    module: torch.nn.Module,
) -> T.Optional[types.ModuleType]:
    """Return the module defining ``module``'s class when it calls the rules.

    transformers >= 5.16 calls the gated delta rules and the conv update as
    module-level functions of the modeling file instead of binding them on
    the GDN module. A GDN module is recognized by its ``conv1d`` / ``A_log``
    / ``dt_bias`` members next to those module-level functions.
    """
    if not all(
        hasattr(module, attr) for attr in ("conv1d", "A_log", "dt_bias")
    ):
        return None
    modeling = sys.modules.get(type(module).__module__)
    if modeling is None or not all(
        callable(getattr(modeling, name, None)) for name in _MODULE_LEVEL_RULES
    ):
        return None
    return modeling


def _dispatch_to_active_module(attr: str, original: T.Callable) -> T.Callable:
    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        if not _ACTIVE_GDN_MODULES:
            return original(*args, **kwargs)
        return getattr(_ACTIVE_GDN_MODULES[-1], attr)(*args, **kwargs)

    return dispatch


def _route_module_level_rules(modeling: types.ModuleType) -> None:
    """Make the modeling file's rule functions call the running module's shims.

    Outside a reified GDN module forward the original functions run, so
    models that were not reified keep the HF behaviour.
    """
    if getattr(modeling, "_t2n_gdn_rules_routed", False):
        return
    for name, attr in _MODULE_LEVEL_RULES.items():
        setattr(
            modeling,
            name,
            _dispatch_to_active_module(attr, getattr(modeling, name)),
        )
    modeling._t2n_gdn_rules_routed = True


def _track_active_module(module: torch.nn.Module) -> None:
    if getattr(module, "_t2n_gdn_active_tracked", False):
        return

    def enter(mod, _args):
        _ACTIVE_GDN_MODULES.append(mod)

    def leave(_mod, _args, _out):
        _ACTIVE_GDN_MODULES.pop()

    module.register_forward_pre_hook(enter)
    module.register_forward_hook(leave, always_call=True)
    module._t2n_gdn_active_tracked = True


def reify_gated_delta_net(model, *, neutralize_gqa_repeat: bool) -> int:
    """Reify the gated delta rule and causal conv update as tract ops.

    Applies to every module of ``model`` that binds both
    ``recurrent_gated_delta_rule`` and ``chunk_gated_delta_rule``
    (the shared HF gated-delta-net module contract, see module
    docstring), and to GDN modules whose modeling file calls them as
    module-level functions (transformers >= 5.16): those functions are
    routed to the shims of the GDN module whose forward is running.
    Returns the number of modules reified.

    ``neutralize_gqa_repeat`` controls the pre-rule q/k
    repeat_interleave neutralization: HF repeat-interleaves q/k to the
    value-head count before calling the rule; setting ``num_k_heads =
    num_v_heads`` makes that repeat an identity so the exported graph
    never materializes the broadcast (the tract op does the group
    indexing itself, the shim repeats internally for eager exactness).
    This aliasing is ONLY safe when the module reads ``num_k_heads``
    nowhere else after ``__init__`` (true for Qwen3.5 / Qwen3.5-MoE,
    whose projections are separate). Qwen3-Next fuses its projections
    and ``fix_query_key_value_ordering`` reads ``num_k_heads`` at every
    forward, so it MUST pass ``False``: the shim then receives HF's
    already-repeated q/k and the op resolves GQA groups to 1, which is
    correct but materializes the broadcast in the graph.
    """
    patch_linear_cache_for_export()

    n_reified = 0
    n_materialized = 0
    n_gqa = 0
    for module in model.modules():
        modeling = (
            None
            if _binds_rules_as_attributes(module)
            else _modeling_module_with_rules(module)
        )
        if _binds_rules_as_attributes(module) or modeling is not None:
            shim = GatedDeltaNetRecurrentReified()
            module.recurrent_gated_delta_rule = shim
            module.chunk_gated_delta_rule = shim
            # Neutralize HF's pre-rule q/k repeat_interleave (GQA):
            # `forward` only repeats when num_v_heads // num_k_heads > 1,
            # so overriding num_k_heads makes the repeat an identity. The
            # shim then receives hk-head q/k at the traced boundary (the
            # exported graph never materializes the broadcast; the tract
            # op does the group indexing) and repeats internally so its
            # eager math stays HF-exact.
            if (
                neutralize_gqa_repeat
                and getattr(module, "num_k_heads", None) is not None
                and module.num_v_heads != module.num_k_heads
            ):
                module.num_k_heads = module.num_v_heads
                n_gqa += 1
            conv_shim = CausalConvUpdateReified()
            module.t2n_conv_update_shim = conv_shim
            module.causal_conv1d_update = _make_conv_update(conv_shim)
            if modeling is not None:
                _route_module_level_rules(modeling)
                _track_active_module(module)
            n_reified += 1
            for owner, attr in (
                (getattr(module, "conv1d", None), "weight"),
                (getattr(module, "conv1d", None), "bias"),
                (module, "A_log"),
                (module, "dt_bias"),
            ):
                if owner is not None:
                    n_materialized += _materialize_offloaded(owner, attr)
    LOGGER.info(
        "reified the gated delta rule and causal conv in %d "
        "linear-attention modules (%d small offloaded params "
        "materialized, %d GQA q/k repeats neutralized)",
        n_reified,
        n_materialized,
        n_gqa,
    )
    return n_reified
