"""Export top-level "glue" modules that NeMo's ``list_export_subnets()`` omits.

Some NeMo models apply a transform inside ``model.forward`` that sits
*between* the standard exportable subnets (encoder / decoder / joint) and is
not itself a member of any of them.  ``list_export_subnets()`` therefore never
reports it, so the exporter silently drops it and each subnet's ``check_io``
still passes exactly (nothing exercises the top-level forward).  The
multilingual Nemotron ASR family is the motivating case: a ``prompt_kernel``
MLP conditions the encoder output on a language id before the decoder consumes
it (see :class:`PromptKernelSubnet`).

This module provides a small, architecture-agnostic mechanism to declare such
glue as its own exportable subnet:

- Subclass :class:`GlueSubnet` (a plain ``nn.Module`` that also carries the
  minimal exportable metadata the export/inspection pipeline reads:
  ``input_names``, ``output_names``, ``input_types``, ``input_example`` and
  ``dynamic_shapes_for_export``).
- Register a builder with :func:`register_glue_subnet`, keyed by model class
  name(s).  Class *names* (not imported classes) are used so architectures
  whose class is not importable in the running NeMo (e.g. an unreleased
  ``EncDecRNNTBPEModelWithPrompt``) can still be matched.

``iter_all_glue_subnets`` is called by
:func:`torch_to_nnef_nemo.export.iter_nemo_model_subnets` after the native
subnets are yielded, so the glue appears in both export and inspection with no
extra wiring. ``after_subnet`` records which native subnet's output the glue
consumes (documentation / ordering intent), not the emission position.

Alternatively the exporter can *fuse* the glue into its ``after_subnet`` (see
:func:`fuse_glues_into` and :class:`FusedGlueSubnet`): instead of a separate
``prompt`` artifact, the encoder subnet itself gains the glue's extra inputs
(e.g. ``lang_id``) and applies the transform on its output. Then downstream
consumers need no extra subnet, only the extra input.
"""

from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)

# Nemotron multilingual: "auto" language slot (see prompt_dictionary). Used as
# the default/example lang id; the exported subnet takes it as a runtime input.
DEFAULT_PROMPT_LANG_ID = 101


class GlueSubnet(nn.Module):
    """Base class for a top-level glue module exported as its own subnet.

    Subclasses set the class-level metadata and implement ``forward`` +
    ``input_example``.  The instance is handed to the same export path as the
    NeMo-native subnets, so it must expose the attributes the pipeline reads.

    Attributes:
        name: subnet name (becomes ``<name>.nnef.tgz`` and the inspection key).
        after_subnet: name of the standard subnet whose output this glue
            consumes (ordering intent / documentation).
        input_names / output_names: NNEF IO names. Mirror the surrounding
            subnets' names so axis symbols unify across the chain.
    """

    name: str = "glue"
    after_subnet: str = "encoder"
    input_names: T.List[str] = []
    output_names: T.List[str] = []

    #: ``{name: NeuralType}`` used only to derive axis-symbol kinds; may be {}.
    input_types: T.Dict[str, T.Any]
    #: ``{input_name: [dynamic_axis_index, ...]}`` (NeMo dynamic-axes shape).
    nemo_dynamic_axes: T.Dict[str, T.List[int]]

    def dynamic_shapes_for_export(
        self, use_dynamo: bool = False
    ) -> T.Dict[str, T.List[int]]:
        return {k: list(v) for k, v in self.nemo_dynamic_axes.items()}

    def input_example(self, max_batch: int = 1, **_: T.Any) -> T.List[object]:
        raise NotImplementedError


GlueSubnetBuilder = T.Callable[[nn.Module], T.List[GlueSubnet]]

#: model class name -> builder returning the glue subnets for that model.
GLUE_SUBNET_BUILDERS: T.Dict[str, GlueSubnetBuilder] = {}


def register_glue_subnet(
    *model_class_names: str,
) -> T.Callable[[GlueSubnetBuilder], GlueSubnetBuilder]:
    """Register a glue-subnet builder for one or more model class names."""

    def _decorator(builder: GlueSubnetBuilder) -> GlueSubnetBuilder:
        for cls_name in model_class_names:
            GLUE_SUBNET_BUILDERS[cls_name] = builder
        return builder

    return _decorator


def _builder_for_model(model: nn.Module) -> T.Optional[GlueSubnetBuilder]:
    """Resolve a builder by walking the model's MRO class names."""
    for cls in type(model).__mro__:
        builder = GLUE_SUBNET_BUILDERS.get(cls.__name__)
        if builder is not None:
            return builder
    return None


def build_glue_subnets(model: nn.Module) -> T.List[GlueSubnet]:
    """Build the registered glue subnets for ``model`` once (``[]`` if none).

    Warns when the model looks prompt-conditioned (has a ``prompt_kernel``) but
    no builder matched its class name: that means the language head would be
    silently dropped, which is the exact failure this module exists to prevent.
    """
    builder = _builder_for_model(model)
    if builder is None:
        if hasattr(model, "prompt_kernel"):
            LOGGER.warning(
                "%s exposes a `prompt_kernel` but no glue-subnet builder is "
                "registered for its class name; the language head will NOT be "
                "exported. Register one with register_glue_subnet(...).",
                type(model).__name__,
            )
        return []
    subs = builder(model)
    for glue in subs:
        glue.eval()
    return subs


def _emit(
    glue: GlueSubnet,
) -> T.Tuple[str, GlueSubnet, T.List[object], T.Dict[str, T.List[int]]]:
    return (
        glue.name,
        glue,
        glue.input_example(),
        glue.dynamic_shapes_for_export(),
    )


def iter_all_glue_subnets(
    model: nn.Module,
    allow: T.Optional[T.Set[str]] = None,
) -> T.Iterator[
    T.Tuple[str, GlueSubnet, T.List[object], T.Dict[str, T.List[int]]]
]:
    """Yield ``(name, glue, input_example, nemo_dynamic_axes)`` for each glue.

    Tuple shape matches ``iter_nemo_model_subnets`` so the caller treats glue
    subnets identically to native ones. Builds the glue list once. ``allow``
    filters by subnet name (mirrors the native subnets' ``only_subnets``).
    """
    for glue in build_glue_subnets(model):
        if allow is not None and glue.name not in allow:
            continue
        yield _emit(glue)


def iter_glue_subnets_after(
    model: nn.Module,
    subnet_name: str,
    allow: T.Optional[T.Set[str]] = None,
) -> T.Iterator[
    T.Tuple[str, GlueSubnet, T.List[object], T.Dict[str, T.List[int]]]
]:
    """Yield glue subnets whose ``after_subnet`` equals ``subnet_name``."""
    for glue in build_glue_subnets(model):
        if glue.after_subnet != subnet_name:
            continue
        if allow is not None and glue.name not in allow:
            continue
        yield _emit(glue)


class FusedGlueSubnet(nn.Module):
    """A native subnet with a :class:`GlueSubnet` fused onto its first output.

    Used when a glue is folded into its parent subnet instead of exported as a
    separate artifact (e.g. the language head into the encoder). The fused
    module exposes the native inputs plus the glue's *extra* inputs (every glue
    input after the first, which is the one that consumes the native output),
    and the native output names. ``forward`` runs the native subnet, applies the
    glue to its first output, and passes the remaining outputs through.
    """

    def __init__(
        self,
        native: nn.Module,
        glue: GlueSubnet,
        native_input_example: T.Sequence[object],
    ):
        super().__init__()
        self.native = native
        self.glue = glue
        self._n_native = len(native_input_example)
        native_names = list(native.input_names[: self._n_native])
        extra_names = list(glue.input_names[1:])
        self.input_names = native_names + extra_names
        self.output_names = list(native.output_names)
        self._example = list(native_input_example) + list(
            glue.input_example()[1:]
        )

    @property
    def input_types(self) -> T.Dict[str, T.Any]:
        # Delegate to the native subnet so audio/time axis symbols are derived
        # identically to the un-fused encoder; glue extras (e.g. lang_id) have
        # no dynamic axis and are tolerated when absent.
        return getattr(self.native, "input_types", {})

    def input_example(self, *_: T.Any, **__: T.Any) -> T.List[object]:
        return list(self._example)

    def dynamic_shapes_for_export(
        self, use_dynamo: bool = False
    ) -> T.Dict[str, T.Any]:
        return self.native.dynamic_shapes_for_export(use_dynamo)

    def forward(self, *inputs: torch.Tensor):
        native_inputs = inputs[: self._n_native]
        extra_inputs = inputs[self._n_native :]
        outs = self.native(*native_inputs)
        if not isinstance(outs, tuple):
            outs = (outs,)
        conditioned = self.glue(outs[0], *extra_inputs)
        return (conditioned, *outs[1:])


def fuse_glues_into(
    native: nn.Module,
    subnet_name: str,
    native_input_example: T.Sequence[object],
    glues: T.Sequence[GlueSubnet],
) -> T.Tuple[nn.Module, T.List[object], bool]:
    """Fuse every glue whose ``after_subnet == subnet_name`` onto ``native``.

    Returns ``(module, input_example, matched)``: the (possibly wrapped) module,
    its input example (native inputs + fused glues' extra inputs), and whether
    any glue was fused. When none match, returns ``native`` unchanged.
    """
    module = native
    example = list(native_input_example)
    matched = False
    for glue in glues:
        if glue.after_subnet != subnet_name:
            continue
        module = FusedGlueSubnet(module, glue, example)
        example = module.input_example()
        matched = True
    return module, example, matched


class PromptKernelSubnet(GlueSubnet):
    """Language-conditioning ``prompt_kernel`` head, exported as a subnet.

    Multilingual Nemotron ASR (``EncDecRNNTBPEModelWithPrompt``) refines the
    encoder output with a per-frame MLP that takes a one-hot language id::

        encoded[B, T, D] ++ onehot(lang, P)[B, T, P]
            -> Linear(D + P, H) -> ReLU -> Linear(H, D)  ->  encoded[B, T, D]

    The MLP *replaces* the encoder output (no residual); it then feeds the
    decoder. The encoder subnet emits (and the decoder consumes) the encoded
    tensor channels-first as ``[B, D, T]``, so this wrapper transposes around
    the MLP. The language is exposed as a runtime ``lang_id`` integer input.

    Rather than materialize the one-hot and concatenate (which needs an
    ``expand`` over the dynamic batch/time axes that tract cannot resolve at
    plan time), the first linear is split algebraically: it applies to the
    encoded features, and the one-hot's contribution reduces to a per-language
    bias vector added to every frame. This is numerically identical to the
    concat form but introduces no ``expand`` -- so it runs with a symbolic
    batch and is pulse-safe.
    """

    name = "prompt"
    after_subnet = "encoder"

    def __init__(
        self,
        prompt_kernel: nn.Module,
        num_prompts: int,
        d_model: int,
        default_lang_id: int = DEFAULT_PROMPT_LANG_ID,
        encoded_type: T.Optional[T.Any] = None,
    ):
        super().__init__()
        self.prompt_kernel = prompt_kernel
        self.num_prompts = int(num_prompts)
        self.d_model = int(d_model)
        self.default_lang_id = int(default_lang_id)
        if not 0 <= self.default_lang_id < self.num_prompts:
            # Out of range -> the example one-hot is all zeros; check_io would
            # still pass but silently exercise null conditioning.
            LOGGER.warning(
                "default lang id %d is outside [0, %d); the example one-hot "
                "will be all zeros. Check the model's prompt_dictionary.",
                self.default_lang_id,
                self.num_prompts,
            )
        # Mirror the encoder->decoder handoff names so the axis symbols
        # (namespaced by input name + kind) unify across the chain:
        # encoder emits "outputs", decoder consumes "encoder_outputs".
        self.input_names = ["encoder_outputs", "lang_id"]
        self.output_names = ["outputs"]
        # Only input_types is read downstream (by build_dynamic_axes, to derive
        # the axis-symbol kind). Providing the encoder's "outputs" type makes
        # the time axis resolve to the same symbol the decoder uses.
        self.input_types = (
            {"encoder_outputs": encoded_type}
            if encoded_type is not None
            else {}
        )
        # Batch (0) and time (2) of the encoded tensor are dynamic, matching
        # the decoder's "encoder_outputs" ([0, 2]).
        self.nemo_dynamic_axes = {"encoder_outputs": [0, 2]}

    def _param_dtype(self) -> torch.dtype:
        try:
            return next(self.prompt_kernel.parameters()).dtype
        except StopIteration:
            return torch.float32

    def input_example(
        self, max_batch: int = 1, seq_len: int = 16, **_: T.Any
    ) -> T.List[object]:
        encoded = torch.zeros(
            max_batch, self.d_model, seq_len, dtype=self._param_dtype()
        )
        lang_id = torch.tensor([self.default_lang_id], dtype=torch.int64)
        return [encoded, lang_id]

    def forward(
        self, encoder_outputs: torch.Tensor, lang_id: torch.Tensor
    ) -> torch.Tensor:
        # encoder_outputs: [B, D, T] channels-first ; lang_id: [1] int
        feats = encoder_outputs.transpose(1, 2)  # [B, T, D]
        lin0 = self.prompt_kernel[0]  # Linear(D + P, H); D cols first
        weight = lin0.weight  # [H, D + P]
        # Feature half of the first linear (the concat's encoded part).
        hidden = F.linear(feats, weight[:, : self.d_model], lin0.bias)
        # One-hot half reduces to selecting one column-block of the weight,
        # i.e. a per-language bias [1, H] broadcast over batch and time.
        slots = torch.arange(self.num_prompts, device=encoder_outputs.device)
        onehot = (slots == lang_id.reshape(-1, 1)).to(feats.dtype)  # [1, P]
        # F.linear(onehot, W_lang) == onehot @ W_lang.T, avoiding an explicit
        # transpose op (older tract cores lack aten::t).
        lang_bias = F.linear(onehot, weight[:, self.d_model :])  # [1, H]
        hidden = hidden + lang_bias.unsqueeze(1)  # [B, T, H] + [1, 1, H]
        # Remaining layers (activation + final linear).
        conditioned = self.prompt_kernel[1:](hidden)  # [B, T, D]
        return conditioned.transpose(1, 2)  # [B, D, T]


def _resolve_default_lang_id(model: nn.Module) -> int:
    """Best-effort read of the ``auto`` language slot from the model config."""
    cfg = getattr(model, "cfg", None)
    prompt_dict = None
    if cfg is not None:
        prompt_dict = getattr(cfg, "prompt_dictionary", None)
        if prompt_dict is None and hasattr(cfg, "get"):
            prompt_dict = cfg.get("prompt_dictionary", None)
    if prompt_dict is not None:
        try:
            if "auto" in prompt_dict:
                return int(prompt_dict["auto"])
        except (TypeError, KeyError):
            pass
    return DEFAULT_PROMPT_LANG_ID


@register_glue_subnet(
    "EncDecRNNTBPEModelWithPrompt",
    "EncDecHybridRNNTCTCBPEModelWithPrompt",
)
def build_prompt_kernel_subnets(model: nn.Module) -> T.List[GlueSubnet]:
    """Build the ``prompt`` subnet for a prompt-conditioned Nemotron model."""
    prompt_kernel = getattr(model, "prompt_kernel", None)
    if prompt_kernel is None:
        LOGGER.warning(
            "%s has no `prompt_kernel`; skipping prompt subnet export",
            type(model).__name__,
        )
        return []
    linears = [m for m in prompt_kernel.modules() if isinstance(m, nn.Linear)]
    if not linears:
        LOGGER.warning(
            "prompt_kernel has no Linear layers; skipping prompt subnet export"
        )
        return []
    d_model = linears[-1].out_features
    num_prompts = linears[0].in_features - d_model
    if num_prompts <= 0:
        LOGGER.warning(
            "prompt_kernel input (%d) <= encoder dim (%d); "
            "cannot infer one-hot width, skipping prompt subnet export",
            linears[0].in_features,
            d_model,
        )
        return []
    encoded_type = None
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        try:
            encoded_type = encoder.output_types.get("outputs")
        except (AttributeError, TypeError):
            encoded_type = None
    if encoded_type is None:
        # Without the encoder's neural type the time axis falls back to a
        # generic "DIM" symbol instead of the decoder's "...__TIME"; the shape
        # config can still unify them, but streaming users should double-check.
        LOGGER.warning(
            "could not read encoder output neural type; prompt subnet axis "
            "symbols may not auto-unify with the decoder (regenerate the shape "
            "config for streaming)."
        )
    return [
        PromptKernelSubnet(
            prompt_kernel=prompt_kernel,
            num_prompts=num_prompts,
            d_model=d_model,
            default_lang_id=_resolve_default_lang_id(model),
            encoded_type=encoded_type,
        )
    ]
