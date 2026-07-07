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

``iter_glue_subnets_after`` is called by
:func:`torch_to_nnef_nemo.export.iter_nemo_model_subnets` right after the
standard subnet named by ``GlueSubnet.after_subnet`` is yielded, so the glue
appears in both export and inspection with no extra wiring.
"""

from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn

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
            consumes; the glue is emitted immediately after it.
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


def iter_glue_subnets_after(
    model: nn.Module,
    subnet_name: str,
    allow: T.Optional[T.Set[str]] = None,
) -> T.Iterator[
    T.Tuple[str, GlueSubnet, T.List[object], T.Dict[str, T.List[int]]]
]:
    """Yield glue subnets attached after ``subnet_name``.

    Yields ``(name, glue_module, input_example, nemo_dynamic_axes)`` matching
    the tuple shape of ``iter_nemo_model_subnets`` so the caller can treat glue
    subnets identically to native ones. ``allow`` filters by subnet name.
    """
    builder = _builder_for_model(model)
    if builder is None:
        return
    for glue in builder(model):
        if glue.after_subnet != subnet_name:
            continue
        if allow is not None and glue.name not in allow:
            continue
        glue.eval()
        yield (
            glue.name,
            glue,
            glue.input_example(),
            glue.dynamic_shapes_for_export(),
        )


class PromptKernelSubnet(GlueSubnet):
    """Language-conditioning ``prompt_kernel`` head, exported as a subnet.

    Multilingual Nemotron ASR (``EncDecRNNTBPEModelWithPrompt``) refines the
    encoder output with a per-frame MLP that takes a one-hot language id::

        encoded[B, T, D] ++ onehot(lang, P)[B, T, P]
            -> Linear(D + P, H) -> ReLU -> Linear(H, D)  ->  encoded[B, T, D]

    The MLP *replaces* the encoder output (no residual); it then feeds the
    decoder. The encoder subnet emits (and the decoder consumes) the encoded
    tensor channels-first as ``[B, D, T]``, so this wrapper transposes around
    the MLP. The language is exposed as a runtime ``lang_id`` integer input and
    the one-hot is built in-graph, so a single export serves any language.
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
        # Mirror the encoder->decoder handoff names so the axis symbols
        # (namespaced by input name + kind) unify across the chain:
        # encoder emits "outputs", decoder consumes "encoder_outputs".
        self.input_names = ["encoder_outputs", "lang_id"]
        self.output_names = ["outputs"]
        self.input_types = (
            {"encoder_outputs": encoded_type}
            if encoded_type is not None
            else {}
        )
        self.output_types = (
            {"outputs": encoded_type} if encoded_type is not None else {}
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
        slots = torch.arange(self.num_prompts, device=encoder_outputs.device)
        # [1, P] -> broadcast to [B, T, P]; time-invariant so pulse-safe.
        onehot = (slots.unsqueeze(0) == lang_id.reshape(-1, 1)).to(feats.dtype)
        onehot = onehot.unsqueeze(1).expand(
            feats.shape[0], feats.shape[1], self.num_prompts
        )
        conditioned = self.prompt_kernel(torch.cat([feats, onehot], dim=-1))
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
    return [
        PromptKernelSubnet(
            prompt_kernel=prompt_kernel,
            num_prompts=num_prompts,
            d_model=d_model,
            default_lang_id=_resolve_default_lang_id(model),
            encoded_type=encoded_type,
        )
    ]
