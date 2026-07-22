"""Joint export of multimodal transformer models to NNEF.

A multimodal model is exported as two (or more) coordinated NNEF graphs:

- the decoder graph (the existing :class:`LLMExporter` path) which consumes
  modality embeddings as inputs, and
- one encoder graph per modality (vision tower / audio tower + projector) which
  produces those embeddings from raw modality input.

The graphs share an
:class:`~torch_to_nnef_llm.models.handlers.base.EmbeddingContract` and are tied
together at export time by a ``multimodal.json`` manifest so a downstream
runtime can chain them.
"""

import json
import logging
import typing as T
from dataclasses import dataclass
from pathlib import Path

import torch

from torch_to_nnef.exceptions import T2NErrorConsistency, T2NErrorMisuse
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractNNEF,
)
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef_llm.config import DtypeStr, ExportDirStruct
from torch_to_nnef_llm.exporter import (
    DEFAULT_HF_DOWNLOAD_N_RETRIES,
    LM_VAR_SCHEME,
    TYPE_OPTIONAL_DEVICE_MAP,
    LLMExporter,
    _normalize_dump_kwargs,
    _resolve_attn_implementation,
)
from torch_to_nnef_llm.models.base import BaseEncoder, update_forward_signature
from torch_to_nnef_llm.models.handlers import (
    EmbeddingContract,
    EncoderHandler,
    MultiModalArchitectureHandler,
    get_encoder_handlers,
)

LOGGER = logging.getLogger(__name__)

#: Manifest filename tying the encoder graphs to the decoder graph.
MANIFEST_NAME = "multimodal.json"

#: Sub-directory holding the decoder graph inside the export directory.
DECODER_DIRNAME = "decoder"

#: Above this parameter count, ``check_io`` roughly doubles peak RAM (the torch
#: model stays resident while the tract subprocess loads the NNEF), so we warn.
LARGE_MODEL_CHECK_IO_WARN_PARAMS = 2_000_000_000


@dataclass
class EncoderArtifact:
    """One exported encoder graph plus the contracts it satisfies."""

    label: str
    rel_path: str
    contracts: T.List[EmbeddingContract]


def build_manifest(
    *,
    config,
    decoder_rel_path: str,
    encoders: T.List[EncoderArtifact],
    inputs_dtype_str: str,
) -> T.Dict[str, T.Any]:
    """Build the ``multimodal.json`` payload (pure, no I/O).

    ``config`` is the HuggingFace model config; placeholder token ids are read
    from it via each contract's ``placeholder_token_id_attr``.
    """
    encoder_entries: T.List[T.Dict[str, T.Any]] = []
    injection_layers: T.Dict[str, T.List[int]] = {}
    for artifact in encoders:
        for contract in artifact.contracts:
            placeholder_token_id = getattr(
                config, contract.placeholder_token_id_attr, None
            )
            if placeholder_token_id is None:
                raise T2NErrorConsistency(
                    f"config has no '{contract.placeholder_token_id_attr}' for "
                    f"the {contract.modality!r} contract: the decoder cannot "
                    "splice embeddings without a placeholder token id"
                )
            entry: T.Dict[str, T.Any] = {
                "modality": contract.modality,
                "path": artifact.rel_path,
                "placeholder_token_id": placeholder_token_id,
                "outputs": [
                    {
                        "name": contract.output_name,
                        "feeds": contract.input_name,
                        "shape": [
                            contract.dynamic_axis,
                            contract.hidden_size,
                        ],
                        "dtype": inputs_dtype_str,
                    }
                ],
            }
            if contract.injection_layers:
                # DeepStack: extra residual streams injected at the given
                # decoder layer indices. The i-th stream is emitted by the
                # encoder as ``out_<modality>_deepstack_<i>`` and fed to the
                # decoder input ``in_<modality>_deepstack_<i>``, injected at
                # ``injection_layers[i]``. They have their own token axis.
                deepstack_axis = (
                    contract.deepstack_dynamic_axis or contract.dynamic_axis
                )
                entry["deepstack"] = [
                    {
                        "layer": layer,
                        "name": f"out_{contract.modality}_deepstack_{i}",
                        "feeds": f"in_{contract.modality}_deepstack_{i}",
                        "shape": [deepstack_axis, contract.hidden_size],
                        "dtype": inputs_dtype_str,
                    }
                    for i, layer in enumerate(contract.injection_layers)
                ]
                injection_layers[contract.modality] = list(
                    contract.injection_layers
                )
            encoder_entries.append(entry)
    manifest: T.Dict[str, T.Any] = {
        "decoder": {"path": decoder_rel_path},
        "encoders": encoder_entries,
    }
    if injection_layers:
        manifest["injection_layers"] = injection_layers
    return manifest


def _prefer_sdpa_attention(module: torch.nn.Module) -> None:
    """Switch a module's eager attention to SDPA in place.

    Vision towers (e.g. SigLIP/Idefics3) default to eager attention, which
    overflows in fp16 (its explicit QK^T/softmax) and which
    ``force_attention_inner_in_f32`` cannot reach (that flag only rewrites the
    SDPA op). The attention forward dispatches on the config's
    ``_attn_implementation`` at call time, so flipping it to ``"sdpa"`` routes
    both the torch reference
    (CPU SDPA upcasts internally to f32) and the exported graph (the f32 SDPA
    fragment) through the numerically stable path.
    """
    for sub in module.modules():
        config = getattr(sub, "config", None)
        if (
            config is not None
            and getattr(config, "_attn_implementation", None) == "eager"
        ):
            config._attn_implementation = "sdpa"


class MultiModalExporter:
    """Orchestrate joint export of a multimodal model's encoder(s) + decoder."""

    def __init__(self, decoder_exporter: LLMExporter):
        self.decoder_exporter = decoder_exporter
        self.config_helper = decoder_exporter.model_infos
        model_type = self.config_helper.conf.model_type
        encoder_handler_classes = get_encoder_handlers(model_type)
        self.encoder_handlers: T.List[EncoderHandler] = [
            cls() for cls in encoder_handler_classes
        ]
        self.handler = MultiModalArchitectureHandler(
            decoder_handler=self.config_helper.handler,
            encoder_handlers=self.encoder_handlers,
        )

    @classmethod
    def load(cls, *args, **kwargs) -> "MultiModalExporter":
        """Load like :meth:`LLMExporter.load`, returning a joint exporter.

        Defaults ``force_module_dtype`` to f32. bfloat16 (the native dtype of
        most multimodal checkpoints) cannot round-trip through the numpy-based
        NNEF representation yet (no numpy bf16), and CPU f16 hits layer-norm /
        attention issues, so f32 is the only dtype that reliably exports today.
        f32 makes a multi-billion-param model heavy (see the check_io RAM
        warning in ``export``); pass ``no_verify=True`` for large models to skip
        the tract subprocess, or export on a larger-RAM host.
        """
        kwargs.setdefault("force_module_dtype", "f32")
        return cls(LLMExporter.load(*args, **kwargs))

    @property
    def hf_model_causal(self):
        return self.decoder_exporter.hf_model_causal

    @property
    def contracts(self) -> T.List[EmbeddingContract]:
        return self.handler.contracts(self.config_helper)

    def _export_one_encoder(
        self,
        handler: EncoderHandler,
        inference_target: TractNNEF,
        export_dirpath: Path,
        naming_scheme: VariableNamingScheme,
    ) -> EncoderArtifact:
        label = handler.MODALITY
        model_dir = export_dirpath / label
        model_dir.mkdir(parents=True, exist_ok=True)

        handler.prepare_model_for_export(self.hf_model_causal)
        encoder_module = handler.get_encoder_module(self.hf_model_causal)
        # `export()` already routed fp16 models to SDPA and set the f32
        # accumulation flags on the shared target; here we only pick the check
        # tolerance.
        is_f16 = any(
            p.dtype == torch.float16 for p in encoder_module.parameters()
        )
        wrapper = BaseEncoder(encoder_module, handler)
        io_spec = handler.build_input_spec(
            config_helper=self.config_helper,
            inputs_dtype=self.decoder_exporter.inputs_dtype,
        )
        update_forward_signature(wrapper, io_spec)
        inference_target.dynamic_axes = io_spec.dynamic_axes
        # Encoder towers accumulate more f32 attention drift than the decoder,
        # so each encoder handler declares its own check_io tolerance; fp16
        # towers additionally accumulate fp16 rounding through the deep stack
        # and need the loosest tract tolerance to verify.
        if inference_target.check_io:
            tolerance = handler.CHECK_IO_TOLERANCE
            if is_f16:
                tolerance = TractCheckTolerance.ULTRA
            inference_target.check_io_tolerance = TractCheckTolerance(tolerance)

        # NOTE: encoder towers export uncompressed on purpose. Weight
        # compression (e.g. Q4_0) is applied to the LLM decoder only, via
        # `dump_with_inference_target`; the vision/audio towers are small and
        # the LLM-oriented compression schemes are not meant for them.
        LOGGER.info("exporting '%s' encoder graph", label)
        export_model_to_nnef(
            model=wrapper,
            args=io_spec.inputs,
            inference_target=inference_target,
            file_path_export=model_dir / "model.nnef.tgz",
            input_names=io_spec.input_names,
            output_names=io_spec.output_names,
            nnef_variable_naming_scheme=naming_scheme,
        )
        return EncoderArtifact(
            label=label,
            rel_path=f"{label}/model.nnef.tgz",
            contracts=handler.contracts(self.config_helper),
        )

    def _write_manifest(
        self,
        export_dirpath: Path,
        encoders: T.List[EncoderArtifact],
    ) -> Path:
        manifest = build_manifest(
            config=self.hf_model_causal.config,
            decoder_rel_path=f"{DECODER_DIRNAME}/model.nnef.tgz",
            encoders=encoders,
            inputs_dtype_str=DtypeStr.from_torch_dtype(
                self.decoder_exporter.inputs_dtype
            ).value,
        )
        manifest_path = export_dirpath / MANIFEST_NAME
        with manifest_path.open("w", encoding="utf8") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest_path

    def export(
        self,
        export_dirpath: T.Union[str, Path],
        inference_target: TractNNEF,
        naming_scheme: VariableNamingScheme = LM_VAR_SCHEME,
        **decoder_kwargs,
    ) -> Path:
        """Export the decoder graph, each encoder graph, and the manifest."""
        export_dirpath = Path(export_dirpath)
        if not self.encoder_handlers:
            LOGGER.warning(
                "no encoder handler registered for model_type '%s'; "
                "exporting decoder only",
                self.config_helper.conf.model_type,
            )
        # the projected encoder embeddings are spliced into the decoder token
        # sequence, so their hidden size must equal the decoder's; catch a
        # config mismatch here instead of at a confusing downstream shape error.
        decoder_hidden = self.config_helper.decoder_conf.hidden_size
        for contract in self.contracts:
            if contract.hidden_size != decoder_hidden:
                raise T2NErrorConsistency(
                    f"{contract.modality!r} encoder hidden_size "
                    f"{contract.hidden_size} != decoder hidden_size "
                    f"{decoder_hidden}: projected embeddings cannot be spliced "
                    "into the decoder sequence"
                )
        export_dirpath.mkdir(parents=True, exist_ok=True)

        # fp16 decoder + towers: eager attention overflows in fp16 (its explicit
        # QK^T/softmax) and `force_attention_inner_in_f32` only covers the SDPA
        # op, so route the whole model to SDPA and keep normalization, attention
        # and matmul accumulation in f32. Both the torch reference (CPU SDPA
        # upcasts internally) and the exported graph then agree, and the deep
        # fp16 stacks verify against tract at its loosest tolerance -- so fp16
        # export is checkable end to end (halving RAM vs f32).
        is_f16 = any(
            p.dtype == torch.float16
            for p in self.hf_model_causal.parameters()
        )
        if is_f16:
            _prefer_sdpa_attention(self.hf_model_causal)
            inference_target.force_norm_in_f32 = True
            inference_target.force_attention_inner_in_f32 = True
            inference_target.force_linear_accumulation_in_f32 = True
            if inference_target.check_io:
                inference_target.check_io_tolerance = TractCheckTolerance.ULTRA

        n_params = self.decoder_exporter.model_n_params
        if (
            getattr(inference_target, "check_io", False)
            and not is_f16
            and n_params > LARGE_MODEL_CHECK_IO_WARN_PARAMS
        ):
            LOGGER.warning(
                "check_io on a %.1fB-param f32 model: peak RAM is roughly 2x "
                "the weight size because the torch model stays resident while "
                "the tract subprocess loads the NNEF. Pass `-dt f16` to halve "
                "RAM (verified), no_verify=True to skip the tract check, or "
                "export on a larger-RAM host.",
                n_params / 1e9,
            )

        # FLAT so the decoder graph lands at decoder/model.nnef.tgz, matching
        # the manifest path (DEEP would nest it under decoder/model/).
        decoder_kwargs.setdefault("export_dir_struct", ExportDirStruct.FLAT)
        self.decoder_exporter.dump_with_inference_target(
            inference_target=inference_target,
            export_dirpath=export_dirpath / DECODER_DIRNAME,
            naming_scheme=naming_scheme,
            **decoder_kwargs,
        )

        encoders = [
            self._export_one_encoder(
                handler, inference_target, export_dirpath, naming_scheme
            )
            for handler in self.encoder_handlers
        ]
        manifest_path = self._write_manifest(export_dirpath, encoders)
        LOGGER.info("wrote multimodal manifest: %s", manifest_path)
        return export_dirpath


#: kwargs consumed by `build_inference_target` (not by the decoder dump call).
_INFERENCE_TARGET_ONLY_KWARGS = (
    "tract_specific_path",
    "tract_specific_version",
    "tract_specific_properties",
    "force_f32_attention",
    "force_f32_linear_accumulator",
    "force_f32_normalization",
    "reify_sdpa_operator",
    "tract_check_io_tolerance",
)


def dump_multimodal(
    model_slug: T.Optional[str] = None,
    local_dir: T.Optional[Path] = None,
    force_module_dtype: T.Optional[DtypeStr] = None,
    force_inputs_dtype: T.Optional[DtypeStr] = None,
    merge_peft: T.Optional[bool] = None,
    num_logits_to_keep: T.Union[int, str] = 1,
    device_map: TYPE_OPTIONAL_DEVICE_MAP = None,
    hf_download_n_retries: int = DEFAULT_HF_DOWNLOAD_N_RETRIES,
    trust_remote_code: bool = True,
    upcast_quant: T.Optional[T.Sequence[str]] = None,
    attn_implementation: T.Optional[str] = None,
    experts_implementation: T.Optional[str] = "auto",
    **kwargs,
) -> T.Tuple[T.Union[Path, None], "MultiModalExporter"]:
    """Export a multimodal model as coordinated NNEF graphs + a manifest.

    Mirrors :func:`~torch_to_nnef_llm.exporter.dump_llm`, but loads via
    :class:`MultiModalExporter` and writes the vision/audio encoder graph(s),
    the LLM decoder graph, and a ``multimodal.json`` manifest tying them
    together.
    """
    attn_implementation = _resolve_attn_implementation(
        attn_implementation,
        kwargs.get("reify_sdpa_operator"),
    )
    exporter = MultiModalExporter.load(
        model_slug,
        local_dir,
        force_module_dtype=force_module_dtype,
        force_inputs_dtype=force_inputs_dtype,
        merge_peft=merge_peft,
        num_logits_to_keep=num_logits_to_keep,
        device_map=device_map,
        hf_download_n_retries=hf_download_n_retries,
        trust_remote_code=trust_remote_code,
        upcast_quant=upcast_quant,
        attn_implementation=attn_implementation,
        experts_implementation=experts_implementation,
    )
    dump_kwargs = _normalize_dump_kwargs(kwargs)
    target_kwargs = {
        key: dump_kwargs.pop(key)
        for key in _INFERENCE_TARGET_ONLY_KWARGS
        if key in dump_kwargs
    }
    inference_target = exporter.decoder_exporter.build_inference_target(
        **target_kwargs,
        no_verify=dump_kwargs.get("no_verify", False),
        compression_method=dump_kwargs.get("compression_method"),
        compression_registry=dump_kwargs.get("compression_registry"),
    )
    # the manifest records the decoder at ``decoder/model.nnef.tgz``; force the
    # FLAT layout so the graph lands there (DEEP would nest it one level down).
    dump_kwargs.pop("export_dir_struct", None)
    if "export_dirpath" not in dump_kwargs:
        raise T2NErrorMisuse("dump_multimodal requires 'export_dirpath'")
    export_dirpath = dump_kwargs.pop("export_dirpath")
    exporter.export(
        export_dirpath=export_dirpath,
        inference_target=inference_target,
        **dump_kwargs,
    )
    return (
        Path(export_dirpath) if export_dirpath else None,
        exporter,
    )
