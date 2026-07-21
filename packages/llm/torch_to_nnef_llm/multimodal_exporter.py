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

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractNNEF,
)
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef_llm.config import DtypeStr, ExportDirStruct
from torch_to_nnef_llm.exporter import LM_VAR_SCHEME, LLMExporter
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
            encoder_entries.append(
                {
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
            )
            if contract.injection_layers:
                injection_layers[contract.modality] = list(
                    contract.injection_layers
                )
    manifest: T.Dict[str, T.Any] = {
        "decoder": {"path": decoder_rel_path},
        "encoders": encoder_entries,
    }
    if injection_layers:
        manifest["injection_layers"] = injection_layers
    return manifest


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
        """Load like :meth:`LLMExporter.load`, returning a joint exporter."""
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
        wrapper = BaseEncoder(encoder_module, handler)
        io_spec = handler.build_input_spec(
            config_helper=self.config_helper,
            inputs_dtype=self.decoder_exporter.inputs_dtype,
        )
        update_forward_signature(wrapper, io_spec)
        inference_target.dynamic_axes = io_spec.dynamic_axes
        # Encoder towers accumulate more f32 attention drift than the decoder,
        # so each encoder handler declares its own check_io tolerance.
        if inference_target.check_io:
            inference_target.check_io_tolerance = TractCheckTolerance(
                handler.CHECK_IO_TOLERANCE
            )

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
        export_dirpath.mkdir(parents=True, exist_ok=True)

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
