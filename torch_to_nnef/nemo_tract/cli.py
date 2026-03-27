"""CLI for NeMo→NNEF export.

This module owns the full CLI surface: argument parsing, runtime
orchestration (``run_export``), and the ``main`` entry point.
The programmatic export API lives in ``export.py``.
"""

from __future__ import annotations

import argparse
import datetime
import json
from enum import Enum
import logging
import os
import shlex
import sys
import textwrap
import typing as T
from dataclasses import asdict
from pathlib import Path

import torch

from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractCli,
    TractNNEF,
)
from torch_to_nnef.log import init_log, set_lib_log_level
from torch_to_nnef.nemo_tract.axis_registry import (
    AxisSymbolRegistry,
    load_axis_symbol_registry,
)
from torch_to_nnef.nemo_tract.config import (
    CompressionConfig,
    InspectionConfig,
    LogConfig,
    ModelSelectionConfig,
    NamingPrecisionConfig,
    NemoExportConfig,
    NemoTractConfig,
    OutputConfig,
    SdpaConfig,
    SubnetSelectionConfig,
    TractBinaryConfig,
)
from torch_to_nnef.nemo_tract.constants import (
    NEMO_INPUT_SYMBOL_SEPARATOR as _SEP,
)
from torch_to_nnef.nemo_tract.export import export_nemo_from_model
from torch_to_nnef.nemo_tract.config import InspectFormat
from torch_to_nnef.nemo_tract.inspect import run_inspection
from torch_to_nnef.nemo_tract.model_loader import (
    load_asr_model_from_nemo_slug,
    load_asr_model_from_path,
)
from torch_to_nnef.nemo_tract.provider import NemoProvider
from torch_to_nnef.nemo_tract.registry_utils import (
    dump_registry_from_signatures,
    tie_batch_symbols_in_registry,
    validate_registry_against_signatures,
)
from torch_to_nnef.nemo_tract.wrappers import WrapPreprocessorCast, use_pytorch_sdpa
from torch_to_nnef.remodeler import Stage, save_config
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import SemanticVersion, normalize_cli_list_option

LOGGER = logging.getLogger(__name__)


def setup_inference_target_from_cli_args(cfg: NemoTractConfig) -> TractNNEF:
    """Build a TractNNEF inference target from CLI configuration."""
    if cfg.tract.tract_specific_version:
        assert cfg.tract.tract_specific_path is None, (
            "set either version or path"
        )
        inference_target = TractNNEF(
            SemanticVersion.from_str(cfg.tract.tract_specific_version)
            if isinstance(cfg.tract.tract_specific_version, str)
            else cfg.tract.tract_specific_version
        )
    elif cfg.tract.tract_specific_path:
        expanded = os.path.expandvars(str(cfg.tract.tract_specific_path))
        tract_cli_path = Path(expanded).expanduser().resolve()
        if not tract_cli_path.exists() or not tract_cli_path.is_file():
            raise FileNotFoundError(
                "Invalid --tract-specific-path: "
                f"'{cfg.tract.tract_specific_path}' "
                f"-> '{tract_cli_path}' does not exist or is not a file"
            )
        tract_cli = TractCli(tract_cli_path)
        inference_target = TractNNEF(
            tract_cli.version,
            specific_tract_binary_path=tract_cli_path,
        )
    else:
        inference_target = TractNNEF.latest()

    if cfg.tract.tract_check_io_tolerance == "skip":
        inference_target.check_io = False
    else:
        inference_target.check_io_tolerance = cfg.tract.tract_check_io_tolerance

    if cfg.sdpa.tract_reify_sdpa:
        inference_target.reify_sdpa_operator = True
        if (
            not cfg.sdpa.force_sdpa_pytorch
            and inference_target.version < "0.23.0"
        ):
            LOGGER.warning(
                "Reifying sdpa without forcing pytorch implementation "
                "may export no sdpa ops depending on model."
            )
    return inference_target


def _prepare_model_dtype_and_wrappers(asr_model, cfg: NemoTractConfig):
    """Apply dtype conversions and wrappers based on CLI configuration."""
    if cfg.sdpa.force_sdpa_pytorch:
        use_pytorch_sdpa(asr_model)
    asr_model.eval()
    if cfg.naming.data_type == "float16":
        asr_model = asr_model.half()
        asr_model.preprocessor.to(dtype=torch.float32)
    if cfg.naming.data_type in ["float16", "mixed"] and hasattr(
        asr_model, "preprocessor"
    ):
        asr_model.preprocessor = WrapPreprocessorCast(
            asr_model.preprocessor, dtype=torch.float16
        )
    return asr_model


def _normalize_tolerance(cfg: NemoTractConfig) -> None:
    """Coerce ``tract_check_io_tolerance`` string to enum when needed."""
    if (
        isinstance(cfg.tract.tract_check_io_tolerance, str)
        and cfg.tract.tract_check_io_tolerance != "skip"
    ):
        cfg.tract.tract_check_io_tolerance = TractCheckTolerance(
            cfg.tract.tract_check_io_tolerance
        )


def _build_axis_registry(
    cfg: NemoTractConfig, asr_model, inference_target
) -> AxisSymbolRegistry:
    """Discover signatures and build an AxisSymbolRegistry from CLI config."""
    provider = NemoProvider(
        inference_target=inference_target,
        skip_preprocessor=cfg.subnet.skip_preprocessor,
        split_joint_decoder=cfg.subnet.split_joint_decoder,
        float_dtype=(
            torch.float16
            if cfg.naming.data_type == "float16"
            else torch.float32
        ),
        only_subnets=cfg.subnet.only_subnets,
    )
    raw_sigs = provider.discover_signatures(asr_model, Stage.RAW)
    if cfg.inspect.shape_config is None:
        default_axis_reg = dump_registry_from_signatures(raw_sigs)
        return tie_batch_symbols_in_registry(default_axis_reg)
    axis_reg = load_axis_symbol_registry(cfg.inspect.shape_config)
    validate_registry_against_signatures(raw_sigs, axis_reg)
    return axis_reg


def _prepare_export_dir_and_logging(
    cfg: NemoTractConfig, export_dir: Path
) -> None:
    """Create export directory and attach a file logger when exporting."""
    if cfg.inspect.inspect_signatures or cfg.inspect.dry_run:
        return
    assert not export_dir.exists(), f"export_dir '{export_dir}' must not exist"
    export_dir.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(export_dir / "nemo_tract_export.log")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s,%(msecs)d %(levelname)-8s "
            "[%(filename)s:%(lineno)d] %(message)s",
            "%Y-%m-%d:%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)


def _maybe_dump_export_config(cfg: NemoTractConfig, export_dir: Path) -> None:
    """Write CLI configuration JSON to the export directory."""
    if cfg.inspect.inspect_signatures or cfg.inspect.dry_run:
        return
    with (export_dir / "export_config.json").open("w", encoding="utf8") as fh:
        payload = asdict(cfg)

        def _coerce(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            if isinstance(o, dict):
                return {k: _coerce(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_coerce(v) for v in o]
            return o

        json.dump(_coerce(payload), fh, indent=2)


def _init_logging_and_export_dir(cfg: NemoTractConfig) -> Path:
    """Initialize logging and create export directory; return the path."""
    level = logging.DEBUG if cfg.log.verbose else logging.INFO
    set_lib_log_level(level)
    export_dir = Path(cfg.output.export_dir)
    _prepare_export_dir_and_logging(cfg, export_dir)
    LOGGER.info("started nemo_tract export with config: %s", cfg)
    return export_dir


def _write_config_header(fh, model_label: str, now: str, cmd: str) -> None:
    fh.write(
        textwrap.dedent(
            f"""\
# '{model_label}' shapes config generated on '{now}'
# Command:
#   {cmd}
# Edit dims/symbols as needed. Keys must match subnet/input names.
#
# Optional: per-subnet 'outputs_keep' filters exported outputs;
# if not set, all outputs declared by the subnet are kept.
"""
        )
        + "\n"
    )


def _write_config_example_block(fh) -> None:
    fh.write(
        textwrap.dedent(
            f"""\
# Config example (structured):
# encoder:
#   inputs:
#     audio_signal:
#       original_shape:
#         [AUDIO_SIGNAL{_SEP}BATCH, 128, AUDIO_SIGNAL{_SEP}TIME]
#       collapse_dims: [AUDIO_SIGNAL{_SEP}BATCH]
#     length:
#       original_shape: [LENGTH{_SEP}BATCH]
#       collapse_dims: [LENGTH{_SEP}BATCH]
#       bind_scalar_to_dim_size: encoder.audio_signal.AUDIO_SIGNAL{_SEP}TIME
# decoder_joint:
#   inputs:
#     encoder_outputs:
#       original_shape:
#         [ENCODER_OUTPUTS{_SEP}BATCH, 1024, ENCODER_OUTPUTS{_SEP}TIME]
#       collapse_dims:
#         [ENCODER_OUTPUTS{_SEP}BATCH, ENCODER_OUTPUTS{_SEP}TIME]

# decoder:
#   # Optionally unify symbols with 'renamed_symbols' if needed.
#   # Aliases in 'renamed_symbols' are accepted for any symbol.
#   # Optionally select exported outputs (default: keep all)
#   outputs_keep: [LOG_PROBS, STATES_0, STATES_1]
#   inputs:
#     targets:
#       original_shape: [TARGETS{_SEP}BATCH, TARGETS{_SEP}TIME]
#       # Aliases are accepted when listed in renamed_symbols
#       collapse_dims: [BATCH]
#     states_0:
#       original_shape: [2, STATES_0{_SEP}BATCH, 640]
#       collapse_dims: [BATCH]
#     states_1:
#       original_shape: [2, STATES_1{_SEP}BATCH, 640]
#       collapse_dims: [BATCH]
#   # Binding can also use alias symbols listed in renamed_symbols
"""
        )
        + "\n\n"
    )


def _dump_shape_config_template(
    *,
    cfg: NemoTractConfig,
    registry: AxisSymbolRegistry,
    model_label: str,
) -> None:
    """Generate and write a structured shape-config template to file."""
    cfg.inspect.dump_shape_config.parent.mkdir(parents=True, exist_ok=True)
    with cfg.inspect.dump_shape_config.open("w", encoding="utf8") as fh:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        cmd = " ".join(shlex.quote(a) for a in sys.argv)
        _write_config_header(fh, model_label, now, cmd)
        _write_config_example_block(fh)
        save_config(cfg.inspect.dump_shape_config, registry, stream=fh)


def _normalize_inspect_stages(cfg: NemoTractConfig):
    """Return list of Stage values or None from CLI args."""
    raw_stages = cfg.inspect.inspect_stages or None
    if not raw_stages:
        return None
    if any(s == "all" for s in raw_stages):
        return list(Stage)
    return [Stage(s) for s in raw_stages]


def _run_inspection_flow(
    *,
    cfg: NemoTractConfig,
    axis_reg: AxisSymbolRegistry,
    asr_model,
    inference_target,
    model_label: str,
) -> None:
    """Execute the inspection flow including optional template dump."""
    if cfg.inspect.dump_shape_config is not None:
        _dump_shape_config_template(
            cfg=cfg,
            registry=axis_reg,
            model_label=model_label,
        )
    run_inspection(
        asr_model=asr_model,
        inference_target=inference_target,
        skip_preprocessor=cfg.subnet.skip_preprocessor,
        split_joint_decoder=cfg.subnet.split_joint_decoder,
        float_dtype=(
            torch.float16
            if cfg.naming.data_type == "float16"
            else torch.float32
        ),
        only_subnets=cfg.subnet.only_subnets,
        stages=_normalize_inspect_stages(cfg),
        fmt=cfg.inspect.inspect_format,
        to_path=cfg.inspect.inspect_output,
        diff=cfg.inspect.inspect_diff,
        axis_registry=axis_reg,
        model_label=model_label,
    )


def run_export(cfg: NemoTractConfig) -> None:
    """Orchestrate a full NeMo→NNEF export from a CLI configuration."""
    init_log()
    export_dir = _init_logging_and_export_dir(cfg)

    asr_model = (
        load_asr_model_from_path(cfg.model.model_path)
        if cfg.model.model_path is not None
        else load_asr_model_from_nemo_slug(cfg.model.model_slug)
    )
    asr_model = _prepare_model_dtype_and_wrappers(asr_model, cfg)
    _normalize_tolerance(cfg)
    inference_target = setup_inference_target_from_cli_args(cfg)
    _maybe_dump_export_config(cfg, export_dir)
    axis_reg = _build_axis_registry(cfg, asr_model, inference_target)

    if cfg.inspect.inspect_signatures or cfg.inspect.dry_run:
        model_label = (
            str(Path(cfg.model.model_path).resolve())
            if cfg.model.model_path
            else str(cfg.model.model_slug)
        )
        _run_inspection_flow(
            cfg=cfg,
            axis_reg=axis_reg,
            asr_model=asr_model,
            inference_target=inference_target,
            model_label=model_label,
        )
        if cfg.inspect.dry_run:
            return

    export_nemo_from_model(
        model=asr_model,
        target=inference_target,
        export_dir=export_dir,
        axis_reg=axis_reg,
        cfg=NemoExportConfig(
            pretrained_name=cfg.model.model_slug,
            naming_scheme=cfg.naming.naming_scheme,
            data_type=cfg.naming.data_type,
            subnet=cfg.subnet,
            compression=cfg.compression,
        ),
    )


def add_inspection_args(parser: argparse.ArgumentParser) -> None:
    """Register inspection and dry-run arguments."""
    parser.add_argument(
        "--inspect-signatures",
        action="store_true",
        help=(
            "Inspect per-subnetwork IO signatures without exporting. "
            "Shows shapes, dtypes, and names at chosen stages."
        ),
    )
    parser.add_argument(
        "--inspect-stage",
        dest="inspect_stages",
        action="append",
        default=None,
        choices=[st.value for st in Stage] + ["all"],
        help=(
            "Which stage(s) to display: raw|collapsed|bound|final|all. "
            "Repeat flag to show multiple; default is final."
        ),
    )
    parser.add_argument(
        "--inspect-format",
        dest="inspect_format",
        default=InspectFormat.HUMAN_RICH.value,
        choices=[f.value for f in InspectFormat],
        help="Inspection output format (human, human-rich, or json).",
    )
    parser.add_argument(
        "--inspect-output",
        dest="inspect_output",
        type=Path,
        default=None,
        help=(
            "Optional file path to write inspection output. If omitted, "
            "prints to stdout."
        ),
    )
    parser.add_argument(
        "--inspect-diff",
        action="store_true",
        help=(
            "When two stages are selected, print concise diffs per IO. "
            "(human format only)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run model loading and inspection pipeline without writing any "
            "export artifacts. Implies --inspect-signatures."
        ),
    )


def parse_config() -> NemoTractConfig:
    """Build and parse CLI args into a NemoTractConfig dataclass."""
    parser = argparse.ArgumentParser(
        description="Export NeMo ASR model to NNEF format using TractNNEF."
    )
    parser.add_argument(
        "-s",
        "--model-slug",
        type=str,
        default="*",
        help=(
            "The model slug for the NeMo ASR model to export."
            " If unknown, leave blank to select interactively."
        ),
    )
    parser.add_argument(
        "-p",
        "--model-path",
        type=str,
        help="Path to a local .nemo file.",
    )
    parser.add_argument(
        "-e",
        "--export-dir",
        type=Path,
        required=True,
        help="Directory to save the exported NNEF files.",
    )
    parser.add_argument(
        "--skip-preprocessor",
        action="store_true",
        help="Skip exporting the preprocessor subnet.",
    )
    parser.add_argument(
        "--split-joint-decoder",
        action="store_true",
        help="Split the joint and decoder subnets during export.",
    )
    parser.add_argument(
        "--force-sdpa-pytorch",
        action="store_true",
        help=(
            "Force sdpa to use PyTorch implementation (recommended before"
            " stable Tract SDPA support)."
        ),
    )
    parser.add_argument(
        "--tract-reify-sdpa",
        action="store_true",
        help="Force SDPA reification in NNEF (auto-enabled for Tract>=0.23.0).",
    )
    parser.add_argument(
        "--tract-specific-version",
        type=str,
        default=None,
        help="Use a specific Tract version (semantic).",
    )
    parser.add_argument(
        "--tract-specific-path",
        type=str,
        default=None,
        help="Use a specific Tract binary at path.",
    )
    parser.add_argument(
        "--naming-scheme",
        type=str,
        default=VariableNamingScheme.NATURAL_VERBOSE_CAMEL.value,
        choices=[vns.value for vns in VariableNamingScheme],
        help="NNEF variable naming scheme.",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="float32",
        choices=["float32", "float16", "mixed"],
        help="Data type for export.",
    )
    parser.add_argument(
        "-tt",
        "--tract-check-io-tolerance",
        default=TractCheckTolerance.APPROXIMATE.value,
        choices=[t.value for t in TractCheckTolerance] + ["skip"],
        help="Tract check IO tolerance level.",
    )
    parser.add_argument(
        "--compress-registry",
        type=str,
        default=CompressionConfig().compress_registry,
        help="Compression registry for exported NNEF subnets.",
    )
    parser.add_argument(
        "--compress-method",
        type=str,
        default=None,
        help="Compression method for exported NNEF subnets.",
    )
    parser.add_argument(
        "--dump-checked-io",
        required=False,
        default=False,
        action="store_true",
        help="Dump tested IO to export_dir/test for checking.",
    )

    # Inspection / dry-run controls
    add_inspection_args(parser)

    parser.add_argument(
        "--only-subnet",
        dest="only_subnets",
        action="append",
        default=None,
        help=(
            "Export only the specified subnet(s). Repeat the flag or use a "
            "comma-separated list to include multiple (e.g. --only-subnet "
            "encoder --only-subnet decoder or --only-subnet encoder,decoder)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display debug information.",
    )
    parser.add_argument(
        "--shape-config",
        type=Path,
        default=None,
        help=(
            "Optional YAML/JSON mapping of input name → symbolic dims\n"
            "e.g. encoder.audio_signal: [AUDIO_SIGNAL::BATCH,128,"
            "AUDIO_SIGNAL::TIME]"
        ),
    )
    parser.add_argument(
        "--dump-shape-config",
        type=Path,
        default=None,
        help=(
            "Write a template shapes YAML (nested by subnet) built from the "
            "current model inspect (raw stage)."
        ),
    )

    ns = parser.parse_args()

    only_subnets: T.Optional[T.List[str]] = normalize_cli_list_option(
        getattr(ns, "only_subnets", None)
    )
    inspect_stages: T.Optional[T.List[str]] = (
        list(ns.inspect_stages) if getattr(ns, "inspect_stages", None) else None
    )

    return NemoTractConfig(
        model=ModelSelectionConfig(
            model_slug=ns.model_slug,
            model_path=ns.model_path,
        ),
        output=OutputConfig(export_dir=ns.export_dir),
        subnet=SubnetSelectionConfig(
            skip_preprocessor=ns.skip_preprocessor,
            split_joint_decoder=ns.split_joint_decoder,
            only_subnets=only_subnets,
        ),
        sdpa=SdpaConfig(
            force_sdpa_pytorch=ns.force_sdpa_pytorch,
            tract_reify_sdpa=ns.tract_reify_sdpa,
        ),
        tract=TractBinaryConfig(
            tract_specific_version=ns.tract_specific_version,
            tract_specific_path=ns.tract_specific_path,
            tract_check_io_tolerance=ns.tract_check_io_tolerance,
        ),
        naming=NamingPrecisionConfig(
            naming_scheme=VariableNamingScheme(ns.naming_scheme),
            data_type=ns.data_type,
        ),
        compression=CompressionConfig(
            compress_registry=ns.compress_registry,
            compress_method=ns.compress_method,
            dump_checked_io=ns.dump_checked_io,
        ),
        inspect=InspectionConfig(
            inspect_signatures=ns.inspect_signatures,
            inspect_stages=inspect_stages,
            inspect_format=InspectFormat(ns.inspect_format),
            inspect_output=ns.inspect_output,
            inspect_diff=ns.inspect_diff,
            shape_config=ns.shape_config,
            dump_shape_config=ns.dump_shape_config,
            dry_run=ns.dry_run,
        ),
        log=LogConfig(verbose=ns.verbose),
    )


def main() -> None:
    """CLI entry point: parse config and run export."""
    cfg = parse_config()
    run_export(cfg)
