"""CLI parser for NeMo→NNEF export.

This module builds the argparse CLI and returns a `NemoTractCliArgs`
dataclass. Runtime logic lives in `entry.py`.
"""

from __future__ import annotations

import argparse
import typing as T
from pathlib import Path

from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef.nemo_tract.config import (
    CompressionConfig,
    InspectionConfig,
    LogConfig,
    ModelSelectionConfig,
    NamingPrecisionConfig,
    NemoTractConfig,
    OutputConfig,
    SdpaConfig,
    SubnetSelectionConfig,
    TractBinaryConfig,
)
from torch_to_nnef.nemo_tract.entry import run_export
from torch_to_nnef.remodeler import Stage
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import normalize_cli_list_option


def add_inspection_args(parser: argparse.ArgumentParser) -> None:
    """Register inspection and dry-run arguments.

    Args:
        parser: The CLI parser to augment with inspection flags.
    """
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
        default=InspectionConfig().inspect_format,
        choices=["human", "human-rich", "json"],
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
    """Build and parse CLI args into a structured dataclass.

    Returns:
        Parsed CLI options packed in a `NemoTractCliArgs` dataclass.
    """
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
        help=("Path to a local .nemo file."),
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
        help=(
            "Force SDPA reification in NNEF (auto-enabled for Tract>=0.23.0)."
        ),
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

    # Filter which NeMo subnets to export
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

    # Normalize list-type options parsed with action=append
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
            naming_scheme=ns.naming_scheme,
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
            inspect_format=ns.inspect_format,
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
