import argparse
import datetime
import json
import logging
import os
import shlex
import sys
import textwrap
import typing as T
from pathlib import Path

import torch

from torch_to_nnef.compress import DEFAULT_COMPRESSION_REGISTRY
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
from torch_to_nnef.nemo_tract.constants import (
    NEMO_INPUT_SYMBOL_SEPARATOR as _SEP,
)
from torch_to_nnef.nemo_tract.export import export_nemo_asr_model
from torch_to_nnef.nemo_tract.inspect import (
    InspectFormat,
    run_inspection,
)
from torch_to_nnef.nemo_tract.model_loader import (
    load_asr_model_from_nemo_slug,
    load_asr_model_from_path,
)
from torch_to_nnef.nemo_tract.provider import NemoProvider
from torch_to_nnef.nemo_tract.registry_utils import (
    dump_registry_from_signatures,
    validate_registry_against_signatures,
)
from torch_to_nnef.nemo_tract.wrappers import (
    WrapPreprocessorCast,
    use_pytorch_sdpa,
)
from torch_to_nnef.remodeler import Stage, save_config
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import SemanticVersion, normalize_cli_list_option

LOGGER = logging.getLogger(__name__)


def add_inspection_args(parser: argparse.ArgumentParser) -> None:
    """Register inspection and dry-run arguments.

    Args:
        parser: The CLI parser to augment with inspection flags.

    Notes:
        These flags enable signature inspection at various stages without
        affecting the actual export behavior (unless --dry-run is used).
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


def parser_cli():
    """Build the CLI parser for NeMo ASR model export to NNEF format."""
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
        default=DEFAULT_COMPRESSION_REGISTRY,
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

    # Inspection / dry-run controls (Phase 0)
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
            f"e.g. encoder.audio_signal: [AUDIO_SIGNAL{_SEP}BATCH,128,"
            f"AUDIO_SIGNAL{_SEP}TIME]"
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

    return parser.parse_args()


def setup_inference_target_from_cli_args(args) -> TractNNEF:
    """Setup TractNNEF inference target from CLI arguments."""
    if args.tract_specific_version:
        assert args.tract_specific_path is None, "set either version or path"
        inference_target = TractNNEF(
            SemanticVersion.from_str(args.tract_specific_version)
            if isinstance(args.tract_specific_version, str)
            else args.tract_specific_version
        )
    elif args.tract_specific_path:
        # Expand env vars and user home (e.g. "$HOME" or "~/"), then resolve
        expanded = os.path.expandvars(str(args.tract_specific_path))
        tract_cli_path = Path(expanded).expanduser().resolve()
        if not tract_cli_path.exists() or not tract_cli_path.is_file():
            raise FileNotFoundError(
                f"Invalid --tract-specific-path: '{args.tract_specific_path}' "
                f"-> '{tract_cli_path}' does not exist or is not a file"
            )
        tract_cli = TractCli(tract_cli_path)
        inference_target = TractNNEF(
            tract_cli.version,
            specific_tract_binary_path=tract_cli_path,
        )
    else:
        inference_target = TractNNEF.latest()
    if args.tract_check_io_tolerance == "skip":
        inference_target.check_io = False
    else:
        inference_target.check_io_tolerance = args.tract_check_io_tolerance

    if args.tract_reify_sdpa:
        inference_target.reify_sdpa_operator = True
        if not args.force_sdpa_pytorch and inference_target.version < "0.23.0":
            LOGGER.warning(
                "Reifying sdpa without forcing pytorch implementation "
                "may export no sdpa ops depending on model."
            )
    return inference_target


def _prepare_export_dir_and_logging(args, export_dir: Path) -> None:
    """Create export directory and attach file logger when exporting.

    Args:
        args: Parsed CLI args.
        export_dir: Target directory for export artifacts.
    """
    if args.inspect_signatures or args.dry_run:
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


def _maybe_dump_export_config(args, export_dir: Path) -> None:
    """Write CLI configuration to export directory if exporting."""
    if args.inspect_signatures or args.dry_run:
        return
    with (export_dir / "export_config.json").open("w", encoding="utf8") as fh:
        json.dump(
            {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
            },
            fh,
            indent=2,
        )


def _prepare_model_dtype_and_wrappers(asr_model, args):
    """Apply dtype conversions and wrappers based on CLI args."""
    if args.force_sdpa_pytorch:
        use_pytorch_sdpa(asr_model)
    asr_model.eval()
    if args.data_type == "float16":
        asr_model = asr_model.half()
        asr_model.preprocessor.to(dtype=torch.float32)
    if args.data_type in ["float16", "mixed"] and hasattr(
        asr_model, "preprocessor"
    ):
        asr_model.preprocessor = WrapPreprocessorCast(
            asr_model.preprocessor, dtype=torch.float16
        )
    return asr_model


def _normalize_inspect_stages(args):
    """Return list of Stage or None based on CLI args."""
    raw_stages = args.inspect_stages or None
    if not raw_stages:
        return None
    if any(s == "all" for s in raw_stages):
        return list(Stage)
    return [Stage(s) for s in raw_stages]


def _build_axis_registry_from_args(args) -> T.Optional[AxisSymbolRegistry]:
    """Load optional shape-config into an AxisSymbolRegistry.

    Returns None if no shape-config was provided.
    """
    if args.shape_config is None:
        return None
    return load_axis_symbol_registry(args.shape_config)


def _normalize_tolerance(args) -> None:
    """Normalize `tract_check_io_tolerance` CLI option to enum when needed."""
    if (
        isinstance(args.tract_check_io_tolerance, str)
        and args.tract_check_io_tolerance != "skip"
    ):
        args.tract_check_io_tolerance = TractCheckTolerance(
            args.tract_check_io_tolerance
        )


def _compute_log_level(args) -> int:
    """Return logging level based on verbosity flag."""
    return logging.DEBUG if args.verbose else logging.INFO


def _init_logging_and_export_dir(args) -> Path:
    """Initialize logging and export directory; return export_dir Path."""
    set_lib_log_level(_compute_log_level(args))
    export_dir = Path(args.export_dir)
    _prepare_export_dir_and_logging(args, export_dir)
    LOGGER.info("started nemo_tract export with args: %s", args)
    return export_dir


def _dump_shape_config_template(
    *,
    args,
    asr_model,
    inference_target,
    model_label: str,
) -> None:
    """Generate and write a structured shape-config template to file.

    This encapsulates the long YAML dump logic to keep `main()` small.
    """
    # Use provider-agnostic remodeler discovery (NeMo provider here)
    provider = NemoProvider(
        inference_target=inference_target,
        skip_preprocessor=args.skip_preprocessor,
        split_joint_decoder=args.split_joint_decoder,
        float_dtype=(
            torch.float16 if args.data_type == "float16" else torch.float32
        ),
        only_subnets=args.only_subnets,
    )
    snaps = provider.discover_signatures(asr_model, Stage.RAW)
    registry = dump_registry_from_signatures(snaps)

    args.dump_shape_config.parent.mkdir(parents=True, exist_ok=True)
    with args.dump_shape_config.open("w", encoding="utf8") as fh:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        cmd = " ".join(shlex.quote(a) for a in sys.argv)
        _write_config_header(fh, model_label, now, cmd)
        _write_config_example_block(fh)
        # Dump remodeler registry to YAML content after header
        save_config(args.dump_shape_config, registry, stream=fh)


def _build_nested_template_dict(snaps, args) -> dict[str, dict]:
    """Build nested dict for template from collected signatures."""
    nested: dict[str, dict] = {}
    for ss in snaps:
        bucket = nested.setdefault(ss.name, {})
        inputs_map: dict = bucket.setdefault("inputs", {})
        # Always include outputs_keep pre-filled; easier to remove than add
        if isinstance(ss.outputs, list) and ss.outputs:
            bucket["outputs_keep"] = [o.name for o in ss.outputs]
        else:
            bucket["outputs_keep"] = []
        for i in ss.inputs:
            dims = [
                int(d) if isinstance(d, int) else str(d)
                for d in (i.shape or [])
            ]
            entry: dict = {}
            entry["original_shape"] = dims
            entry["collapse_dims"] = []
            inputs_map[i.name] = entry
        batch_syms = []
        for i in ss.inputs:
            for d in i.shape or []:
                if (
                    isinstance(d, str)
                    and d.upper().endswith(f"{_SEP}BATCH")
                    and d not in batch_syms
                ):
                    batch_syms.append(d)
        if ss.name in ("decoder", "decoder_joint") and len(batch_syms) > 1:
            bucket["renamed_symbols"] = {"BATCH": batch_syms}
    return nested


def _write_config_header(fh, model_label: str, now: str, cmd: str) -> None:
    """Write the header section of the template file."""
    header = f"""
    # '{model_label}' shapes config generated on '{now}'
    # Command:
    #   {cmd}
    # Edit dims/symbols as needed. Keys must match subnet/input names.
    #
    # Optional: per-subnet 'outputs_keep' filters exported outputs;
    # if not set, all outputs declared by the subnet are kept.
    """.strip()
    fh.write(textwrap.dedent(header) + "\n")


def _write_config_example_block(fh) -> None:
    """Write the example config block for guidance."""
    example = f"""
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
    """.strip()
    fh.write(textwrap.dedent(example) + "\n\n")


def _run_inspection_flow(
    *,
    args,
    asr_model,
    inference_target,
    model_label: str,
) -> None:
    """Execute the inspection flow including optional template dump."""
    norm_stages = _normalize_inspect_stages(args)
    fmt_enum = InspectFormat(args.inspect_format)
    axis_reg = _build_axis_registry_from_args(args)
    if args.dump_shape_config is not None:
        _dump_shape_config_template(
            args=args,
            asr_model=asr_model,
            inference_target=inference_target,
            model_label=model_label,
        )
    # Validate shape-config early to catch mistakes before printing results
    if axis_reg is not None:
        provider = NemoProvider(
            inference_target=inference_target,
            skip_preprocessor=args.skip_preprocessor,
            split_joint_decoder=args.split_joint_decoder,
            float_dtype=(
                torch.float16 if args.data_type == "float16" else torch.float32
            ),
            only_subnets=args.only_subnets,
        )
        raw_sigs = provider.discover_signatures(asr_model, Stage.RAW)
        validate_registry_against_signatures(raw_sigs, axis_reg)

    run_inspection(
        asr_model=asr_model,
        inference_target=inference_target,
        skip_preprocessor=args.skip_preprocessor,
        split_joint_decoder=args.split_joint_decoder,
        float_dtype=(
            torch.float16 if args.data_type == "float16" else torch.float32
        ),
        only_subnets=args.only_subnets,
        stages=norm_stages,
        fmt=fmt_enum,
        to_path=args.inspect_output,
        diff=args.inspect_diff,
        axis_registry=axis_reg,
        model_label=model_label,
    )


def _call_export(
    *,
    asr_model,
    inference_target,
    export_dir: Path,
    args,
    float_dtype: torch.dtype,
) -> None:
    """Thin wrapper to perform the export with provided dtype and args."""
    export_nemo_asr_model(
        asr_model,
        inference_target,
        export_dir,
        nnef_variable_naming_scheme=VariableNamingScheme(args.naming_scheme),
        compress_registry=args.compress_registry,
        compress_method=args.compress_method,
        skip_preprocessor=args.skip_preprocessor,
        split_joint_decoder=args.split_joint_decoder,
        only_subnets=args.only_subnets,
        extra_cfg={"pretrained_name": args.model_slug},
        float_dtype=float_dtype,
        dump_checked_io=args.dump_checked_io,
        axis_registry=(
            load_axis_symbol_registry(args.shape_config)
            if args.shape_config is not None
            else None
        ),
    )


def _parse_args():
    """Initialize logging and parse CLI arguments."""
    init_log()
    return parser_cli()


def main():
    args = _parse_args()
    # Normalize early so subsequent logic and config dump see final form
    args.only_subnets = normalize_cli_list_option(args.only_subnets)
    export_dir = _init_logging_and_export_dir(args)
    # ensure that the model is loaded on CPU
    asr_model = (
        load_asr_model_from_path(args.model_path)
        if args.model_path is not None
        else load_asr_model_from_nemo_slug(args.model_slug)
    )

    asr_model = _prepare_model_dtype_and_wrappers(asr_model, args)

    _normalize_tolerance(args)

    inference_target = setup_inference_target_from_cli_args(args)

    _maybe_dump_export_config(args, export_dir)

    # If in inspection mode (explicit or via dry-run), run the inspector.
    if args.inspect_signatures or args.dry_run:
        # Determine a human-friendly model label for inspection header
        model_label = (
            str(Path(args.model_path).resolve())
            if args.model_path
            else str(args.model_slug)
        )
        _run_inspection_flow(
            args=args,
            asr_model=asr_model,
            inference_target=inference_target,
            model_label=model_label,
        )
        if args.dry_run:
            return

    if args.data_type == "mixed":
        try:
            # pylint: disable=import-outside-toplevel
            from torch import autocast

            LOGGER.info("exporting with mixed precision using autocast")
            LOGGER.warning(
                "mixed precision export is experimental "
                "(not supported by tract)"
            )
            with autocast(device_type="cpu", dtype=torch.float16):
                _call_export(
                    asr_model=asr_model,
                    inference_target=inference_target,
                    export_dir=export_dir,
                    args=args,
                    float_dtype=torch.float16,
                )
        except ImportError as ie:
            raise ImportError(
                "To use mixed precision export please install recent torch"
            ) from ie
    else:
        _call_export(
            asr_model=asr_model,
            inference_target=inference_target,
            export_dir=export_dir,
            args=args,
            float_dtype=(
                torch.float16 if args.data_type == "float16" else torch.float32
            ),
        )
