"""Export entry and runtime orchestration for NeMo→NNEF CLI."""

from __future__ import annotations

import datetime
import json
import logging
import os
import shlex
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

import torch
from torch import autocast

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
from torch_to_nnef.nemo_tract.config import NemoExportConfig, NemoTractConfig
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
    tie_batch_symbols_in_registry,
    validate_registry_against_signatures,
)
from torch_to_nnef.nemo_tract.wrappers import (
    WrapPreprocessorCast,
    use_pytorch_sdpa,
)
from torch_to_nnef.remodeler import Stage, save_config
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import SemanticVersion

LOGGER = logging.getLogger(__name__)


def setup_inference_target_from_cli_args(args: NemoTractConfig) -> TractNNEF:
    """Setup TractNNEF inference target from CLI arguments."""
    if args.tract.tract_specific_version:
        assert args.tract.tract_specific_path is None, (
            "set either version or path"
        )
        inference_target = TractNNEF(
            SemanticVersion.from_str(args.tract.tract_specific_version)
            if isinstance(args.tract.tract_specific_version, str)
            else args.tract.tract_specific_version
        )
    elif args.tract.tract_specific_path:
        # Expand env vars and user home (e.g. "$HOME" or "~/"), then resolve
        expanded = os.path.expandvars(str(args.tract.tract_specific_path))
        tract_cli_path = Path(expanded).expanduser().resolve()
        if not tract_cli_path.exists() or not tract_cli_path.is_file():
            raise FileNotFoundError(
                "Invalid --tract-specific-path: "
                f"'{args.tract.tract_specific_path}' "
                f"-> '{tract_cli_path}' does not exist or is not a file"
            )
        tract_cli = TractCli(tract_cli_path)
        inference_target = TractNNEF(
            tract_cli.version,
            specific_tract_binary_path=tract_cli_path,
        )
    else:
        inference_target = TractNNEF.latest()
    if args.tract.tract_check_io_tolerance == "skip":
        inference_target.check_io = False
    else:
        inference_target.check_io_tolerance = (
            args.tract.tract_check_io_tolerance
        )

    if args.sdpa.tract_reify_sdpa:
        inference_target.reify_sdpa_operator = True
        if (
            not args.sdpa.force_sdpa_pytorch
            and inference_target.version < "0.23.0"
        ):
            LOGGER.warning(
                "Reifying sdpa without forcing pytorch implementation "
                "may export no sdpa ops depending on model."
            )
    return inference_target


def _prepare_export_dir_and_logging(
    args: NemoTractConfig, export_dir: Path
) -> None:
    """Create export directory and attach file logger when exporting.

    Args:
        args: Parsed CLI args.
        export_dir: Target directory for export artifacts.
    """
    if args.inspect.inspect_signatures or args.inspect.dry_run:
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


def _maybe_dump_export_config(args: NemoTractConfig, export_dir: Path) -> None:
    """Write CLI configuration to export directory if exporting."""
    if args.inspect.inspect_signatures or args.inspect.dry_run:
        return
    with (export_dir / "export_config.json").open("w", encoding="utf8") as fh:
        payload = asdict(args)

        def _coerce(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, dict):
                return {k: _coerce(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_coerce(v) for v in o]
            return o

        json.dump(_coerce(payload), fh, indent=2)


def _prepare_model_dtype_and_wrappers(asr_model, args: NemoTractConfig):
    """Apply dtype conversions and wrappers based on CLI args."""
    if args.sdpa.force_sdpa_pytorch:
        use_pytorch_sdpa(asr_model)
    asr_model.eval()
    if args.naming.data_type == "float16":
        asr_model = asr_model.half()
        asr_model.preprocessor.to(dtype=torch.float32)
    if args.naming.data_type in ["float16", "mixed"] and hasattr(
        asr_model, "preprocessor"
    ):
        asr_model.preprocessor = WrapPreprocessorCast(
            asr_model.preprocessor, dtype=torch.float16
        )
    return asr_model


def _normalize_inspect_stages(args: NemoTractConfig):
    """Return list of Stage or None based on CLI args."""
    raw_stages = args.inspect.inspect_stages or None
    if not raw_stages:
        return None
    if any(s == "all" for s in raw_stages):
        return list(Stage)
    return [Stage(s) for s in raw_stages]


def _build_axis_registry_from_args(
    args: NemoTractConfig, asr_model, inference_target
) -> AxisSymbolRegistry:
    """Load optional shape-config into an AxisSymbolRegistry.

    Returns None if no shape-config was provided.
    """
    provider = NemoProvider(
        inference_target=inference_target,
        skip_preprocessor=args.subnet.skip_preprocessor,
        split_joint_decoder=args.subnet.split_joint_decoder,
        float_dtype=(
            torch.float16
            if args.naming.data_type == "float16"
            else torch.float32
        ),
        only_subnets=args.subnet.only_subnets,
    )
    raw_sigs = provider.discover_signatures(asr_model, Stage.RAW)
    if args.inspect.shape_config is None:
        default_axis_reg = dump_registry_from_signatures(raw_sigs)
        # Auto-alias namespaced batch symbols to unified BATCH per subnet
        return tie_batch_symbols_in_registry(default_axis_reg)

    axis_reg = load_axis_symbol_registry(args.inspect.shape_config)
    validate_registry_against_signatures(raw_sigs, axis_reg)
    return axis_reg


def _normalize_tolerance(args: NemoTractConfig) -> None:
    """Normalize `tract_check_io_tolerance` CLI option to enum when needed."""
    if (
        isinstance(args.tract.tract_check_io_tolerance, str)
        and args.tract.tract_check_io_tolerance != "skip"
    ):
        args.tract.tract_check_io_tolerance = TractCheckTolerance(
            args.tract.tract_check_io_tolerance
        )


def _compute_log_level(args: NemoTractConfig) -> int:
    """Return logging level based on verbosity flag."""
    return logging.DEBUG if args.log.verbose else logging.INFO


def _init_logging_and_export_dir(args: NemoTractConfig) -> Path:
    """Initialize logging and export directory; return export_dir Path."""
    set_lib_log_level(_compute_log_level(args))
    export_dir = Path(args.output.export_dir)
    _prepare_export_dir_and_logging(args, export_dir)
    LOGGER.info("started nemo_tract export with config: %s", args)
    return export_dir


def _dump_shape_config_template(
    *,
    args: NemoTractConfig,
    registry: AxisSymbolRegistry,
    model_label: str,
) -> None:
    """Generate and write a structured shape-config template to file."""
    args.inspect.dump_shape_config.parent.mkdir(parents=True, exist_ok=True)
    with args.inspect.dump_shape_config.open("w", encoding="utf8") as fh:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        cmd = " ".join(shlex.quote(a) for a in sys.argv)
        _write_config_header(fh, model_label, now, cmd)
        _write_config_example_block(fh)
        save_config(args.inspect.dump_shape_config, registry, stream=fh)


def _write_config_header(fh, model_label: str, now: str, cmd: str) -> None:
    """Write the header section of the template file."""
    header = textwrap.dedent(
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
    fh.write(header + "\n")


def _write_config_example_block(fh) -> None:
    """Write the example config block for guidance."""
    example = textwrap.dedent(
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
    fh.write(example + "\n\n")


def _run_inspection_flow(
    *,
    args: NemoTractConfig,
    axis_reg,
    asr_model,
    inference_target,
    model_label: str,
) -> None:
    """Execute the inspection flow including optional template dump."""
    norm_stages = _normalize_inspect_stages(args)
    fmt_enum = InspectFormat(args.inspect.inspect_format)
    if args.inspect.dump_shape_config is not None:
        _dump_shape_config_template(
            args=args,
            registry=axis_reg,
            model_label=model_label,
        )

    run_inspection(
        asr_model=asr_model,
        inference_target=inference_target,
        skip_preprocessor=args.subnet.skip_preprocessor,
        split_joint_decoder=args.subnet.split_joint_decoder,
        float_dtype=(
            torch.float16
            if args.naming.data_type == "float16"
            else torch.float32
        ),
        only_subnets=args.subnet.only_subnets,
        stages=norm_stages,
        fmt=fmt_enum,
        to_path=args.inspect.inspect_output,
        diff=args.inspect.inspect_diff,
        axis_registry=axis_reg,
        model_label=model_label,
    )


def run_export(cfg: NemoTractConfig) -> None:
    """Run export/inspection from a prebuilt configuration."""
    init_log()
    export_dir = _init_logging_and_export_dir(cfg)

    # ensure that the model is loaded on CPU
    asr_model = (
        load_asr_model_from_path(cfg.model.model_path)
        if cfg.model.model_path is not None
        else load_asr_model_from_nemo_slug(cfg.model.model_slug)
    )

    asr_model = _prepare_model_dtype_and_wrappers(asr_model, cfg)

    _normalize_tolerance(cfg)

    inference_target = setup_inference_target_from_cli_args(cfg)

    _maybe_dump_export_config(cfg, export_dir)

    axis_reg = _build_axis_registry_from_args(cfg, asr_model, inference_target)
    # If in inspection mode (explicit or via dry-run), run the inspector.
    if cfg.inspect.inspect_signatures or cfg.inspect.dry_run:
        # Determine a human-friendly model label for inspection header
        model_label = (
            str(Path(cfg.model.model_path).resolve())
            if cfg.model.model_path
            else str(cfg.model.model_slug)
        )
        _run_inspection_flow(
            args=cfg,
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


def export_nemo_from_model(
    *,
    model,
    target: TractNNEF,
    export_dir: Path,
    axis_reg: AxisSymbolRegistry,
    cfg: NemoExportConfig,
) -> None:
    """Export a prepared model using a configuration.

    Derives the working dtype from ``cfg.data_type`` and wraps the call in
    ``autocast`` when ``data_type`` is ``"mixed"``.
    """
    float_dtype = (
        torch.float16 if cfg.data_type in ("float16", "mixed") else torch.float32
    )

    def _do_export() -> None:
        export_nemo_asr_model(
            model,
            target,
            export_dir,
            nnef_variable_naming_scheme=VariableNamingScheme(cfg.naming_scheme),
            compress_registry=cfg.compression.compress_registry,
            compress_method=cfg.compression.compress_method,
            skip_preprocessor=cfg.subnet.skip_preprocessor,
            split_joint_decoder=cfg.subnet.split_joint_decoder,
            only_subnets=cfg.subnet.only_subnets,
            extra_cfg={"pretrained_name": cfg.pretrained_name},
            float_dtype=float_dtype,
            dump_checked_io=cfg.compression.dump_checked_io,
            axis_registry=axis_reg,
        )

    if cfg.data_type == "mixed":
        LOGGER.info("exporting with mixed precision using autocast")
        LOGGER.warning(
            "mixed precision export is experimental (not supported by tract)"
        )
        with autocast(device_type="cpu", dtype=torch.float16):
            _do_export()
    else:
        _do_export()
