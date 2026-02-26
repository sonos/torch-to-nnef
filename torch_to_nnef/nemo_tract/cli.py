import argparse
import json
import logging
from pathlib import Path

import torch

from torch_to_nnef.compress import DEFAULT_COMPRESSION_REGISTRY
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractCli,
    TractNNEF,
)
from torch_to_nnef.log import init_log, set_lib_log_level
from torch_to_nnef.nemo_tract.export import export_nemo_asr_model
from torch_to_nnef.nemo_tract.model_loader import load_asr_model_from_nemo_slug
from torch_to_nnef.nemo_tract.wrappers import (
    WrapPreprocessorCast,
    use_pytorch_sdpa,
)
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import SemanticVersion

LOGGER = logging.getLogger(__name__)


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
        "--collapse-batch-dim",
        action="store_true",
        help=(
            "Remove batch dimension from exported subnet interfaces and "
            "hide batch-only length inputs (length, target_length, ...)."
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

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display debug information.",
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
        tract_cli_path = Path(args.tract_specific_path)
        assert tract_cli_path.exists(), tract_cli_path
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


def main():
    init_log()
    args = parser_cli()
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    set_lib_log_level(log_level)
    export_dir = Path(args.export_dir)
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
    LOGGER.info("started nemo_tract export with args: %s", args)
    # ensure that the model is loaded on CPU
    asr_model = load_asr_model_from_nemo_slug(args.model_slug)

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

    if (
        isinstance(args.tract_check_io_tolerance, str)
        and args.tract_check_io_tolerance != "skip"
    ):
        args.tract_check_io_tolerance = TractCheckTolerance(
            args.tract_check_io_tolerance
        )

    inference_target = setup_inference_target_from_cli_args(args)

    with (export_dir / "export_config.json").open("w", encoding="utf8") as fh:
        json.dump(
            {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
            },
            fh,
            indent=2,
        )

    def call_export(float_dtype=torch.float32):
        export_nemo_asr_model(
            asr_model,
            inference_target,
            export_dir,
            nnef_variable_naming_scheme=VariableNamingScheme(
                args.naming_scheme
            ),
            compress_registry=args.compress_registry,
            compress_method=args.compress_method,
            skip_preprocessor=args.skip_preprocessor,
            split_joint_decoder=args.split_joint_decoder,
            extra_cfg={"pretrained_name": args.model_slug},
            float_dtype=float_dtype,
            dump_checked_io=args.dump_checked_io,
            collapse_batch_dim=args.collapse_batch_dim,
        )

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
                call_export(float_dtype=torch.float16)
        except ImportError as ie:
            raise ImportError(
                "To use mixed precision export please install recent torch"
            ) from ie
    else:
        call_export(
            float_dtype=torch.float16
            if args.data_type == "float16"
            else torch.float32
        )
