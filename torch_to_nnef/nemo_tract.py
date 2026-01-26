"""Support for NVIDIA NeMo models export to NNEF (with TractNNEF focus).

Provide utilities to export NeMo models, particularly ASR models,
to the NNEF format using TractNNEF.
Includes functions to handle model subnets, dynamic axes, and
custom extensions required for the export process.

"""

import argparse
import json
import logging
import typing as T
from contextlib import contextmanager
from pathlib import Path

import torch

from torch_to_nnef._optional_types import (
    InjectedLightningModule,
    InjectedNemoModule,
    InjectedOmegaConfModule,
)
from torch_to_nnef.compress import (
    DEFAULT_COMPRESSION_REGISTRY,
    dynamic_load_registry,
)
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractCli,
    TractNNEF,
)
from torch_to_nnef.log import init_log, set_lib_log_level
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import (
    INJECTED,
    SemanticVersion,
    T2NExtra,
    require_extra_decorator,
)

# https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
PARAKEET_V3_SLUG = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_110M_SLUG = "parakeet-tdt_ctc-110m"
# https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
NEMOTRON_0_6B = "nvidia/nemotron-speech-streaming-en-0.6b"


LOGGER = logging.getLogger(__name__)


@contextmanager
@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="nemo")
@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="pytorch_lightning")
def exportable_nemo_net(
    output_name,
    model,
    input_example,
    use_dynamo=False,
    *,
    nemo: InjectedNemoModule = INJECTED,
    pytorch_lightning: InjectedLightningModule = INJECTED,
):
    """Context manager to follow export way of nemo models.

    see: nemo.core.classes.Exportable._export
    """
    typecheck = nemo.core.classes.typecheck
    exportable_class = nemo.core.classes.exportable.Exportable
    parse_input_example = nemo.utils.export_utils.parse_input_example
    wrap_forward_method = nemo.utils.export_utils.wrap_forward_method
    my_args = {"use_dynamo": use_dynamo}

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    exportables = []
    for m in model.modules():
        if isinstance(m, exportable_class):
            exportables.append(m)

    forward_method = None
    old_forward_method = None
    try:
        # Disable typechecks
        typecheck.set_typecheck_enabled(enabled=False)

        # Allow user to completely override forward method to export
        forward_method, old_forward_method = wrap_forward_method(model)

        # Set module mode
        with (
            torch.inference_mode(),
            torch.no_grad(),
            torch.jit.optimized_execution(True),
            pytorch_lightning.core.module._jit_is_scripting(),
        ):
            if input_example is None:
                input_example = model.input_module.input_example()

            # Run (posibly overridden) prepare methods before calling forward()
            for ex in exportables:
                ex._prepare_for_export(**my_args, noreplace=True)
            model._prepare_for_export(
                output=output_name, input_example=input_example, **my_args
            )

            input_list, input_dict = parse_input_example(input_example)
            output_example = model.forward(*input_list, **input_dict)
            if not isinstance(output_example, tuple):
                output_example = (output_example,)

            # dynamic axis is a mapping from input/output_name
            # => list of "dynamic" indices
            dynamic_axes = model.dynamic_shapes_for_export(use_dynamo)
            yield input_example, output_example, dynamic_axes
    finally:
        typecheck.enable_wrapping(enabled=True)
        typecheck.set_typecheck_enabled(enabled=True)
        if forward_method:
            type(model).forward = old_forward_method
        model._export_teardown()


def iter_nemo_model_subnets(model, input_example=None):
    """Iterator over exportable subnets of a nemo model.

    Yields:
        subnet_name: name of the subnet
        subnet: the subnet module
        input_example: input example for the subnet
        dynamic_axes: dynamic axes info for the subnet

    see: nemo.core.classes.Exportable.export

    """
    for subnet_name in model.list_export_subnets():
        subnet = model.get_export_subnet(subnet_name)
        if subnet_name == "decoder_joint":
            input_example = None  # reset input example for joint
            # because need more parameters than encoder output only
        with exportable_nemo_net(subnet_name, subnet, input_example) as (
            #  pylint: disable-next=redefined-argument-from-local
            input_example,
            out_example,
            dynamic_axes,
        ):
            if len(input_example) > len(subnet.input_names):
                # if < that means some inputs are optional
                raise RuntimeError(
                    "declared input names:",
                    subnet.input_names,
                    f"but expected {len(input_example)} inputs",
                )
            yield subnet_name, subnet, input_example, dynamic_axes
            # Propagate input example
            # (default scenario, may need to be overriden)
            if input_example is not None:
                input_example = out_example


def build_dynamic_axes(subnet, nemo_dynamic_axes):  # noqa: MC0001
    """Build dynamic axes mapping and custom extensions for nemo subnet."""
    dynamic_axes = {}
    # Assume each input always start by Batch dimension
    custom_extensions = set()

    def build_partial_dynamic_axes(
        iname: str, symbols: T.Union[str, T.List[str]], suffix: str = ""
    ):
        siname = iname + suffix
        dynamic_axes[siname] = {}
        for axis in nemo_dynamic_axes[iname]:
            if symbols[axis] in "BSA":
                custom_extensions.add(f"tract_assert {symbols[axis]} >= 1")
            dynamic_axes[siname][axis] = symbols[axis]

    for iname in subnet.input_names:
        if iname in nemo_dynamic_axes:
            if not nemo_dynamic_axes[iname]:
                continue
            symbols = ""
            if iname == "input_signal":
                symbols = "BA"  # Batch, audio Frames
            elif iname == "audio_signal":
                assert max(nemo_dynamic_axes[iname]) < 3
                symbols = "BFS"  # Batch, Features, Stream
            elif iname == "length":
                assert max(nemo_dynamic_axes[iname]) < 1
                symbols = "B"  # Batch
            elif iname == "encoder_outputs":
                # example: B,512,(S+7)/8,F32
                symbols = "BHR"  # Batch, High End Features, ReducedStream
            elif iname == "targets":
                symbols = "BT"  # Batch, TargetInfo
            elif iname == "target_length":
                symbols = "B"  # Batch
            elif iname == "input_states_1":
                symbols = ["STATES_1_DIM_1", "B", "STATES_1_DIM_2"]
            elif iname == "input_states_2":
                symbols = ["STATES_2_DIM_1", "B", "STATES_2_DIM_2"]
            elif iname == "states":
                build_partial_dynamic_axes(
                    iname,
                    ["STATES_1_DIM_1", "B", "STATES_1_DIM_2"],
                    suffix="_0",
                )
                build_partial_dynamic_axes(
                    iname,
                    ["STATES_2_DIM_1", "B", "STATES_2_DIM_2"],
                    suffix="_1",
                )
                continue
            elif iname == "decoder_outputs":
                # Batch, output decoder, Unknown, Time dimension decoder
                symbols = "BOUT"
            else:
                raise NotImplementedError(
                    f"cannot guess dynamic axis symbols for input '{iname}'"
                )
            build_partial_dynamic_axes(iname, symbols)
    return dynamic_axes, custom_extensions


ExportParameters = T.NamedTuple(
    "ExportParameters",
    [
        ("name", str),
        ("model", torch.nn.Module),
        ("test_input", object),
        ("inference_target", InferenceTarget),
        ("input_names", list),
        ("output_names", list),
        ("custom_extensions", list),
        ("allow_same_io_names", bool),
        ("specific_tract_properties", dict),
    ],
)


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="nemo")
def build_custom_subnet_tract_properties(
    subnet_name, subnet, *, nemo: InjectedNemoModule = INJECTED
):
    """Build custom tract properties for nemo subnet."""
    return {
        "subnet_name": subnet_name,
        "n_parameters": sum(_.numel() for _ in subnet.parameters()),
        "nemo_version": nemo.__version__,
    }


def iter_export_params_for_generic_nemo_asr_model(
    asr_model,
    inference_target,
    skip_preprocessor: bool = False,
    split_joint_decoder: bool = False,
) -> T.Iterator[ExportParameters]:
    """Iterator over export parameters for a generic NeMo ASR model.

    Args:
        asr_model: The NeMo ASR model to export.
        inference_target: The target inference type.
        skip_preprocessor: Whether to skip exporting the preprocessor subnet.
        split_joint_decoder:
            Whether to split the joint and decoder subnets exported.

    Yields:
        ExportParameters for each subnet of the ASR model, with the preprocessor
    """
    asr_model.eval()
    inps = asr_model.preprocessor.input_example()

    if not skip_preprocessor:
        with exportable_nemo_net(
            "preprocessor", asr_model.preprocessor, inps
        ) as (
            input_example,
            _,
            nemo_dynamic_axes,
        ):
            dynamic_axes, custom_extensions = build_dynamic_axes(
                asr_model.preprocessor, nemo_dynamic_axes
            )

            subnet_name = "preprocessor"
            yield ExportParameters(
                name=subnet_name,
                model=asr_model.preprocessor,
                test_input=inps,
                inference_target=inference_target.with_dynamic_axes(
                    dynamic_axes
                ),
                input_names=asr_model.preprocessor.input_names[: len(inps)],
                output_names=asr_model.preprocessor.output_names,
                custom_extensions=list(custom_extensions),
                allow_same_io_names=False,  # not used for preprocessor export
                specific_tract_properties=build_custom_subnet_tract_properties(
                    subnet_name, asr_model.preprocessor
                ),
            )

    for (
        subnet_name,
        subnet,
        input_example,
        nemo_dynamic_axes,
    ) in iter_nemo_model_subnets(asr_model):
        dynamic_axes, custom_extensions = build_dynamic_axes(
            subnet, nemo_dynamic_axes
        )
        inames = subnet.input_names[: len(input_example)]
        onames = [
            # ensure that nop input to output
            # are not force renamed ...
            "target_length"
            if on == "prednet_lengths" and "target_length" in inames
            else on
            for on in subnet.output_names
        ]
        if split_joint_decoder and subnet_name == "decoder_joint":
            # split into decoder and joint
            # assume last two inputs are encoder_outputs
            # and encoder_output_length
            decoder_input_example = asr_model.decoder.input_example()
            joint_input_example = asr_model.joint.input_example()

            decoder_inames = asr_model.decoder.input_names[
                : len(decoder_input_example)
            ]

            joint_inames = asr_model.joint.input_names[
                : len(joint_input_example)
            ]
            use_dynamo = False
            dynamic_axes, custom_extensions = build_dynamic_axes(
                asr_model.decoder,
                asr_model.decoder.dynamic_shapes_for_export(use_dynamo),
            )
            # decoder part
            yield ExportParameters(
                name="decoder",
                model=asr_model.decoder,
                test_input=decoder_input_example,
                inference_target=inference_target.with_dynamic_axes(
                    {
                        k: v
                        for k, v in dynamic_axes.items()
                        if k not in ("encoder_outputs", "encoder_output_length")
                    }
                ),
                input_names=decoder_inames,
                output_names=onames,
                custom_extensions=list(custom_extensions),
                allow_same_io_names=True,
                specific_tract_properties=build_custom_subnet_tract_properties(
                    "decoder", asr_model.decoder
                ),
            )

            dynamic_axes, custom_extensions = build_dynamic_axes(
                asr_model.joint,
                asr_model.joint.dynamic_shapes_for_export(use_dynamo),
            )
            # joint part
            yield ExportParameters(
                name="joint",
                model=asr_model.joint,
                test_input=joint_input_example,
                inference_target=inference_target.with_dynamic_axes(
                    {
                        k: v
                        for k, v in dynamic_axes.items()
                        if k in ("encoder_outputs", "encoder_output_length")
                    }
                ),
                input_names=joint_inames,
                output_names=onames,
                custom_extensions=list(custom_extensions),
                allow_same_io_names=True,
                specific_tract_properties=build_custom_subnet_tract_properties(
                    "joint", asr_model.joint
                ),
            )
            continue

        yield ExportParameters(
            name=subnet_name,
            model=subnet,
            test_input=input_example,
            inference_target=inference_target.with_dynamic_axes(dynamic_axes),
            input_names=inames,
            output_names=onames,
            custom_extensions=list(custom_extensions),
            allow_same_io_names=(subnet_name == "decoder_joint"),
            specific_tract_properties=build_custom_subnet_tract_properties(
                subnet_name, subnet
            ),
        )


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="omegaconf")
def export_nemo_asr_model(
    asr_model,
    inference_target,
    export_dir: Path,
    compress_registry: str,
    compress_method: T.Optional[str] = None,
    skip_preprocessor: bool = False,
    split_joint_decoder: bool = False,
    extra_cfg: T.Optional[T.Dict[str, T.Any]] = None,
    *,
    omegaconf: InjectedOmegaConfModule = INJECTED,
    **kwargs,
):
    """Export a generic NeMo ASR model to NNEF format using TractNNEF.

    Args:
        asr_model: The NeMo ASR model to export.
        inference_target: The inference target configuration for export.
        export_dir: Directory where the exported NNEF files will be saved.
        skip_preprocessor: If True, skip exporting the preprocessor subnet.
        split_joint_decoder: Whether to split the joint&decoder subnets export.
        compress_registry: Compression registry for the exported NNEF subnets.
        compress_method: Compression method for the exported NNEF subnets.
            if None, no compression is applied.
        extra_cfg: Additional configuration to save alongside the model.
        omegaconf: Injected OmegaConf module.
        kwargs: Additional keyword arguments to pass to the export function.
    """
    with (export_dir / "model_config.json").open("w", encoding="utf8") as fh:
        cfg = omegaconf.OmegaConf.to_container(asr_model.cfg)
        if extra_cfg is not None:
            cfg.update(extra_cfg)
        json.dump(cfg, fh, indent=2)
    if compress_method:
        LOGGER.info("use compresssion: %s", compress_method)
        registry = dynamic_load_registry(compress_registry)
        asr_model = registry[compress_method](
            asr_model,
            export_dirpath=export_dir,
        )
        LOGGER.info("successfully applied compression: %s", compress_method)

    for export_params in iter_export_params_for_generic_nemo_asr_model(
        asr_model,
        inference_target,
        skip_preprocessor=skip_preprocessor,
        split_joint_decoder=split_joint_decoder,
    ):
        LOGGER.info("start subnet export: %s", export_params.name)
        export_model_to_nnef(
            model=export_params.model,
            args=export_params.test_input,
            inference_target=export_params.inference_target.with_specific_properties(
                export_params.specific_tract_properties
            ),
            input_names=export_params.input_names,
            output_names=export_params.output_names,
            file_path_export=export_dir / f"{export_params.name}.nnef.tgz",
            custom_extensions=export_params.custom_extensions,
            allow_same_io_names=export_params.allow_same_io_names,
            **kwargs,
        )
        LOGGER.info("exported subnet: %s with success", export_params.name)


def parser_cli():
    parser = argparse.ArgumentParser(
        description="Export NeMo ASR model to NNEF format using TractNNEF."
    )
    parser.add_argument(
        "-s",
        "--model-slug",
        type=str,
        required=True,
        help="The model slug for the NeMo ASR model to export.",
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
        "-n",
        "--naming-scheme",
        default=VariableNamingScheme.NATURAL_VERBOSE_CAMEL.value,
        choices=[vns.value for vns in VariableNamingScheme],
        help="display debug information",
    )
    parser.add_argument(
        "--tract-specific-path",
        required=False,
        help="tract specific path (instead of latest version)",
    )
    parser.add_argument(
        "--tract-specific-version",
        required=False,
        help="tract specific version",
    )
    parser.add_argument(
        "-tt",
        "--tract-check-io-tolerance",
        default=TractCheckTolerance.APPROXIMATE.value,
        choices=[t.value for t in TractCheckTolerance],
        help="tract check io tolerance level",
    )

    parser.add_argument(
        "--compress-registry",
        type=str,
        default=DEFAULT_COMPRESSION_REGISTRY,
        help="compression registry for the exported nnef subnets",
    )

    parser.add_argument(
        "--compress-method",
        type=str,
        default=None,
        help="compression method for the exported nnef subnets",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="display debug information",
    )

    return parser.parse_args()


def setup_inference_target_from_cli_args(args) -> TractNNEF:
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
    inference_target.check_io_tolerance = args.tract_check_io_tolerance
    return inference_target


@require_extra_decorator(
    extra=T2NExtra.NEMO_TRACT, module="nemo.collections.asr", kw="nemo_asr"
)
def main(*, nemo_asr: InjectedNemoModule = INJECTED):
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
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=args.model_slug
    )
    if isinstance(args.tract_check_io_tolerance, str):
        args.tract_check_io_tolerance = TractCheckTolerance(
            args.tract_check_io_tolerance
        )

    inference_target = setup_inference_target_from_cli_args(args)
    export_nemo_asr_model(
        asr_model,
        inference_target,
        export_dir,
        nnef_variable_naming_scheme=VariableNamingScheme(args.naming_scheme),
        compress_registry=args.compress_registry,
        compress_method=args.compress_method,
        skip_preprocessor=args.skip_preprocessor,
        split_joint_decoder=args.split_joint_decoder,
        extra_cfg={"pretrained_name": args.model_slug},
    )


if __name__ == "__main__":
    main()
