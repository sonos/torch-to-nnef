"""Support for NVIDIA NeMo models export to NNEF (with TractNNEF focus).

Provide utilities to export NeMo models, particularly ASR models,
to the NNEF format using TractNNEF.
Includes functions to handle model subnets, dynamic axes, and
custom extensions required for the export process.

"""

import argparse
import datetime
import json
import logging
import sys
import typing as T
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from torch_to_nnef._optional_types import (
    HuggingFaceHubModule,
    InjectedHuggingFaceHubModule,
    InjectedLightningModule,
    InjectedNemoModule,
    InjectedOmegaConfModule,
    InjectedQuestionaryModule,
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
    build_io,
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
    batch_size: int = 1,
    float_dtype: T.Optional[torch.dtype] = None,
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
                if float_dtype is None:
                    try:
                        fdtype = next(model.input_module.parameters()).dtype
                    except StopIteration:
                        fdtype = torch.float32
                else:
                    fdtype = float_dtype
                LOGGER.debug("Generating dummy input... %s", float_dtype)
                input_example = model.input_example(max_batch=batch_size)
                # Cast to correct dtype (usualy float16 if not float16)
                if fdtype != torch.float32:
                    input_example = [
                        ie.to(fdtype)
                        if isinstance(ie, torch.Tensor)
                        and ie.dtype == torch.float32
                        else ie
                        for ie in input_example
                    ]

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


def iter_nemo_model_subnets(
    model,
    input_example=None,
    float_dtype: T.Optional[torch.dtype] = None,
    split_joint_decoder: bool = False,
    remove_unused_inputs: bool = True,
    apply_sequential_examples: bool = False,
):
    """Iterator over exportable subnets of a nemo model.

    Args:
        model: NeMo model to iterate over.
        input_example: Optional input example to use for export.
        float_dtype: Optional float dtype to use for export.
        split_joint_decoder: To split joint decoder subnets (if encountered).
        remove_unused_inputs: To remove unused inputs from subnet exports.
        apply_sequential_examples: If True, use sequential input examples
            for each subnet.

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
        with exportable_nemo_net(
            subnet_name,
            subnet,
            input_example,
            # NOTE: Investigate
            # set this to 3 it highlight issue wih batch dim
            # being wrongly concretized in 'encoder'
            batch_size=1,
            float_dtype=float_dtype,
        ) as (
            #  pylint: disable-next=redefined-argument-from-local
            input_example,
            out_example,
            dynamic_axes,
        ):
            if subnet_name == "decoder_joint":
                if split_joint_decoder:
                    # split into decoder and joint
                    # inputs are force generated by nemo .input_example()
                    decoder = DecoderWithoutTargetLength(subnet.decoder)
                    yield (
                        "decoder",
                        decoder,
                        decoder.input_example(),
                        decoder.dynamic_shapes_for_export(False),
                    )
                    yield (
                        "joint",
                        subnet.joint,
                        subnet.joint.input_example(),
                        subnet.joint.dynamic_shapes_for_export(False),
                    )
                    continue
                subnet = DecoderWithoutTargetLength(subnet)
                input_example = subnet.filter_original_input_example(
                    input_example
                )
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
            if input_example is not None and apply_sequential_examples:
                input_example = out_example
            else:
                input_example = None


def build_dynamic_axes(subnet, nemo_dynamic_axes):  # noqa: MC0001
    """Build dynamic axes mapping and custom extensions for nemo subnet.

    Args:
        subnet: nemo subnet module
        nemo_dynamic_axes: dynamic axes info from nemo export
    Returns:
        dynamic_axes: dynamic axes mapping for torch_to_nnef
        custom_extensions: custom extensions for torch_to_nnef

    Note:
        this code will not scale well and should be refactored when more
        nemo models are supported.
    """
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
    remove_unused_inputs: bool = True,
    float_dtype: T.Optional[torch.dtype] = None,
) -> T.Iterator[ExportParameters]:
    """Iterator over export parameters for a generic NeMo ASR model.

    Args:
        asr_model: The NeMo ASR model to export.
        inference_target: The target inference type.
        skip_preprocessor: Whether to skip exporting the preprocessor subnet.
        split_joint_decoder:
            Whether to split the joint and decoder subnets exported.
        remove_unused_inputs:
            Whether to remove unused inputs from the exported model.
        float_dtype: Optional float dtype to use for export.

    Yields:
        ExportParameters for each subnet of the ASR model, with the preprocessor
    """
    asr_model.eval()

    if not skip_preprocessor:
        inps = asr_model.preprocessor.input_example()
        if hasattr(asr_model.preprocessor, "featurizer") and hasattr(
            asr_model.preprocessor.featurizer, "dither"
        ):
            # disable dither for export
            if asr_model.preprocessor.featurizer.dither != 0.0:
                LOGGER.info("disabling dither for preprocessor export")
            asr_model.preprocessor.featurizer.dither = 0.0
        if hasattr(asr_model.preprocessor, "featurizer") and hasattr(
            asr_model.preprocessor.featurizer, "pad_to"
        ):
            if asr_model.preprocessor.featurizer.pad_to != 0.0:
                LOGGER.info("disabling pad_to for preprocessor export")
            asr_model.preprocessor.featurizer.pad_to = 0

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
    ) in iter_nemo_model_subnets(
        asr_model,
        float_dtype=float_dtype,
        split_joint_decoder=split_joint_decoder,
        remove_unused_inputs=remove_unused_inputs,
    ):
        dynamic_axes, custom_extensions = build_dynamic_axes(
            subnet, nemo_dynamic_axes
        )
        yield ExportParameters(
            name=subnet_name,
            model=subnet,
            test_input=input_example,
            inference_target=inference_target.with_dynamic_axes(dynamic_axes),
            input_names=subnet.input_names[: len(input_example)],
            output_names=subnet.output_names,
            custom_extensions=list(custom_extensions),
            allow_same_io_names=False,
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
    float_dtype: T.Optional[torch.dtype] = None,
    remove_unused_inputs: bool = True,
    dump_checked_io: bool = False,
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
        float_dtype: Optional float dtype to use for export.
        remove_unused_inputs: To remove unused inputs in the exported model.
            This happen for decoder subnets that do not use target_length.
        dump_checked_io: Whether to dump checked input/output examples.
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
        float_dtype=float_dtype,
        remove_unused_inputs=remove_unused_inputs,
    ):
        LOGGER.info("start subnet export: %s", export_params.name)
        if dump_checked_io:
            test_dir = export_dir / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            build_io(
                export_params.model,
                export_params.test_input,
                io_npz_path=test_dir / f"{export_params.name}_checked_io.npz",
                input_names=export_params.input_names,
                output_names=export_params.output_names,
            )
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
    """Build the CLI parser for NeMo ASR model export to NNEF format."""
    parser = argparse.ArgumentParser(
        description="Export NeMo ASR model to NNEF format using TractNNEF."
    )
    parser.add_argument(
        "-s",
        "--model-slug",
        type=str,
        default="*",
        help="The model slug for the NeMo ASR model to export."
        "if you don't know just live it blank (select box will be proposed)",
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
        help="Forcing sdpa to use PyTorch implementation."
        " (likely more efficent, once stable tract side)",
    )
    parser.add_argument(
        "-dt",
        "--data-type",
        type=str,
        choices=["float32", "float16", "mixed"],
        help="Data of most weights for export (experimental).",
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
        choices=[t.value for t in TractCheckTolerance] + ["skip"],
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
        "--dump-checked-io",
        required=False,
        default=False,
        action="store_true",
        help="dump tested io to the given path for checking purpose",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="display debug information",
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
    return inference_target


class WrapPreprocessorCast(torch.nn.Module):
    """Wraps the preprocessor to add a cast to float32 at the output."""

    def __init__(self, preprocessor: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.preprocessor = preprocessor
        self.dtype = dtype

    def input_example(self):
        return self.preprocessor.input_example()

    def _export_teardown(self):
        self.preprocessor._export_teardown()

    def _prepare_for_export(self, *args, **kwargs):
        self.preprocessor._prepare_for_export(*args, **kwargs)

    def dynamic_shapes_for_export(self, *args, **kwargs):
        return self.preprocessor.dynamic_shapes_for_export(*args, **kwargs)

    @property
    def input_names(self):
        return self.preprocessor.input_names

    @property
    def output_names(self):
        return self.preprocessor.output_names

    def forward(self, *args, **kwargs):
        x = self.preprocessor(*args, **kwargs)
        return tuple([x[0].to(self.dtype)] + list(x)[1:])


class DecoderWithoutTargetLength(torch.nn.Module):
    """Wraps the decoder or joint+decoder for export.

    This remove the parameters 'target_length' that are
    not needed during inference...

    Enabled classes:
        `nemo.collections.asr.modules.rnnt.RNNTDecoderJoint`
        `nemo.collections.asr.modules.rnnt.RNNTDecoder`

    Alter forward by auto adding the target_length parameter
    based on the shape of the input tensors (Batch size).
    as an array of shape (batch_size, 1) full of ones.
    Then remove it from the output (this is the 2nd argument).
    This is only applied for enabled classes.

    This should lead at export to a complete removal
    of the unused target_length.

    """

    FILTER_ARGUMENT = "target_length"
    FILTER_OUTPUT = "prednet_lengths"

    @require_extra_decorator(
        extra=T2NExtra.NEMO_TRACT, module="nemo.collections.asr", kw="nemo_asr"
    )
    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        nemo_asr: InjectedNemoModule = INJECTED,
    ):
        super().__init__()
        self.decoder = decoder
        self.active_fitering = isinstance(
            decoder,
            (
                nemo_asr.modules.rnnt.RNNTDecoderJoint,
                nemo_asr.modules.rnnt.RNNTDecoder,
            ),
        )

    def _infer_batch_size(self, args, kwargs):
        """Infer batch size from the first Tensor found.

        This avoids relying on positional conventions.
        """
        for v in args:
            if torch.is_tensor(v):
                return v.shape[0], v
        for v in kwargs.values():
            if torch.is_tensor(v):
                return v.shape[0], v
        raise RuntimeError("Cannot infer batch size: no Tensor inputs found")

    @property
    def input_names(self):
        if self.active_fitering:
            return [
                name
                for name in self.decoder.input_names
                if name != self.FILTER_ARGUMENT
            ]
        return self.decoder.input_names

    @property
    def output_names(self):
        def rename_state(name: str) -> str:
            if name == "states":
                return "out_states"
            return name

        if self.active_fitering:
            return [
                rename_state(_)
                for _ in self.decoder.output_names
                if _ != self.FILTER_OUTPUT
            ]
        return self.decoder.output_names

    @property
    def index_arg_to_remove(self) -> int:
        if self.active_fitering:
            for idx, name in enumerate(self.decoder.input_names):
                if name == self.FILTER_ARGUMENT:
                    return idx
        raise RuntimeError(
            f"Cannot find argument named {self.FILTER_ARGUMENT} to remove"
        )

    @property
    def index_output_to_remove(self) -> int:
        if self.active_fitering:
            for idx, name in enumerate(self.decoder.output_names):
                if name == self.FILTER_OUTPUT:
                    return idx
        raise RuntimeError(
            f"Cannot find output named {self.FILTER_OUTPUT} to remove"
        )

    def input_example(self):
        if not self.active_fitering:
            return self.decoder.input_example()
        return self.filter_original_input_example(self.decoder.input_example())

    def filter_original_input_example(
        self, inputs: T.List[torch.Tensor]
    ) -> T.List[torch.Tensor]:
        """Filter out target_length from inputs."""
        filtered_inputs = []
        for name, tensor in zip(self.decoder.input_names, inputs):
            if name != self.FILTER_ARGUMENT:
                filtered_inputs.append(tensor)
        return filtered_inputs

    def forward(self, *args, **kwargs):
        if not self.active_fitering:
            return self.decoder(*args, **kwargs)

        assert self.FILTER_ARGUMENT not in kwargs
        batch_size, ref_tensor = self._infer_batch_size(args, kwargs)

        target_length = torch.ones(
            (batch_size, 1),
            device=ref_tensor.device,
            dtype=ref_tensor.dtype,
        )
        to_rm_in_idx = self.index_arg_to_remove
        if len(args) > to_rm_in_idx:
            args = list(args)
            args.insert(to_rm_in_idx, target_length)
            args = tuple(args)
        else:
            kwargs = dict(kwargs)
            kwargs[self.FILTER_ARGUMENT] = target_length

        outs = self.decoder(*args, **kwargs)

        # If decoder returns multiple outputs:
        # (logits, target_length, *states)
        # Drop ONLY the target_length (2nd output)
        to_rm_out_idx = self.index_output_to_remove
        return tuple(
            list(outs[:to_rm_out_idx]) + list(outs[to_rm_out_idx + 1 :])
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder, name)


def use_pytorch_sdpa(model: torch.nn.Module):
    """Modify the model to use PyTorch sdpa implementations where applicable.

    This leverage attention modules set in NeMo with
    specific use_pytorch_sdpa flag.
    """
    # pylint: disable=import-outside-toplevel
    from nemo.collections.asr.parts.submodules.multi_head_attention import (
        MultiHeadAttention,
    )

    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            module.use_pytorch_sdpa = True


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="questionary")
def ask_model_selector(
    pretrained_model_info_list,
    *,
    questionary: InjectedQuestionaryModule = INJECTED,
    **kwargs,
):
    # create the question object
    question = questionary.select(
        " ∵ What model do you want to export ? "
        "(those starting with nvidia/ are from 🤗Hub)",
        qmark="",
        choices=[
            questionary.Choice(
                title=pretrained_model_info.pretrained_model_name,
                description=pretrained_model_info.description,
            )
            for pretrained_model_info in pretrained_model_info_list
        ],
        use_jk_keys=False,
        use_search_filter=True,
        **kwargs,
    )

    # prompt the user for an answer
    return question.ask()


def nemo_asr_hg_list(huggingface_hub: HuggingFaceHubModule):
    """Return the list of available NeMo ASR models from HuggingFace."""
    hugging_face_hub_model_list = []

    # Query HF for NVIDIA ASR models that declare NeMo compatibility
    models = huggingface_hub.list_models(
        author="nvidia",
        task="automatic-speech-recognition",
        library="nemo",
        full=True,
    )

    for m in models:
        # Skip archived / gated / unusable entries early
        if getattr(m, "gated", False):
            continue

        tags = set(m.tags or [])

        # Heuristic: ensure this is an ASR *model*,
        # not speech translation / TTS / etc.
        if "automatic-speech-recognition" not in tags:
            continue

        jtags = "tags:" + ", ".join(sorted(tags))

        def last_modified_str(dt: datetime.datetime) -> str:
            fmt = "%Y-%m-%d"
            return f"last modified: {dt.strftime(fmt)}"

        hugging_face_hub_model_list.append(
            HGModelInfo(
                pretrained_model_name=m.modelId,
                organization="nvidia",
                description="HuggingFace (compatibility only guessed): "
                + (m.cardData.get("description") if m.cardData else jtags)
                + last_modified_str(m.lastModified),
                tags=sorted(tags),
                pipeline_tag=m.pipeline_tag,
                library_name=m.library_name,
                sha=m.sha,
                last_modified=m.lastModified,
            )
        )

    # Sort to match NeMo-style deterministic output
    hugging_face_hub_model_list.sort(
        key=lambda x: x.last_modified, reverse=True
    )

    return hugging_face_hub_model_list


@dataclass(frozen=True)
class HGModelInfo:
    pretrained_model_name: str
    organization: str
    description: str
    tags: T.List[str]
    pipeline_tag: str
    library_name: str
    sha: str
    last_modified: datetime.datetime


@require_extra_decorator(
    extra=T2NExtra.NEMO_TRACT, module="nemo.collections.asr", kw="nemo_asr"
)
@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="huggingface_hub")
def load_asr_model_from_nemo_slug(
    model_slug: str,
    *,
    nemo_asr: InjectedNemoModule = INJECTED,
    huggingface_hub: InjectedHuggingFaceHubModule = INJECTED,
):
    """Load a NeMo ASR model from a given model slug."""
    # pylint: disable=import-outside-toplevel
    from huggingface_hub import errors

    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_slug, map_location=torch.device("cpu")
        )
    except (errors.RepositoryNotFoundError, FileNotFoundError):
        LOGGER.error("Could not find model with slug: %s", model_slug)
        if model_slug != "*":
            while True:
                resp = (
                    input("Do you want to list available models? (y/n): ")
                    .strip()
                    .lower()
                )
                if resp in ("y", "n"):
                    break
            if resp == "n":
                LOGGER.info("User chose not to list available models. Exiting.")
                sys.exit(1)

        available_models = (
            nemo_asr_hg_list(huggingface_hub)
            + nemo_asr.models.ASRModel.list_available_models()
        )
        model_slug = ask_model_selector(available_models)
        LOGGER.info("selected model slug: %s", model_slug)
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_slug,
            map_location=torch.device("cpu"),
        )
    return asr_model


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


if __name__ == "__main__":
    main()
