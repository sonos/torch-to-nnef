import json
import logging
import typing as T
from collections import OrderedDict
from contextlib import contextmanager, suppress
from pathlib import Path

import torch

from torch_to_nnef._optional_types import (
    InjectedLightningModule,
    InjectedNemoModule,
    InjectedOmegaConfModule,
)
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import (
    build_io,
)
from torch_to_nnef.utils import INJECTED, T2NExtra, require_extra_decorator

from .wrappers import (
    CollapseBatchDimWrapper,
    DecoderWithoutTargetLength,
    WrapAudioPreprocessor,
    decoder_fix_input_example_batch_size,
)

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
            imod = model
            if hasattr(imod, "input_module"):
                imod = model.input_module
            if input_example is None:
                if float_dtype is None:
                    try:
                        fdtype = next(imod.parameters()).dtype
                    except StopIteration:
                        fdtype = torch.float32
                else:
                    fdtype = float_dtype
                LOGGER.debug("Generating dummy input... %s", float_dtype)
                input_example = imod.input_example(max_batch=batch_size)
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
                if hasattr(ex, "_prepare_for_export"):
                    ex._prepare_for_export(**my_args, noreplace=True)

            if hasattr(model, "_prepare_for_export"):
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
        if hasattr(model, "_export_teardown"):
            model._export_teardown()


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="nemo")
def iter_nemo_model_subnets(
    model,
    input_example=None,
    float_dtype: T.Optional[torch.dtype] = None,
    split_joint_decoder: bool = False,
    remove_unused_inputs: bool = True,
    apply_sequential_examples: bool = False,
    batch_size: int = 3,
    *,
    nemo: InjectedNemoModule = INJECTED,
):
    """Iterator over exportable subnets of a nemo model."""
    subnet_names = model.list_export_subnets()
    allow_same_io_names = [False] * len(subnet_names)
    nemo_model_mod = nemo.collections.asr.models
    if isinstance(
        model, nemo_model_mod.classification_models.EncDecClassificationModel
    ):
        subnet_names = ["encoder", "decoder"]
        allow_same_io_names = [True, False]

        # Get the class you want to patch
        cls = model.encoder.__class__

        # Capture original property getter BEFORE patching
        orig_fget = cls.output_types.fget  # <-- important

        def patched_output_types(self):
            original = orig_fget(self)  # call original getter

            new = OrderedDict()
            for k, v in original.items():
                if k == "encoded_lengths":
                    new["length"] = v
                else:
                    new[k] = v
            return new

        # Patch the right attribute name
        cls.output_types = property(patched_output_types)

    # Handle CTC families (EncDecCTCModel and EncDecCTCModelBPE)
    EncDecCTCModel = None
    EncDecCTCModelBPE = None
    # Newer NeMo often exposes ctc_models submodule
    try:
        EncDecCTCModel = getattr(
            nemo_model_mod.ctc_models, "EncDecCTCModel", None
        )
        EncDecCTCModelBPE = getattr(
            nemo_model_mod.ctc_models, "EncDecCTCModelBPE", None
        )
    except AttributeError:  # pragma: no cover - defensive
        pass
    # Fallback: sometimes classes are directly under models
    if EncDecCTCModel is None:
        EncDecCTCModel = getattr(nemo_model_mod, "EncDecCTCModel", None)
    if EncDecCTCModelBPE is None:
        EncDecCTCModelBPE = getattr(nemo_model_mod, "EncDecCTCModelBPE", None)

    if (
        (EncDecCTCModel is not None and isinstance(model, EncDecCTCModel))
        or (
            EncDecCTCModelBPE is not None
            and isinstance(model, EncDecCTCModelBPE)
        )
    ):
        subnet_names = ["encoder", "decoder"]
        allow_same_io_names = [True, False]

        # Patch encoder output_types to map encoded_lengths -> length
        cls = model.encoder.__class__
        try:
            orig_fget = cls.output_types.fget  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - defensive
            orig_fget = None

        def patched_output_types_ctc(self):
            original = (
                orig_fget(self)
                if orig_fget is not None
                else getattr(self, "output_types", {})
            )
            try:
                items = original.items()
            except Exception:  # pragma: no cover - defensive
                return original
            new = OrderedDict()
            for k, v in items:
                new["length" if k == "encoded_lengths" else k] = v
            return new

        with suppress(Exception):  # pragma: no cover - defensive
            cls.output_types = property(patched_output_types_ctc)  # type: ignore[attr-defined]

    for subnet_name, sio in zip(subnet_names, allow_same_io_names):
        subnet = model.get_export_subnet(subnet_name)
        if subnet_name == "decoder_joint":
            input_example = None  # reset input example for joint
            # because need more parameters than encoder output only
        with exportable_nemo_net(
            subnet_name,
            subnet,
            input_example,
            batch_size=batch_size,
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
                    decoder = subnet.decoder
                    if remove_unused_inputs:
                        decoder = DecoderWithoutTargetLength(decoder)
                    yield (
                        "decoder",
                        decoder,
                        decoder_fix_input_example_batch_size(
                            decoder.input_example(max_batch=batch_size),
                            batch_size=batch_size,
                        ),
                        decoder.dynamic_shapes_for_export(False),
                        sio,
                    )
                    yield (
                        "joint",
                        subnet.joint,
                        subnet.joint.input_example(max_batch=batch_size),
                        subnet.joint.dynamic_shapes_for_export(False),
                        sio,
                    )
                    continue
                if remove_unused_inputs:
                    subnet = DecoderWithoutTargetLength(subnet)
                    input_example = subnet.filter_original_input_example(
                        input_example
                    )
                input_example = decoder_fix_input_example_batch_size(
                    input_example,
                    batch_size=batch_size,
                )
            if len(input_example) > len(subnet.input_names):
                # if < that means some inputs are optional
                raise RuntimeError(
                    "declared input names:",
                    subnet.input_names,
                    f"but expected {len(input_example)} inputs",
                )
            yield subnet_name, subnet, input_example, dynamic_axes, sio
            # Propagate input example
            # (default scenario, may need to be overriden)
            if input_example is not None and apply_sequential_examples:
                input_example = out_example
            else:
                input_example = None


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
            elif iname == "encoder_output":
                # example: B,512,(S+7)/8,F32
                symbols = "BHR"  # Batch, High End Features, ReducedStream
            else:
                raise NotImplementedError(
                    f"cannot guess dynamic axis symbols for input '{iname}'"
                )
            build_partial_dynamic_axes(iname, symbols)
    return dynamic_axes, custom_extensions


def _collapse_axes_for_input(
    orig_axes: T.Dict[int, str],
    *,
    full_axes_spec: T.Optional[T.Sequence[str]] = None,
    assume_batch_at0: bool = True,
) -> T.Dict[int, str]:
    """Collapse a single input axes mapping by removing 'B' and reindexing."""
    # Determine batch positions
    if full_axes_spec is not None:
        b_positions = [
            ix for ix, sym in enumerate(full_axes_spec) if sym == "B"
        ]
    else:
        pairs_for_b = sorted(orig_axes.items(), key=lambda kv: kv[0])
        b_positions = [ix for ix, sym in pairs_for_b if sym == "B"]
        if assume_batch_at0 and 0 not in b_positions:
            b_positions = [0] + b_positions

    # Build a full axes symbol list to reason about final rank
    # If full_axes_spec wasn't provided, derive a minimal plausible one.
    if full_axes_spec is None:
        max_idx = max(orig_axes) if orig_axes else -1
        full_axes = [None] * (max_idx + 1)
        for i, s in orig_axes.items():
            full_axes[i] = s
        # Fill any unknowns with a generic symbol ('?'), treat as non-B
        full_axes_spec = [s if s is not None else "?" for s in full_axes]

    # Create a mapping old_index->new_index, removing B-dims
    new_map: T.Dict[int, int] = {}
    new_i = 0
    for i, sym in enumerate(full_axes_spec):
        if sym == "B":
            continue
        new_map[i] = new_i
        new_i += 1

    # Translate the original dynamic axes into the collapsed indexing
    collapsed: T.Dict[int, str] = {}
    for i, sym in orig_axes.items():
        if full_axes_spec[i] == "B":
            # drop this dimension entirely
            continue
        collapsed[new_map[i]] = sym
    return collapsed


def collapse_dynamic_axes_mapping(
    nemo_dynamic_axes: T.Dict[str, T.Dict[int, str]],
    input_names: T.Sequence[str],
) -> T.Dict[str, T.Dict[int, str]]:
    """Collapse mapping for all inputs keeping only the exposed inputs."""
    # Heuristic: when batch is hidden, the external interface has no B;
    # keep only entries for exposed input names.
    full_axes_by_name: T.Dict[str, T.Sequence[str]] = {}
    for name, axes in nemo_dynamic_axes.items():
        # Derive a plausible full axes spec from dynamic mapping
        max_idx = max(axes) if axes else -1
        spec = ["?"] * (max_idx + 1)
        for i, s in axes.items():
            spec[i] = s
        full_axes_by_name[name] = tuple(spec)

    collapsed: T.Dict[str, T.Dict[int, str]] = {}
    for name in input_names:
        axes = nemo_dynamic_axes.get(name)
        if not axes:
            continue
        collapsed[name] = _collapse_axes_for_input(
            axes,
            full_axes_spec=(full_axes_by_name or {}).get(name),
        )
    return collapsed


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


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="nemo")
def iter_export_params_for_generic_nemo_asr_model(
    asr_model,
    inference_target,
    skip_preprocessor: bool = False,
    split_joint_decoder: bool = False,
    remove_unused_inputs: bool = True,
    collapse_batch_dim: bool = False,
    float_dtype: T.Optional[torch.dtype] = None,
    *,
    nemo: InjectedNemoModule = INJECTED,
) -> T.Iterator[ExportParameters]:
    """Iterator over export parameters for a generic NeMo ASR model."""
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

        if isinstance(
            asr_model.preprocessor,
            nemo.collections.asr.modules.audio_preprocessing.AudioPreprocessor,
        ):
            asr_model.preprocessor = WrapAudioPreprocessor(
                asr_model.preprocessor
            )

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
            model = asr_model.preprocessor
            input_names = model.input_names[: len(input_example)]
            output_names = model.output_names
            test_input = inps or input_example
            dyn = dynamic_axes
            if collapse_batch_dim:
                # Wrap and collapse axes
                model = CollapseBatchDimWrapper(model, dynamic_axes)
                input_names = model.input_names
                output_names = model.output_names
                test_input = model.input_example()
                dyn = collapse_dynamic_axes_mapping(dynamic_axes, input_names)

            yield ExportParameters(
                name=subnet_name,
                model=model,
                test_input=test_input,
                inference_target=inference_target.with_dynamic_axes(dyn),
                input_names=input_names,
                output_names=output_names,
                custom_extensions=list(custom_extensions),
                allow_same_io_names=False,  # not used for preprocessor export
                specific_tract_properties=build_custom_subnet_tract_properties(
                    subnet_name, model
                ),
            )

    for (
        subnet_name,
        subnet,
        input_example,
        nemo_dynamic_axes,
        allow_same_io_names,
    ) in iter_nemo_model_subnets(
        asr_model,
        float_dtype=float_dtype,
        split_joint_decoder=split_joint_decoder,
        remove_unused_inputs=remove_unused_inputs,
    ):
        dynamic_axes, custom_extensions = build_dynamic_axes(
            subnet, nemo_dynamic_axes
        )

        model = subnet
        test_input = input_example
        input_names = subnet.input_names[: len(input_example)]
        output_names = subnet.output_names
        # Limit dynamic axes to the inputs we are actually exposing
        dyn = {k: v for k, v in dynamic_axes.items() if k in input_names}

        if collapse_batch_dim:
            model = CollapseBatchDimWrapper(subnet, dynamic_axes)
            test_input = model.input_example()
            input_names = model.input_names
            output_names = model.output_names
            # Use wrapper's collapsed dynamic mapping for correctness
            dyn = model.dynamic_shapes_for_export()

        yield ExportParameters(
            name=subnet_name,
            model=model,
            test_input=test_input,
            inference_target=inference_target.with_dynamic_axes(dyn),
            input_names=input_names,
            output_names=output_names,
            custom_extensions=list(custom_extensions),
            allow_same_io_names=allow_same_io_names,
            specific_tract_properties=build_custom_subnet_tract_properties(
                subnet_name, model
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
    collapse_batch_dim: bool = False,
    *,
    omegaconf: InjectedOmegaConfModule = INJECTED,
    **kwargs,
):
    """Export a generic NeMo ASR model to NNEF format using TractNNEF."""
    with (export_dir / "model_config.json").open("w", encoding="utf8") as fh:
        cfg = omegaconf.OmegaConf.to_container(asr_model.cfg)
        if extra_cfg is not None:
            cfg.update(extra_cfg)
        json.dump(cfg, fh, indent=2)
    if compress_method:
        from torch_to_nnef.compress import dynamic_load_registry

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
        collapse_batch_dim=collapse_batch_dim,
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
