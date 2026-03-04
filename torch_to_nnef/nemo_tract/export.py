import json
import logging
import typing as T
from collections import OrderedDict
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import torch

from torch_to_nnef._optional_types import (
    InjectedLightningModule,
    InjectedNemoModule,
    InjectedOmegaConfModule,
)
from torch_to_nnef.compress import dynamic_load_registry
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import (
    build_io,
)
from torch_to_nnef.nemo_tract.axes import collapse_dynamic_axes_mapping
from torch_to_nnef.nemo_tract.dynaxes import (
    build_dynamic_axes as build_dynamic_axes_for_subnet,
)
from torch_to_nnef.nemo_tract.wrappers import (
    CollapseBatchDimWrapper,
    DecoderWithoutTargetLength,
    WrapAudioPreprocessor,
    decoder_fix_input_example_batch_size,
)
from torch_to_nnef.utils import INJECTED, T2NExtra, require_extra_decorator

LOGGER = logging.getLogger(__name__)


def _patch_encoder_output_types(
    cls, *, from_key: str = "encoded_lengths", to_key: str = "length"
):
    """Patch encoder.output_types to remap a key.

    (e.g., encoded_lengths -> length).

    Resilient to cases where output_types is not a property;
    falls back gracefully.
    """
    try:
        orig_fget = cls.output_types.fget  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - defensive
        orig_fget = None

    def patched_output_types(self):
        original = (
            orig_fget(self)
            if orig_fget is not None
            else getattr(self, "output_types", {})
        )
        try:
            items = original.items()
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            return original
        new = OrderedDict()
        for k, v in items:
            new[to_key if k == from_key else k] = v
        return new

    with suppress(AttributeError, TypeError):
        cls.output_types = property(patched_output_types)  # type: ignore[attr-defined]


def _resolve_ctc_model_classes(nemo_models_mod):
    """Resolve CTC model classes across NeMo layouts."""
    cls_enc_dec_ctc_model = None
    cls_enc_dec_ctc_model_bpe = None
    try:
        cls_enc_dec_ctc_model = getattr(
            nemo_models_mod.ctc_models, "EncDecCTCModel", None
        )
        cls_enc_dec_ctc_model_bpe = getattr(
            nemo_models_mod.ctc_models, "EncDecCTCModelBPE", None
        )
    except AttributeError:  # pragma: no cover - defensive
        pass
    if cls_enc_dec_ctc_model is None:
        cls_enc_dec_ctc_model = getattr(nemo_models_mod, "EncDecCTCModel", None)
    if cls_enc_dec_ctc_model_bpe is None:
        cls_enc_dec_ctc_model_bpe = getattr(
            nemo_models_mod, "EncDecCTCModelBPE", None
        )
    return cls_enc_dec_ctc_model, cls_enc_dec_ctc_model_bpe


def _pick_for_classification(model, nemo_models_mod):
    """Specialize EncDecClassificationModel & patch encoder outputs."""
    cls_enc_dec_cls = (
        nemo_models_mod.classification_models.EncDecClassificationModel
    )
    if isinstance(model, cls_enc_dec_cls):
        subnet_names = ["encoder", "decoder"]
        allow_same_io_names = [True, False]
        _patch_encoder_output_types(model.encoder.__class__)
        return subnet_names, allow_same_io_names
    return None


def _pick_for_ctc(model, nemo_models_mod):
    """Specialize subnets for CTC families and patch encoder outputs."""
    cls_ctc, cls_ctc_bpe = _resolve_ctc_model_classes(nemo_models_mod)
    if (cls_ctc is not None and isinstance(model, cls_ctc)) or (
        cls_ctc_bpe is not None and isinstance(model, cls_ctc_bpe)
    ):
        subnet_names = ["encoder", "decoder"]
        allow_same_io_names = [True, False]
        _patch_encoder_output_types(model.encoder.__class__)
        return subnet_names, allow_same_io_names
    return None


def _disable_training(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="nemo")
def _collect_exportables(model, *, nemo: InjectedNemoModule = INJECTED):
    exportable_class = nemo.core.classes.exportable.Exportable
    exportables = []
    for m in model.modules():
        if isinstance(m, exportable_class):
            exportables.append(m)
    return exportables


def _get_target_float_dtype(
    imod, float_dtype: T.Optional[torch.dtype] = None
) -> torch.dtype:
    if float_dtype is None:
        try:
            fdtype = next(imod.parameters()).dtype
        except StopIteration:
            fdtype = torch.float32
    else:
        fdtype = float_dtype
    return fdtype


def _maybe_cast_float_inputs(
    input_example: T.List[torch.Tensor], fdtype: torch.dtype
) -> T.List[torch.Tensor]:
    if fdtype != torch.float32:
        input_example = [
            ie.to(fdtype)
            if isinstance(ie, torch.Tensor) and ie.dtype == torch.float32
            else ie
            for ie in input_example
        ]
    return input_example


def _prepare_input_example_for_export(
    model: torch.nn.Module,
    input_example: T.Optional[T.List[torch.tensor]],
    float_dtype: T.Optional[torch.dtype],
    batch_size: int,
):
    imod = model
    if hasattr(imod, "input_module"):
        imod = model.input_module
    if input_example is None:
        fdtype = _get_target_float_dtype(imod, float_dtype)
        LOGGER.debug("Generating dummy input... %s", fdtype)
        # Cast to correct dtype (usualy float16 if not float16)
        input_example = _maybe_cast_float_inputs(
            imod.input_example(max_batch=batch_size), fdtype
        )
    return input_example


def _prepare_for_export(
    model, exportables, output_name, input_example, my_args
):
    # Run (posibly overridden) prepare methods before calling forward()
    for ex in exportables:
        if hasattr(ex, "_prepare_for_export"):
            ex._prepare_for_export(**my_args, noreplace=True)

    if hasattr(model, "_prepare_for_export"):
        model._prepare_for_export(
            output=output_name, input_example=input_example, **my_args
        )


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="nemo")
def _build_output_example(
    model, input_example, *, nemo: InjectedNemoModule = INJECTED
):
    parse_input_example = nemo.utils.export_utils.parse_input_example
    input_list, input_dict = parse_input_example(input_example)
    output_example = model.forward(*input_list, **input_dict)
    if not isinstance(output_example, tuple):
        output_example = (output_example,)
    return output_example


@dataclass(frozen=True)
class ExportContext:
    input_example: T.List[torch.Tensor]
    output_example: T.Tuple[torch.Tensor, ...]
    dynamic_axes: T.Dict[str, T.Dict[int, str]]


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

    It prepare model by switching mode to eval,
    disabling typechecks and wrapping forward method for tracing
    by PyTorch export tools.

    Mostly borrowed from nemo codebase logic (with more modularity).
        see: nemo.core.classes.Exportable._export

    Yield:
        ExportContext with input_example, output_example and dynamic_axes
        ready for export.

    """
    typecheck = nemo.core.classes.typecheck
    wrap_forward_method = nemo.utils.export_utils.wrap_forward_method
    my_args = {"use_dynamo": use_dynamo}

    _disable_training(model)
    exportables = _collect_exportables(model)

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
            input_example = _prepare_input_example_for_export(
                model, input_example, float_dtype, batch_size
            )
            _prepare_for_export(
                model, exportables, output_name, input_example, my_args
            )
            output_example = _build_output_example(model, input_example)
            # dynamic axis is a mapping from input/output_name
            # => list of "dynamic" indices
            dynamic_axes = model.dynamic_shapes_for_export(use_dynamo)

            yield ExportContext(input_example, output_example, dynamic_axes)
    finally:
        typecheck.enable_wrapping(enabled=True)
        typecheck.set_typecheck_enabled(enabled=True)
        if forward_method:
            type(model).forward = old_forward_method
        if hasattr(model, "_export_teardown"):
            model._export_teardown()


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="nemo")
def _pick_subnets_names_and_allowed_io_names_overlap(
    model, *, nemo: InjectedNemoModule = INJECTED
):
    nemo_model_mod = nemo.collections.asr.models
    # Default from model
    subnet_names = model.list_export_subnets()
    allow_same_io_names = [False] * len(subnet_names)
    # Specialize for known families
    spec = _pick_for_classification(model, nemo_model_mod)
    if spec is not None:
        return spec
    spec = _pick_for_ctc(model, nemo_model_mod)
    if spec is not None:
        return spec
    return subnet_names, allow_same_io_names


def iter_nemo_model_subnets(
    model,
    input_example=None,
    float_dtype: T.Optional[torch.dtype] = None,
    split_joint_decoder: bool = False,
    remove_unused_inputs: bool = True,
    apply_sequential_examples: bool = False,
    batch_size: int = 3,
):
    """Iterator over exportable subnets of a nemo model."""
    subnet_names, allow_same_io_names = (
        _pick_subnets_names_and_allowed_io_names_overlap(model)
    )
    for subnet_name, sio in zip(subnet_names, allow_same_io_names):
        subnet = model.get_export_subnet(subnet_name)
        if subnet_name == "decoder_joint":
            input_example = None  # reset: joint needs more than encoder out
        with exportable_nemo_net(
            subnet_name,
            subnet,
            input_example,
            batch_size=batch_size,
            float_dtype=float_dtype,
        ) as ctx:
            if subnet_name == "decoder_joint":
                yield from iter_decoder_joint_subnets(
                    subnet,
                    ctx.input_example,
                    ctx.dynamic_axes,
                    batch_size=batch_size,
                    remove_unused_inputs=remove_unused_inputs,
                    split_joint_decoder=split_joint_decoder,
                    allow_same_io_names=sio,
                )
                continue

            input_example = ctx.input_example
            if len(input_example) > len(subnet.input_names):
                # if < that means some inputs are optional
                raise RuntimeError(
                    "declared input names:",
                    subnet.input_names,
                    f"but expected {len(input_example)} inputs",
                )
            yield subnet_name, subnet, input_example, ctx.dynamic_axes, sio
            # Propagate input example
            # (default scenario, may need to be overriden)
            if input_example is not None and apply_sequential_examples:
                input_example = ctx.output_example
            else:
                input_example = None


def build_dynamic_axes(
    subnet,
    nemo_dynamic_axes,
    input_example: T.Optional[T.Sequence[object]] = None,
):  # noqa: MC0001
    return build_dynamic_axes_for_subnet(
        subnet, nemo_dynamic_axes, input_example
    )


def iter_decoder_joint_subnets(
    subnet,
    input_example,
    ctx_dynamic_axes,
    *,
    batch_size: int,
    remove_unused_inputs: bool,
    split_joint_decoder: bool,
    allow_same_io_names: bool,
):
    """Yield export tuples for the decoder_joint case.

    - If split_joint_decoder is True: yields separate decoder and joint entries
      with their own input_examples and dynamic axes.
    - Otherwise: optionally remove unused inputs, fix batch size on the input
      example, validate arity, and yield a single decoder_joint entry using the
      context-provided dynamic axes.
    """
    if split_joint_decoder:
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
            allow_same_io_names,
        )
        yield (
            "joint",
            subnet.joint,
            subnet.joint.input_example(max_batch=batch_size),
            subnet.joint.dynamic_shapes_for_export(False),
            allow_same_io_names,
        )
        return

    # Not splitting: keep decoder_joint together
    if remove_unused_inputs:
        subnet = DecoderWithoutTargetLength(subnet)
        input_example = subnet.filter_original_input_example(input_example)
    input_example = decoder_fix_input_example_batch_size(
        input_example, batch_size=batch_size
    )

    if len(input_example) > len(subnet.input_names):
        # if < that means some inputs are optional
        raise RuntimeError(
            "declared input names:",
            subnet.input_names,
            f"but expected {len(input_example)} inputs",
        )

    yield (
        "decoder_joint",
        subnet,
        input_example,
        ctx_dynamic_axes,
        allow_same_io_names,
    )


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
def build_preprocessor_export_params(
    asr_model,
    inference_target,
    collapse_batch_dim=False,
    *,
    nemo: InjectedNemoModule = INJECTED,
) -> T.Iterator[ExportParameters]:
    """Build export parameters for the preprocessor of a NeMo ASR model."""
    inps = asr_model.preprocessor.input_example()
    if hasattr(asr_model.preprocessor, "featurizer"):
        asr_model.preprocessor.featurizer.training = False
        if hasattr(asr_model.preprocessor.featurizer, "dither"):
            # disable dither for export
            if asr_model.preprocessor.featurizer.dither != 0.0:
                LOGGER.info("disabling dither for preprocessor export")
            asr_model.preprocessor.featurizer.dither = 0
        if hasattr(asr_model.preprocessor.featurizer, "pad_to"):
            if asr_model.preprocessor.featurizer.pad_to != 0.0:
                LOGGER.info("disabling pad_to for preprocessor export")
            asr_model.preprocessor.featurizer.pad_to = 0

    if isinstance(
        asr_model.preprocessor,
        nemo.collections.asr.modules.audio_preprocessing.AudioPreprocessor,
    ):
        asr_model.preprocessor = WrapAudioPreprocessor(asr_model.preprocessor)

    with exportable_nemo_net(
        "preprocessor", asr_model.preprocessor, inps
    ) as ctx:
        # Stay inside NeMo export context while yielding parameters,
        # so the caller performs export with typechecks disabled and
        # wrapped forward in place.
        input_example = ctx.input_example
        dynamic_axes, custom_extensions = build_dynamic_axes(
            asr_model.preprocessor, ctx.dynamic_axes, input_example
        )

        subnet_name = "preprocessor"
        model = asr_model.preprocessor
        input_names = model.input_names[: len(input_example)]
        output_names = model.output_names
        # Use the context-provided input_example to ensure consistency between
        # the dynamic axes and the actual IO used during export.
        test_input = input_example
        dyn = dynamic_axes
        if collapse_batch_dim:
            # Wrap and collapse axes. Use the wrapper's own dynamic-axes view
            # to reflect the exposed ranks accurately (mirrors generic path).
            model = CollapseBatchDimWrapper(model, dynamic_axes)
            input_names = model.input_names
            output_names = model.output_names
            test_input = model.input_example()
            dyn = model.dynamic_shapes_for_export()

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


def iter_export_params_for_generic_nemo_asr_model(
    asr_model,
    inference_target,
    skip_preprocessor: bool = False,
    split_joint_decoder: bool = False,
    remove_unused_inputs: bool = True,
    collapse_batch_dim: bool = False,
    float_dtype: T.Optional[torch.dtype] = None,
) -> T.Iterator[ExportParameters]:
    """Iterator over export parameters for a generic NeMo ASR model."""
    asr_model.eval()

    if not skip_preprocessor:
        # Yield preprocessor export params while NeMo export context is active
        yield from build_preprocessor_export_params(
            asr_model, inference_target, collapse_batch_dim
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
            subnet, nemo_dynamic_axes, input_example
        )

        model = subnet
        test_input = input_example
        input_names = subnet.input_names[: len(input_example)]
        output_names = subnet.output_names

        # Limit dynamic axes to the inputs we are actually exposing.
        # Preserve suffixed variants (e.g., states_0, states_1) even if the
        # base name (states) is in input_names to match flattened graph IO.
        def _base_name_of(k: str, _names=subnet.input_names) -> str:
            for nm in _names:
                if k == nm or k.startswith(nm + "_"):
                    return nm
            if "_" in k:
                return k.split("_", 1)[0]
            return k

        dyn = {
            k: v
            for k, v in dynamic_axes.items()
            if (k in input_names) or (_base_name_of(k) in input_names)
        }

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
