import json
import logging
import typing as T
from collections import OrderedDict
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import autocast

from torch_to_nnef.compress import dynamic_load_registry
from torch_to_nnef.exceptions import T2NErrorInvalidArgument
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import build_io
from torch_to_nnef.model_wrapper import build_new_names_and_elements
from torch_to_nnef.remodeler import prepare_subnet_export
from torch_to_nnef.remodeler.adapter import RenameOutputs
from torch_to_nnef.utils import (
    INJECTED,
    T2NExtra,
    check_torch_ecosystem,
    require_extra_decorator,
)
from torch_to_nnef_nemo._optional_types import (
    InjectedLightningModule,
    InjectedNemoModule,
    InjectedOmegaConfModule,
)
from torch_to_nnef_nemo.axis_registry import AxisSymbolRegistry
from torch_to_nnef_nemo.config import NemoExportConfig
from torch_to_nnef_nemo.dynaxes import (
    build_dynamic_axes as build_dynamic_axes_for_subnet,
)
from torch_to_nnef_nemo.glue_subnets import (
    build_glue_subnets,
    fuse_glues_into,
    iter_all_glue_subnets,
)
from torch_to_nnef_nemo.wrappers import (
    WrapAudioPreprocessor,
    WrapPreprocessorCast,
    decoder_fix_input_example_batch_size,
)

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
        _patch_encoder_output_types(model.encoder.__class__)
        return subnet_names
    return None


def _pick_for_ctc(model, nemo_models_mod):
    """Specialize subnets for CTC families and patch encoder outputs."""
    cls_ctc, cls_ctc_bpe = _resolve_ctc_model_classes(nemo_models_mod)
    if (cls_ctc is not None and isinstance(model, cls_ctc)) or (
        cls_ctc_bpe is not None and isinstance(model, cls_ctc_bpe)
    ):
        subnet_names = ["encoder", "decoder"]
        _patch_encoder_output_types(model.encoder.__class__)
        return subnet_names
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
def _pick_subnets_names(model, *, nemo: InjectedNemoModule = INJECTED):
    nemo_model_mod = nemo.collections.asr.models
    # Default from model
    subnet_names = model.list_export_subnets()
    # Specialize for known families
    spec = _pick_for_classification(model, nemo_model_mod)
    if spec is not None:
        return spec
    spec = _pick_for_ctc(model, nemo_model_mod)
    if spec is not None:
        return spec
    return subnet_names


def iter_nemo_model_subnets(
    model,
    input_example=None,
    float_dtype: T.Optional[torch.dtype] = None,
    split_joint_decoder: bool = False,
    remove_unused_inputs: bool = True,
    apply_sequential_examples: bool = False,
    batch_size: int = 3,
    only_subnets: T.Optional[T.Collection[str]] = None,
    fuse_glue: bool = False,
):
    """Iterator over exportable subnets of a nemo model.

    When ``fuse_glue`` is set, each registered glue module (e.g. the language
    prompt head) is folded into its ``after_subnet`` (so the encoder gains a
    ``lang_id`` input and applies the transform on its output) instead of being
    emitted as a separate subnet.
    """
    subnet_names = _pick_subnets_names(model)
    allow: T.Optional[set[str]] = None
    if only_subnets is not None:
        allow = set(only_subnets)
    glues = build_glue_subnets(model) if fuse_glue else []
    fused_parents: set[str] = set()
    for subnet_name in subnet_names:
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
                for (
                    name,
                    _subnet,
                    _input_example,
                    _dyn_axes,
                ) in iter_decoder_joint_subnets(
                    subnet,
                    ctx.input_example,
                    ctx.dynamic_axes,
                    batch_size=batch_size,
                    remove_unused_inputs=remove_unused_inputs,
                    split_joint_decoder=split_joint_decoder,
                ):
                    if allow is None or name in allow:
                        yield name, _subnet, _input_example, _dyn_axes
                continue

            # Filter non-joint subnets early if a restriction is provided
            if allow is not None and subnet_name not in allow:
                input_example = None
                continue

            input_example = ctx.input_example
            if len(input_example) > len(subnet.input_names):
                # if < that means some inputs are optional
                raise T2NErrorInvalidArgument(
                    f"Declared input names: {subnet.input_names} "
                    f"but received {len(input_example)} inputs. "
                    "Some inputs may be optional; verify subnet interface."
                )
            out_subnet, out_example = subnet, input_example
            if fuse_glue:
                # Fold matching glue(s) into this subnet: it keeps its name but
                # gains the glue's extra inputs (e.g. lang_id). Done inside the
                # NeMo export context so the native forward stays wrapped.
                out_subnet, out_example, matched = fuse_glues_into(
                    subnet, subnet_name, input_example, glues
                )
                if matched:
                    fused_parents.add(subnet_name)
            yield subnet_name, out_subnet, out_example, ctx.dynamic_axes
            # Propagate input example
            # (default scenario, may need to be overriden)
            if input_example is not None and apply_sequential_examples:
                input_example = ctx.output_example
            else:
                input_example = None

    # Top-level "glue" modules NeMo's list_export_subnets() omits (e.g. a
    # language prompt head applied between encoder and decoder).
    if not fuse_glue:
        # Emit each as its own subnet, after the native ones, built once, and
        # independent of their export context, so they honor `only_subnets`.
        yield from iter_all_glue_subnets(model, allow)
    else:
        # Fused above; warn for any glue whose parent wasn't exported here and
        # fall back to a standalone subnet so it is never silently dropped.
        for glue in glues:
            if glue.after_subnet not in fused_parents:
                LOGGER.warning(
                    "glue '%s' targets subnet '%s' which was not exported as a "
                    "fusable subnet; emitting it standalone instead.",
                    glue.name,
                    glue.after_subnet,
                )
                if allow is None or glue.name in allow:
                    yield (
                        glue.name,
                        glue,
                        glue.input_example(),
                        glue.dynamic_shapes_for_export(),
                    )


def build_dynamic_axes(
    subnet,
    nemo_dynamic_axes,
    input_example: T.Optional[T.Sequence[object]] = None,
):
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
        yield (
            "decoder",
            decoder,
            decoder_fix_input_example_batch_size(
                decoder.input_example(max_batch=batch_size),
                batch_size=batch_size,
            ),
            decoder.dynamic_shapes_for_export(False),
        )
        yield (
            "joint",
            subnet.joint,
            subnet.joint.input_example(max_batch=batch_size),
            subnet.joint.dynamic_shapes_for_export(False),
        )
        return

    # Not splitting: keep decoder_joint together
    input_example = decoder_fix_input_example_batch_size(
        input_example, batch_size=batch_size
    )

    if len(input_example) > len(subnet.input_names):
        # if < that means some inputs are optional
        raise T2NErrorInvalidArgument(
            f"Declared input names: {subnet.input_names} "
            f"but received {len(input_example)} inputs. "
            "Some inputs may be optional; verify subnet interface."
        )

    yield (
        "decoder_joint",
        subnet,
        input_example,
        ctx_dynamic_axes,
    )


@dataclass(frozen=True)
class ExportParameters:
    name: str
    model: torch.nn.Module
    test_input: object
    inference_target: InferenceTarget
    input_names: list
    output_names: list
    custom_extensions: list
    specific_tract_properties: dict

    def display(self):
        def display_inp(inp):
            if isinstance(inp, torch.Tensor):
                return f"Tensor(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            if isinstance(inp, (list, tuple)):
                return (
                    f"{type(inp).__name__}"
                    f"([{', '.join(display_inp(i) for i in inp)}])"
                )
            return repr(inp)

        print("name", self.name)
        print("model", repr(self.model.__class__))
        print("test_input", display_inp(self.test_input))
        print("inference_target", repr(self.inference_target))
        print("input_names", self.input_names)
        print("output_names", self.output_names)
        print("custom_extensions", self.custom_extensions)
        print("specific_tract_properties", self.specific_tract_properties)


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
    *,
    nemo: InjectedNemoModule = INJECTED,
    axis_registry=None,
) -> T.Iterator[ExportParameters]:
    """Build export parameters for the preprocessor of a NeMo ASR model."""
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

    inps = asr_model.preprocessor.input_example()
    if (
        isinstance(
            asr_model.preprocessor,
            nemo.collections.asr.modules.audio_preprocessing.AudioPreprocessor,
        )
        and inps is None
    ):
        asr_model.preprocessor = WrapAudioPreprocessor(asr_model.preprocessor)
        inps = asr_model.preprocessor.input_example()
        assert inps is not None, "input_example must be provided by the wrapper"

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
        prepared = prepare_subnet_export(
            model=asr_model.preprocessor,
            test_input=input_example,
            input_names=asr_model.preprocessor.input_names[
                : len(input_example)
            ],
            output_names=asr_model.preprocessor.output_names,
            subnet_name=subnet_name,
            dyn=dynamic_axes,
            custom_extensions=list(custom_extensions),
            axis_registry=axis_registry,
        )

        yield ExportParameters(
            name=subnet_name,
            model=prepared.model,
            test_input=prepared.test_input,
            inference_target=inference_target.with_dynamic_axes(prepared.dyn),
            input_names=prepared.input_names,
            output_names=prepared.output_names,
            custom_extensions=prepared.custom_extensions,
            specific_tract_properties=build_custom_subnet_tract_properties(
                subnet_name, prepared.model
            ),
        )


def iter_export_params_for_generic_nemo_asr_model(
    asr_model,
    inference_target,
    skip_preprocessor: bool = False,
    split_joint_decoder: bool = False,
    remove_unused_inputs: bool = True,
    float_dtype: T.Optional[torch.dtype] = None,
    only_subnets: T.Optional[T.Collection[str]] = None,
    axis_registry=None,
    fuse_glue: bool = False,
) -> T.Iterator[ExportParameters]:
    """Iterator over export parameters for a generic NeMo ASR model."""
    asr_model.eval()

    # Optionally export preprocessor (unless filtered out explicitly)
    if not skip_preprocessor and (
        only_subnets is None or "preprocessor" in set(only_subnets)
    ):
        # Yield preprocessor export params while NeMo export context is active
        yield from build_preprocessor_export_params(
            asr_model, inference_target, axis_registry=axis_registry
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
        only_subnets=only_subnets,
        fuse_glue=fuse_glue,
    ):
        dynamic_axes, custom_extensions = build_dynamic_axes(
            subnet, nemo_dynamic_axes, input_example
        )

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

        prepared = prepare_subnet_export(
            model=subnet,
            test_input=input_example,
            input_names=input_names,
            output_names=output_names,
            subnet_name=subnet_name,
            dyn=dyn,
            custom_extensions=list(custom_extensions),
            axis_registry=axis_registry,
        )

        yield ExportParameters(
            name=subnet_name,
            model=prepared.model,
            test_input=prepared.test_input,
            inference_target=inference_target.with_dynamic_axes(prepared.dyn),
            input_names=prepared.input_names,
            output_names=prepared.output_names,
            custom_extensions=prepared.custom_extensions,
            specific_tract_properties=build_custom_subnet_tract_properties(
                subnet_name, prepared.model
            ),
        )


def _find_roots_to_rename(
    inter: set[str], output_names: T.List[str]
) -> set[str]:
    """Map colliding flattened names back to their pre-flattened root names."""
    roots: set[str] = set()
    for flat_name in inter:
        if flat_name in output_names:
            roots.add(flat_name)
        else:
            for oname in output_names:
                if flat_name.startswith(f"{oname}_"):
                    roots.add(oname)
                    break
    return roots


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
    only_subnets: T.Optional[T.Collection[str]] = None,
    fuse_glue: bool = False,
    *,
    omegaconf: InjectedOmegaConfModule = INJECTED,
    axis_registry=None,
    **kwargs,
):
    """Export a generic NeMo ASR model to NNEF format using TractNNEF."""
    check_torch_ecosystem()
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
        only_subnets=only_subnets,
        axis_registry=axis_registry,
        fuse_glue=fuse_glue,
    ):
        LOGGER.info("start subnet export: %s", export_params.name)
        model = export_params.model
        input_names = export_params.input_names
        output_names = export_params.output_names

        # Avoid name collisions between inputs and outputs for NNEF
        # (e.g., both named 'length').  This is an export-time concern
        # and does not leak into inspection or shape-config generation.
        #
        # output_names may still be pre-flattened (e.g. "states" for a
        # tuple that becomes "states_0", "states_1" after flattening).
        # We must check the *flattened* names to catch collisions that
        # only appear after container expansion.
        with torch.no_grad():
            _test_outs = model(*export_params.test_input)
        if isinstance(_test_outs, torch.Tensor):
            _test_outs = (_test_outs,)
        flat_output_names, _, _, _ = build_new_names_and_elements(
            output_names,
            _test_outs,
            default_element_name_tmpl="output_{}",
        )
        inter = set(input_names).intersection(set(flat_output_names))
        if inter:
            roots_to_rename = _find_roots_to_rename(inter, output_names)
            rename_map = {n: f"{n}_out" for n in roots_to_rename}
            model = RenameOutputs(model, rename_map)
            output_names = [rename_map.get(n, n) for n in output_names]

        if dump_checked_io:
            test_dir = export_dir / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            build_io(
                model,
                export_params.test_input,
                input_bundle_path=test_dir
                / f"{export_params.name}_inputs_checked.npz",
                output_bundle_path=test_dir
                / f"{export_params.name}_outputs_checked.npz",
                input_names=input_names,
                output_names=output_names,
            )
        export_model_to_nnef(
            model=model,
            args=export_params.test_input,
            inference_target=export_params.inference_target.with_specific_properties(
                export_params.specific_tract_properties
            ),
            input_names=input_names,
            output_names=output_names,
            file_path_export=export_dir / f"{export_params.name}.nnef.tgz",
            custom_extensions=export_params.custom_extensions,
            # NeMo subnets may declare inputs consumed via control flow
            # that PyTorch tracing cannot capture (e.g. 'length' used
            # for masking). Allow the pruned-input mismatch.
            check_io_names_qte_match=False,
            **kwargs,
        )
        LOGGER.info("exported subnet: %s with success", export_params.name)


def export_nemo_from_model(
    *,
    model,
    target: InferenceTarget,
    export_dir: Path,
    axis_reg: T.Optional[AxisSymbolRegistry] = None,
    cfg: NemoExportConfig,
) -> None:
    """Export a NeMo ASR model using a structured configuration.

    This is the public programmatic API.  Handles dtype preparation
    (``float16`` / ``mixed``) internally so callers don't need to
    apply ``model.half()`` or ``WrapPreprocessorCast`` themselves.
    """
    model.eval()
    if cfg.data_type == "float16":
        model = model.half()
        if hasattr(model, "preprocessor"):
            model.preprocessor.to(dtype=torch.float32)
    if cfg.data_type in ("float16", "mixed") and hasattr(model, "preprocessor"):
        model.preprocessor = WrapPreprocessorCast(
            model.preprocessor, dtype=torch.float16
        )

    float_dtype = (
        torch.float16
        if cfg.data_type in ("float16", "mixed")
        else torch.float32
    )

    def _do_export() -> None:
        export_nemo_asr_model(
            model,
            target,
            export_dir,
            nnef_variable_naming_scheme=cfg.naming_scheme,
            compress_registry=cfg.compression.compress_registry,
            compress_method=cfg.compression.compress_method,
            skip_preprocessor=cfg.subnet.skip_preprocessor,
            split_joint_decoder=cfg.subnet.split_joint_decoder,
            only_subnets=cfg.subnet.only_subnets,
            fuse_glue=cfg.subnet.fuse_prompt_into_encoder,
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
