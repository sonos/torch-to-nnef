import json
import logging
import re
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
from torch_to_nnef.exceptions import T2NErrorInvalidArgument
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import build_io
from torch_to_nnef.nemo_tract.dynaxes import (
    build_dynamic_axes as build_dynamic_axes_for_subnet,
)
from torch_to_nnef.nemo_tract.wrappers import (
    BoundaryAdapter,
    RenameOutputs,
    WrapAudioPreprocessor,
    decoder_fix_input_example_batch_size,
)
from torch_to_nnef.utils import (
    INJECTED,
    T2NExtra,
    check_torch_ecosystem,
    require_extra_decorator,
)

LOGGER = logging.getLogger(__name__)


def _rewrite_assertions_with_renames(
    assertions: list[str], rename_map: dict[str, list[str]] | None
) -> list[str]:
    """Rewrite assertion symbol names based on a rename mapping.

    Args:
        assertions: List of assertion strings, e.g. "tract_assert U = BATCH".
        rename_map: Mapping of target symbol to list of source symbols
            that should be rewritten to the target. Comparison is
            case-insensitive; rewritten symbols are emitted uppercased.

    Returns:
        A list of assertions with symbols rewritten according to
        the provided mapping. Unknown tokens are left unchanged.
    """
    if not rename_map:
        return list(assertions)

    inv: dict[str, str] = {}
    for tgt, srcs in (rename_map or {}).items():
        t_u = str(tgt).upper()
        for s in srcs or []:
            inv[str(s).upper()] = t_u

    # Replace only identifier-like tokens to avoid altering operators
    ident = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

    def _sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        return inv.get(tok.upper(), tok)

    out: list[str] = []
    for a in assertions:
        out.append(ident.sub(_sub, str(a)))
    return out


def _batch_equal_assertions_for_subnet(
    subnet_name: str, dyn: dict[str, dict[int, str]] | None
) -> set[str]:
    """Emit tract_assert equality constraints when batch-like symbols need it.

    - For decoder-like subnets, batch dims across inputs must be equal; add
      equality assertions to help Tract unify distinct symbols at typecheck.
    - For other subnets, we stay hands-off to preserve independence unless the
      model semantics require otherwise in the future.
    """
    if not dyn:
        return set()
    needs_tie = subnet_name in {"decoder", "decoder_joint"}
    if not needs_tie:
        return set()
    # Collect batch-like symbols across inputs
    batch_syms: list[str] = []
    for axes in dyn.values():
        for s in (axes or {}).values():
            su = str(s).upper()
            if su == "B" or "BATCH" in su:
                batch_syms.append(str(s))
    uniq = []
    for s in batch_syms:
        if s not in uniq:
            uniq.append(s)
    if len(uniq) <= 1:
        return set()
    ref = uniq[0]
    # Use single '=' for compatibility with tract assertion parser
    return {f"tract_assert {s} = {ref}" for s in uniq[1:]}


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
):
    """Iterator over exportable subnets of a nemo model."""
    subnet_names = _pick_subnets_names(model)
    allow: T.Optional[set[str]] = None
    if only_subnets is not None:
        allow = set(only_subnets)
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
            yield subnet_name, subnet, input_example, ctx.dynamic_axes
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
        model = asr_model.preprocessor
        input_names = model.input_names[: len(input_example)]
        output_names = model.output_names
        # If any outputs share names with inputs, rename outputs
        # to avoid collision
        inter = set(input_names).intersection(set(output_names))
        if inter:
            rename_map = {n: f"{n}_out" for n in inter}
            model = RenameOutputs(model, rename_map)
            output_names = [rename_map.get(n, n) for n in output_names]
        # Use the context-provided input_example to ensure consistency between
        # the dynamic axes and the actual IO used during export.
        test_input = input_example
        dyn = dynamic_axes
        # Config-driven boundary adapter: apply per-input batch collapse
        # and tuple flattening
        if axis_registry is not None and getattr(
            axis_registry, "input_collapse_dims", None
        ):
            collapse_map = (
                getattr(axis_registry, "input_collapse_dims", {}) or {}
            )
            binds_map = getattr(axis_registry, "bind_to_dim", {}) or {}
            rename_map = (
                getattr(axis_registry, "renamed_symbols_per_subnet", {}) or {}
            ).get(subnet_name, {})
            model = BoundaryAdapter(
                model,
                subnet_name,
                test_input,
                dyn,
                {k: set(v) for k, v in collapse_map.items()},
                binds_map,
                rename_map,
                outputs_keep=(
                    getattr(axis_registry, "outputs_keep_per_subnet", {}) or {}
                ).get(subnet_name, []),
            )
            input_names = model.input_names
            test_input = list(model.input_example())
            dyn = model.dynamic_shapes_for_export()
            # Apply symbol renames for Tract-facing dynamic axes
            if rename_map:
                renamed_dyn: dict[str, dict[int, str]] = {}
                inv: dict[str, str] = {}
                for tgt, srcs in rename_map.items():
                    for s in srcs:
                        inv[s.upper()] = tgt.upper()
                for name, axes in (dyn or {}).items():
                    mapped: dict[int, str] = {}
                    for i, s in (axes or {}).items():
                        su = str(s).upper()
                        mapped[i] = inv.get(su, str(s))
                    renamed_dyn[name] = mapped
                dyn = renamed_dyn

        # Augment custom extensions and consolidate with renames
        custom_ext = set(custom_extensions)
        custom_ext |= _batch_equal_assertions_for_subnet(subnet_name, dyn)
        custom_ext = set(
            _rewrite_assertions_with_renames(
                list(custom_ext),
                (
                    getattr(axis_registry, "renamed_symbols_per_subnet", {})
                    or {}
                ).get(subnet_name, {}),
            )
        )

        yield ExportParameters(
            name=subnet_name,
            model=model,
            test_input=test_input,
            inference_target=inference_target.with_dynamic_axes(dyn),
            input_names=input_names,
            output_names=output_names,
            custom_extensions=list(custom_ext),
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
    float_dtype: T.Optional[torch.dtype] = None,
    only_subnets: T.Optional[T.Collection[str]] = None,
    axis_registry=None,
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
        # Keep namespaced dims; we'll add targeted equality assertions below

        # Config-driven boundary adapter: apply per-input batch collapse
        # and tuple flattening
        if axis_registry is not None and getattr(
            axis_registry, "input_collapse_dims", None
        ):
            collapse_map = (
                getattr(axis_registry, "input_collapse_dims", {}) or {}
            )
            binds_map = getattr(axis_registry, "bind_to_dim", {}) or {}
            rename_map = (
                getattr(axis_registry, "renamed_symbols_per_subnet", {}) or {}
            ).get(subnet_name, {})
            model = BoundaryAdapter(
                model,
                subnet_name,
                test_input,
                dyn,
                {k: set(v) for k, v in collapse_map.items()},
                binds_map,
                rename_map,
                outputs_keep=(
                    getattr(axis_registry, "outputs_keep_per_subnet", {}) or {}
                ).get(subnet_name, []),
            )
            input_names = model.input_names
            test_input = list(model.input_example())
            dyn = model.dynamic_shapes_for_export()
            # Apply symbol renames for Tract-facing dynamic axes
            if rename_map:
                renamed_dyn: dict[str, dict[int, str]] = {}
                inv: dict[str, str] = {}
                for tgt, srcs in rename_map.items():
                    for s in srcs:
                        inv[s.upper()] = tgt.upper()
                for name, axes in (dyn or {}).items():
                    mapped: dict[int, str] = {}
                    for i, s in (axes or {}).items():
                        su = str(s).upper()
                        mapped[i] = inv.get(su, str(s))
                    renamed_dyn[name] = mapped
                dyn = renamed_dyn

        # Avoid name collisions between inputs and outputs (e.g., 'length').
        inter = set(input_names).intersection(set(output_names))
        if inter:
            rename_map = {n: f"{n}_out" for n in inter}
            model = RenameOutputs(model, rename_map)
            output_names = [rename_map.get(n, n) for n in output_names]

        # Augment custom extensions with decoder batch equalities where needed
        custom_ext = set(custom_extensions)
        custom_ext |= _batch_equal_assertions_for_subnet(subnet_name, dyn)

        yield ExportParameters(
            name=subnet_name,
            model=model,
            test_input=test_input,
            inference_target=inference_target.with_dynamic_axes(dyn),
            input_names=input_names,
            output_names=output_names,
            custom_extensions=list(custom_ext),
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
    only_subnets: T.Optional[T.Collection[str]] = None,
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
    ):
        LOGGER.info("start subnet export: %s", export_params.name)
        if dump_checked_io:
            test_dir = export_dir / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            build_io(
                export_params.model,
                export_params.test_input,
                input_bundle_path=test_dir
                / f"{export_params.name}_inputs_checked.npz",
                output_bundle_path=test_dir
                / f"{export_params.name}_outputs_checked.npz",
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
            **kwargs,
        )
        LOGGER.info("exported subnet: %s with success", export_params.name)
