import contextlib
import importlib
import logging as log
import os
import typing as T
from collections.abc import KeysView, ValuesView
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import torch
from nnef_tools.model import Graph
from torch.onnx import TrainingMode  # type: ignore
from torch.onnx.utils import select_model_mode_for_export  # type: ignore

from torch_to_nnef.dtypes import is_quantized_dtype
from torch_to_nnef.exceptions import (
    T2NErrorInvalidArgument,
    T2NErrorNotImplemented,
)
from torch_to_nnef.inference_target import InferenceTarget
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.log import set_lib_log_level
from torch_to_nnef.model_wrapper import unfold_model_io
from torch_to_nnef.nnef_graph import TorchToNGraphExtractor
from torch_to_nnef.nnef_io.writer import Writer as NNEFWriter
from torch_to_nnef.nnef_io.writer import (
    write_nnef_tensor,
    write_tensor_quantization_infos,
)
from torch_to_nnef.op.fragment import FRAGMENTS, Fragment
from torch_to_nnef.op.quantized import torch_qtensor_to_ntensor
from torch_to_nnef.tensor import (
    OpaqueTensorRef,
    QTensor,
    apply_name_to_tensor_in_module,
    set_opaque_tensor_in_params_as_ref,
)
from torch_to_nnef.tensor.updater import ModTensorUpdater
from torch_to_nnef.torch_graph import harden_jit_for_export
from torch_to_nnef.torch_graph.ir_naming import (
    DEFAULT_VARNAME_SCHEME,
    VariableNamingScheme,
)
from torch_to_nnef.utils import dedup_list, ensure_tuple_io, torch_version

LOGGER = log.getLogger(__name__)


def export_model_to_nnef(
    model: torch.nn.Module,
    args,  # args pushed with *args in forward of module
    file_path_export: T.Union[Path, str],
    inference_target: InferenceTarget,
    input_names: T.Optional[T.List[str]] = None,
    output_names: T.Optional[T.List[str]] = None,
    compression_level: T.Optional[int] = 0,
    log_level: int = log.INFO,
    nnef_variable_naming_scheme: VariableNamingScheme = DEFAULT_VARNAME_SCHEME,
    check_io_names_qte_match: bool = True,
    debug_bundle_path: T.Optional[Path] = None,
    custom_extensions: T.Optional[T.List[str]] = None,
    allow_same_io_names: bool = False,
    auto_harden_jit: bool = True,
    load_extra_op_modules: T.Optional[T.List[str]] = None,
    discover_extra_entrypoints: bool = False,
    strict_extra_imports: bool = False,
    skip_eager_forward: bool = False,
) -> Path:
    """Main entrypoint of this library.

    Export any torch.nn.Module to NNEF file format archive

    Args:
        model: a nn.Module that have a `.forward` function
            with only tensor arguments and outputs
            (no tuple, list, dict or objects)
            Only this function will be serialized

        args: a flat ordered list of tensors for each forward inputs of `model`
            this list can not be of dynamic size (at serialization it will be
            fixed to quantity of tensor provided)
            WARNING! tensor size in args will increase export time so take that
            in consideration for dynamic axes

        file_path_export: target path for the exported model.
            Accepted forms are:
            - ".../model.nnef" → base path; creates:
                • directory when `compression_level is None`
                • archive "model.nnef.tar" when `compression_level == 0`
                • archive "model.nnef.tgz" when `compression_level in 1..9`
            - ".../model.nnef.tgz" → treated as a request to use base name
              "model.nnef"; the actual artifact still follows the rule above
              (directory, .tar, or .tgz) depending on `compression_level`.
            Any other suffix pattern is rejected.

        inference_target:
            can be `torch_to_nnef.TractNNEF` or `torch_to_nnef.KhronosNNEF`
            for each you can specify version targeted:
            - KhronosNNEF is the least maintained so far,
                and is checked against nnef-tools PyTorch interpreter
            - TractNNEF is our main focus at SONOS,
              it is checked against tract inference engine
              among key paramters there is
                feature_flags: Optional[Set[str]],
                that may contains tract specifics
                dynamic_axes: Optional
                  By default the exported model will have
                  the shapes of all input and output tensors set
                  to exactly match those given in args.
                  To specify axes of tensors as dynamic
                  (i.e. known only at runtime)
                  set dynamic_axes to a dict with schema:
                      KEY (str):
                        an input or output name. Each name must also
                        be provided in input_names or output_names.
                      VALUE (dict or list): If a dict, keys are axis indices
                        and values are axis names. If a list, each element is
                        an axis index.


        input_names: Optional list of names for args, it replaces
            variable inputs names traced from graph
            (if set it must have the same size as number of args)

        output_names: Optional list of names for outputs of `model.forward`,
            it replaces variable output names traced from graph
            (if set it must have the same size as number of outputs)

        compression_level: Optional[int] = 0
            If None, writes an uncompressed `.nnef` directory.
            If 0, writes an uncompressed tar archive `.nnef.tar`.
            If 1..9, writes a gzip-compressed tar archive `.nnef.tgz` with the
            given compression level.

        log_level: int,
            logger level for `torch_to_nnef` following Python
            standard logging level can be set to:
            INFO, WARN, DEBUG ...

        nnef_variable_naming_scheme:
            Possible choices NNEF variables naming schemes are:
            - "raw": Taking variable names from traced graph debugName directly
            - "natural_verbose": that try to provide nn.Module exported
              variable naming consistency
            - "natural_verbose_camel": that try to provide nn.Module exported
              variable naming consistency but with more consice camelCase
              variable pattern
            - "numeric": that try to be as concise as possible

        check_io_names_qte_match: (default: True)
            During the tracing process of the torch graph
            One or more input provided can be removed if not contributing to
            generate outputs while check_io_names_qte_match is True we ensure
            that this input and output quantity remain constant with numbers in
            `input_names` and `output_names`.

        debug_bundle_path: Optional[Path]
            if specified it should create an archive bundle with all needed
            information to allow easier debug.

        custom_extensions: Optional[List[str]]
            allow to add a set of extensions as defined in
            (https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html)
            Useful to set specific extensions like for example:
            'extension tract_assert S >= 0'
            those assertion allows to add limitation on dynamic shapes
            that are not expressed in traced graph
            (like for example maximum number of tokens for an LLM)
        allow_same_io_names: bool
            by default input and output names must be different
            to avoid simplification of the graph that would
            merge those tensors silently.
            If you really want to have same names for inputs
            and outputs set this flag to True.
            Some libs like 'nvidia/nemo' use this pattern.
            (note that it only make sense if it's a no operation)

        auto_harden_jit: bool (default: True)
            When `model` is a `torch.jit.ScriptModule`, automatically
            run `harden_jit_for_export` to specialize its graph for the
            given example inputs (freeze + size folds + constant folds
            + tuple round-trip + data-dependent If fold). Each pass is
            a no-op on graphs that don't carry the relevant pattern, so
            the wrapper is safe to apply unconditionally; turn it off
            to drive the chain manually for fine-grained control.

        load_extra_op_modules: Optional[List[str]]
            Optional list of Python module paths to import before
            tracing/export. Importing a module that calls
            `torch_to_nnef.op.extras.register("<name>")` registers a
            handler for `t2n_extra::<name>` custom ops so they are
            translated during export. You can also provide the same list
            via the `TORCH_TO_NNEF_EXTRA_MODULES` environment variable
            (comma-separated).

        discover_extra_entrypoints: bool (default: False)
            Auto-discover and import installed plugins that declare a
            Python entry point under the `torch_to_nnef.extras` group. The
            entry point value should be a module path that performs the
            `extras.register` calls on import.

        strict_extra_imports: bool (default: False)
            If True, fail the export when an extra-op module fails to import
            (from `load_extra_op_modules`, env var, or entry points). When
            False, log a warning and continue.

        skip_eager_forward: bool (default: False)
            Skip running a real eager forward to infer outputs and attempt a
            meta-only forward instead (meta tensors). When False, exporter runs
            an eager forward and falls back to a meta forward if the eager run
            fails (e.g., a `t2n_extra` op lacks a CPU kernel).

    Returns:
        Path: the path to the exported artifact.
            - If `compression_level is None`: returns the
              `.nnef` directory path.
            - If `compression_level == 0`: returns the
              `.nnef.tar` archive path.
            - If `compression_level in 1..9`: returns the
              `.nnef.tgz` archive path.

    Raises:
        torch_to_nnef.exceptions.T2NError
            If something fail during the export process we try to provide
            dedicated exceptions (easier to control programmatically)

    Examples:
        For example this function can be used to export
        as simple perceptron model:

        >>> import os
        >>> import tarfile
        >>> import tempfile
        >>> from torch import nn
        >>> mod = nn.Sequential(nn.Linear(1, 5), nn.ReLU())
        >>> export_path = tempfile.mktemp(suffix=".nnef.tgz")
        >>> inference_target = TractNNEF.latest()
        >>> _ = export_model_to_nnef(
        ...   mod,
        ...   torch.rand(3, 1),
        ...   export_path,
        ...   inference_target,
        ...   compression_level=0,
        ...   input_names=["inp"],
        ...   output_names=["out"]
        ... )
        >>> os.chdir(export_path.rsplit("/", maxsplit=1)[0])
        >>> tarfile.open(export_path).extract("graph.nnef")
        >>> "graph network(inp) -> (out)" in open("graph.nnef").read()
        True

    """
    if isinstance(file_path_export, str):
        file_path_export = Path(file_path_export)
    set_lib_log_level(log_level)
    if isinstance(input_names, KeysView):
        input_names = list(input_names)
    if isinstance(output_names, KeysView):
        output_names = list(output_names)
    args = tuple(args) if isinstance(args, ValuesView) else args

    # Auto-apply the JIT-only export hardening chain when the input is
    # a `torch.jit.ScriptModule`. The chain is a no-op on already-clean
    # graphs, so the only cost on a well-behaved JIT artifact is the
    # graph walk; on artifacts whose Python source isn't on the import
    # path (e.g. `silero_vad.jit`) it's the difference between a
    # successful export and a `ModuleNotFoundError`. Opt out via
    # `auto_harden_jit=False` to drive the chain manually.
    if auto_harden_jit and isinstance(model, torch.jit.ScriptModule):
        LOGGER.info(
            "Detected torch.jit.ScriptModule; auto-applying "
            "harden_jit_for_export. Pass auto_harden_jit=False to "
            "drive the chain manually."
        )
        model = harden_jit_for_export(model, args)

    mod_tensor_updater = ModTensorUpdater(
        model,
        add_buffers=False,
        add_unregistred_tensor=False,
        disable_requires_grad=True,
    )

    if custom_extensions is not None and not isinstance(
        custom_extensions, list
    ):
        raise T2NErrorInvalidArgument(
            (
                "custom extensions should be a list, because some extensions "
                "may be order sensitive (in tract)."
            )
        )

    # Optionally load external custom-op handler modules. Handlers registered
    # via `torch_to_nnef.op.extras.register` become available once their
    # defining module is imported. This can be driven via the function
    # parameter or the `TORCH_TO_NNEF_EXTRA_MODULES` environment variable
    # (comma-separated list of module paths).
    mods_to_import: T.List[str] = []
    # Disallow custom-op module usage on legacy torch versions that don't
    # expose the `torch.library` surface (authors often define ops at import).
    supports_t2n_custom = (
        torch_version() >= "2.0.0"
        and hasattr(torch, "library")
        and hasattr(torch.library, "Library")
    )
    if not supports_t2n_custom:
        requested = (
            bool(load_extra_op_modules)
            or bool(os.environ.get("TORCH_TO_NNEF_EXTRA_MODULES"))
            or bool(discover_extra_entrypoints)
        )
        if requested:
            raise T2NErrorInvalidArgument(
                "Custom-op handler modules requested but torch < 2.0.0 or "
                "missing torch.library API. Upgrade torch or run without "
                "custom-op modules."
            )
    if load_extra_op_modules:
        mods_to_import.extend(load_extra_op_modules)
    env_mods = os.environ.get("TORCH_TO_NNEF_EXTRA_MODULES")
    if env_mods:
        mods_to_import.extend(
            [m.strip() for m in env_mods.split(",") if m.strip()]
        )
    # Discover installed plugins via entry points.
    if discover_extra_entrypoints:
        try:
            eps = importlib_metadata.entry_points()
            # Newer Python returns a Selection; older returns dict; handle both.
            group = (
                eps.select(group="torch_to_nnef.extras")
                if hasattr(eps, "select")
                else eps.get("torch_to_nnef.extras", [])
            )
            for ep in group:  # type: ignore[assignment]
                # Entry point values can be "pkg.mod:obj"; import module part.
                val = getattr(ep, "value", None) or getattr(ep, "module", "")
                mod_path = val.split(":", 1)[0]
                if mod_path:
                    mods_to_import.append(mod_path)
        except Exception as err:  # pragma: no cover
            LOGGER.debug("Entry point discovery failed: %s", err)

    # Deduplicate while preserving order.
    seen = set()
    mods_to_import = [
        m for m in mods_to_import if not (m in seen or seen.add(m))
    ]
    if mods_to_import:
        LOGGER.debug("Extra-op handler modules to import: %s", mods_to_import)
    for mod in mods_to_import:
        try:
            importlib.import_module(mod)
            LOGGER.info("Loaded extra op module: %s", mod)
        except Exception as err:  # pragma: no cover - defensive logging only
            msg = f"Failed to import extra op module '{mod}': {err}"
            if strict_extra_imports:
                raise T2NErrorInvalidArgument(msg) from err
            LOGGER.warning(msg)

    # Run forward once to capture outputs under safe modes
    args = ensure_tuple_io(args)

    def _try_forward(_model, _args):
        with (
            select_model_mode_for_export(_model, TrainingMode.EVAL),
            torch.no_grad(),
            torch.inference_mode(),
        ):
            return _model(*_args)

    # Run forward once to capture outputs under safe modes
    if skip_eager_forward:
        # Try a meta-only dry run to infer output structures and shapes.
        try:
            meta_args = tuple(
                (
                    torch.empty_like(a, device="meta")
                    if isinstance(a, torch.Tensor)
                    else a
                )
                for a in args
            )
            outs = _try_forward(model, meta_args)
        except Exception as err:
            raise T2NErrorInvalidArgument(
                "skip_eager_forward requested but meta forward failed; "
                "provide CPU/meta kernels for custom ops or disable the flag. "
                f"Error: {err}"
            ) from err
    else:
        try:
            outs = _try_forward(model, args)
        except Exception as eager_err:
            # Fallback to meta forward for custom ops lacking CPU kernels.
            try:
                meta_args = tuple(
                    (
                        torch.empty_like(a, device="meta")
                        if isinstance(a, torch.Tensor)
                        else a
                    )
                    for a in args
                )
                outs = _try_forward(model, meta_args)
                LOGGER.info(
                    "Eager forward failed; fell back to meta forward: %s",
                    eager_err,
                )
            except Exception as meta_err:
                raise eager_err from meta_err

    # Normalize and validate IO names and shapes
    apply_name_to_tensor_in_module(model)
    outs = ensure_tuple_io(outs)
    _check_io_names(input_names, output_names, allow_same_io_names)

    LOGGER.info(
        "start parse PyTorch model to be exported at %s", file_path_export
    )
    if not any(s == ".nnef" for s in file_path_export.suffixes):
        raise T2NErrorInvalidArgument(
            "`file_path_export` should end with '.nnef' or '.nnef.tgz',"
            f" but found: {file_path_export.suffixes}"
        )

    with (
        _unsupported_module_alerter(inference_target),
        select_model_mode_for_export(model, TrainingMode.EVAL),
    ):
        model_info = _unfold_and_prepare_model_io(
            model, args, outs, input_names, output_names
        )
        input_names = model_info.input_names
        output_names = model_info.output_names

        (
            nnef_graph,
            active_custom_extensions,
            active_custom_fragments,
        ) = _build_nnef_graph_and_fragments(
            model_info,
            model,
            inference_target,
            nnef_variable_naming_scheme,
            check_io_names_qte_match,
            input_names,
            output_names,
            custom_extensions,
        )

        nnef_exp_file_path, archive_format = _compute_archive_settings(
            file_path_export, compression_level
        )

        NNEFWriter(
            compression=compression_level,
            fragments=active_custom_fragments,
            generate_custom_fragments=False,
            extensions=list(active_custom_extensions),
            version_custom_fragments=None,
            inference_target=inference_target,
            archive_format=archive_format,
        )(nnef_graph, str(nnef_exp_file_path))

        _log_extensions_and_success(
            active_custom_extensions, inference_target, nnef_exp_file_path
        )

        exported_filepath = _finalize_export_path(
            file_path_export,
            nnef_exp_file_path,
            compression_level,
            archive_format,
        )

        with _fixed_backend():
            inference_target.post_export(
                model_info,
                nnef_graph,
                exported_filepath,
                debug_bundle_path=debug_bundle_path,
            )
    mod_tensor_updater.restore_require_grad()
    return exported_filepath


def _unfold_and_prepare_model_io(
    model: torch.nn.Module,
    args: T.Tuple[T.Any, ...],
    outs: T.Tuple[T.Any, ...],
    input_names: T.Optional[T.List[str]],
    output_names: T.Optional[T.List[str]],
):
    """Unfold IO structures and set opaque tensor params as references."""
    set_opaque_tensor_in_params_as_ref(model)
    model_info = unfold_model_io(model, args, outs, input_names, output_names)
    return model_info


def _build_nnef_graph_and_fragments(
    model_info,
    model: torch.nn.Module,
    inference_target: InferenceTarget,
    nnef_variable_naming_scheme: VariableNamingScheme,
    check_io_names_qte_match: bool,
    input_names: T.Optional[T.List[str]],
    output_names: T.Optional[T.List[str]],
    custom_extensions: T.Optional[T.List[str]],
):
    """Build the NNEF graph, collect active extensions and fragments."""
    inference_target.pre_trace(model, input_names, output_names)

    graph_extractor = TorchToNGraphExtractor(
        model_info.model,
        model_info.flat_inputs,
        inference_target=inference_target,
        nnef_variable_naming_scheme=nnef_variable_naming_scheme,
        check_io_names_qte_match=check_io_names_qte_match,
        forced_inputs_names=input_names,
        forced_outputs_names=output_names,
    )
    nnef_graph = graph_extractor.parse()

    active_custom_extensions = _get_active_custom_extensions(graph_extractor)
    inference_target.post_trace(nnef_graph, active_custom_extensions)
    if custom_extensions is not None:
        active_custom_extensions = dedup_list(
            active_custom_extensions + custom_extensions
        )

    active_custom_fragments = inference_target.specific_fragments(model)
    active_custom_fragments.update(
        _get_active_custom_fragments(graph_extractor)
    )
    del graph_extractor
    return nnef_graph, active_custom_extensions, active_custom_fragments


def _compute_archive_settings(
    file_path_export: Path, compression_level: T.Optional[int]
) -> T.Tuple[Path, T.Optional[str]]:
    """Decide writer path and archive format."""
    nnef_exp_file_path = _real_export_path(file_path_export, compression_level)
    original_suffixes = file_path_export.suffixes
    wants_tgz = any(s == ".tgz" for s in original_suffixes)
    archive_format = None
    if compression_level is not None:
        if wants_tgz:
            archive_format = "tgz"
        else:
            archive_format = (
                "tgz"
                if (compression_level and compression_level > 0)
                else "tar"
            )
    return nnef_exp_file_path, archive_format


def _log_extensions_and_success(
    active_custom_extensions, inference_target, nnef_exp_file_path: Path
) -> None:
    if len(active_custom_extensions) > 0:
        LOGGER.info(
            (
                "The exported NNEF model need special custom extensions "
                "such as %s, be sure to use the inference engine you "
                "specified: %s"
            ),
            active_custom_extensions,
            inference_target,
        )
    LOGGER.info(
        "model exported successfully as NNEF at: %s", nnef_exp_file_path
    )


def _finalize_export_path(
    file_path_export: Path,
    nnef_exp_file_path: Path,
    compression_level: T.Optional[int],
    archive_format: T.Optional[str],
) -> Path:
    """Return the final exported filepath and emit an informative log."""
    if compression_level is not None:
        if archive_format == "tgz":
            suf = ".tgz"
        elif archive_format == "tar":
            suf = ".tar"
        else:
            suf = (
                ".tgz"
                if (compression_level and compression_level > 0)
                else ".tar"
            )
        exported_filepath = file_path_export.parent / (
            nnef_exp_file_path.name + suf
        )
        LOGGER.info(
            "created archive: %s (compression=%s)",
            exported_filepath,
            compression_level,
        )
    else:
        exported_filepath = nnef_exp_file_path
        LOGGER.info("exported directory: %s", exported_filepath)
    return exported_filepath


def _check_io_names(
    input_names: T.Optional[T.List[str]],
    output_names: T.Optional[T.List[str]],
    allow_same_io_names: bool = False,
):
    if input_names and len(set(input_names)) != len(input_names):
        raise T2NErrorInvalidArgument(
            "Each str in input_names must be different"
        )

    if output_names and len(set(output_names)) != len(output_names):
        raise T2NErrorInvalidArgument(
            "Each str in output_names must be different"
        )

    if (
        input_names
        and output_names
        and len(set(output_names + input_names))
        != len(input_names + output_names)
    ):
        collisions = sorted(set(input_names).intersection(set(output_names)))
        if allow_same_io_names:
            LOGGER.warning(
                "Input and output names overlap: %s. This may cause variable "
                "shadowing in inference engines, leading to misbinding, "
                "incorrect dynamic-shape facts, or optimizer "
                "mis-simplification. "
                "Prefer distinct IO names or rename outputs at export.",
                collisions,
            )
        else:
            raise T2NErrorInvalidArgument(
                "input_names and output_names must be different "
                "(else it could lead to wrong simplification of the graph)"
            )


def _real_export_path(
    file_path_export: Path, compression_level: T.Optional[int] = None
) -> Path:
    """Canonicalize the working export path used by the NNEF writer.

    If the target path ends with `.tgz` (i.e., a user passed
    `.../model.nnef.tgz`),
    always treat it as the base `.../model.nnef` path for the writer, regardless
    of `compression_level`. This lets callers use the suffix to express intent
    for the final artifact format, while the writer always receives the base
    directory path.
    """
    nnef_exp_file_path = Path(file_path_export)
    # Strip only the last suffix if it's .tgz
    # (e.g., model.nnef.tgz -> model.nnef)
    if nnef_exp_file_path.suffix == ".tgz":
        nnef_exp_file_path = nnef_exp_file_path.with_suffix("")
    return nnef_exp_file_path


def _get_active_custom_extensions(graph_extractor):
    return dedup_list(
        [
            ext
            for _ in graph_extractor.activated_custom_fragment_keys
            for ext in (FRAGMENTS[_] if isinstance(_, str) else _).extensions
        ]
    )


def _get_active_custom_fragments(graph_extractor):
    active_custom_fragments = {}
    for _ in graph_extractor.activated_custom_fragment_keys:
        if isinstance(_, Fragment):
            active_custom_fragments[_.name] = _.definition
        else:
            active_custom_fragments[_] = FRAGMENTS[_].definition
    return active_custom_fragments


_Tensor = T.TypeVar("_Tensor", bound=torch.Tensor)


def _default_filter_key(key):
    return True


def iter_torch_tensors_from_disk(
    store_filepath: Path,
    filter_key: T.Optional[T.Callable[[str], bool]] = None,
    map_location: T.Union[str, torch.device] = "cpu",
) -> T.Iterator[T.Tuple[str, _Tensor]]:
    """Iter on torch tensors from disk .safetensors, .pt, pth, .bin.

    Args:
        store_filepath: path to the container file holding PyTorch tensors
            (.pt, .pth, .bin and .safetensors)
        filter_key:
            if set, this function filter over tensor by name
            stored in those format
        map_location:
            device mapping used by torch.load for .pt/.pth/.bin files
            (default: "cpu").

    Yields:
       provide each tensor that are validated by filter within store filepath
       one at a time as tuple with name first then the torch.Tensor itself

    """
    if filter_key is None:
        filter_key = _default_filter_key

    if store_filepath.name.endswith(".safetensors"):
        # pylint: disable-next=import-outside-toplevel
        from safetensors import safe_open

        with safe_open(store_filepath, framework="pt", device="cpu") as fh:
            for key in fh.keys():  # noqa: SIM118
                if filter_key(key):
                    yield key, fh.get_tensor(key)
    elif any(store_filepath.name.endswith(_) for _ in [".pt", ".pth", ".bin"]):
        # Always load tensors to the requested device (default CPU) to avoid
        # device-specific state and environments lacking CUDA.
        res = torch.load(store_filepath, map_location=map_location)
        if isinstance(res, torch.nn.Module):
            for key, tensor in res.named_parameters():
                if filter_key(key):
                    yield key, tensor
        elif hasattr(res, "items"):
            for key, tensor in res.items():
                if not filter_key(key):
                    continue
                if isinstance(tensor, torch.Tensor):
                    yield key, tensor
                else:
                    LOGGER.warning(
                        "Skipping non-tensor entry from state dict: %s "
                        "(type=%s)",
                        key,
                        type(tensor),
                    )
        else:
            raise T2NErrorNotImplemented(type(res))


def export_tensors_from_disk_to_nnef(
    store_filepath: T.Union[Path, str],  # either statedict or safetensors
    output_dir: T.Union[Path, str],
    filter_key: T.Optional[T.Callable[[str], bool]] = None,
    fn_check_found_tensors: T.Optional[
        T.Callable[[T.Dict[str, _Tensor]], bool]
    ] = None,
    map_location: T.Union[str, torch.device] = "cpu",
) -> T.Dict[str, _Tensor]:
    """Export any statedict or safetensors file torch.Tensors to NNEF .dat file.

    Args:
        store_filepath:
            the filepath that hold the .safetensors , .pt or .bin
            containing the state dict
        output_dir:
            directory to dump the NNEF tensor .dat files
        filter_key:
            An optional function to filter specific keys to be exported
        fn_check_found_tensors:
            post checking function to ensure all requested tensors have
            effectively been dumped
        map_location:
            device mapping used by torch.load for .pt/.pth/.bin files
            (default: "cpu").

    Returns:
        a dict of tensor name as key and torch.Tensor values,
            identical to `torch_to_nnef.export.export_tensors_to_nnef`

    Examples:
        Simple filtered example

        >>> import tempfile
        >>> from torch import nn
        >>> class Mod(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.a = nn.Linear(1, 5)
        ...         self.b = nn.Linear(5, 1)
        ...
        ...     def forward(self, x):
        ...         return self.b(self.a(x))
        >>> mod = Mod()
        >>> pt_path = tempfile.mktemp(suffix=".pt")
        >>> nnef_dir = tempfile.mkdtemp(suffix="_nnef")
        >>> torch.save(mod.state_dict(), pt_path)
        >>> def check(ts):
        ...     assert all(_.startswith("a.") for _ in ts)
        >>> exported_tensors = export_tensors_from_disk_to_nnef(
        ...     pt_path,
        ...     nnef_dir,
        ...     lambda x: x.startswith("a."),
        ...     check
        ... )
        >>> list(exported_tensors.keys())
        ['a.weight', 'a.bias']
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if isinstance(store_filepath, str):
        store_filepath = Path(store_filepath)
    to_export = {}
    for key, tensor in iter_torch_tensors_from_disk(  # type: ignore
        store_filepath, filter_key, map_location
    ):
        to_export[key] = tensor

    if fn_check_found_tensors is not None:
        fn_check_found_tensors(to_export)
    return export_tensors_to_nnef(to_export, output_dir)


def export_tensors_to_nnef(
    name_to_torch_tensors: T.Dict[str, _Tensor],
    output_dir: Path,
) -> T.Dict[str, _Tensor]:
    """Export any torch.Tensors list to NNEF .dat file.

    Args:
        name_to_torch_tensors: dict
            A map of name (that will be used to define .dat filename)
            and tensor values (that can also be special torch_to_nnef tensors)
        output_dir:
            directory to dump the NNEF tensor .dat files

    Returns:
        a dict of tensor name as key and torch.Tensor values,
            identical to `torch_to_nnef.export.export_tensors_to_nnef`

    Examples:
        Simple example

        >>> import tempfile
        >>> from torch import nn
        >>> class Mod(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.a = nn.Linear(1, 5)
        ...         self.b = nn.Linear(5, 1)
        ...
        ...     def forward(self, x):
        ...         return self.b(self.a(x))
        >>> mod = Mod()
        >>> nnef_dir = tempfile.mkdtemp(suffix="_nnef")
        >>> exported_tensors = export_tensors_to_nnef(
        ...     {k: v for k, v in mod.named_parameters() if k.startswith("b.")},
        ...     nnef_dir,
        ... )
        >>> list(exported_tensors.keys())
        ['b.weight', 'b.bias']
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    assert output_dir.exists(), output_dir
    for tensor_name, tensor in name_to_torch_tensors.items():
        if isinstance(tensor, (QTensor, OpaqueTensorRef)):
            if isinstance(tensor, OpaqueTensorRef):
                tensor = tensor.q_tensor
            tensor.write_in_file(output_dir, tensor_name)
        else:
            is_qtype = is_quantized_dtype(tensor.dtype)
            np_tensor = tensor.cpu().detach().numpy()
            if is_qtype:
                nnef_tensor = torch_qtensor_to_ntensor(
                    Graph(), tensor, tensor_name
                )
                if tensor.dtype == torch.quint8:
                    quant_filename = output_dir / "graph.quant"
                    with quant_filename.open("a", encoding="utf8") as fh:
                        write_tensor_quantization_infos(nnef_tensor, fh)
                else:
                    # NOTE: 2024-10-14: no engine support
                    # other torch built-in Q dtype
                    raise T2NErrorNotImplemented(tensor.dtype)
            filename = f"{tensor_name}.dat"
            write_nnef_tensor(
                np.asarray(np_tensor, order="C"),
                output_dir / filename,
                quantized=is_qtype,
            )
    return name_to_torch_tensors


@contextlib.contextmanager
def _unsupported_module_alerter(inference_target: InferenceTarget):
    """Temporarily raise for unsupported nn.utils.rnn utilities.

    Notes:
    - This performs a process-wide monkeypatch during export and restores
      originals on exit. It is not thread-safe; avoid concurrent exports.
    - The patching only applies when targeting TractNNEF.
    """

    class UnsupportedRaise:
        def __init__(self, msg) -> None:
            self.msg = msg

        def __call__(self, *args: T.Any, **kwds: T.Any) -> T.Any:
            raise T2NErrorNotImplemented(self.msg)

    orig_pack = None
    orig_pad = None
    did_patch_pack = False
    did_patch_pad = False

    if isinstance(inference_target, TractNNEF):
        # Patch pack_padded_sequence
        rnnmod = getattr(torch.nn.utils, "rnn", None)
        if rnnmod is not None and hasattr(rnnmod, "pack_padded_sequence"):
            orig_pack = rnnmod.pack_padded_sequence
            rnnmod.pack_padded_sequence = UnsupportedRaise(
                (
                    "'nn.utils.rnn.pack_padded_sequence' not supported by "
                    "tract yet. Contribution welcome."
                )
            )
            did_patch_pack = True
        # Patch pad_packed_sequence
        if rnnmod is not None and hasattr(rnnmod, "pad_packed_sequence"):
            orig_pad = rnnmod.pad_packed_sequence
            rnnmod.pad_packed_sequence = UnsupportedRaise(
                "'nn.utils.rnn.pad_packed_sequence' not supported by tract yet."
                " Contribution welcome."
            )
            did_patch_pad = True
    try:
        yield
    finally:
        if isinstance(inference_target, TractNNEF):
            rnnmod = getattr(torch.nn.utils, "rnn", None)
            if rnnmod is not None:
                if did_patch_pack and orig_pack is not None:
                    rnnmod.pack_padded_sequence = orig_pack
                if did_patch_pad and orig_pad is not None:
                    rnnmod.pad_packed_sequence = orig_pad


@contextlib.contextmanager
def _fixed_backend():
    """Controled backend in order to limit volatility of kernel selection.

    Useful in case of checks between PyTorch and targeted inference
    outputs.

    """
    if torch_version() >= "2.3.0":
        # pylint: disable-next=import-outside-toplevel
        from torch.nn.attention import SDPBackend, sdpa_kernel

        kwargs = {}
        if torch_version() >= "2.6.0":
            kwargs["set_priority"] = True

        with sdpa_kernel(
            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION], **kwargs
        ):
            yield None
    else:
        yield None
