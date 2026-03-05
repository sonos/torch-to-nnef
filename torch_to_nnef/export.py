import contextlib
import logging as log
import typing as T
from collections.abc import KeysView, ValuesView
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
from torch_to_nnef.torch_graph.ir_naming import (
    DEFAULT_VARNAME_SCHEME,
    VariableNamingScheme,
)
from torch_to_nnef.utils import dedup_list, torch_version, ensure_tuple_io

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

    Returns:
        Path: the path to the exported artifact.
            - If `compression_level is None`: returns the `.nnef` directory path.
            - If `compression_level == 0`: returns the `.nnef.tar` archive path.
            - If `compression_level in 1..9`: returns the `.nnef.tgz` archive path.

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
        >>> export_model_to_nnef(
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
    if isinstance(args, ValuesView):
        args = tuple(args)
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
            "custom extensions should be a list, "
            "because some extensions may be order sensitive (in tract)."
        )
    args = ensure_tuple_io(args)

    with (
        select_model_mode_for_export(model, TrainingMode.EVAL),
        torch.no_grad(),
        torch.inference_mode(),
    ):
        outs = model(*args)
    apply_name_to_tensor_in_module(model)
    # Normalize single-output or mapping-like outputs into a tuple for
    # downstream processing.
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
        set_opaque_tensor_in_params_as_ref(model)
        # may unfold io structures {
        model_info = unfold_model_io(
            model, args, outs, input_names, output_names
        )
        input_names = model_info.input_names
        output_names = model_info.output_names
        # }
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

        active_custom_extensions = _get_active_custom_extensions(
            graph_extractor
        )
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
        nnef_exp_file_path = _real_export_path(
            file_path_export, compression_level
        )

        # NNEFWriter: using version sometime create conflict with ops
        # hence set to None
        # Decide archive format from user path and compression_level
        original_suffixes = file_path_export.suffixes
        wants_tgz = any(s == ".tgz" for s in original_suffixes)
        archive_format = None
        if compression_level is not None:
            if wants_tgz:
                archive_format = "tgz"
            else:
                archive_format = "tgz" if (compression_level and compression_level > 0) else "tar"

        NNEFWriter(
            compression=compression_level,
            fragments=active_custom_fragments,
            generate_custom_fragments=False,
            extensions=list(active_custom_extensions),
            version_custom_fragments=None,
            inference_target=inference_target,
            archive_format=archive_format,
        )(nnef_graph, str(nnef_exp_file_path))

        if len(active_custom_extensions) > 0:
            LOGGER.info(
                "The exported NNEF model need special custom extensions "
                "such as %s, be sure to use the inference engine "
                "you specified: %s",
                active_custom_extensions,
                inference_target,
            )
        LOGGER.info(
            "model exported successfully as NNEF at: %s", nnef_exp_file_path
        )
        if compression_level is not None:
            # Follow same format decision
            if archive_format == "tgz":
                suf = ".tgz"
            elif archive_format == "tar":
                suf = ".tar"
            else:
                suf = ".tgz" if (compression_level and compression_level > 0) else ".tar"
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
        with _fixed_backend():
            inference_target.post_export(
                model_info,
                nnef_graph,
                exported_filepath,
                debug_bundle_path=debug_bundle_path,
            )
    mod_tensor_updater.restore_require_grad()
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

    If the target path ends with `.tgz` (i.e., a user passed `.../model.nnef.tgz`),
    always treat it as the base `.../model.nnef` path for the writer, regardless
    of `compression_level`. This lets callers use the suffix to express intent
    for the final artifact format, while the writer always receives the base
    directory path.
    """
    nnef_exp_file_path = Path(file_path_export)
    # Strip only the last suffix if it's .tgz (e.g., model.nnef.tgz -> model.nnef)
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
                        "Skipping non-tensor entry from state dict: %s (type=%s)",
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
                "'nn.utils.rnn.pack_padded_sequence' not supported by tract yet."
                " Contribution welcome."
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
