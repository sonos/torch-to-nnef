from collections.abc import KeysView, ValuesView
import contextlib
import logging as log
import typing as T
from pathlib import Path

import numpy as np
import torch
from nnef_tools.model import Graph
from torch.onnx import TrainingMode  # type: ignore
from torch.onnx.utils import select_model_mode_for_export  # type: ignore

from torch_to_nnef.custom_nnef_writer import Writer as NNEFWriter
from torch_to_nnef.custom_nnef_writer import (
    write_nnef_tensor,
    write_tensor_quantization_infos,
)
from torch_to_nnef.dtypes import is_quantized_dtype
from torch_to_nnef.exceptions import (
    TorchToNNEFInvalidArgument,
    TorchToNNEFNotImplementedError,
)
from torch_to_nnef.inference_target import InferenceTarget
from torch_to_nnef.model_wrapper import may_wrap_model_to_flatten_io
from torch_to_nnef.nnef_graph import TorchToNGraphExtractor
from torch_to_nnef.op.fragment import FRAGMENTS, Fragment
from torch_to_nnef.op.quantized import torch_qtensor_to_ntensor
from torch_to_nnef.tensor import (
    set_opaque_tensor_in_params_as_ref,
    apply_name_to_tensor_in_module,
    QTensor,
    OpaqueTensorRef,
)
from torch_to_nnef.tensor.updater import ModTensorUpdater
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import dedup_list, torch_version

LOGGER = log.getLogger(__name__)


def export_model_to_nnef(
    model: torch.nn.Module,
    args,  # args pushed with *args in forward of module
    file_path_export: Path,
    inference_target: InferenceTarget,
    input_names: T.Optional[T.List[str]] = None,
    output_names: T.Optional[T.List[str]] = None,
    compression_level: int = 0,
    log_level: int = log.INFO,
    nnef_variable_naming_scheme: VariableNamingScheme = VariableNamingScheme.default(),
    check_io_names_qte_match: bool = True,
    debug_bundle_path: T.Optional[Path] = None,
    custom_extensions: T.Optional[T.List[str]] = None,
):
    """Main entrypoint of this library

    Export any torch.nn.Module to NNEF file format archive

    Args:
        model: an nn.Module that have a `.forward` function
            with only tensor arguments and outputs
            (no tuple, list, dict or objects)
            Only this function will be serialised

        args: a flat ordered list of tensors for each forward inputs of `model`
            this list can not be of dynamic size (at serialization it will be
            fixed to quantity of tensor provided)
            WARNING! tensor size in args will increase export time so take that
            in consideration for dynamic axes

        file_path_export: a Path to the exported NNEF serialized model archive.
            It must by convention end with `.nnef.tgz` suffixes

        inference_target: can be either `torch_to_nnef.TractNNEF` or `torch_to_nnef.KhronosNNEF`
            for each you can specify version targeted:
            - KhronosNNEF is least maintained so far, and is checked against nnef-tools PyTorch interpreter
            - TractNNEF is our main focus at SONOS, it is checked against tract inference engine
                among key paramters there is
                    feature_flags: Optional[Set[str]], that may contains tract specifics
                    dynamic_axes: Optional
                        By default the exported model will have the shapes of all input
                        and output tensors set to exactly match those given in args.
                        To specify axes of tensors as dynamic (i.e. known only at run-time)
                        set dynamic_axes to a dict with schema:
                            KEY (str): an input or output name. Each name must also
                                be provided in input_names or output_names.
                            VALUE (dict or list): If a dict, keys are axis indices
                                and values are axis names. If a list, each element is
                                an axis index.

        specific_tract_binary_path: Optional[Path] ideal to check io against new tract versions


        input_names: Optional list of names for args, it replaces
            variable inputs names traced from graph
            (if set it must have same size as number of args)

        output_names: Optional list of names for outputs of `model.forward`,
            it replaces variable output names traced from graph
            (if set it must have same size as number of outputs)

        compression_level: int (>= 0)
            compression level of tar.gz (higher is more compressed)

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
            information to allows easier debug.

        custom_extensions: Optional[List[str]]
            allow to add a set of extensions as defined in
            (https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html)
            Useful to set specific extensions like for example:
            'extension tract_assert S >= 0'
            those assertion allows to add limitation on dynamic shapes
            that are not expressed in traced graph
            (like for example maximum number of tokens for an LLM)
    """
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
        raise TorchToNNEFInvalidArgument(
            "custom extensions should be a list, "
            "because some extensions may be order sensitive (in tract)."
        )
    if isinstance(args, (torch.Tensor, int, float, bool, dict)) or (
        hasattr(args, "__getitem__")
        and hasattr(args, "items")
        and not isinstance(args, torch.Tensor)
    ):
        args = (args,)
    outs = model(*args)
    apply_name_to_tensor_in_module(model)
    if isinstance(outs, (torch.Tensor, int, float, bool, dict)) or (
        hasattr(args, "__getitem__")
        and hasattr(args, "items")
        and not isinstance(args, torch.Tensor)
    ):
        outs = (outs,)
    check_io_names(input_names, output_names)

    LOGGER.info(
        f"start parse PyTorch model to be exported at {file_path_export}"
    )
    if not any(s == ".nnef" for s in file_path_export.suffixes):
        raise TorchToNNEFInvalidArgument(
            "`file_path_export` should end with '.nnef' or '.nnef.tgz',"
            f" but found: {file_path_export.suffixes}"
        )
    with select_model_mode_for_export(model, TrainingMode.EVAL):
        set_opaque_tensor_in_params_as_ref(model)
        model, args, input_names, output_names = may_wrap_model_to_flatten_io(
            model, args, outs, input_names, output_names
        )
        inference_target.pre_trace(model, input_names, output_names)

        graph_extractor = TorchToNGraphExtractor(
            model,
            args,
            inference_target=inference_target,
            nnef_variable_naming_scheme=nnef_variable_naming_scheme,
            check_io_names_qte_match=check_io_names_qte_match,
            forced_inputs_names=input_names,
            forced_outputs_names=output_names,
        )
        nnef_graph = graph_extractor.parse()

        active_custom_extensions = get_active_custom_extensions(graph_extractor)
        inference_target.post_trace(nnef_graph, active_custom_extensions)
        if custom_extensions is not None:
            active_custom_extensions = dedup_list(
                active_custom_extensions + custom_extensions
            )

        active_custom_fragments = inference_target.specific_fragments(model)
        active_custom_fragments.update(
            get_active_custom_fragments(graph_extractor)
        )
        del graph_extractor
        nnef_exp_file_path = real_export_path(
            file_path_export, compression_level
        )

        NNEFWriter(
            compression=compression_level,
            fragments=active_custom_fragments,
            generate_custom_fragments=False,
            extensions=list(active_custom_extensions),
            version_custom_fragments=None,  # using version sometime create conflict with ops
            inference_target=inference_target,
        )(nnef_graph, str(nnef_exp_file_path))

        if len(active_custom_extensions) > 0:
            LOGGER.info(
                "The exported NNEF model need special custom extensions "
                f"such as {active_custom_extensions}, be sure "
                f"to use the inference engine you specified: {inference_target}"
            )
        LOGGER.info(
            f"model exported successfully as NNEF at: {nnef_exp_file_path}"
        )
        exported_filepath = file_path_export.parent / (
            nnef_exp_file_path.name + ".tgz"
        )
        with fixed_backend():
            inference_target.post_export(
                model,
                nnef_graph,
                args,
                exported_filepath,
                debug_bundle_path=debug_bundle_path,
            )
    mod_tensor_updater.restore_require_grad()


@contextlib.contextmanager
def fixed_backend():
    """Controled backend in order to limit volatility of kernel selection

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


def check_io_names(
    input_names: T.Optional[T.List[str]], output_names: T.Optional[T.List[str]]
):
    if input_names and len(set(input_names)) != len(input_names):
        raise TorchToNNEFInvalidArgument(
            "Each str in input_names must be different"
        )

    if output_names and len(set(output_names)) != len(output_names):
        raise TorchToNNEFInvalidArgument(
            "Each str in output_names must be different"
        )

    if (
        input_names
        and output_names
        and len(set(output_names + input_names))
        != len(input_names + output_names)
    ):
        raise TorchToNNEFInvalidArgument(
            "input_names and output_names must be different "
            "(else it could lead to wrong simplification of the graph)"
        )


def real_export_path(
    file_path_export: Path, compression_level: T.Optional[int] = None
) -> Path:
    nnef_exp_file_path = file_path_export
    if compression_level is not None:
        nnef_exp_file_path = Path(nnef_exp_file_path)
        if nnef_exp_file_path.suffix == ".tgz":
            nnef_exp_file_path = nnef_exp_file_path.with_suffix("")
    return nnef_exp_file_path


def get_active_custom_extensions(graph_extractor):
    return dedup_list(
        [
            ext
            for _ in graph_extractor.activated_custom_fragment_keys
            for ext in (FRAGMENTS[_] if isinstance(_, str) else _).extensions
        ]
    )


def get_active_custom_fragments(graph_extractor):
    active_custom_fragments = {}
    for _ in graph_extractor.activated_custom_fragment_keys:
        if isinstance(_, Fragment):
            active_custom_fragments[_.name] = _.definition
        else:
            active_custom_fragments[_] = FRAGMENTS[_].definition
    return active_custom_fragments


def set_lib_log_level(log_level):
    logger = log.getLogger("torch_to_nnef")
    logger.setLevel(log_level)


_Tensor = T.TypeVar("_Tensor", bound=torch.Tensor)


def _default_filter_key(key):
    return True


def iter_torch_tensors_from_disks(
    store_filepath, filter_key: T.Optional[T.Callable[[str], bool]] = None
) -> T.Iterator[T.Tuple[str, _Tensor]]:
    if filter_key is None:
        filter_key = _default_filter_key

    if store_filepath.name.endswith(".safetensors"):
        # pylint: disable-next=import-outside-toplevel
        from safetensors import safe_open

        with safe_open(store_filepath, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                if filter_key(key):
                    yield key, fh.get_tensor(key)
    elif any(store_filepath.name.endswith(_) for _ in [".pt", ".pth", ".bin"]):
        res = torch.load(store_filepath)
        if isinstance(res, torch.nn.Module):
            for key, tensor in res.named_parameters():
                if filter_key(key):
                    yield key, tensor
        elif hasattr(res, "items"):
            for key, tensor in res.items():
                if filter_key(key):
                    yield key, tensor
        else:
            raise TorchToNNEFNotImplementedError(type(res))


def export_tensors_from_disk_to_nnef(
    store_filepath: Path,  # either statedict or safetensors
    output_dir: Path,
    filter_key: T.Optional[T.Callable[[str], bool]] = None,
    fn_check_found_tensors: T.Optional[
        T.Callable[[T.Dict[str, _Tensor]], bool]
    ] = None,
):
    """Export any statedict or safetensors file torch.Tensors to NNEF .dat file

    Args:
        store_filepath:
            the filepath that hold the .safetensors , .pt or .bin containing the state dict
        output_dir:
            directory to dump the NNEF tensor .dat files
        fn_check_found_tensors:
            post checking function to ensure all requested tensors have effectively been dumped
    """
    to_export = {}
    for key, tensor in iter_torch_tensors_from_disks(  # type: ignore
        store_filepath, filter_key
    ):
        to_export[key] = tensor

    if fn_check_found_tensors is not None:
        fn_check_found_tensors(to_export)
    return export_tensors_to_nnef(to_export, output_dir)


def export_tensors_to_nnef(
    name_to_torch_tensors: T.Dict[str, _Tensor],
    output_dir: Path,
) -> T.Dict[str, _Tensor]:
    """Export any torch.Tensors list to NNEF .dat file"""
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
                    # NOTE: 2024-10-14: no engine support other torch built-in Q dtype
                    raise TorchToNNEFNotImplementedError(tensor.dtype)
            filename = f"{tensor_name}.dat"
            write_nnef_tensor(
                np.asarray(np_tensor, order="C"),
                output_dir / filename,
                quantized=is_qtype,
            )
    return name_to_torch_tensors
