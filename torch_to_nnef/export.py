import typing as T
from pathlib import Path

import nnef
import torch
from nnef_tools.io.nnef.writer import Writer as NNEFWriter
from torch.onnx import TrainingMode  # type: ignore
from torch.onnx.utils import (  # type: ignore
    _decide_input_format,
    _validate_dynamic_axes,
    select_model_mode_for_export,
)

# from . import __version__
from torch_to_nnef import tract
from torch_to_nnef.log import log
from torch_to_nnef.nnef_graph import TorchToNGraphExtractor
from torch_to_nnef.op.fragment import FRAGMENTS

LOGGER = log.getLogger(__name__)


def apply_dynamic_shape_in_nnef(dynamic_axes, nnef_graph):
    custom_extensions = set()
    for node_name, named_dims in dynamic_axes.items():
        for inp_tensor in nnef_graph.inputs:
            if inp_tensor.name == node_name:
                LOGGER.debug("found matching node element")
                assert len(inp_tensor.producers) == 1
                external_op = inp_tensor.producers[0]
                assert external_op.type == "external"
                for axis, axis_name in named_dims.items():
                    if len(axis_name) != 1:
                        raise ValueError(
                            "axis_name in dynamic_axes must "
                            "be of length 1 to follow tract convention "
                            f"but was given '{axis_name}' "
                            f"in dynamic_axes={dynamic_axes}"
                        )
                    external_op.attribs["shape"] = [
                        nnef.Identifier(str(axis_name))
                        if idx == axis
                        else dim_size
                        for idx, dim_size in enumerate(
                            external_op.attribs["shape"]
                        )
                    ]
                    custom_extensions.add("tract_pulse_streaming_symbol")
                break
    return custom_extensions


def export_model_to_nnef(
    model: torch.nn.Module,
    args,  # args pushed with *args in forward of module
    file_path_export: Path,
    input_names: T.Optional[T.List[str]],
    output_names: T.Optional[T.List[str]],
    dynamic_axes=None,
    compression_level: int = 0,
    log_level: int = log.INFO,
    check_same_io_as_tract: bool = False,
    debug_bundle_path: T.Optional[Path] = None,
    renaming_scheme: str = "numeric",
    check_io_names_qte_match: bool = True,
):
    """Main entrypoint of this library

    Export any torch.nn.Module to NNEF file format

    """
    logger = log.getLogger("torch_to_nnef")
    logger.setLevel(log_level)
    LOGGER.info(
        f"start parse Pytorch model to be exported at {file_path_export}"
    )
    assert file_path_export.with_suffix(
        ".nnef"
    ), "export filepath should end with '.nnef'"
    with select_model_mode_for_export(model, TrainingMode.EVAL):
        args = _decide_input_format(model, args)
        if dynamic_axes is None:
            dynamic_axes = {}
        _validate_dynamic_axes(dynamic_axes, model, input_names, output_names)
        if isinstance(args, (torch.Tensor, int, float, bool)):
            args = (args,)

        graph_extractor = TorchToNGraphExtractor(
            model,
            args,
            renaming_scheme=renaming_scheme,
            check_io_names_qte_match=check_io_names_qte_match,
        )
        nnef_graph = graph_extractor.parse(
            input_names,
            output_names,
        )

        active_custom_fragments = {
            _: FRAGMENTS[_].definition
            for _ in graph_extractor.activated_custom_fragment_keys
        }
        active_custom_extensions = {
            ext
            for _ in graph_extractor.activated_custom_fragment_keys
            for ext in FRAGMENTS[_].extensions
        }
        if dynamic_axes is not None:
            custom_extensions = apply_dynamic_shape_in_nnef(
                dynamic_axes, nnef_graph
            )
            active_custom_extensions.update(custom_extensions)

        if len(active_custom_extensions) > 0:
            LOGGER.warning(
                "The exported NNEF model need special custom extensions "
                f"such as {active_custom_extensions} be sure "
                "to use an inference engine that support them"
            )

        NNEFWriter(
            compression=compression_level,
            fragments=active_custom_fragments,
            generate_custom_fragments=len(active_custom_fragments) > 0,
            extensions=list(active_custom_extensions),
            version_custom_fragments=None,  # using version sometime create conflict with ops
        )(nnef_graph, str(file_path_export))
        LOGGER.info(
            f"model exported successfully as NNEF at: {file_path_export}"
        )
    if check_same_io_as_tract:
        tract.assert_io_and_debug_bundle(
            model,
            args,
            file_path_export.with_suffix(".nnef.tgz"),
            debug_bundle_path=debug_bundle_path,
            input_names=input_names,
            output_names=output_names,
        )
