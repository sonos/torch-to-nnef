import typing as T
from pathlib import Path

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
from torch_to_nnef.nnef_graph import GraphExtractor
from torch_to_nnef.op.fragment import FRAGMENTS

LOGGER = log.getLogger(__name__)


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

        graph_extractor = GraphExtractor(
            model, args, renaming_scheme=renaming_scheme
        )
        nnef_graph = graph_extractor.parse(
            input_names,
            output_names,
        )

        active_custom_fragments = {
            _: FRAGMENTS[_]
            for _ in graph_extractor.activated_custom_fragment_keys
        }

        NNEFWriter(
            compression=compression_level,
            fragments=active_custom_fragments,
            generate_custom_fragments=len(active_custom_fragments) > 0,
            # could be better integrated by exposed extensions deps in active_custom_fragments
            extensions=["tract_registry tract_core"]
            if len(active_custom_fragments) > 0
            else [],
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
