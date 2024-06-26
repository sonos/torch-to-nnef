import logging as log
import typing as T
from pathlib import Path

import torch
from torch.onnx import TrainingMode  # type: ignore
from torch.onnx.utils import (  # type: ignore
    _decide_input_format,
    select_model_mode_for_export,
)

from torch_to_nnef.custom_nnef_writer import Writer as NNEFWriter
from torch_to_nnef.inference_target import InferenceTarget
from torch_to_nnef.nnef_graph import TorchToNGraphExtractor
from torch_to_nnef.op.fragment import FRAGMENTS
from torch_to_nnef.utils import torch_version

LOGGER = log.getLogger(__name__)


def export_model_to_nnef(
    model: torch.nn.Module,
    args,  # args pushed with *args in forward of module
    file_path_export: Path,
    inference_target: InferenceTarget,
    input_names: T.Optional[T.List[str]],
    output_names: T.Optional[T.List[str]],
    compression_level: int = 0,
    log_level: int = log.INFO,
    renaming_scheme: str = "numeric",
    check_io_names_qte_match: bool = True,
    # SONOS tract specific:
    debug_bundle_path: T.Optional[Path] = None,
):
    """Main entrypoint of this library

    Export any torch.nn.Module to NNEF file format

    """
    logger = log.getLogger("torch_to_nnef")
    logger.setLevel(log_level)

    LOGGER.info(
        f"start parse Pytorch model to be exported at {file_path_export}"
    )
    assert any(
        s == ".nnef" for s in file_path_export.suffixes
    ), f"export filepath should end with '.nnef' or '.nnef.tgz', but found: {file_path_export.suffixes}"
    with select_model_mode_for_export(model, TrainingMode.EVAL):
        if "1.8.0" <= torch_version() < "1.12.0":
            # change starting in 1.12.0 for recurent layers make it unsuitable
            args = _decide_input_format(model, args)
        inference_target.pre_trace(model, input_names, output_names)
        if isinstance(args, (torch.Tensor, int, float, bool)):
            args = (args,)

        graph_extractor = TorchToNGraphExtractor(
            model,
            args,
            inference_target=inference_target,
            renaming_scheme=renaming_scheme,
            check_io_names_qte_match=check_io_names_qte_match,
        )
        nnef_graph = graph_extractor.parse(input_names, output_names)

        active_custom_fragments = {
            _: FRAGMENTS[_].definition
            for _ in graph_extractor.activated_custom_fragment_keys
        }
        active_custom_extensions = {
            ext
            for _ in graph_extractor.activated_custom_fragment_keys
            for ext in FRAGMENTS[_].extensions
        }
        inference_target.post_trace(nnef_graph, active_custom_extensions)

        if len(active_custom_extensions) > 0:
            LOGGER.warning(
                "The exported NNEF model need special custom extensions "
                f"such as {active_custom_extensions} be sure "
                "to use an inference engine that support them"
            )
        custom_framgnent_names = list(active_custom_fragments.keys())
        nnef_exp_file_path = file_path_export
        if compression_level is not None:
            nnef_exp_file_path = Path(nnef_exp_file_path)
            if nnef_exp_file_path.suffix == ".tgz":
                nnef_exp_file_path = nnef_exp_file_path.with_suffix("")

        NNEFWriter(
            compression=compression_level,
            fragments=active_custom_fragments,
            fragment_dependencies={
                # this trick ensure all requested fragment are exported
                _: custom_framgnent_names
                for _ in custom_framgnent_names
            },
            generate_custom_fragments=False,
            extensions=list(active_custom_extensions),
            version_custom_fragments=None,  # using version sometime create conflict with ops
        )(nnef_graph, str(nnef_exp_file_path))
        LOGGER.info(
            f"model exported successfully as NNEF at: {nnef_exp_file_path}"
        )
        exported_filepath = file_path_export.parent / (
            nnef_exp_file_path.name + ".tgz"
        )
        inference_target.post_export(
            model,
            nnef_graph,
            args,
            exported_filepath,
            debug_bundle_path,
        )
