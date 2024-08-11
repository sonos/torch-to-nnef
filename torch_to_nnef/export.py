import logging as log
import os
import typing as T
from pathlib import Path

import nnef
import torch
from torch.onnx import TrainingMode  # type: ignore
from torch.onnx.utils import (  # type: ignore
    _decide_input_format,
    _validate_dynamic_axes,
    select_model_mode_for_export,
)

# from . import __version__
from torch_to_nnef import tract
from torch_to_nnef.custom_nnef_writer import Writer as NNEFWriter
from torch_to_nnef.exceptions import (
    DynamicShapeValue,
    StrictNNEFSpecError,
    TractError,
)
from torch_to_nnef.nnef_graph import TorchToNGraphExtractor
from torch_to_nnef.op.fragment import FRAGMENTS
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import torch_version

LOGGER = log.getLogger(__name__)


def apply_dynamic_shape_in_nnef(dynamic_axes, nnef_graph):
    custom_extensions = set()
    for node_name, named_dims in dynamic_axes.items():
        for inp_tensor in nnef_graph.inputs:
            if inp_tensor.name == node_name:
                LOGGER.debug("found matching node element")
                assert len(inp_tensor.producers) == 1
                external_op = inp_tensor.producers[0]
                assert external_op.type in [
                    "external",
                    "tract_core_external",
                ], external_op.type
                for axis, axis_name in named_dims.items():
                    if len(axis_name) != 1:
                        raise DynamicShapeValue(
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
                    if tract.tract_version() < "0.18.2":
                        custom_extensions.add("tract_pulse_streaming_symbol")
                    else:
                        custom_extensions.add(f"tract_symbol {axis_name}")
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
    renaming_scheme: VariableNamingScheme = VariableNamingScheme.default(),
    check_io_names_qte_match: bool = True,
    nnef_spec_strict: bool = False,
    # SONOS tract specific:
    debug_bundle_path: T.Optional[Path] = None,
    check_same_io_as_tract: bool = False,
    use_specific_tract_binary: T.Optional[Path] = None,
    tract_feature_flags: T.Optional[T.Set[str]] = None,
    custom_extensions: T.Optional[T.Set[str]] = None,
):
    """Main entrypoint of this library

    Export any torch.nn.Module to NNEF file format

    """
    logger = log.getLogger("torch_to_nnef")
    logger.setLevel(log_level)
    if nnef_spec_strict and check_same_io_as_tract:
        LOGGER.warning(
            "Activated `nnef_spec_strict=True` and `check_same_io_as_tract=True`"
            " but NNEF specification limits dynamic shape export."
            "You may be unable to run with it's full expressivity within tract."
        )
    if nnef_spec_strict and dynamic_axes:
        raise StrictNNEFSpecError(
            "NNEF spec does not allow dynamic_axes "
            "(use either dynamic_axes=None or set nnef_spec_strict=False)"
        )

    if use_specific_tract_binary is not None:
        assert use_specific_tract_binary.exists(), use_specific_tract_binary
        os.environ["TRACT_PATH"] = str(use_specific_tract_binary.absolute())
        LOGGER.info(f"use tract:{use_specific_tract_binary.absolute()}")

    if tract_feature_flags is None:
        tract_feature_flags = set()
    else:
        LOGGER.info(f"use tract features flags: {tract_feature_flags}")

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
            nnef_spec_strict=nnef_spec_strict,
            # has_dynamic_axes: bool alter export "operations"
            # if False to use fixed shape provided by input sample
            # Limitation: for now there is no proper tracing of dyn dim
            # within the computational graph
            # hence NO finegrain modification per axis referenced is applied
            has_dynamic_axes=bool(dynamic_axes),
            # specific feature flags from tract
            # by example 'complex'
            tract_feature_flags=tract_feature_flags,
            forced_inputs_names=input_names,
            forced_outputs_names=output_names,
        )
        nnef_graph = graph_extractor.parse()

        active_custom_fragments = {
            _: FRAGMENTS[_].definition
            for _ in graph_extractor.activated_custom_fragment_keys
        }
        active_custom_extensions = {
            ext
            for _ in graph_extractor.activated_custom_fragment_keys
            for ext in FRAGMENTS[_].extensions
        }
        if custom_extensions is not None:
            active_custom_extensions.update(custom_extensions)

        if dynamic_axes is not None:
            active_custom_extensions.update(
                apply_dynamic_shape_in_nnef(dynamic_axes, nnef_graph)
            )

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
            target_tract=not nnef_spec_strict,
        )(nnef_graph, str(nnef_exp_file_path))
        LOGGER.info(
            f"model exported successfully as NNEF at: {nnef_exp_file_path}"
        )
    if check_same_io_as_tract:
        # CHECK input and output are different
        _output_names = {str(t.name) for t in nnef_graph.outputs}
        _input_names = {str(t.name) for t in nnef_graph.inputs}
        if len(_output_names.difference(_input_names)) == 0:
            raise TractError(
                "Tract does not support input passed as output without transform: "
                f"outputs={_output_names} inputs={_input_names}"
            )
        tract.assert_io_and_debug_bundle(
            model,
            args,
            (file_path_export.parent / (nnef_exp_file_path.name + ".tgz")),
            debug_bundle_path=debug_bundle_path,
            input_names=input_names,
            output_names=output_names,
        )
