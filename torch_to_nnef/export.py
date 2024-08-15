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

from torch_to_nnef import tract
from torch_to_nnef.custom_nnef_writer import Writer as NNEFWriter
from torch_to_nnef.exceptions import (
    DynamicShapeValue,
    StrictNNEFSpecError,
    TorchToNNEFInvalidArgument,
)
from torch_to_nnef.model_wrapper import may_wrap_model_to_flatten_io
from torch_to_nnef.nnef_graph import TorchToNGraphExtractor
from torch_to_nnef.op.fragment import FRAGMENTS
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import flatten_tuple_or_list_with_idx, torch_version

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

        input_names: Optional list of names for args, it replaces
            variable inputs names traced from graph
            (if set it must have same size as number of args)

        output_names: Optional list of names for outputs of `model.forward`,
            it replaces variable output names traced from graph
            (if set it must have same size as number of outputs)

        dynamic_axes: Optional (only possible if `nnef_spec_strict` is False).
            By default the exported model will have the shapes of all input
            and output tensors set to exactly match those given in args.
            To specify axes of tensors as dynamic (i.e. known only at run-time)
            set dynamic_axes to a dict with schema:
                KEY (str): an input or output name. Each name must also
                    be provided in input_names or output_names.

                VALUE (dict or list): If a dict, keys are axis indices
                    and values are axis names. If a list, each element is
                    an axis index.

        compression_level: int (>= 0)
            compression level of tar.gz (higher is more compressed)

        log_level: int,
            logger level for `torch_to_nnef` following Python
            standard logging level can be set to:
            INFO, WARN, DEBUG ...

        renaming_scheme:
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

        nnef_spec_strict: bool (default: False)
            If False we can use ["tract"](https://github.com/sonos/tract/)
            specific operator set that extend strict NNEF specification
            else we restrict ourselves to NNEF 1.0.5
            tract version used is the one in your $PATH cli or if specified
            $TRACT_PATH

        debug_bundle_path: Optional[Path]
            if specified it should create an archive bundle with all needed
            information to allows easier debug.

        check_same_io_as_tract: bool
            check if given provided `args` we get same outputs in PyTorch
            and in tract

        use_specific_tract_binary: Optional[Path]
            Use a specific tract binary based on provided Path (
                It overwrite $PATH / $TRACT_PATH if set
            )

        tract_feature_flags: Optional[Set[str]]
            tract offer some feature flags such as: 'complex'

        custom_extensions: Optional[Set[str]]
            allow to add a set of extensions as defined in
            (https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html)
            Useful to set specific extensions like for example:
            'extension tract_assert S >= 0'
            those assertion allows to add limitation on dynamic shapes
            that are not expressed in traced graph
            (like for example maximum number of tokens for an LLM)
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
    if isinstance(args, (torch.Tensor, int, float, bool)):
        args = (args,)
    outs = model(*args)
    if isinstance(outs, (torch.Tensor, int, float, bool)):
        outs = (outs,)
    check_io_types(args, outs)
    check_io_names(input_names, output_names)

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
    if not any(s == ".nnef" for s in file_path_export.suffixes):
        raise TorchToNNEFInvalidArgument(
            "`file_path_export` should end with '.nnef' or '.nnef.tgz',"
            f" but found: {file_path_export.suffixes}"
        )
    with select_model_mode_for_export(model, TrainingMode.EVAL):
        if "1.8.0" <= torch_version() < "1.12.0":
            # change starting in 1.12.0 for recurent layers make it unsuitable
            args = _decide_input_format(model, args)
        if dynamic_axes is None:
            dynamic_axes = {}
        _validate_dynamic_axes(dynamic_axes, model, input_names, output_names)

        model, args, input_names, output_names = may_wrap_model_to_flatten_io(
            model, args, outs, input_names, output_names
        )

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
            LOGGER.info(
                "The exported NNEF model need special custom extensions "
                f"such as {active_custom_extensions} be sure "
                "to use an inference engine that support them"
            )
        custom_fragment_names = list(active_custom_fragments.keys())
        nnef_exp_file_path = real_export_path(
            file_path_export, compression_level
        )

        NNEFWriter(
            compression=compression_level,
            fragments=active_custom_fragments,
            fragment_dependencies={
                # this trick ensure all requested fragment are exported
                _: custom_fragment_names
                for _ in custom_fragment_names
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
        tract.assert_io_and_debug_bundle(
            model,
            args,
            (file_path_export.parent / (nnef_exp_file_path.name + ".tgz")),
            debug_bundle_path=debug_bundle_path,
            input_names=input_names,
            output_names=output_names,
        )


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


PRIMITIVE_IO_TYPES = (torch.Tensor,)
SUPPORTED_IO_TYPES = PRIMITIVE_IO_TYPES + (tuple, list)


def check_io_types(args, outs):
    for ix, a in enumerate(args):
        if isinstance(a, PRIMITIVE_IO_TYPES) or (
            isinstance(a, (tuple, list))
            and all(
                isinstance(ax, torch.Tensor)
                for _, ax in flatten_tuple_or_list_with_idx(a)
            )
        ):
            continue
        raise TorchToNNEFInvalidArgument(
            f"Provided args[{ix}] is of type {type(a)}"
            f" but only {SUPPORTED_IO_TYPES} is supported."
            " (you can use a wrapper module to comply to this rule.)"
        )
    for ix, o in enumerate(outs):
        if isinstance(o, PRIMITIVE_IO_TYPES) or (
            isinstance(o, (tuple, list))
            and all(
                isinstance(ox, torch.Tensor)
                for _, ox in flatten_tuple_or_list_with_idx(o)
            )
        ):
            continue
        raise TorchToNNEFInvalidArgument(
            f"Obtained model outputs[{ix}] is of type {type(o)}"
            f" but only {SUPPORTED_IO_TYPES} are supported."
            " (you can use a wrapper module to comply to this rule.)"
        )


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
