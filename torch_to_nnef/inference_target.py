"""Targeted inference engine.

We mainly focus our effort to best support SONOS 'tract' inference engine.

Stricter Khronos NNEF specification mode also exist but is less extensively tested


"""

import enum
import logging as log
import typing as T
from pathlib import Path

import nnef
from nnef_tools.model import Graph as NGraph
from torch import nn
from torch.onnx.utils import _validate_dynamic_axes  # type: ignore

from torch_to_nnef import tract
from torch_to_nnef.exceptions import (
    DynamicShapeValue,
    TorchToNNEFNotImplementedError,
    TractError,
)
from torch_to_nnef.utils import SemanticVersion

LOGGER = log.getLogger(__name__)


class InferenceTarget:
    def __init__(
        self, version: T.Union[SemanticVersion, str], check_io: bool = False
    ):
        self.version = (
            SemanticVersion.from_str(version)
            if isinstance(version, str)
            else version
        )
        assert isinstance(self.version, SemanticVersion), self.version
        self.check_io = check_io

    @property
    def has_dynamic_axes(self) -> bool:
        return False

    def pre_trace(
        self,
        model: nn.Module,
        input_names: T.Optional[T.List[str]],
        output_names: T.Optional[T.List[str]],
    ):
        pass

    def post_trace(
        self, nnef_graph: NGraph, active_custom_extensions: T.Set[str]
    ):
        pass

    def post_export(
        self,
        model: nn.Module,
        nnef_graph: NGraph,
        args: T.List[T.Any],
        exported_filepath: Path,
        debug_bundle_path: T.Optional[Path] = None,
    ):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.version.to_str()}>"


class KhronosNNEF(InferenceTarget):
    """
    in case of check_io=True
        we perform evaluation against nnef_tool nnef to pytorch converter.
        And access original and reloaded pytorch model provide same outputs
    """

    LATEST_KNOWN_STABLE_VERSION = SemanticVersion.from_str("1.0.5")

    @classmethod
    def latest(cls):
        return cls(cls.LATEST_KNOWN_STABLE_VERSION)


class TractFeatureFlag(enum.Enum):
    DEFAULT = "default"
    COMPLEX = "complex"


class TractNNEF(InferenceTarget):
    LATEST_KNOWN_STABLE_VERSION = SemanticVersion.from_str("0.21.6")

    def __init__(
        self,
        version: T.Union[str, SemanticVersion],
        feature_flags: T.Optional[T.Set[str]] = None,
        check_io: bool = True,
        dynamic_axes: T.Optional[T.Dict[str, T.Dict[int, str]]] = None,
        specific_tract_binary_path: T.Optional[Path] = None,
    ):
        super().__init__(version, check_io)
        self.feature_flags = feature_flags or set()
        self.dynamic_axes = dynamic_axes or {}
        if self.feature_flags:
            LOGGER.info(f"use tract features flags: {self.feature_flags}")

        if specific_tract_binary_path is None:
            if self.feature_flags:
                raise TorchToNNEFNotImplementedError(
                    "feature_flags need specific_tract_binary_path provided"
                )
            tract_cli = tract.TractCli.download(self.version)
            # we can not check easily feature flags compat so it's left
        else:
            tract_cli = tract.TractCli(specific_tract_binary_path)
        LOGGER.info(f"use tract:{tract_cli.tract_path.absolute()}")
        self.tract_cli = tract_cli
        assert tract_cli.version == self.version

    @property
    def has_dynamic_axes(self) -> bool:
        return bool(self.dynamic_axes)

    @classmethod
    def latest(cls):
        return cls(cls.LATEST_KNOWN_STABLE_VERSION)

    def pre_trace(
        self,
        model: nn.Module,
        input_names: T.Optional[T.List[str]],
        output_names: T.Optional[T.List[str]],
    ):
        _validate_dynamic_axes(
            self.dynamic_axes, model, input_names, output_names
        )

    def post_trace(self, nnef_graph, active_custom_extensions):
        if self.dynamic_axes is not None:
            custom_extensions = apply_dynamic_shape_in_nnef(
                self.dynamic_axes, nnef_graph
            )
            active_custom_extensions.update(custom_extensions)

    def post_export(
        self,
        model: nn.Module,
        nnef_graph: NGraph,
        args: T.List[T.Any],
        exported_filepath: Path,
        debug_bundle_path: T.Optional[Path] = None,
    ):
        if self.check_io:
            # CHECK input and output are different
            input_names = [str(t.name) for t in nnef_graph.inputs]
            output_names = [str(t.name) for t in nnef_graph.outputs]
            _output_names = set(output_names)
            _input_names = set(input_names)
            if len(_output_names.difference(_input_names)) == 0:
                raise TractError(
                    "Tract does not support input passed as output without any transform: "
                    f"outputs={_output_names} inputs={_input_names}"
                )
            tract.assert_io_and_debug_bundle(
                model,
                args,
                exported_filepath,
                debug_bundle_path=debug_bundle_path,
                input_names=input_names,
                output_names=output_names,
                tract_cli=self.tract_cli,
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
