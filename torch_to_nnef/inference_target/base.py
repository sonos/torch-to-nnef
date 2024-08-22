import typing as T
from pathlib import Path

from nnef_tools.model import Graph as NGraph
from torch import nn

from torch_to_nnef.utils import SemanticVersion


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
