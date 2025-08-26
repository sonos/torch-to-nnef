import logging
import typing as T
from pathlib import Path

from nnef_tools.model import Graph as NGraph
from torch import nn

from torch_to_nnef.utils import SemanticVersion

LOGGER = logging.getLogger(__name__)


class InferenceTarget:
    """Base abstract class to implement a new inference engine target."""

    # each implementation should specify
    OFFICIAL_SUPPORTED_VERSIONS: T.List[SemanticVersion] = []

    def __init__(
        self, version: T.Union[SemanticVersion, str], check_io: bool = False
    ):
        """Init InferenceTarget.

        Each inference engine is supposed to have at least a version
        and a way to check output given an input.

        """
        self.version = (
            SemanticVersion.from_str(version)
            if isinstance(version, str)
            else version
        )
        assert isinstance(self.version, SemanticVersion), self.version
        newest_supported = self.OFFICIAL_SUPPORTED_VERSIONS[0]
        if self.version > newest_supported:
            LOGGER.warning(
                "`torch_to_nnef` maintainers did not tests "
                "inference target '%s' "
                "beyond '%s', "
                "but you requested upper version: '%s', "
                "some features may be missing",
                self.__class__.__name__,
                newest_supported.to_str(),
                self.version.to_str(),
            )
        oldest_supported = self.OFFICIAL_SUPPORTED_VERSIONS[-1]
        if self.version < oldest_supported:
            LOGGER.warning(
                "`torch_to_nnef` maintainers do not tests (anymore) "
                "inference target '%s' beyond '%s', "
                "but you requested lower version: '%s', ",
                self.__class__.__name__,
                oldest_supported.to_str(),
                self.version.to_str(),
            )
        self.check_io = check_io

    @property
    def has_dynamic_axes(self) -> bool:
        """Define if user request dynamic axes to be in the NNEF graph.

        Some inference engines may not support it hence False by default.
        """
        return False

    def specific_fragments(self, model: nn.Module) -> T.Dict[str, str]:
        """Optional custom fragments to pass."""
        return {}

    def pre_trace(
        self,
        model: nn.Module,
        input_names: T.Optional[T.List[str]],
        output_names: T.Optional[T.List[str]],
    ):
        """Get called just before PyTorch graph is traced.

        (after auto wrapper)
        """

    def post_trace(
        self, nnef_graph: NGraph, active_custom_extensions: T.List[str]
    ):
        """Get called just after PyTorch graph is parsed."""

    def post_export(
        self,
        model: nn.Module,
        nnef_graph: NGraph,
        args: T.List[T.Any],
        exported_filepath: Path,
        debug_bundle_path: T.Optional[Path] = None,
    ):
        """Get called after NNEF model asset is generated.

        This is typically where check_io is effectively applied.
        """

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.version.to_str()}>"
