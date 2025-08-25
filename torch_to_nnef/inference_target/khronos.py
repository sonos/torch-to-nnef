import json
import logging as log
import shutil
import subprocess
import tempfile
import typing as T
from pathlib import Path

import torch
from nnef_tools.interpreter.pytorch import NNEFModule
from nnef_tools.model import Graph as NGraph
from torch import nn

from torch_to_nnef.exceptions import (
    T2NErrorKhronosInterpreterDiffValue,
    T2NErrorKhronosNNEFModuleLoad,
)
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.utils import SemanticVersion, cd

LOGGER = log.getLogger(__name__)


class KhronosNNEF(InferenceTarget):
    """Khronos Specification compliant NNEF asset build.

    in case of check_io=True
        we perform evaluation against nnef_tool nnef to pytorch converter.
        And access original and reloaded pytorch model provide same outputs

    """

    OFFICIAL_SUPPORTED_VERSIONS = [
        SemanticVersion.from_str(version)
        for version in [
            "1.0.5",
        ]
    ]

    @classmethod
    def latest(cls):
        return cls(cls.OFFICIAL_SUPPORTED_VERSIONS[0])

    def __init__(
        self, version: T.Union[SemanticVersion, str], check_io: bool = True
    ):
        super().__init__(version, check_io)

    def post_export(
        self,
        model: nn.Module,
        nnef_graph: NGraph,
        args: T.List[T.Any],
        exported_filepath: Path,
        debug_bundle_path: T.Optional[Path] = None,
    ):
        """Check io via the Torch interpreter of NNEF-Tools."""
        if self.check_io:
            with tempfile.TemporaryDirectory() as td:
                # reader decompression ill written in nnef-tools
                # so uncompress here to avoid issue
                with cd(td):
                    subprocess.check_output(
                        ["tar", "-xzf", str(exported_filepath)]
                    )
                try:
                    nnef_mod = NNEFModule(td)
                except Exception as exp:
                    self._maybe_dump_debug_bundle(
                        debug_bundle_path, td, exported_filepath
                    )
                    raise T2NErrorKhronosNNEFModuleLoad(
                        "unable to instanciate NNEFModule"
                    ) from exp
                interpreter_outs = nnef_mod(*args)
                reference_outs = model(*args)
                if not isinstance(reference_outs, tuple):
                    reference_outs = (reference_outs,)
                for idx, (ref, obs) in enumerate(
                    zip(reference_outs, interpreter_outs)
                ):
                    if not torch.allclose(ref, obs, equal_nan=True):
                        self._maybe_dump_debug_bundle(
                            debug_bundle_path, td, exported_filepath
                        )
                        raise T2NErrorKhronosInterpreterDiffValue(
                            f"outputs[{idx}] is different "
                            f"expected:{ref} but got: {obs}"
                        )

    def _maybe_dump_debug_bundle(
        self,
        debug_bundle_path: T.Optional[Path],
        td: str,
        exported_filepath: Path,
    ):
        if debug_bundle_path:
            debug_bundle_path.mkdir(exist_ok=True, parents=True)
            with (debug_bundle_path / "engine.json").open(
                "w", encoding="utf8"
            ) as fh:
                json.dump(
                    {
                        "inference_target": str(self.__class__.__name__),
                        "inference_version": self.version.to_str(),
                    },
                    fh,
                )
            shutil.copytree(td, debug_bundle_path / "nnef")
            shutil.copy(exported_filepath, debug_bundle_path / "nnef")
