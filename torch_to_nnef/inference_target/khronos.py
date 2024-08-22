import logging as log
import subprocess
import tempfile
import typing as T
from pathlib import Path

import torch
from nnef_tools.interpreter.pytorch import NNEFModule
from nnef_tools.model import Graph as NGraph
from torch import nn

from torch_to_nnef.exceptions import KhronosInterpreterDiffValueError
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.utils import SemanticVersion

LOGGER = log.getLogger(__name__)


class KhronosNNEF(InferenceTarget):
    """
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
        """Check io via the Torch interpreter of NNEF-Tools"""
        if self.check_io:
            with tempfile.TemporaryDirectory() as td:
                # reader decompression ill written in nnef-tools
                # so uncompress here to avoid issue
                subprocess.check_output(
                    f"cd {td} && tar -xvzf {exported_filepath}", shell=True
                )
                nnef_mod = NNEFModule(td)
                interpreter_outs = nnef_mod(*args)
                reference_outs = model(*args)
                if not isinstance(reference_outs, tuple):
                    reference_outs = (reference_outs,)
                for idx, (ref, obs) in enumerate(
                    zip(reference_outs, interpreter_outs)
                ):
                    if not torch.allclose(ref, obs, equal_nan=True):
                        raise KhronosInterpreterDiffValueError(
                            f"outputs[{idx}] is different expected:{ref} but got: {obs}"
                        )
