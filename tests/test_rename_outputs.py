import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_to_nnef.exceptions import T2NErrorInvalidArgument
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.nemo_tract.wrappers import RenameOutputs


class _IdentityWithNames(nn.Module):
    def __init__(self, input_names, output_names):
        super().__init__()
        self.input_names = list(input_names)
        self.output_names = list(output_names)

    def forward(self, x):  # noqa: D401
        return x


def test_rename_outputs_wrapper_names():
    base = _IdentityWithNames(["length"], ["length", "other"])
    wrap = RenameOutputs(base, {"length": "length_out"})
    assert wrap.input_names == ["length"]
    assert wrap.output_names == ["length_out", "other"]


def test_export_with_renamed_outputs_succeeds():
    m = _IdentityWithNames(["length"], ["length"]).eval()
    x = torch.ones(1, 1)

    # Without renaming, export should fail due to IO name collision
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "model.nnef"
        with pytest.raises(T2NErrorInvalidArgument):
            export_model_to_nnef(
                model=m,
                args=x,
                file_path_export=out,
                input_names=m.input_names,
                output_names=m.output_names,
                inference_target=TractNNEF.latest(),
                compression_level=0,
            )

    # With RenameOutputs, export should succeed
    rm = RenameOutputs(m, {"length": "length_out"}).eval()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "model.nnef"
        exported = export_model_to_nnef(
            model=rm,
            args=x,
            file_path_export=out,
            input_names=rm.input_names,
            output_names=rm.output_names,
            inference_target=TractNNEF.latest(),
            compression_level=0,
        )
        assert exported.exists()
