import logging as log
from pathlib import Path

import pytest
import torch
from nnef_tools.io.nnef.writer import tempfile

from torch_to_nnef.exceptions import StrictNNEFSpecError
from torch_to_nnef.export import export_model_to_nnef


def test_should_fail_compliance_NNEF_with_dyn_axes():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_input = torch.rand(1, 10, 100)
        model = torch.nn.Sequential(torch.nn.Conv1d(10, 20, 3))
        export_path = Path(tmpdir) / "model.nnef"

        with pytest.raises(StrictNNEFSpecError):
            export_model_to_nnef(
                model=model,
                args=test_input,
                file_path_export=export_path,
                input_names=["input"],
                output_names=["output"],
                log_level=log.WARNING,
                nnef_spec_strict=True,
                dynamic_axes={"input": {1: "S"}},
            )
