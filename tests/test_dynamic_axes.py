"""Tests dynamic_axes."""
import os

import pytest
import torch
from torchaudio import models as audio_mdl

from .utils import check_model_io_test, set_seed  # noqa: E402

set_seed(int(os.environ.get("SEED", 25)))

INPUT_AND_MODELS = [
    (torch.rand(1, 1, 100, 64), model)
    for model in [
        audio_mdl.DeepSpeech(64, n_hidden=256),
    ]
]


@pytest.mark.parametrize("test_input,model", INPUT_AND_MODELS)
def test_tricky_export(test_input, model):
    """Test simple models"""
    check_model_io_test(
        model=model, test_input=test_input, dynamic_axes={"input_0": {2: "S"}}
    )
