from pathlib import Path
from copy import deepcopy
import subprocess

import pytest
import torch
from torch.nn import functional as F

from tests.wrapper import TernaryPrimitive
from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
)


FORCE_F32_INFERENCES = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX)
for inf in FORCE_F32_INFERENCES:
    inf.force_attention_inner_in_f32 = True

attn_test_suite = TestSuiteInferenceExactnessBuilder(
    FORCE_F32_INFERENCES + TRACT_INFERENCES_TO_TESTS_APPROX
)


attn_test_suite.add(
    # q, k, v
    (
        torch.arange(12).reshape(1, 3, 4).half(),
        torch.arange(12).reshape(1, 3, 4).half(),
        torch.arange(12).reshape(1, 3, 4).half(),
    ),
    TernaryPrimitive(op=F.scaled_dot_product_attention),
)


def check_contains_f32_upcast_attn(inference_target, path):
    assert path.exists()
    graph_filename = "graph.nnef"
    subprocess.check_call(["tar", "-xzf", path, graph_filename])
    graph_filepath = Path(graph_filename)
    graph_content = graph_filepath.read_text()
    try:
        if inference_target.force_attention_inner_in_f32:
            assert (
                "fragment scaled_dot_product_attention_3d_f16_df32("
                in graph_content
            )
        else:
            assert (
                "fragment scaled_dot_product_attention_3d_f16(" in graph_content
            )
    finally:
        graph_filepath.unlink()


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    attn_test_suite.test_samples,
    ids=attn_test_suite.ids,
)
def test_upcast_f32_attn(id, test_input, model, inference_target):
    """Test simple models"""
    check_model_io_test(
        model=model,
        test_input=test_input,
        inference_target=inference_target,
        callback=check_contains_f32_upcast_attn,
    )
