from pathlib import Path
from copy import deepcopy
import subprocess

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.wrapper import TernaryPrimitive, TensorFnPrimitive
from tests.utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
)
from torch_to_nnef.inference_target.tract import TractCheckTolerance


FORCE_F32_INFERENCES = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX)
for inf in FORCE_F32_INFERENCES:
    inf.force_attention_inner_in_f32 = True
    inf.force_norm_in_f32 = True

attn_test_suite = TestSuiteInferenceExactnessBuilder(
    FORCE_F32_INFERENCES + TRACT_INFERENCES_TO_TESTS_APPROX
)


def set_inference_supper(inference_target):
    new_inference_target = deepcopy(inference_target)
    new_inference_target.check_io = False
    return new_inference_target


attn_test_suite.add(
    # q, k, v
    (
        torch.arange(12).reshape(1, 3, 4).half(),
        torch.arange(12).reshape(1, 3, 4).half(),
        torch.arange(12).reshape(1, 3, 4).half(),
    ),
    TernaryPrimitive(op=F.scaled_dot_product_attention),
)

bn_test_suite = TestSuiteInferenceExactnessBuilder(
    FORCE_F32_INFERENCES + TRACT_INFERENCES_TO_TESTS_APPROX
)
bn_test_suite.add(
    (torch.arange(12).reshape(1, 3, 4).half()),
    nn.BatchNorm1d(3),
)
bn_test_suite.add(
    (torch.arange(12).reshape(1, 3, 4).half()),
    TensorFnPrimitive("norm", kwargs=dict(p=2, dim=1, keepdim=True)),
)
gn = nn.GroupNorm(num_groups=3, num_channels=6, eps=0.0)
gn.requires_grad_ = False
gn.eval()
bn_test_suite.add(
    torch.arange(12).reshape(1, 6, 2).half(),
    gn,
)

try:
    from torch.nn.utils import weight_norm as wn

    bn_test_suite.add(
        torch.rand(1, 1, 5, 5).half(),
        wn(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=False,
            ).half(),
            dim=2,
        ),
        inference_modifier=set_inference_supper,
    )
except ImportError as exp:
    print("not yet weight_norm import:", exp)


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


def check_contains_f32_upcast_norm(inference_target, path):
    assert path.exists()
    graph_filename = "graph.nnef"
    subprocess.check_call(["tar", "-xzf", path, graph_filename])
    graph_filepath = Path(graph_filename)
    graph_content = graph_filepath.read_text()
    try:
        elms_to_be_found = ["to = 'f32'", "to = 'f16'"]
        if inference_target.force_norm_in_f32:
            assert all(
                elm in graph_content
                for elm in elms_to_be_found + ["tract_core_cast"]
            )
        else:
            assert not any(elm in graph_content for elm in elms_to_be_found)
    finally:
        graph_filepath.unlink()


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    bn_test_suite.test_samples,
    ids=bn_test_suite.ids,
)
def test_upcast_f32_bn(id, test_input, model, inference_target):
    """Test simple models"""
    check_model_io_test(
        model=model,
        test_input=test_input,
        inference_target=inference_target,
        callback=check_contains_f32_upcast_norm,
    )


def test_layer_norm_f16_unsupported_in_torch():
    """Check no layer norm support for f16"""
    with pytest.raises(RuntimeError) as excinfo:
        check_model_io_test(
            nn.LayerNorm(4, 3),
            test_input=(torch.arange(12).reshape(1, 3, 4).half()),
            inference_target=FORCE_F32_INFERENCES[0],
        )
    assert "\"LayerNormKernelImpl\" not implemented for 'Half'" in str(
        excinfo.value
    )
