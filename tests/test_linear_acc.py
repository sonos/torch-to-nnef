"""Tests simple accumulator option."""

import os
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 25)))


def _f16_cpu_conv_supported() -> bool:
    try:
        with torch.inference_mode():
            nn.Conv2d(1, 1, 1).half()(torch.zeros(1, 1, 2, 2).half())
        return True
    except RuntimeError:
        return False


_F16_CONV_CPU_OK = _f16_cpu_conv_supported()

tract_latest = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX[0])
tract_latest.force_linear_accumulation_in_f32 = True

test_suite = TestSuiteInferenceExactnessBuilder([tract_latest])

if tract_latest.version >= "0.21.11":
    mod = nn.Linear(3, 4)
    mod = mod.half()
    inp = torch.arange(6).reshape(1, 2, 3).half()
    try:
        with torch.inference_mode():
            mod(inp)
        test_suite.add(inp, mod)
    except RuntimeError as exp:
        print("failed to add the test because torch:", exp)

    # `conv` shares the `force_linear_accumulation_in_f32` option with
    # `linear`: an fp16 conv accumulator diverges from PyTorch's CPU f16
    # kernels (which accumulate in f32), so we upcast the conv to f32 and
    # cast the result back. Cover the plain conv and the transposed (deconv)
    # paths, both reached through `aten::_convolution`. Old torch (e.g. 1.13)
    # lacks CPU f16 conv, so guard like the linear case above.
    conv_inp = torch.randn(1, 4, 8, 8).half()
    for conv_mod in (
        nn.Conv2d(4, 8, kernel_size=3, padding=1).half(),
        nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2).half(),
    ):
        try:
            with torch.inference_mode():
                conv_mod(conv_inp)
            test_suite.add(conv_inp, conv_mod)
        except RuntimeError as exp:
            print("failed to add the conv test because torch:", exp)


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_linear_accumulate_f32_export(id, test_input, model, inference_target):
    """Test simple aten PyTorch core."""
    check_model_io_test(
        model=model, test_input=test_input, inference_target=inference_target
    )


def _read_graph_nnef_from_archive(path: Path) -> str:
    with tarfile.open(path, "r:*") as tf:
        for member in tf.getmembers():
            if member.name.endswith("graph.nnef"):
                f = tf.extractfile(member)
                assert f is not None
                return f.read().decode("utf-8")
    raise AssertionError("graph.nnef not found in exported archive")


def _check_conv_upcast_emitted(inference_target, path):
    graph_content = _read_graph_nnef_from_archive(path)
    sandwich = ["tract_core_cast", "to = 'f32'", "to = 'f16'"]
    if inference_target.force_linear_accumulation_in_f32:
        assert all(elm in graph_content for elm in sandwich), graph_content
    else:
        assert not any(elm in graph_content for elm in sandwich)


@pytest.mark.skipif(
    condition=tract_latest.version < "0.21.11" or not _F16_CONV_CPU_OK,
    reason="conv f32 accum needs tract>=0.21.11 and CPU f16 conv",
)
@pytest.mark.parametrize("force_f32", [True, False])
def test_conv_accumulate_f32_emits_cast_sandwich(force_f32):
    """The f32 sandwich is emitted only when the option is set."""
    inference_target = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX[0])
    inference_target.force_linear_accumulation_in_f32 = force_f32
    # Without the upcast an fp16 conv legitimately diverges from tract, so only
    # numerically check the f32 path; the flag-off case just inspects the graph.
    inference_target.check_io = force_f32
    conv = nn.Conv2d(4, 8, kernel_size=3, padding=1).half()
    conv_inp = torch.randn(1, 4, 8, 8).half()
    check_model_io_test(
        model=conv,
        test_input=conv_inp,
        inference_target=inference_target,
        callback_post_export=_check_conv_upcast_emitted,
    )
