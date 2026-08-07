import pytest
import torch

from tests.utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


class InPlaceCopyIntoZerosBuffer(torch.nn.Module):
    """In-place copy_ into a lazily created zeros buffer.

    This is the exact pattern of HF transformers linear-attention caches
    (`LinearAttentionLayer.lazy_initialization` + `update_conv_state` /
    `update_recurrent_state`): a `torch.zeros` buffer is created, then
    `copy_`d into, then read back. The traced value flowing out of
    `aten::copy_` must be the SOURCE, not the destination's pre-write
    zeros.
    """

    def forward(self, x):
        buf = torch.zeros(1, 3)
        buf.copy_(x[:, -3:])
        y = buf * 2.0
        state = torch.zeros_like(x[:, :1])
        state.copy_(y[:, :1] + 1.0)
        return y, state


class InPlaceCopyWithDtypeCast(torch.nn.Module):
    """`copy_` casts the source to the destination dtype."""

    def forward(self, x):
        buf = torch.zeros(1, 3, dtype=torch.float16)
        buf.copy_(x[:, -3:])
        return buf.float() * 2.0


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_copy_inplace_into_zeros_buffer(inference_target):
    inp = torch.arange(6.0).reshape(1, 6)
    check_model_io_test(
        model=InPlaceCopyIntoZerosBuffer(),
        test_input=inp,
        inference_target=inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_copy_inplace_with_dtype_cast(inference_target):
    inp = torch.arange(6.0).reshape(1, 6)
    check_model_io_test(
        model=InPlaceCopyWithDtypeCast(),
        test_input=inp,
        inference_target=inference_target,
    )
