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


def _skip_if_scatter_declutter_bug(inference_target):
    """Skip when tract cannot declutter the scatter these graphs emit.

    The `*_scatter` write lowers to a sliced assignment whose declutter trips
    tract's own PushSliceUp pass before 0.23 (`Error at stage "declutter" ->
    running pass PushSliceUp -> Condition failed: boundaries[0] ==
    0.to_dim()`). The 0.23 line handles it, so only the legacy target is
    skipped rather than dropping the coverage entirely. Same shape as the
    tract-0.22.1 skip in `packages/llm/tests/test_llm_cli.py`.
    """
    if inference_target.version < "0.23.0":
        pytest.skip("tract PushSliceUp declutter bug (fixed in the 0.23 line)")


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_copy_inplace_into_zeros_buffer(inference_target):
    _skip_if_scatter_declutter_bug(inference_target)
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


class IndexedWriteIntoZerosBuffer(torch.nn.Module):
    """Indexed write through a view chain, read back via the parent.

    This is HF's gated-delta-rule output pattern
    (`core_attn_out[:, :, i] = ...` then reading `core_attn_out`): the
    write goes through slice/select views, so the tracer does NOT
    SSA-rename the parent read; `functionalize_view_inplace_copy` must
    rewrite it into select_scatter/slice_scatter.
    """

    def forward(self, x):
        buf = torch.zeros(1, 4, 1, 16)
        buf[:, :, 0] = x.squeeze(2) * 2.0
        return buf.transpose(1, 2) + 1.0


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_indexed_write_into_zeros_buffer(inference_target):
    _skip_if_scatter_declutter_bug(inference_target)
    inp = torch.ones(1, 4, 1, 16)
    check_model_io_test(
        model=IndexedWriteIntoZerosBuffer(),
        test_input=inp,
        inference_target=inference_target,
    )
