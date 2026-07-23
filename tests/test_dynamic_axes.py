"""Tests dynamic_axes."""

import os
from functools import partial

import pytest
import torch
from torch import nn
from torchaudio import models as audio_mdl

from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.utils import torch_version

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    change_dynamic_axes,
    check_model_io_test,
    set_seed,
)
from .wrapper import TorchFnPrimitive

set_seed(int(os.environ.get("SEED", 25)))


test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)


dyn_stream_axis1 = {"input_0": {1: "S"}}
dyn_stream_axis2 = {"input_0": {2: "S"}}
dyn_stream_axis3 = {"input_0": {3: "S"}}

test_suite.add(
    torch.rand(1, 1, 100, 64),
    audio_mdl.DeepSpeech(64, n_hidden=256),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)


class MimicShapeOut(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        xsh = x.shape
        return self.op(xsh, dtype=torch.float32)


class LambdaOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, *args):
        return self.op(*args)


test_suite.add(
    torch.rand(1, 2, 3),
    MimicShapeOut(torch.ones),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)

test_suite.add(
    torch.rand(1, 10, 3),
    MimicShapeOut(torch.ones),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)

test_suite.add(
    torch.rand(1, 4, 3),
    MimicShapeOut(partial(torch.full, fill_value=5)),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)


def ge_tract_0_21_5(i):
    return isinstance(i, TractNNEF) and i.version > "0.21.5"


test_suite.add(
    torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]]),
    TorchFnPrimitive("repeat_interleave", opt_kwargs={"repeats": 3, "dim": 2}),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)

if torch_version() > "2.3.0":  # uint64 support in torch
    test_suite.add(
        torch.tensor([[[[1, 2, 3, 4]], [[5, 6, 7, 8]]]]).float(),
        LambdaOp(lambda x: x[..., : x.shape[-1] // 2]),
        inference_conditions=ge_tract_0_21_5,
        inference_modifier=partial(
            change_dynamic_axes, dynamic_axes=dyn_stream_axis3
        ),
    )

    test_suite.add(
        torch.tensor([[[[1, 2, 3, 4]], [[5, 6, 7, 8]]]]).float(),
        LambdaOp(lambda x: x[..., x.shape[-1] // 2 :]),
        inference_conditions=ge_tract_0_21_5,
        inference_modifier=partial(
            change_dynamic_axes, dynamic_axes=dyn_stream_axis3
        ),
    )
test_suite.add(
    torch.rand(2, 1),
    LambdaOp(lambda x: x[:, -3:]),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis1
    ),
)

test_suite.add(
    torch.rand(2, 1),
    LambdaOp(lambda x: x[:, :1000]),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis1
    ),
)

test_suite.add(
    torch.rand(2, 1),
    LambdaOp(lambda x: x.repeat(1, x.shape[-1])),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis1
    ),
)

test_suite.add(
    torch.rand(2, 1),
    LambdaOp(lambda x: x.expand([2, x.shape[1] * 2])),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis1
    ),
)


def trace_tdim_through_arange_fail(inputs_embeds, dummy_past_kv):
    sequence_length = inputs_embeds.shape[1]
    past_seen_tokens = dummy_past_kv.shape[2]
    # past_seen_tokens is extract from shape so TDimed
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + sequence_length,
        device=inputs_embeds.device,
    )
    new_past_seen_tokens = cache_position[0]
    target_length = new_past_seen_tokens + sequence_length + 1
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=-100,
        dtype=torch.float32,
    )
    return causal_mask


def causal_mask_dyn_inference_modifier(inference_target):
    inference_target = change_dynamic_axes(
        inference_target,
        dynamic_axes={"input_0": {1: "S"}, "input_1": {2: "P"}},
    )
    inference_target.custom_extensions = [
        "tract_assert P >= 0",
        "tract_assert S >= 1",
    ]
    return inference_target


test_suite.add(
    (torch.rand(2, 10), torch.rand(1, 4, 5, 3)),
    LambdaOp(trace_tdim_through_arange_fail),
    inference_conditions=lambda i: (
        isinstance(i, TractNNEF) and i.version >= "0.21.8"
    ),
    inference_modifier=causal_mask_dyn_inference_modifier,
)


class WrapperPickLastNonZeros(nn.Module):
    def __init__(self, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x):
        lengths = (
            x.ne(self.pad_idx).sum(dim=1).cpu()
        )  # Sum along the sequence length dimension

        # Use the last hidden state for each sequence
        last_indices = lengths.long() - 1
        batch_size = x.size(0)
        out = x[torch.arange(batch_size), last_indices]
        return out


start_val = 1
end_val = 10
inp = torch.zeros(3, end_val)
inp[0, :-1] = torch.arange(start_val, end_val + start_val - 1)
inp[1, :-3] = torch.arange(start_val, end_val + start_val - 3)
inp[2, :-2] = torch.arange(start_val, end_val + start_val - 2)

test_suite.add(
    inp,
    WrapperPickLastNonZeros(),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis1
    ),
)


# Negative-pad-on-dynamic-axis: PyTorch's `nn.ConstantPad2d` (and the
# underlying `aten::constant_pad_nd`) accepts negative entries that
# crop the corresponding side. T2n decomposes that into a `slice` +
# `pad` pair. Under dynamic axes the slice must use the streaming-aware
# `dyn_slice_begin` form (open-ended end), otherwise the cropped axis
# collapses to a concrete value and the downstream pulse-mode pass
# breaks every subsequent broadcast against the streaming siblings.
# Surfaced first on DPDFNet's `pad_feat` (causal lookahead crop).
test_suite.add(
    torch.rand(1, 1, 100, 64),
    nn.ConstantPad2d((0, 0, -2, 2), 0.0),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis2
    ),
)


# `Tensor.unfold` over a streaming axis: the handler dispatches to the
# `unfold` NNEF fragment (Conv1d-with-identity-kernel), which tract
# pulsifies via its existing `Conv` pulsifier. Without this dispatch
# the static slice-stack form would emit slices with absolute end
# points that tract can't pulse. Two cases:
#   - rank-2 input, unfold over last axis (no permutation needed).
#   - rank-3 input, unfold over middle axis (handler must thread perms).
class UnfoldOverStreamLast(nn.Module):
    def forward(self, x):  # x: (B, T) -> (B, T-2, 3)
        return x.unfold(1, 3, 1)


class UnfoldOverStreamMiddle(nn.Module):
    def forward(self, x):  # x: (B, T, F) -> (B, T-2, F, 3)
        return x.unfold(1, 3, 1)


test_suite.add(
    torch.arange(2 * 12, dtype=torch.float32).reshape(2, 12),
    UnfoldOverStreamLast(),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis1
    ),
)
test_suite.add(
    torch.arange(2 * 12 * 4, dtype=torch.float32).reshape(2, 12, 4),
    UnfoldOverStreamMiddle(),
    inference_conditions=ge_tract_0_21_5,
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis1
    ),
)


# Static-axis shape query feeding a split while another axis is dynamic
# (the 2D-RoPE / attention-head pattern): `x.shape[-1] // 2` must fold to a
# constant even though axis 1 is symbolic, else `split_with_sizes` gets None
# sizes. Guards per-axis dynamic-shape tracking.
class StaticAxisSplitUnderDynamic(nn.Module):
    def forward(self, x):  # x: (B, S, F), S dynamic, F static
        f = x.shape[-1]
        half = f // 2
        a, b = torch.split(x, [half, f - half], dim=-1)
        return a + b


test_suite.add(
    torch.arange(2 * 12 * 8, dtype=torch.float32).reshape(2, 12, 8),
    StaticAxisSplitUnderDynamic(),
    inference_modifier=partial(
        change_dynamic_axes, dynamic_axes=dyn_stream_axis1
    ),
)


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_dynamic_axes_exports(id, test_input, model, inference_target):
    """Test simple models."""
    check_model_io_test(
        model=model,
        test_input=test_input,
        inference_target=inference_target,
        # for convenience of tests we assigned
        # custom_extensions to inference target
        custom_extensions=(
            inference_target.custom_extensions
            if hasattr(inference_target, "custom_extensions")
            else None
        ),
    )
