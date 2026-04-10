from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from tests.utils import TRACT_INFERENCES_TO_TESTS_APPROX, check_model_io_test


class MyModel(nn.Module):
    def forward(self, x):
        # a, b, _ = x.shape
        # return y.expand(a, 1, b, b)
        batch_size, query_length, _ = x.shape
        att_mask = AttentionMaskConverter(is_causal=True)._make_causal_mask(
            (batch_size, query_length),
            x.dtype,
            x.device,
            past_key_values_length=10,
            sliding_window=2047,
        )
        return att_mask


class ReducedForm(nn.Module):
    def forward(self, x):
        _, tgt_len, _ = x.shape
        mask = torch.full(
            (tgt_len, tgt_len), torch.finfo(x.dtype).min, device=x.device
        )
        mask_cond = torch.arange(mask.size(-1), device=x.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        return mask


@pytest.mark.parametrize(
    "inference_target",
    [_ for _ in TRACT_INFERENCES_TO_TESTS_APPROX if _.version > "0.21.6"],
)
def test_tile_and_dyn_dims(inference_target):
    inference_target = deepcopy(inference_target)
    inference_target.dynamic_axes = {"input_0": {0: "B", 1: "S"}}
    check_model_io_test(
        model=ReducedForm(),
        input_names=["input_0"],
        test_input=(torch.rand(1, 250, 4)),
        inference_target=inference_target,
    )
