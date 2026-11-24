"""Generic dynamic-shape regressions exercised by looped attention models."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.inference_target.tract import TractCli


class LeadingExpand(torch.nn.Module):
    def __init__(self, scalar=False):
        super().__init__()
        self.register_buffer(
            "query",
            torch.tensor(2.0) if scalar else torch.arange(8.0).reshape(2, 4),
        )
        self.scalar = scalar

    def forward(self, x):
        b, s = x.shape[:2]
        if self.scalar:
            return self.query.expand(b, s) + x[:, :, 0]
        return self.query.expand(b, s, -1, -1) + x.unsqueeze(2)


class RelativeGather(torch.nn.Module):
    def forward(self, x):
        b, h, s, _ = x.shape
        positions = torch.arange(s, device=x.device)
        distances = positions[:, None] - positions[None, :]
        index = distances.clamp(0, 3).expand(b, h, -1, -1)
        values = x.gather(-1, index)
        return values.masked_fill((distances < 0) | (distances >= 4), 0.0)


class SlidingWindows(torch.nn.Module):
    def forward(self, x, history):
        b, c, s = x.shape
        context = torch.cat((history, x), dim=-1)
        context = F.pad(context, (2, 0))[..., -(s + 2) :]
        windows = context.unfold(-1, 3, 1).flip(-1).transpose(1, 2)
        return windows.reshape(b, s, 2, c // 2, 3)


class TailSlice(torch.nn.Module):
    def __init__(self, sign_changing=False, stride=1):
        super().__init__()
        self.sign_changing = sign_changing
        self.stride = stride

    def forward(self, x, length_source):
        if self.sign_changing:
            return x[-(length_source.shape[0] - 2) :]
        return x[-(length_source.shape[0] + 2) :: self.stride]


def _check_dynamic(tmp_path, model, cases, axes):
    binary = os.environ.get("TRACT_BIN")
    if not binary:
        pytest.skip("set TRACT_BIN for dynamic NNEF runtime validation")
    cli = TractCli(Path(binary))
    names = [f"input_{i}" for i in range(len(cases[0]))]
    target = TractNNEF(
        version=cli.version,
        specific_tract_binary_path=cli.tract_path,
        dynamic_axes=axes,
        check_io=True,
    )
    path = tmp_path / "model.nnef.tgz"
    with torch.no_grad():
        export_model_to_nnef(
            model=model.eval(),
            args=cases[0],
            file_path_export=path,
            inference_target=target,
            input_names=names,
            output_names=["output"],
        )
        for inputs in cases:
            in_path, out_path = (
                tmp_path / "inputs.npz",
                tmp_path / "outputs.npz",
            )
            np.savez(
                in_path,
                **{
                    name: value.numpy()
                    for name, value in zip(names, inputs, strict=True)
                },
            )
            np.savez(out_path, output=model(*inputs).numpy())
            cli.assert_io(
                path,
                in_path,
                out_path,
                check_tolerance=target.check_io_tolerance,
            )


@pytest.mark.parametrize("scalar", [False, True])
def test_expand_multiple_symbolic_leading_axes(tmp_path, scalar):
    _check_dynamic(
        tmp_path,
        LeadingExpand(scalar),
        [(torch.randn(b, s, 4),) for b, s in ((2, 3), (1, 5), (3, 1))],
        {"input_0": {0: "B", 1: "S"}},
    )


@pytest.mark.parametrize("scalar", [False, True])
def test_expand_static_leading_axes(tmp_path, scalar):
    _check_dynamic(
        tmp_path, LeadingExpand(scalar), [(torch.randn(2, 3, 4),)], {}
    )


def test_relative_gather_and_boolean_mask(tmp_path):
    _check_dynamic(
        tmp_path,
        RelativeGather(),
        [(torch.randn(b, 2, s, 4),) for b, s in ((2, 3), (1, 7), (3, 1))],
        {"input_0": {0: "B", 2: "S"}},
    )


def test_padded_dynamic_tail_unfold(tmp_path):
    _check_dynamic(
        tmp_path,
        SlidingWindows(),
        [(torch.randn(1, 4, s), torch.randn(1, 4, 2)) for s in (3, 1, 7)],
        {"input_0": {2: "S"}},
    )


@pytest.mark.parametrize("sign_changing", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
def test_tail_slice_clipping_and_unknown_sign(tmp_path, sign_changing, stride):
    _check_dynamic(
        tmp_path,
        TailSlice(sign_changing, stride),
        [
            (torch.randn(s, 4), torch.ones(n))
            for s, n in ((5, 1), (1, 7), (9, 2), (8, 4))
        ],
        {"input_0": {0: "S"}, "input_1": {0: "N"}},
    )


class UnsortedTopK(torch.nn.Module):
    def __init__(self, largest):
        super().__init__()
        self.largest = largest

    def forward(self, x):
        values, indices = x.topk(3, dim=-1, largest=self.largest, sorted=False)
        # Compare order-independent membership and value/index correspondence.
        chosen = torch.nn.functional.one_hot(indices, num_classes=8).float()
        return (chosen * values.unsqueeze(-1)).sum(-2)


@pytest.mark.parametrize("largest", [False, True])
def test_unsorted_topk_membership_and_values(tmp_path, largest):
    _check_dynamic(
        tmp_path,
        UnsortedTopK(largest),
        [(torch.randn(s, 8),) for s in (3, 1, 7)],
        {"input_0": {0: "S"}},
    )
