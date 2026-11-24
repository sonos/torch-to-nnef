"""Generic dynamic-shape regressions exercised by looped attention models."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tests.utils import TRACT_INFERENCES_TO_TESTS_APPROX
from torch_to_nnef import export_model_to_nnef


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


def _check_dynamic(
    tmp_path, model, cases, axes, inference_target, custom_extensions=None
):
    target = inference_target.with_dynamic_axes(axes)
    cli = target.tract_cli
    names = [f"input_{i}" for i in range(len(cases[0]))]
    path = tmp_path / "model.nnef.tgz"
    with torch.no_grad():
        export_model_to_nnef(
            model=model.eval(),
            args=cases[0],
            file_path_export=path,
            inference_target=target,
            input_names=names,
            output_names=["output"],
            custom_extensions=custom_extensions,
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
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_expand_multiple_symbolic_leading_axes(
    tmp_path, scalar, inference_target
):
    _check_dynamic(
        tmp_path,
        LeadingExpand(scalar),
        [(torch.randn(b, s, 4),) for b, s in ((2, 3), (1, 5), (3, 1))],
        {"input_0": {0: "B", 1: "S"}},
        inference_target,
    )


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_expand_static_leading_axes(tmp_path, scalar, inference_target):
    _check_dynamic(
        tmp_path,
        LeadingExpand(scalar),
        [(torch.randn(2, 3, 4),)],
        {},
        inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_relative_gather_and_boolean_mask(tmp_path, inference_target):
    _check_dynamic(
        tmp_path,
        RelativeGather(),
        [(torch.randn(b, 2, s, 4),) for b, s in ((2, 3), (1, 7), (3, 1))],
        {"input_0": {0: "B", 2: "S"}},
        inference_target,
    )


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_padded_dynamic_tail_unfold(tmp_path, inference_target, request):
    if inference_target.version == "0.22.1":
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="tract 0.22.1 cannot simplify symbolic unfold lengths",
            )
        )
    _check_dynamic(
        tmp_path,
        SlidingWindows(),
        [(torch.randn(1, 4, s), torch.randn(1, 4, 2)) for s in (3, 1, 7)],
        {"input_0": {2: "S"}},
        inference_target,
    )


@pytest.mark.parametrize("sign_changing", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_tail_slice_clipping_and_unknown_sign(
    tmp_path, sign_changing, stride, inference_target, request
):
    if inference_target.version == "0.22.1" and (sign_changing or stride != 1):
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="tract 0.22.1 computes incorrect dynamic slice lengths",
            )
        )
    _check_dynamic(
        tmp_path,
        TailSlice(sign_changing, stride),
        [
            (torch.randn(s, 4), torch.ones(n))
            for s, n in ((5, 1), (1, 7), (9, 2), (8, 4))
        ],
        {"input_0": {0: "S"}, "input_1": {0: "N"}},
        inference_target,
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
@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_unsorted_topk_membership_and_values(
    tmp_path, largest, inference_target
):
    _check_dynamic(
        tmp_path,
        UnsortedTopK(largest),
        [(torch.randn(s, 8),) for s in (3, 1, 7)],
        {"input_0": {0: "S"}},
        inference_target,
    )


class PaddedConvolutionTail(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 4, 3, groups=4, padding=2, bias=False)

    def forward(self, x, history):
        context = torch.cat((history, x), dim=-1)
        result = self.conv(context)[..., : context.shape[-1]]
        result = result[..., -x.shape[-1] :]
        # The consumer requires the tail's symbolic length to remain S.
        return torch.cat((x.unsqueeze(2), result.unsqueeze(2)), dim=2)


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_padded_convolution_tail_with_positive_sequence(
    tmp_path, inference_target, request
):
    if inference_target.version == "0.22.1":
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="tract 0.22.1 cannot resolve constrained tail lengths",
            )
        )
    _check_dynamic(
        tmp_path,
        PaddedConvolutionTail(),
        [(torch.randn(1, 4, s), torch.randn(1, 4, 2)) for s in (3, 1, 7)],
        {"input_0": {2: "S"}},
        inference_target,
        custom_extensions=["tract_assert S >= 1"],
    )


class PossiblyEmptyTail(torch.nn.Module):
    def forward(self, x, length_source):
        return x[-length_source.shape[0] :]


@pytest.mark.parametrize("inference_target", TRACT_INFERENCES_TO_TESTS_APPROX)
def test_tail_of_zero_length_means_entire_input(
    tmp_path, inference_target, request
):
    if inference_target.version == "0.22.1":
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="tract 0.22.1 computes incorrect dynamic slice lengths",
            )
        )
    _check_dynamic(
        tmp_path,
        PossiblyEmptyTail(),
        [(torch.randn(5, 4), torch.ones(n)) for n in (2, 0, 1, 7)],
        {"input_1": {0: "N"}},
        inference_target,
    )
