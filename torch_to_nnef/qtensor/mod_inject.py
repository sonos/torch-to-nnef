"""Advanced QTensor (<= 8bits) with complex quant scheme non torch native"""

import typing as T

import torch
from torch import nn
from torch.nn import functional as F

from torch_to_nnef.qtensor.base import QTensor


class WeightInputedConv1d(nn.Module):
    CP_ATTRS = [
        "bias",
        "padding_mode",
        "stride",
        "dilation",
        "padding",
        "groups",
        "_reversed_padding_repeated_twice",
    ]

    def __init__(self, conv_nn: nn.Conv1d):
        super().__init__()
        for attr in self.CP_ATTRS:
            setattr(self, attr, getattr(conv_nn, attr))

    def _conv_forward(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        bias: T.Optional[torch.Tensor],
    ):
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    inp,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            inp,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(inp, weight, self.bias)


class WeightInputedLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    CP_ATTRS = ["bias", "in_features", "out_features"]

    def __init__(self, linear: nn.Linear):
        super().__init__()
        for attr in self.CP_ATTRS:
            setattr(self, attr, getattr(linear, attr))

    def forward(self, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(inp, weight, self.bias)


class QWeightedOp(nn.Module):
    def __init__(self, mod: nn.Module, weight_mod: QTensor):
        super().__init__()
        self.mod = mod
        self.weight_mod = weight_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mod(x, self.weight_mod())

    def __getattr__(self, name):
        mod_dic = self.__dict__
        if name in mod_dic:
            return mod_dic[name]
        mod_dic = mod_dic["_modules"]
        if name in mod_dic:
            return mod_dic[name]
        mod_dic = mod_dic["mod"].__dict__
        if name in mod_dic:
            return mod_dic[name]
        raise AttributeError(f"{name} not found")


class WeightInputedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse

    def forward(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
    ):
        return F.embedding(
            inp,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def replace_nn_ops(module: nn.Module, q_weight: QTensor) -> nn.Module:
    if isinstance(module, nn.Embedding):
        assert (
            module.weight.shape == q_weight().shape
        ), f"{module.weight.shape} == {q_weight().shape}"
        return QWeightedOp(WeightInputedEmbedding(module), q_weight)
    if isinstance(module, nn.Conv1d):
        assert (
            module.weight.shape == q_weight().shape
        ), f"{module.weight.shape} == {q_weight().shape}"
        return QWeightedOp(WeightInputedConv1d(module), q_weight)
    if isinstance(module, nn.Linear):
        assert (
            module.weight.shape == q_weight().shape
        ), f"{module.weight.shape} == {q_weight().shape}"
        return QWeightedOp(WeightInputedLinear(module), q_weight)
    raise NotImplementedError(module)
