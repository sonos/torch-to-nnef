"""Tests for torch_safe_load (weights_only-by-default torch.load)."""

import torch

from torch_to_nnef.utils import torch_safe_load


def test_torch_safe_load_plain_tensor(tmp_path):
    path = tmp_path / "tensor.pt"
    torch.save(torch.tensor([1.0, 2.0, 3.0]), path)
    out = torch_safe_load(path)
    assert torch.equal(out, torch.tensor([1.0, 2.0, 3.0]))


def test_torch_safe_load_state_dict(tmp_path):
    path = tmp_path / "sd.pt"
    torch.save(torch.nn.Linear(2, 2).state_dict(), path)
    out = torch_safe_load(path)
    assert set(out) == {"weight", "bias"}


def test_torch_safe_load_falls_back_for_pickled_module(tmp_path):
    # A full nn.Module is not loadable with weights_only=True, so the helper
    # must fall back to full unpickling (and still return the module).
    path = tmp_path / "module.pt"
    torch.save(torch.nn.Linear(2, 2), path)
    out = torch_safe_load(path)
    assert isinstance(out, torch.nn.Linear)
