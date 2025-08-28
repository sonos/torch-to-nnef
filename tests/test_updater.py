import pytest
import torch
from torch import nn

from tests.utils import skipif_unsupported_tensor_updater
from torch_to_nnef.exceptions import T2NErrorInconsistentTensor
from torch_to_nnef.tensor.updater import ModTensorUpdater


class CustomMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(3, 10, 3),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Conv1d(10, 10, 1),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Conv1d(10, 10, 1),
            nn.BatchNorm1d(10),
            nn.Conv1d(10, 3, 1),
            nn.ReLU(),
        )
        self.last_lin_proj = nn.Linear(3, 5)
        self.embedding = nn.Embedding(5, 3)

    def forward(self, inp: torch.Tensor):
        x = self.embedding(inp)
        x = x.permute(0, 2, 1)
        x = self.seq(x)
        x = x.permute(0, 2, 1)
        x = self.last_lin_proj(x)
        return x


@skipif_unsupported_tensor_updater
def test_parameter_update_by_ref_with_tied():
    model = CustomMod()
    model.last_lin_proj.weight = model.embedding.weight  # Tied parameters
    sample0 = torch.tensor([[1, 3, 4, 0, 1, 2]])
    _ = model(sample0)
    param_updaters = ModTensorUpdater(model)
    old_data = model.last_lin_proj.weight.clone()
    assert id(model.last_lin_proj.weight) == id(model.embedding.weight)
    param_updaters.update_by_ref(
        ref=model.embedding.weight,
        new_tensor=torch.arange(5 * 3).reshape(5, 3).float(),
    )
    assert id(model.last_lin_proj.weight) == id(model.embedding.weight)
    assert not torch.eq(old_data, model.last_lin_proj.weight).all()
    assert torch.eq(
        torch.arange(5 * 3).reshape(5, 3).float(), model.last_lin_proj.weight
    ).all()


@skipif_unsupported_tensor_updater
def test_parameter_update_by_name_with_tied():
    model = CustomMod()
    model.last_lin_proj.weight = model.embedding.weight  # Tied parameters
    sample0 = torch.tensor([[1, 3, 4, 0, 1, 2]])
    _ = model(sample0)
    param_updaters = ModTensorUpdater(model)

    old_data = model.last_lin_proj.weight.clone()
    assert id(model.last_lin_proj.weight) == id(model.embedding.weight)

    new_tensor = torch.arange(5 * 3).reshape(5, 3).float()
    param_updaters.update_by_name(
        name="embedding.weight", new_tensor=new_tensor
    )

    assert id(model.last_lin_proj.weight) == id(model.embedding.weight)
    assert not torch.eq(old_data, model.last_lin_proj.weight).all()
    assert torch.eq(new_tensor, model.last_lin_proj.weight).all()


@skipif_unsupported_tensor_updater
def test_parameter_update_by_name_with_tied_modified_separatly():
    model = CustomMod()
    model.last_lin_proj.weight = model.embedding.weight  # Tied parameters
    sample0 = torch.tensor([[1, 3, 4, 0, 1, 2]])
    _ = model(sample0)
    param_updaters = ModTensorUpdater(model)

    old_data = model.last_lin_proj.weight.clone()
    assert id(model.last_lin_proj.weight) == id(model.embedding.weight)

    new_tensor = torch.arange(5 * 3).reshape(5, 3).float()
    param_updaters.update_by_name(
        name="embedding.weight", new_tensor=new_tensor, tie_replacements=False
    )

    assert id(model.last_lin_proj.weight) != id(model.embedding.weight)
    assert torch.eq(old_data, model.last_lin_proj.weight).all()
    assert not torch.eq(new_tensor, model.last_lin_proj.weight).all()


@skipif_unsupported_tensor_updater
def test_parameter_update_by_name_with_tied_modified_separatly_meta():
    model = CustomMod()
    model.last_lin_proj.weight = model.embedding.weight  # Tied parameters
    sample0 = torch.tensor([[1, 3, 4, 0, 1, 2]])
    _ = model(sample0)
    param_updaters = ModTensorUpdater(model)
    assert id(model.last_lin_proj.weight) == id(model.embedding.weight)

    new_tensor = torch.arange(5 * 2).reshape(5, 2).float()
    with pytest.raises(T2NErrorInconsistentTensor):
        param_updaters.update_by_name(
            name="embedding.weight",
            new_tensor=new_tensor,
        )

    new_tensor = torch.arange(5 * 3).reshape(5, 3).half()
    with pytest.raises(T2NErrorInconsistentTensor):
        param_updaters.update_by_name(
            name="embedding.weight",
            new_tensor=new_tensor,
        )

    new_tensor = torch.arange(5 * 3).reshape(5, 3).float().to("meta")
    with pytest.raises(T2NErrorInconsistentTensor):
        param_updaters.update_by_name(
            name="embedding.weight",
            new_tensor=new_tensor,
        )
    new_tensor = torch.arange(5 * 3).reshape(5, 3).float().to("meta")
    param_updaters.update_by_name(
        name="embedding.weight",
        new_tensor=new_tensor,
        enforce_tensor_consistency=False,
    )


@skipif_unsupported_tensor_updater
def test_add_parameter_if_unset():
    model = CustomMod()
    model.last_lin_proj.weight = model.embedding.weight  # Tied parameters
    sample0 = torch.tensor([[1, 3, 4, 0, 1, 2]])
    _ = model(sample0)
    param_updaters = ModTensorUpdater(model, add_parameter_if_unset=False)
    assert id(model.last_lin_proj.weight) == id(model.embedding.weight)

    new_tensor = torch.arange(5 * 3).reshape(5, 3).float()
    with pytest.raises(ValueError):
        param_updaters.update_by_name(
            name="embedding.weight",
            new_tensor=new_tensor,
        )

    param_updaters.update_by_name(
        name="embedding.weight",
        new_tensor=nn.Parameter(new_tensor),
    )
