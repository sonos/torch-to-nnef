from dataclasses import dataclass

import pytest

import torch
from torch import nn

from torch_to_nnef.exceptions import InconsistentTensorError
from torch_to_nnef.utils import (
    NamedItem,
    ReactiveNamedItemDict,
    flatten_dict_tuple_or_list,
)
from torch_to_nnef.tensor.updater import ModTensorUpdater


@dataclass
class MyDataItem(NamedItem):
    name: str


def test_named_items_base():
    nos = ReactiveNamedItemDict(
        [
            MyDataItem("a"),
            MyDataItem("b"),
            MyDataItem("c"),
        ]
    )
    assert len(nos) == 3


def test_named_items_rename():
    nos = ReactiveNamedItemDict(
        [
            MyDataItem("a"),
            MyDataItem("b"),
            MyDataItem("c"),
        ]
    )
    nitem = nos[-1]
    nitem.name = "d"
    assert nos.get_by_name("c") is None
    assert nos.get_by_name("d") is nitem


def test_named_items_multiple_ordered_set_of_same_item():
    items = [
        MyDataItem("a"),
        MyDataItem("b"),
        MyDataItem("c"),
    ]
    nos1 = ReactiveNamedItemDict(items)
    nos2 = ReactiveNamedItemDict(items)
    nitem = nos1[-1]
    nitem.name = "d"
    assert nos1.get_by_name("c") is None
    assert nos1.get_by_name("d") is nitem
    assert nos2.get_by_name("c") is None
    assert nos2.get_by_name("d") is nitem


FLATTEN_LIST_IOS = []


def _add_flatten_example(inp, out):
    """micro fn to clarify notation in code"""
    FLATTEN_LIST_IOS.append((inp, out))


_add_flatten_example(
    inp=[1, 2, 3],
    out=(((list,), (0,), 1), ((list,), (1,), 2), ((list,), (2,), 3)),
)
_add_flatten_example(
    inp=[[1], 2, 3],
    out=(((list, list), (0, 0), 1), ((list,), (1,), 2), ((list,), (2,), 3)),
)
_add_flatten_example(
    inp=[[1, 2], 2, 3],
    out=(
        ((list, list), (0, 0), 1),
        ((list, list), (0, 1), 2),
        ((list,), (1,), 2),
        ((list,), (2,), 3),
    ),
)
_add_flatten_example(
    inp=[[[1], 2], 2, [3]],
    out=(
        ((list, list, list), (0, 0, 0), 1),
        ((list, list), (0, 1), 2),
        ((list,), (1,), 2),
        ((list, list), (2, 0), 3),
    ),
)

_add_flatten_example(
    inp=[{"a": [1, 2], "b": [[3, 4], [5, 6]]}, {"c": 7}],
    out=(
        ((list, dict, list), (0, "a", 0), 1),
        ((list, dict, list), (0, "a", 1), 2),
        ((list, dict, list, list), (0, "b", 0, 0), 3),
        ((list, dict, list, list), (0, "b", 0, 1), 4),
        ((list, dict, list, list), (0, "b", 1, 0), 5),
        ((list, dict, list, list), (0, "b", 1, 1), 6),
        ((list, dict), (1, "c"), 7),
    ),
)

_add_flatten_example(
    inp=[{"a": 1, "b": 3}],
    out=(
        ((list, dict), (0, "a"), 1),
        ((list, dict), (0, "b"), 3),
    ),
)


@pytest.mark.parametrize("inputs,outputs", FLATTEN_LIST_IOS)
def test_flatten(inputs, outputs):
    gen_outs = flatten_dict_tuple_or_list(inputs)
    assert gen_outs == outputs


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


def test_parameter_update_by_name_with_tied_modified_separatly():
    model = CustomMod()
    model.last_lin_proj.weight = model.embedding.weight  # Tied parameters
    sample0 = torch.tensor([[1, 3, 4, 0, 1, 2]])
    _ = model(sample0)
    param_updaters = ModTensorUpdater(model)
    assert id(model.last_lin_proj.weight) == id(model.embedding.weight)

    new_tensor = torch.arange(5 * 2).reshape(5, 2).float()
    with pytest.raises(InconsistentTensorError):
        param_updaters.update_by_name(
            name="embedding.weight",
            new_tensor=new_tensor,
        )

    new_tensor = torch.arange(5 * 3).reshape(5, 3).half()
    with pytest.raises(InconsistentTensorError):
        param_updaters.update_by_name(
            name="embedding.weight",
            new_tensor=new_tensor,
        )

    new_tensor = torch.arange(5 * 3).reshape(5, 3).float().to("meta")
    with pytest.raises(InconsistentTensorError):
        param_updaters.update_by_name(
            name="embedding.weight",
            new_tensor=new_tensor,
        )
    new_tensor = torch.arange(5 * 3).reshape(5, 3).float().to("meta")
    param_updaters.update_by_name(
        name="embedding.weight",
        new_tensor=new_tensor,
        enforce_same_shape_dtype_device=False,
    )


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
