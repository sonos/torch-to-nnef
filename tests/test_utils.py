import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef.utils import (
    NamedItem,
    ReactiveNamedItemDict,
    check_torch_ecosystem,
    flatten_dict_tuple_or_list,
)


@pytest.mark.parametrize(
    "torch_version,audio_version,compatible",
    [
        ("2.7.0", "2.7.0", True),
        ("2.7.0", "2.6.0", False),
        ("2.10.0", "2.11.0", False),
        ("2.11.0", "2.11.0", True),
        ("2.13.0", "2.11.0", True),
        ("2.14.1", "2.11.0", True),
        ("2.13.0", "2.10.0", False),
    ],
)
def test_torchaudio_compatibility(
    monkeypatch, torch_version, audio_version, compatible
):
    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setitem(
        sys.modules, "torchaudio", SimpleNamespace(__version__=audio_version)
    )
    monkeypatch.setitem(sys.modules, "torchvision", None)
    if compatible:
        check_torch_ecosystem()
    else:
        with pytest.raises(T2NErrorMisuse, match="incompatible with torch"):
            check_torch_ecosystem()


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
    """Micro fn to clarify notation in code."""
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
