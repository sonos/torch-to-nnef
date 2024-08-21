from dataclasses import dataclass

from torch_to_nnef.utils import NamedItem, NamedItemOrderedSet


@dataclass
class MyDataItem(NamedItem):
    name: str


def test_named_items_base():
    nos = NamedItemOrderedSet(
        [
            MyDataItem("a"),
            MyDataItem("b"),
            MyDataItem("c"),
        ]
    )
    assert len(nos) == 3


def test_named_items_rename():
    nos = NamedItemOrderedSet(
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
    nos1 = NamedItemOrderedSet(items)
    nos2 = NamedItemOrderedSet(items)
    nitem = nos1[-1]
    nitem.name = "d"
    assert nos1.get_by_name("c") is None
    assert nos1.get_by_name("d") is nitem
    assert nos2.get_by_name("c") is None
    assert nos2.get_by_name("d") is nitem
