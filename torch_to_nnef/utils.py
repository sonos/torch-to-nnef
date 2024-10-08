import contextlib
import functools
import logging
import os
import typing as T
from abc import ABC
from collections.abc import MutableMapping
from functools import total_ordering

import torch
from torch import _C

from torch_to_nnef.exceptions import (
    DataNodeValueError,
    TorchToNNEFNotImplementedError,
)

LOGGER = logging.getLogger(__name__)

C = T.TypeVar("C")


def cache(func: T.Callable[..., C]) -> C:
    """LRU cache helper that avoid pylint complains"""
    return functools.lru_cache()(func)  # type: ignore


def fullname(o) -> str:
    """Full class name with module path from an object"""
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


@contextlib.contextmanager
def cd(path):
    old_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_path)


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    items: T.List[T.Tuple[str, T.Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_dict_tuple_or_list(
    obj,
    collected_types: T.Optional[T.List[T.Type]] = None,
    collected_idxes: T.Optional[T.List[int]] = None,
    current_idx: int = 0,
) -> T.Tuple[
    T.Tuple[T.Tuple[T.Type, ...], T.Tuple[T.Union[int, str], ...], T.Any], ...
]:
    """Flatten dict/list/tuple recursively, return types, indexes and values

    Flatten in depth first search order

    Args:
        obj: dict/tuple/list or anything else (structure can be arbitrary deep)
            this contains N number of element non dict/list/tuple

        collected_types: do not set
        collected_idxes: do not set
        current_idx: do not set

    Return:
        tuple of N tuples each containing a tuple of:
            types, indexes and the element

    Example:
        If initial obj=[{"a": 1, "b": 3}]
        it will output:
            (
                ((list, dict), (0, "a"), 1),
                ((list, dict), (0, "b"), 3),
            )
    """
    if collected_idxes is None:
        collected_idxes = []
    else:
        collected_idxes = collected_idxes[:]

    if collected_types is None:
        collected_types = []
    else:
        collected_types = collected_types[:]

    collected_types.append(type(obj))
    if isinstance(obj, (tuple, list)):
        collected_idxes.append(current_idx)
        current_idx += 1
        if not obj:
            return ()
        if isinstance(obj[0], (tuple, list, dict)):
            return flatten_dict_tuple_or_list(
                obj[0], collected_types, collected_idxes, 0
            ) + flatten_dict_tuple_or_list(
                obj[1:], collected_types[:-1], collected_idxes[:-1], current_idx
            )
        return (
            (tuple(collected_types), tuple(collected_idxes), obj[0]),
        ) + flatten_dict_tuple_or_list(
            obj[1:], collected_types[:-1], collected_idxes[:-1], current_idx
        )
    if isinstance(obj, dict):
        res = []  # type: ignore
        for k, v in obj.items():
            if isinstance(v, (tuple, list, dict)):
                res += flatten_dict_tuple_or_list(
                    v, collected_types, collected_idxes + [k], 0
                )
            else:
                res += [
                    (tuple(collected_types), tuple(collected_idxes + [k]), v)
                ]
        return tuple(res)
    return ()


@total_ordering
class SemanticVersion:
    """Helper to check a version is higher than another"""

    TAGS = ["major", "minor", "patch"]

    def __init__(self, **kwargs):
        for t in self.TAGS:
            assert isinstance(kwargs[t], int), kwargs[t]
            assert kwargs[t] >= 0, kwargs[t]

        self.version = {t: kwargs[t] for t in self.TAGS}

    @classmethod
    def from_str(cls, version_str, sep="."):
        version_chunks = version_str.strip().split(sep)
        if "-" in version_chunks[-1]:
            version_chunks[-1] = version_chunks[-1].split("-")[0]
        vtags = list(map(int, version_chunks))
        assert len(vtags) == len(cls.TAGS)
        return cls(**dict(zip(cls.TAGS, vtags)))

    def __eq__(self, other: object):
        if isinstance(other, str):
            other = SemanticVersion.from_str(other)
        assert isinstance(other, SemanticVersion), other
        return all(self.version[t] == other.version[t] for t in self.TAGS)

    def __lt__(self, other: object):
        if isinstance(other, str):
            other = SemanticVersion.from_str(other)
        assert isinstance(other, SemanticVersion), other
        for t in self.TAGS:
            if self.version[t] < other.version[t]:
                return True
            if self.version[t] > other.version[t]:
                return False
        return False

    def to_str(self):
        return ".".join(str(self.version[t]) for t in self.TAGS)

    def __repr__(self) -> str:
        return f"<Version {self.to_str()}>"


def torch_version() -> SemanticVersion:
    """Semantic version for torch"""
    return SemanticVersion.from_str(torch.__version__.split("+")[0])


def select_ctx_disable_torch_fn():
    if hasattr(_C, "DisableTorchFunctionSubclass"):  # post torch 2.0.0
        ctx_disable_torch_fn = _C.DisableTorchFunctionSubclass()
    elif hasattr(_C, "DisableTorchFunction"):  # pre torch 2.0.0
        ctx_disable_torch_fn = _C.DisableTorchFunction()
    else:
        raise TorchToNNEFNotImplementedError(
            f"How to disable torch function in torch=={torch_version()}"
        )
    return ctx_disable_torch_fn


class NamedItem(ABC):
    # must implement name attribute
    name: str

    def register_listener_name_change(self, listener):
        if not hasattr(self, "_name_hooks"):
            self._name_hooks = set()
        if listener in self._name_hooks:
            raise DataNodeValueError("Already registered  listener !")
        self._name_hooks.add(listener)

    def detach_listener_name_change(self, listener):
        if not hasattr(self, "_name_hooks"):
            self._name_hooks = set()
        self._name_hooks.remove(listener)

    def __setattr__(self, attr_name, attr_value):
        if attr_name == "name" and hasattr(self, "_name_hooks"):
            for name_hook in self._name_hooks:
                name_hook(self.name, attr_value)
        super().__setattr__(attr_name, attr_value)


class ReactiveNamedItemDict:
    """Named items ordered Dict data structure

    Ensure that 'NO' 2 items are inserted with same 'name' attribute
    and maintains fast name update and with some additive colision
    protections.

    Warning! only aimed at NamedItem subclass.

    Expose a 'list' like interface. (with limited index access)

    """

    def __init__(self, items: T.Optional[T.Iterable[NamedItem]] = None):
        self._map: T.Dict[str, T.Any] = {}
        self._last_inserted_item: T.Optional[NamedItem] = None
        if items is not None:
            self.__add__(items)
        self.avoid_name_collision = False
        self._protected_names: T.Set[str] = set()

    @classmethod
    def from_list(cls, items) -> "ReactiveNamedItemDict":
        if not items:
            return cls()
        return cls() + items

    def _change_name_hook(self, old_name: str, new_name: str):
        """maintain sync between data structure and name changes in items"""
        if old_name in self._protected_names:
            raise DataNodeValueError(
                f"Not allowed to alter protected_name: {old_name}"
            )
        if new_name in self._protected_names:
            raise DataNodeValueError(
                f"Not allowed to alter protected_name: {new_name}"
            )
        if new_name == old_name:
            return
        if new_name in self._map:
            msg = f"node with name:{new_name} overwritten in {self}"
            LOGGER.debug(msg)
            if self.avoid_name_collision:
                raise DataNodeValueError(msg)
        self._map[new_name] = self._map[old_name]
        del self._map[old_name]

    def detach_listener_name_change_for_item(self, item):
        self._map[item.name].detach_listener_name_change(self._change_name_hook)

    def remove(
        self,
        item: NamedItem,
        raise_exception_if_not_found: bool = True,
        raise_exception_if_protected_name: bool = True,
    ):
        if item.name not in self._map:
            msg = f"item '{item.name}' requested for deletion. Not Found !"
            if raise_exception_if_not_found:
                raise DataNodeValueError(msg)
            LOGGER.debug(msg)
            return
        if (
            item.name in self._protected_names
            and not raise_exception_if_protected_name
        ):
            raise DataNodeValueError(f"Not authorized to remove: '{item.name}'")
        self.detach_listener_name_change_for_item(item)
        del self._map[item.name]

    def get_by_name(self, name: str, default: T.Any = None):
        return self._map.get(name, default)

    def contains(self, item: NamedItem, strict: bool = False):
        name_exists = item.name in self._map
        if name_exists and strict:
            return self._map[item.name] == item
        return name_exists

    def append(self, item: NamedItem):
        """Append item to ordered set

        WARNING: This is crucial that all added items use this
        function as it set the hook to listen to name changes
        """
        if item.name in self._map:
            raise DataNodeValueError(
                f"`{item.name}` already exist in container:"
                f" {self._map[item.name]}, "
                f"but tried to add an item: {item} with same name."
            )
        item.register_listener_name_change(self._change_name_hook)
        self._map[item.name] = item
        self._last_inserted_item = item

    def protect_item_names(self, names: T.Iterable[str]):
        self._protected_names.update(set(names))

    def is_empty(self):
        return not bool(self._map)

    def __getitem__(self, index: T.Any):
        if index == -1:
            if self._last_inserted_item is None:
                raise ValueError("No last value found")
            return self._last_inserted_item
        if isinstance(index, (slice, int)):
            return list(self._map.values())[index]
        raise NotImplementedError(index)

    def iter_renamable(self):
        for item_name, item in self._map.items():
            if item_name in self._protected_names:
                continue
            yield item

    def __add__(self, items):
        for item in items:
            self.append(item)
        return self

    def __setitem__(self, index: T.Any, value: NamedItem):
        raise NotImplementedError(
            "Assigning a specific index is not supported to date"
        )

    def __iter__(self):
        yield from self._map.values()

    def __len__(self):
        return len(self._map)

    def __repr__(self):
        names = "\n\t".join(f"'{k}'" for k in self._map)
        protected = ""
        if self._protected_names:
            pnames = ",\n".join(f"\t'{k}'" for k in self._protected_names)
            protected = f"\nprotected_names=[\n{pnames}]\n"
        return f"<ReactiveNamedItemDict ({len(self._map)}) stored_names=[{names}] {protected}>"
