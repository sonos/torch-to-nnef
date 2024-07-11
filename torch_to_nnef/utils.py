import functools
import logging
import typing as T
from abc import ABC
from functools import total_ordering

import torch

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

    def __repr__(self) -> str:
        version_str = ".".join(str(self.version[t]) for t in self.TAGS)
        return f"<Version {version_str}>"


def torch_version() -> SemanticVersion:
    """Semantic version for torch"""
    return SemanticVersion.from_str(torch.__version__.split("+")[0])


class NamedItem(ABC):
    # must implement name attribute
    name: str

    def register_listener_name_change(self, listener):
        if not hasattr(self, "_name_hooks"):
            self._name_hooks = set()
        if listener in self._name_hooks:
            raise ValueError("Already registered  listener !")
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


class NamedItemOrderedSet:
    """Named items ordered Set data structure

    Ensure that no 2 items are inserted with same 'name' attribute
    and maintains fast name update and Set speed benefits

    Warning! only aimed at NamedItem subclass.

    Expose a 'list' like interface. (with limited index access)

    """

    def __init__(self, items: T.Optional[T.Iterable[NamedItem]] = None):
        self._map: T.Dict[str, T.Any] = {}
        self._last_inserted_item: T.Optional[NamedItem] = None
        if items is not None:
            self.__add__(items)

    @classmethod
    def from_list(cls, items) -> "NamedItemOrderedSet":
        if not items:
            return cls()
        return cls() + items

    def _change_name_hook(self, old_name: str, new_name: str):
        """maintain sync between data structure and name changes in items"""
        if new_name == old_name:
            return
        if new_name in self._map:
            LOGGER.debug(f"node with name:{new_name} overwritten in {self}")
        self._map[new_name] = self._map[old_name]
        del self._map[old_name]

    def remove(
        self, item: NamedItem, raise_exception_if_not_found: bool = True
    ):
        if item.name not in self._map:
            msg = f"item '{item.name}' requested for deletion. Not Found !"
            if raise_exception_if_not_found:
                raise ValueError(msg)
            LOGGER.debug(msg)
            return
        self._map[item.name].detach_listener_name_change(self._change_name_hook)
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

        WARNING: This is crutial that all added items use this
        function as it set the hook to listen to name changes
        """
        assert item.name not in self._map, item.name
        item.register_listener_name_change(self._change_name_hook)
        self._map[item.name] = item
        self._last_inserted_item = item

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
        return (
            f"<NamedItemOrderedSet ({len(self._map)}) stored_names=[{names}]>"
        )
