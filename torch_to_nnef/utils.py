import functools
from typing import Callable, TypeVar

T = TypeVar("T")


def cache(func: Callable[..., T]) -> T:
    """LRU cache helper that avoid pylint complains"""
    return functools.lru_cache()(func)  # type: ignore


def fullname(o):
    """Full class name with module path from an object"""
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__
