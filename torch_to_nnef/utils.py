import functools
from typing import Callable, TypeVar

T = TypeVar("T")


def cache(func: Callable[..., T]) -> T:
    return functools.lru_cache()(func)  # type: ignore
