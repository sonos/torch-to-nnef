import functools
from functools import total_ordering
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


def cache(func: Callable[..., T]) -> T:
    """LRU cache helper that avoid pylint complains"""
    return functools.lru_cache()(func)  # type: ignore


def fullname(o) -> str:
    """Full class name with module path from an object"""
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def torch_version_within(lower: str, upper: str) -> bool:
    """lower included but upper not included"""
    torch_version = SemanticVersion.from_str(torch.__version__.split("+")[0])
    lower_version = SemanticVersion.from_str(lower)
    upper_version = SemanticVersion.from_str(upper)
    return lower_version <= torch_version < upper_version


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
