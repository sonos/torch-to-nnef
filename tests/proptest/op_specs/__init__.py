"""Hypothesis-driven primitive op spec catalog.

`REGISTRY` is the single source of truth consumed by
`tests/test_primitive_proptest.py`. The catalog is split into themed
submodules (elementwise, reductions, shape, activation, norm, conv_pool,
specialty, factory); each exposes a single `SPECS` tuple and this module
just concatenates them.
"""

import typing as T

from . import (
    activation,
    conv_pool,
    elementwise,
    factory,
    loss,
    norm,
    reductions,
    shape,
    specialty,
)
from ._common import OpSample, OpSpec

__all__ = ["OpSample", "OpSpec", "REGISTRY"]

# Order kept stable so spec IDs remain unchanged for `pytest -k` filtering
# and Hypothesis's example-DB cache keys.
_MODULES = (
    elementwise,
    reductions,
    shape,
    activation,
    norm,
    conv_pool,
    specialty,
    factory,
    loss,
)

REGISTRY: T.Tuple[OpSpec, ...] = tuple(
    spec for module in _MODULES for spec in module.SPECS
)
