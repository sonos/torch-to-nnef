"""Hypothesis-driven primitive op spec catalog.

`REGISTRY` is the single source of truth consumed by
`tests/test_primitive_proptest.py`. The catalog is split into themed
submodules (elementwise, reductions, shape, activation, norm, conv_pool,
specialty, factory); each exposes a single `SPECS` tuple and this module
just concatenates them.

`gaps` is the one module that does not describe something we ship: its
specs carry `nnef_gap` and exist so the ONNX sweep can measure operators
t2n cannot translate. They are part of the same registry on purpose, so
every consumer (the sweep, the attribution check, the page) sees them
without opting in.
"""

import typing as T

from . import (
    activation,
    conv_pool,
    elementwise,
    factory,
    gaps,
    loss,
    norm,
    reductions,
    shape,
    specialty,
)
from ._common import NnefGap, NnefGapStage, OpSample, OpSpec

__all__ = [
    "NnefGap",
    "NnefGapStage",
    "OpSample",
    "OpSpec",
    "REGISTRY",
]

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
    gaps,
)

REGISTRY: T.Tuple[OpSpec, ...] = tuple(
    spec for module in _MODULES for spec in module.SPECS
)
