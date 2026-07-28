"""Hypothesis-driven primitive op spec catalog.

`REGISTRY` is the single source of truth consumed by
`tests/test_primitive_proptest.py`. The catalog is split into themed
submodules (elementwise, reductions, shape, activation, norm, conv_pool,
specialty, factory); each exposes a single `SPECS` tuple and this module
just concatenates them.

Operators we cannot translate live in these same themed modules, marked
with `nnef_gap`, rather than in a package of their own. That way
implementing one means deleting a field instead of moving a spec between
files, and a spec's home never depends on its support status. The rows
that get no spec at all are recorded in `untranslated.EXCLUDED`.
"""

import typing as T

from . import (
    activation,
    conv_pool,
    elementwise,
    factory,
    linalg,
    loss,
    norm,
    random_sampling,
    reductions,
    shape,
    special,
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
    linalg,
    special,
    random_sampling,
)

REGISTRY: T.Tuple[OpSpec, ...] = tuple(
    spec for module in _MODULES for spec in module.SPECS
)
