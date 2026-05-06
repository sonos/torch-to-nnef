"""Dtype-aware tolerance table for the proptest comparator.

The tolerance levels mirror `TractCheckTolerance` (see
`torch_to_nnef.inference_target.tract`) but are resolved to (rtol, atol)
pairs per dtype, since the meaningful epsilon is dtype-dependent.

Integer and bool dtypes always require bit-exact comparison; tolerance is
ignored for them.
"""

import typing as T
from dataclasses import dataclass

import torch

from torch_to_nnef.dtypes import dtype_is_floating_point
from torch_to_nnef.inference_target.tract import TractCheckTolerance


@dataclass(frozen=True)
class Tol:
    rtol: float
    atol: float


_FLOAT_TOL_TABLE: T.Dict[T.Tuple[torch.dtype, TractCheckTolerance], Tol] = {
    # f64: same scale as f32 entries; tract may downcast to f32 in
    # practice on most ops, but f64 inputs/outputs flow through the
    # NPZ pipeline and the comparator should accept the same tolerance.
    (torch.float64, TractCheckTolerance.EXACT): Tol(0.0, 0.0),
    (torch.float64, TractCheckTolerance.APPROXIMATE): Tol(1e-6, 1e-6),
    (torch.float64, TractCheckTolerance.CLOSE): Tol(1e-5, 1e-5),
    (torch.float64, TractCheckTolerance.VERY): Tol(1e-4, 1e-4),
    (torch.float64, TractCheckTolerance.SUPER): Tol(1e-3, 1e-3),
    (torch.float64, TractCheckTolerance.ULTRA): Tol(1e-2, 1e-2),
    (torch.float32, TractCheckTolerance.EXACT): Tol(0.0, 0.0),
    (torch.float32, TractCheckTolerance.APPROXIMATE): Tol(1e-6, 1e-6),
    (torch.float32, TractCheckTolerance.CLOSE): Tol(1e-5, 1e-5),
    (torch.float32, TractCheckTolerance.VERY): Tol(1e-4, 1e-4),
    (torch.float32, TractCheckTolerance.SUPER): Tol(1e-3, 1e-3),
    (torch.float32, TractCheckTolerance.ULTRA): Tol(1e-2, 1e-2),
    (torch.float16, TractCheckTolerance.EXACT): Tol(0.0, 0.0),
    (torch.float16, TractCheckTolerance.APPROXIMATE): Tol(1e-3, 1e-3),
    (torch.float16, TractCheckTolerance.CLOSE): Tol(5e-3, 5e-3),
    (torch.float16, TractCheckTolerance.VERY): Tol(1e-2, 1e-2),
    (torch.float16, TractCheckTolerance.SUPER): Tol(5e-2, 5e-2),
    (torch.float16, TractCheckTolerance.ULTRA): Tol(1e-1, 1e-1),
    (torch.bfloat16, TractCheckTolerance.EXACT): Tol(0.0, 0.0),
    (torch.bfloat16, TractCheckTolerance.APPROXIMATE): Tol(5e-3, 5e-3),
    (torch.bfloat16, TractCheckTolerance.CLOSE): Tol(1e-2, 1e-2),
    (torch.bfloat16, TractCheckTolerance.VERY): Tol(5e-2, 5e-2),
    (torch.bfloat16, TractCheckTolerance.SUPER): Tol(1e-1, 1e-1),
    (torch.bfloat16, TractCheckTolerance.ULTRA): Tol(2e-1, 2e-1),
}


def lookup_tol(dtype: torch.dtype, level: TractCheckTolerance) -> Tol:
    """Return the (rtol, atol) pair for a dtype at a given tolerance level.

    Integer/bool dtypes always return `Tol(0, 0)` regardless of level.

    Raises:
        KeyError: if the dtype is unknown to the proptest layer.
    """
    if not dtype_is_floating_point(dtype):
        return Tol(0.0, 0.0)
    return _FLOAT_TOL_TABLE[(dtype, level)]
