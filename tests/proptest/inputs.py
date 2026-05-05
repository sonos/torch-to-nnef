"""Tensor-value strategies for hypothesis-driven primitive tests.

`tensor_st` builds a torch tensor from a numpy strategy and then converts via
``torch.from_numpy`` so that no torch RNG state is touched (which would leak
across hypothesis examples). The dtype parameter accepts EITHER a concrete
``torch.dtype`` OR a hypothesis strategy returning one, mirroring the
``hypothesis.extra.numpy.arrays(dtype=...)`` idiom.

For ops that need N tensors to share a dtype (binary same-dtype, matmul, cat,
where, etc.), the recommended pattern is to draw the dtype ONCE in the
composite via ``draw(dtype_st([...]))`` and thread the concrete value into
each ``tensor_st`` call. Drawing the dtype inside each ``tensor_st`` would
yield independent draws per tensor.
"""

import typing as T
from dataclasses import dataclass

import numpy as np
import torch
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

DtypeArg = T.Union[torch.dtype, st.SearchStrategy[torch.dtype]]


@dataclass(frozen=True)
class Interval:
    """Closed interval used to constrain a tensor's value range.

    For ops with restricted input domains (e.g. log on (0, +inf), acos on
    [-1, 1], atanh on (-1, 1) strict). Bounds are inclusive; if you need a
    strict lower bound, pass min + epsilon.
    """

    min: float
    max: float


# Mapping from torch dtypes to numpy dtypes for hypothesis numpy extra.
# torch.bfloat16 has no direct numpy dtype; we generate float32 and downcast.
_TORCH_TO_NUMPY_DTYPE: T.Dict[torch.dtype, np.dtype] = {
    torch.float32: np.dtype("float32"),
    torch.float16: np.dtype("float16"),
    torch.float64: np.dtype("float64"),
    torch.int64: np.dtype("int64"),
    torch.int32: np.dtype("int32"),
    torch.int16: np.dtype("int16"),
    torch.int8: np.dtype("int8"),
    torch.uint8: np.dtype("uint8"),
    torch.bool: np.dtype("bool_"),
}


def dtype_st(
    allowed: T.Sequence[torch.dtype],
) -> st.SearchStrategy[torch.dtype]:
    """Strategy returning one of the listed torch dtypes."""
    return st.sampled_from(list(allowed))


def _resolve_dtype(
    draw: T.Callable[[st.SearchStrategy[T.Any]], T.Any],
    dtype: DtypeArg,
) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return draw(dtype)


def _quantize_to_dtype(value: float, np_dtype: np.dtype) -> float:
    """Round a float to the nearest exactly-representable value of np_dtype.

    Hypothesis ``floats(width=...)`` rejects min/max bounds that are not
    exactly representable at the given width. This helper makes the domain
    bounds safe for any float width without forcing every caller to
    pre-quantize.
    """
    return float(np.array(value, dtype=np_dtype).item())


def _elements_strategy(
    np_dtype: np.dtype,
    finite: bool,
    domain: T.Optional[Interval],
    allow_subnormal: bool,
) -> st.SearchStrategy[T.Any]:
    if np_dtype == np.dtype("bool_"):
        return st.booleans()
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        lo = int(domain.min) if domain is not None else info.min
        hi = int(domain.max) if domain is not None else info.max
        return st.integers(
            min_value=max(lo, info.min),
            max_value=min(hi, info.max),
        )
    # Float path.
    width_bits = {
        np.dtype("float16"): 16,
        np.dtype("float32"): 32,
        np.dtype("float64"): 64,
    }[np_dtype]
    kwargs: T.Dict[str, T.Any] = {
        "width": width_bits,
        "allow_nan": not finite,
        "allow_infinity": not finite,
        "allow_subnormal": allow_subnormal,
    }
    if domain is not None:
        # Quantize the domain bounds to be exactly representable at this
        # float width; otherwise hypothesis raises InvalidArgument.
        kwargs["min_value"] = _quantize_to_dtype(domain.min, np_dtype)
        kwargs["max_value"] = _quantize_to_dtype(domain.max, np_dtype)
    return st.floats(**kwargs)


@st.composite
def tensor_st(
    draw,
    shape: T.Tuple[int, ...],
    dtype: DtypeArg,
    finite: bool = True,
    domain: T.Optional[Interval] = None,
    allow_subnormal: bool = False,
) -> torch.Tensor:
    """Strategy returning a torch tensor of the given shape and dtype.

    Args:
        draw: hypothesis composite draw function (injected automatically).
        shape: concrete shape tuple. Use a shape strategy upstream and pass
            the drawn value here.
        dtype: a concrete ``torch.dtype`` or a strategy returning one.
        finite: when True (default), exclude NaN and Inf from the value pool.
            When False, NaN and Inf may be drawn -- only do this for ops whose
            property under test depends on them (e.g. isnan, isinf).
        domain: optional Interval bound on values. For ops with restricted
            input domains (log, sqrt, acos, etc.).
        allow_subnormal: when True, do not exclude subnormals. Default False
            because tract may flush-to-zero on some backends.
    """
    resolved = _resolve_dtype(draw, dtype)

    # bf16 has no numpy dtype; generate as float32 then cast to bf16 via torch.
    if resolved == torch.bfloat16:
        np_dtype = np.dtype("float32")
        cast_to_bf16 = True
    else:
        np_dtype = _TORCH_TO_NUMPY_DTYPE[resolved]
        cast_to_bf16 = False

    elements = _elements_strategy(
        np_dtype, finite=finite, domain=domain, allow_subnormal=allow_subnormal
    )
    arr = draw(npst.arrays(dtype=np_dtype, shape=shape, elements=elements))
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    if cast_to_bf16:
        tensor = tensor.to(torch.bfloat16)
    return tensor
