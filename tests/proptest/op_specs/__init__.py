"""Hypothesis-driven primitive op spec catalog.

``REGISTRY`` is the single source of truth consumed by
``tests/test_primitive_proptest.py``. This package splits the catalog into
themed submodules (elementwise, reductions, shape, activation, norm,
conv_pool, specialty, factory) so each future spec edit touches one small
file rather than the whole registry.
"""

import typing as T

from ._common import OpSample, OpSpec
from .activation import _activation_specs, _softmax_specs
from .conv_pool import (
    _conv3d_pool3d_helpers_specs,
    _depth_conv_pool_specs,
    _pool_specs,
)
from .elementwise import (
    _binary_arith_specs,
    _binary_compare_specs,
    _binary_logical_specs,
    _bitwise_builder_specs,
    _clamp_where_specs,
    _unary_broad_specs,
    _unary_specs,
)
from .factory import (
    _constructors_index_sdpa_specs,
    _fft_specs,
    _glue_specs,
)
from .norm import (
    _depth_norm_topk_cat_specs,
    _norm_conv_matmul_specs,
    _norm_specs,
)
from .reductions import _depth_reduction_dtype_specs, _reduction_specs
from .shape import (
    _concat_split_specs,
    _pad_specs,
    _selector_specs,
    _shape_specs,
    _sort_scatter_specs,
)
from .specialty import (
    _final_specs,
    _prelu_glu_einsum_specs,
    _specialty_specs,
)

__all__ = ["OpSpec", "OpSample", "REGISTRY"]


def _build_registry() -> T.Tuple[OpSpec, ...]:
    specs: T.List[OpSpec] = []
    specs.extend(_unary_specs())
    specs.extend(_unary_broad_specs())
    specs.extend(_binary_arith_specs())
    specs.extend(_binary_compare_specs())
    specs.extend(_binary_logical_specs())
    specs.extend(_reduction_specs())
    specs.extend(_shape_specs())
    specs.extend(_clamp_where_specs())
    specs.extend(_activation_specs())
    specs.extend(_softmax_specs())
    specs.extend(_selector_specs())
    specs.extend(_pool_specs())
    specs.extend(_norm_conv_matmul_specs())
    specs.extend(_concat_split_specs())
    specs.extend(_pad_specs())
    specs.extend(_norm_specs())
    specs.extend(_sort_scatter_specs())
    specs.extend(_conv3d_pool3d_helpers_specs())
    specs.extend(_bitwise_builder_specs())
    specs.extend(_specialty_specs())
    specs.extend(_prelu_glu_einsum_specs())
    specs.extend(_final_specs())
    specs.extend(_constructors_index_sdpa_specs())
    specs.extend(_fft_specs())
    specs.extend(_glue_specs())
    specs.extend(_depth_conv_pool_specs())
    specs.extend(_depth_norm_topk_cat_specs())
    specs.extend(_depth_reduction_dtype_specs())
    return tuple(specs)


REGISTRY: T.Tuple[OpSpec, ...] = _build_registry()
