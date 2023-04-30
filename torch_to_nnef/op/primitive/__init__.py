# pylint: disable=too-many-lines
import logging
import typing as T

from torch_to_nnef.op.primitive.activation import (
    clamp,
    clamp_max,
    clamp_min,
    elu,
    erf,
    gelu,
    glu,
    hardtanh,
    leaky_relu,
    log_softmax,
    prelu,
    selu,
    silu,
    softmax,
    softplus,
)
from torch_to_nnef.op.primitive.axes_change import (
    flatten,
    permute,
    reshape,
    squeeze,
    transpose,
    unsqueeze,
    view,
)
from torch_to_nnef.op.primitive.base import unary_output_op_without_params
from torch_to_nnef.op.primitive.complex import view_as_complex, view_as_real
from torch_to_nnef.op.primitive.concat import cat, roll, stack
from torch_to_nnef.op.primitive.expand import expand, repeat
from torch_to_nnef.op.primitive.fft import fft_fft, fft_ifft, stft
from torch_to_nnef.op.primitive.math import (
    abs,
    div,
    floor_divide,
    log10,
    mul,
    pow_,
    remainder,
    round_,
    rsub,
    trunc,
)
from torch_to_nnef.op.primitive.matmul import (
    _convolution,
    baddbmm,
    einsum,
    linear,
    matmul,
)
from torch_to_nnef.op.primitive.norm import (
    batch_norm,
    group_norm,
    layer_norm,
    norm,
)
from torch_to_nnef.op.primitive.other import (
    contiguous,
    detach,
    dropout,
    external,
    size,
    to,
)
from torch_to_nnef.op.primitive.pad import pad
from torch_to_nnef.op.primitive.pool import (
    adaptive_avg_pool2d,
    avg_pool1d,
    avg_pool2d,
    max_pool1d,
    max_pool2d,
)
from torch_to_nnef.op.primitive.qops import dequantize, quantize_per_tensor
from torch_to_nnef.op.primitive.reducer import (
    argmax,
    argmin,
    max_,
    mean,
    min_,
    reduce_all,
    reduce_any,
    reduce_max,
    reduce_min,
    reduce_sum,
)
from torch_to_nnef.op.primitive.selector import (
    embedding,
    index_,
    masked_fill,
    narrow,
    select,
    slice_,
    where,
)
from torch_to_nnef.op.primitive.split import chunk, split_with_sizes, unbind
from torch_to_nnef.op.primitive.tensor_build import (
    arange,
    copy,
    new_zeros,
    ones,
    zeros,
    zeros_like,
)

# silence pyflakes F401 {
assert view_as_complex
assert view_as_real
assert fft_fft
assert fft_ifft
assert stft

assert softmax
assert softplus
assert elu
assert leaky_relu
assert prelu
assert selu
assert silu
assert gelu
assert erf
assert hardtanh
assert log_softmax
assert glu
assert clamp
assert clamp_min
assert clamp_max

assert norm
assert batch_norm
assert group_norm
assert layer_norm

assert max_pool1d
assert avg_pool1d
assert max_pool2d
assert avg_pool2d
assert adaptive_avg_pool2d

assert arange
assert ones
assert zeros_like
assert new_zeros
assert zeros
assert copy

assert pad

assert expand
assert repeat

assert slice_
assert where
assert narrow
assert select
assert index_
assert embedding
assert masked_fill

assert mul
assert div
assert floor_divide
assert trunc
assert pow_
assert round_
assert remainder
assert rsub
assert abs
assert log10

assert quantize_per_tensor
assert dequantize

assert mean
assert reduce_sum
assert argmax
assert argmin
assert reduce_any
assert reduce_all
assert reduce_max
assert reduce_min
assert max_
assert min_

assert split_with_sizes
assert unbind
assert chunk

assert view
assert transpose
assert permute
assert unsqueeze
assert squeeze
assert flatten
assert reshape

assert _convolution
assert linear
assert einsum
assert matmul
assert baddbmm

assert external
assert dropout
assert detach
assert contiguous
assert to
assert size

assert cat
assert stack
assert roll

# }


LOGGER = logging.getLogger(__name__)

REMAP_ATEN_OP_NAMES = {
    "_relu": "relu",
    "reciprocal": "rcp",
    "clone": "copy",
    "bitwise_not": "not",
    "bitwise_not_cpu": "not",
    "bitwise_cpu": "and",
    "__and_": "and",
    "__or_": "or",
    "less": "lt",
    "greater": "gt",
    "less_equal": "le",
    "greater_equal": "ge",
    "reflection_pad1d": "reflection_padnd",
    "replication_pad1d": "replication_padnd",
    "constant_pad1d": "constant_padnd",
    # avoid to ovewrite python builtin {
    "any": "reduce_any",
    "all": "reduce_all",
    "sum": "reduce_sum",
    "pow": "pow_",
    "max": "max_",
    "min": "min_",
    "slice": "slice_",
    "round": "round_",
    "index": "index_",
    # }
    "bmm": "matmul",  # since NNEF matmul does not care about rank
    "amax": "reduce_max",
}

GENERIC_UNARY_OUTPUT_ATEN_OP_NAMES = [
    "relu",
    "sigmoid",
    "log",
    "exp",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "sign",
    "neg",
    "floor",
    "ceil",
    "sqrt",
    "rsqrt",
    "log2",
    "rcp",
    "not",
    "eq",
    "ne",
    "add",
    "sub",
    "lt",
    "gt",
    "le",
    "ge",
    "and",
    "or",
]


def aten_to_nnef_tensor_and_ops(
    g,
    node,
    name_to_tensor,
    null_ref,
    torch_graph,
    nnef_spec_strict: bool = False,
    has_dynamic_axes: bool = False,
    tract_feature_flags: T.Optional[T.Set[str]] = None,
) -> T.Optional[T.List[str]]:
    """Main primitive dispatcher

    Allow to write in graph any not Quantized Operation from pytorch defined in
    node attribute.

    """
    aten_op_name = node.kind.split("::")[1]

    # remap
    if aten_op_name.endswith("_"):
        aten_op_name = aten_op_name[:-1]
    aten_op_name = REMAP_ATEN_OP_NAMES.get(aten_op_name, aten_op_name)

    if aten_op_name in GENERIC_UNARY_OUTPUT_ATEN_OP_NAMES:
        return unary_output_op_without_params(
            nnef_op_type=aten_op_name,
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
    try:
        return globals()[aten_op_name](
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
            torch_graph=torch_graph,
            nnef_spec_strict=nnef_spec_strict,
            has_dynamic_axes=has_dynamic_axes,
            tract_feature_flags=tract_feature_flags,
        )
    except KeyError as exp:
        torch_graph.printall()
        raise exp
