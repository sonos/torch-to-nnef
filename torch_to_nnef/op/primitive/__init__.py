import logging
import operator
import typing as T
from functools import reduce

from torch_to_nnef.op.primitive import (
    activation,
    axes_change,
    base,
    complex,
    concat,
    expand,
    fft,
    math,
    matmul,
    norm,
    other,
    pad,
    pool,
    qops,
    reducer,
    selector,
    split,
    tensor_build,
)

primitive_ops_registry = reduce(
    operator.add,
    [
        mod.OP_REGISTRY
        for mod in [
            activation,
            axes_change,
            complex,
            concat,
            expand,
            fft,
            math,
            matmul,
            norm,
            other,
            pad,
            pool,
            qops,
            reducer,
            selector,
            split,
            tensor_build,
        ]
    ],
)

LOGGER = logging.getLogger(__name__)

REMAP_ATEN_OP_NAMES = {
    "_relu": "relu",
    "reciprocal": "rcp",
    "bitwise_not": "not",
    "bitwise_not_cpu": "not",
    "bitwise_cpu": "and",
    "__and_": "and",
    "__or_": "or",
    "less": "lt",
    "greater": "gt",
    "less_equal": "le",
    "greater_equal": "ge",
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
        return base.unary_output_op_without_params(
            nnef_op_type=aten_op_name,
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
        )
    try:
        return primitive_ops_registry.get(aten_op_name)(
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
