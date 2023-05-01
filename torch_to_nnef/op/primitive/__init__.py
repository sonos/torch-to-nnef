import logging
import operator
import typing as T
from functools import reduce

from torch_to_nnef.op.primitive import (
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
    unary,
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
            unary,
        ]
    ],
)

LOGGER = logging.getLogger(__name__)


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
    aten_op_id = node.kind.split("::")[1]

    # remap
    if aten_op_id.endswith("_"):
        aten_op_id = aten_op_id[:-1]

    try:
        return primitive_ops_registry.get(aten_op_id)(
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
            torch_graph=torch_graph,
            nnef_spec_strict=nnef_spec_strict,
            has_dynamic_axes=has_dynamic_axes,
            tract_feature_flags=tract_feature_flags,
            aten_op_id=aten_op_id,
        )
    except KeyError as exp:
        torch_graph.printall()
        raise exp
