"""PyTorch Aten::* operators translation."""

import logging
import operator
import typing as T
from functools import reduce

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import InferenceTarget

# pylint: disable-next=redefined-builtin
from torch_to_nnef.op.aten import (
    activation,
    attn,
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
from torch_to_nnef.op.helper import OpHelper

aten_ops_registry = reduce(
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
            attn,
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
    inference_target: InferenceTarget,
) -> T.Optional[T.List[str]]:
    """Main primitive dispatcher.

    Allow to write in graph any not Quantized Operation from pytorch defined in
    node attribute.

    """
    ops_family, aten_op_id = node.kind.split("::")
    assert ops_family == "aten"

    # remap
    if aten_op_id.endswith("_") and not aten_op_id.endswith("__"):
        aten_op_id = aten_op_id[:-1]

    try:
        return aten_ops_registry.get(aten_op_id)(
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            null_ref=null_ref,
            torch_graph=torch_graph,
            inference_target=inference_target,
            aten_op_id=aten_op_id,
            op_helper=OpHelper(
                g, node, name_to_tensor, null_ref, inference_target
            ),
        )
    except KeyError as exp:
        torch_graph.printall()
        raise T2NErrorNotImplemented(f"unregistered {aten_op_id}") from exp
