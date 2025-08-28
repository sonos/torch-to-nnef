import logging

import numpy as np

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import KhronosNNEF, TractNNEF
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_single_output_op,
    cast_to_if_not_dtype_and_variable,
    get_or_add_tensor_variable_in_nnef,
)

LOGGER = logging.getLogger(__name__)

OP_REGISTRY = AtenOpRegistry()


@OP_REGISTRY.register()
def quantize_per_tensor(g, node, name_to_tensor, inference_target, **kwargs):
    """Map PyTorch: 'aten:quantize_per_tensor' to NNEF."""
    (
        input_node,
        scale_node,
        zero_point_node,
        dtype_node,
    ) = node.inputs
    assert dtype_node.data == 13, "is not expected quint8"
    input_node = node.inputs[0]
    tensor = get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor)
    quant_infos = {
        "zero_point": zero_point_node.data,
        "scale": scale_node.data,
        "bits": 8,
        "signed": False,
        "symmetric": False,
        "op-name": "zero_point_linear_quantize",
    }
    if isinstance(inference_target, KhronosNNEF):
        LOGGER.debug(
            "quantize with KhronosNNEF: set quant info on direct output"
        )
        tensor.quant = quant_infos
        return []
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        "tract_core_cast",
        inputs=tensor,
    )
    out.quant = quant_infos
    return ["tract_core"]


@OP_REGISTRY.register()
def dequantize(g, node, name_to_tensor, inference_target, **kwargs):
    """Translate `aten::dequantize` to NNEF.

    We will only handle the case of zero_point affine quantization for now..
    which in reverse of quantization is:

       (x - zero_point) * scale
    """
    input_node = node.inputs[0]
    nnef_tensor = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    if not isinstance(inference_target, TractNNEF):
        raise T2NErrorNotImplemented(inference_target)
    _, fragment_names = cast_to_if_not_dtype_and_variable(
        g,
        name_to_tensor,
        node,
        nnef_tensor,
        cast_to=np.float32,
    )
    return fragment_names
