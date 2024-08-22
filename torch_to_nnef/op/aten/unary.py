from torch_to_nnef.op import helper

REMAP_ATEN_OP_NAMES = {
    "_relu": "relu",
    "reciprocal": "rcp",
    "bitwise_not": "not",
    "bitwise_not_cpu": "not",
    "bitwise_cpu": "and",
    "__and__": "and",
    "__or__": "or",
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


OP_REGISTRY = helper.AtenOpRegistry()


@OP_REGISTRY.register(
    torch_op_ids=GENERIC_UNARY_OUTPUT_ATEN_OP_NAMES
    + list(REMAP_ATEN_OP_NAMES.keys())
)
def generic_unary(aten_op_id, g, node, name_to_tensor, null_ref, **kwargs):
    aten_op_id = REMAP_ATEN_OP_NAMES.get(aten_op_id, aten_op_id)
    return helper.unary_output_op_without_params(
        nnef_op_type=aten_op_id,
        g=g,
        node=node,
        name_to_tensor=name_to_tensor,
        null_ref=null_ref,
    )
