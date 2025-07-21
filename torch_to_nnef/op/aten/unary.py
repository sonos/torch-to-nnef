from torch_to_nnef.op import helper

REMAP_ATEN_OP_NAMES = {
    "__and__": "and",
    "__or__": "or",
    "_relu": "relu",
    "greater": "gt",
    "greater_equal": "ge",
    "less": "lt",
    "less_equal": "le",
    "logical_not": "not",
    "logical_and": "and",
    "logical_or": "or",
    "reciprocal": "rcp",
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
def generic_unary(aten_op_id, node, op_helper, **kwargs):
    aten_op_id = REMAP_ATEN_OP_NAMES.get(aten_op_id, aten_op_id)
    return op_helper.unary_output_op_without_attr(
        nnef_op_type=aten_op_id,
        node=node,
    )
