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
    "minimum": "min",
    "maximum": "max",
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
    """Map PyTorch generic operators to NNEF (direct map).

    List is:
        'aten:relu', 'aten:sigmoid', 'aten:log', 'aten:exp', 'aten:sin',
        'aten:cos', 'aten:tan', 'aten:asin', 'aten:acos', 'aten:atan',
        'aten:sinh', 'aten:cosh', 'aten:tanh', 'aten:asinh', 'aten:acosh',
        'aten:atanh', 'aten:sign', 'aten:neg', 'aten:floor', 'aten:ceil',
        'aten:sqrt', 'aten:rsqrt', 'aten:log2', 'aten:rcp', 'aten:not',
        'aten:eq', 'aten:ne', 'aten:add', 'aten:sub', 'aten:lt', 'aten:gt',
        'aten:le', 'aten:ge', 'aten:and', 'aten:or', 'aten:__and__',
        'aten:__or__', 'aten:_relu', 'aten:greater', 'aten:greater_equal',
        'aten:less', 'aten:less_equal', 'aten:logical_not', 'aten:logical_and',
        'aten:logical_or', 'aten:reciprocal', 'aten:minimum', 'aten:maximum'
    """
    aten_op_id = REMAP_ATEN_OP_NAMES.get(aten_op_id, aten_op_id)
    return op_helper.unary_output_op_without_attr(
        nnef_op_type=aten_op_id,
        node=node,
    )
