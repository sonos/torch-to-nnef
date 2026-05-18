import math

from torch_to_nnef.inference_target import TractNNEF
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

# Ops whose standard NNEF spec name differs from tract's registered op name.
TRACT_OP_ALIASES = {
    # NNEF spec: `rcp`; tract registers `recip` (ops::math::Recip).
    "rcp": "recip",
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
    # `add` and `sub` deliberately left out: PyTorch passes a third input
    # (`alpha`) that this generic handler would silently drop. They have
    # dedicated emitters in `torch_to_nnef/op/aten/math.py`.
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
        'aten:eq', 'aten:ne', 'aten:lt', 'aten:gt',
        'aten:le', 'aten:ge', 'aten:and', 'aten:or', 'aten:__and__',
        'aten:__or__', 'aten:_relu', 'aten:greater', 'aten:greater_equal',
        'aten:less', 'aten:less_equal', 'aten:logical_not', 'aten:logical_and',
        'aten:logical_or', 'aten:reciprocal', 'aten:minimum', 'aten:maximum'
    """
    nnef_name = REMAP_ATEN_OP_NAMES.get(aten_op_id, aten_op_id)
    inference_target = kwargs.get("inference_target")
    if isinstance(inference_target, TractNNEF):
        nnef_name = TRACT_OP_ALIASES.get(nnef_name, nnef_name)

    # PyTorch's silent int->float promotion (`sqrt(int) -> float`,
    # `sin(int) -> float`, ...) is bridged at the NNEF emit layer:
    # the float-result NNEF op names are registered in
    # `OPS_IMPLICIT_CAST_BY_OUTPUT_DTYPE`, and
    # `add_single_output_op_from_nnef_tensors` automatically casts
    # integer inputs to match the trace's recorded output dtype.
    # Falling through to the generic emitter is sufficient here.
    return op_helper.unary_output_op_without_attr(
        nnef_op_type=nnef_name,
        node=node,
    )


@OP_REGISTRY.register()
def positive(node, op_helper, **kwargs):
    """Map `aten::positive` to NNEF.

    `torch.positive(x)` is a no-op identity (parity counterpart of
    `torch.negative`); emit `mul(x, 1.0)` since NNEF's stdlib `copy`
    is not in tract's op set.
    """
    _emit_scalar_mul(node, op_helper, 1.0)


@OP_REGISTRY.register()
def ravel(node, op_helper, **kwargs):
    """Map `aten::ravel(x)` to NNEF: flatten to 1D.

    Equivalent to `flatten(x, 0, -1)`; emit a `reshape` to `[-1]`.
    """
    (input_node,) = node.inputs
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "reshape",
        inputs=op_helper.get_or_add_tensor_variable_in_nnef(input_node),
        attrs={
            "dtype": node.outputs[0].np_dtype,
            "shape": [-1],
            "axis_start": 0,
            "axis_count": -1,
        },
    )


def _emit_scalar_mul(node, op_helper, factor: float):
    """Helper: emit `mul(x, factor)` lifting `factor` to a tensor."""
    inp = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "mul",
        inputs=inp,
        attrs={"y": factor},
    )


@OP_REGISTRY.register()
def deg2rad(node, op_helper, **kwargs):
    """Map `aten::deg2rad(x)` to NNEF: `x * (pi / 180)`."""
    _emit_scalar_mul(node, op_helper, math.pi / 180.0)


@OP_REGISTRY.register()
def rad2deg(node, op_helper, **kwargs):
    """Map `aten::rad2deg(x)` to NNEF: `x * (180 / pi)`."""
    _emit_scalar_mul(node, op_helper, 180.0 / math.pi)


@OP_REGISTRY.register()
def float_power(node, op_helper, **kwargs):
    """Map `aten::float_power(x, y)` to NNEF as `pow(x, y)`.

    `torch.float_power` widens to f64 internally for accuracy; the
    `pow` we emit keeps the trace dtype (f32 in the common case) and
    matches `torch.pow` within tract's normal tolerance.
    """
    a_node, b_node = node.inputs[:2]
    a_ref = op_helper.get_or_add_tensor_variable_in_nnef(a_node)
    b_ref = op_helper.get_or_add_tensor_variable_in_nnef(b_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node, "pow", inputs=[a_ref, b_ref]
    )
