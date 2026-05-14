import torch

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import InferenceTarget, TractNNEF
from torch_to_nnef.op.helper import AtenOpRegistry

OP_REGISTRY = AtenOpRegistry()


def tract_complex_support(inference_target: InferenceTarget) -> bool:
    return (
        isinstance(inference_target, TractNNEF)
        and "complex" in inference_target.feature_flags
        and inference_target.version < "0.20.0"
    )


def is_complex_dtype_and_complex_only_supported_as_lastdim(
    dtype, inference_target: InferenceTarget
) -> bool:
    return dtype in [
        torch.complex64,
        torch.complex128,
    ] and not tract_complex_support(inference_target)


@OP_REGISTRY.register()
def view_as_complex(
    node,
    inference_target,
    torch_graph,
    **kwargs,
):
    """Map PyTorch: 'aten:view_as_complex' to NNEF."""
    if tract_complex_support(inference_target):
        raise T2NErrorNotImplemented("Complex not supported in vanilla spec")
    # in such case we simulate complex with additional last axis being x2
    # 1 for real
    # 1 for imaginary
    # this means that rest of the flow still need to handle this design
    # decision.
    node.inputs[0].dtype = torch.complex64
    torch_graph.remap_node(node.outputs[0], node.inputs[0])
    return []


@OP_REGISTRY.register()
def view_as_real(
    node,
    torch_graph,
    inference_target,
    **kwargs,
):
    """Map PyTorch: 'aten:view_as_real' to NNEF."""
    if tract_complex_support(inference_target):
        raise T2NErrorNotImplemented("Complex not supported by vanilla NNEF")
    # in such case we simulate complex with additional last axis being x2
    # 1 for real
    # 1 for imaginary
    # this means that rest of the flow still need to handle this design
    # decision.
    torch_graph.remap_node(node.outputs[0], node.inputs[0])
    return []


@OP_REGISTRY.register()
def angle(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:angle' to NNEF.

    Calls the `angle` fragment (`atan2(imag, real)` on a `(..., 2)`
    complex layout). The `atan2` fragment is quadrant-aware, so the
    result matches `torch.angle` across the full plane (signed-zero
    edge cases excepted).
    """
    if tract_complex_support(inference_target):
        raise T2NErrorNotImplemented("Complex not supported in vanilla spec")
    (input_node,) = node.inputs
    inp_ref = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "angle",
        inputs=[inp_ref],
        attrs={"axis": input_node.rank - 1},
    )
    return ["atan2", "angle"]


@OP_REGISTRY.register()
def complex(  # pylint: disable=redefined-builtin
    node, op_helper, inference_target, **kwargs
):
    """Map PyTorch: 'aten:complex(real, imag)' to NNEF.

    Stacks `real` / `imag` on a new trailing axis via the `complex`
    fragment so the result matches t2n's `(..., 2)` layout (mirror of
    `polar` without the `cos` / `sin`).
    """
    if tract_complex_support(inference_target):
        raise T2NErrorNotImplemented("Complex not supported in vanilla spec")
    real_node, imag_node = node.inputs
    # Clear the trace's complex64 marker on the IR output: we actually
    # emit a real `(..., 2)` tensor, and downstream implicit-cast logic
    # otherwise tries to coerce the real inputs to `complexf64`.
    node.outputs[0].dtype = torch.float32
    node.outputs[0].shape = list(real_node.shape) + [2]
    real_ref = op_helper.get_or_add_tensor_variable_in_nnef(real_node)
    imag_ref = op_helper.get_or_add_tensor_variable_in_nnef(imag_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "complex",
        inputs=[real_ref, imag_ref],
        attrs={"axis": real_node.rank},
    )
    return ["complex"]


def _emit_conjugate(node, op_helper, inference_target, torch_graph, op_label):
    """Shared body for `aten::conj` / `aten::conj_physical`.

    On a real input we leave the trace untouched (the conjugate of a
    real is itself). On a complex input we route through the
    `conjugate` NNEF fragment which flips the sign of the imag slice
    on the trailing-2 axis.
    """
    if tract_complex_support(inference_target):
        raise T2NErrorNotImplemented("Complex not supported in vanilla spec")
    (input_node,) = node.inputs
    if input_node.dtype not in (torch.complex64, torch.complex128):
        torch_graph.remap_node(node.outputs[0], input_node)
        return []
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    # On a complex input the trailing-2 axis is already part of the
    # t2n IR rank (`view_as_complex` re-tagged the input's dtype to
    # `complex64` without changing its shape), so the complex axis
    # sits at `rank - 1`, not `rank`.
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "conjugate",
        inputs=[inp],
        attrs={"axis": input_node.rank - 1},
    )
    return ["conjugate"]


@OP_REGISTRY.register(["conj", "_conj"])
def conj(node, op_helper, inference_target, torch_graph, **kwargs):
    """Map PyTorch: 'aten:conj' / 'aten::_conj' to NNEF.

    The trace usually pairs `conj` with `resolve_conj` (which stays in
    `identity_remap` as a no-op); we put the actual sign flip here.
    """
    return _emit_conjugate(
        node, op_helper, inference_target, torch_graph, "conj"
    )


@OP_REGISTRY.register()
def conj_physical(node, op_helper, inference_target, torch_graph, **kwargs):
    """Map PyTorch: 'aten:conj_physical' to NNEF.

    Standalone version of `conj`; same code path. Previously routed to
    `identity_remap`, which silently produced the wrong answer for
    complex inputs (returned the input unchanged instead of
    conjugating it).
    """
    return _emit_conjugate(
        node, op_helper, inference_target, torch_graph, "conj_physical"
    )


@OP_REGISTRY.register()
def polar(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:polar(abs, angle)' to NNEF.

    Calls the `polar` fragment which builds `(abs*cos, abs*sin)` on a
    new trailing axis (matching t2n's `(..., 2)` complex layout).
    """
    if tract_complex_support(inference_target):
        raise T2NErrorNotImplemented("Complex not supported in vanilla spec")
    abs_node, angle_node = node.inputs
    # The fragment writes a real `(..., 2)` tensor; clear the trace's
    # `complex64` marker on the output node so downstream implicit-cast
    # logic doesn't try to coerce the real inputs to `complexf64`
    # (tract has no such dtype). Same for the shape: the trace carries
    # the bare complex shape, we actually emit `(..., 2)`.
    node.outputs[0].dtype = torch.float32
    node.outputs[0].shape = list(abs_node.shape) + [2]
    abs_ref = op_helper.get_or_add_tensor_variable_in_nnef(abs_node)
    angle_ref = op_helper.get_or_add_tensor_variable_in_nnef(angle_node)
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "polar",
        inputs=[abs_ref, angle_ref],
        attrs={"axis": abs_node.rank},
    )
    return ["polar"]


@OP_REGISTRY.register()
def sgn(node, op_helper, inference_target, **kwargs):
    """Map PyTorch: 'aten:sgn' to NNEF.

    Real input: alias of `sign` (-1 / 0 / +1).  Complex input
    (stored as `(..., 2)` real): `z / |z|` with `0 -> 0`, routed
    through the `sgn_complex` fragment.
    """
    if tract_complex_support(inference_target):
        raise T2NErrorNotImplemented("Complex not supported in vanilla spec")
    (input_node,) = node.inputs
    inp = op_helper.get_or_add_tensor_variable_in_nnef(input_node)
    if input_node.dtype not in (torch.complex64, torch.complex128):
        op_helper.add_single_output_op_from_nnef_tensors(
            node, "sign", inputs=inp
        )
        return []
    # Complex path: trailing-2 axis already lives in the t2n IR rank
    # (see `_emit_conjugate` for the same convention).
    op_helper.add_single_output_op_from_nnef_tensors(
        node,
        "sgn_complex",
        inputs=[inp],
        attrs={"axis": input_node.rank - 1},
    )
    return ["sgn_complex"]
