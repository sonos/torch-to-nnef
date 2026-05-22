from __future__ import annotations

import torch

from torch_to_nnef.op.extras import register

# Define a simple t2n_extra op and provide CPU + meta behavior so that
# both eager and meta forward paths can succeed during export.
try:
    lib = torch.library.Library("t2n_extra", "DEF")
    lib.define("unit_relu(Tensor x) -> Tensor")

    def _cpu(x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return torch.relu(x)

    def _meta(x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return torch.empty_like(x, device="meta")

    lib.impl("unit_relu", _cpu, "CPU")
    lib.impl_abstract("unit_relu", _meta)

    # Define a meta-only variant to exercise exporter eager→meta fallback.
    lib.define("unit_relu_meta_only(Tensor x) -> Tensor")

    def _meta_only(x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return torch.empty_like(x, device="meta")

    lib.impl_abstract("unit_relu_meta_only", _meta_only)
except Exception:
    pass


@register("unit_relu")
def unit_relu(
    g,
    node,
    name_to_tensor,
    null_ref,
    *,
    torch_graph,
    inference_target,
    op_helper,
    **_,
):
    x = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node=node,
        nnef_op_type="relu",
        inputs=x,
        force_full_output_tensor_name=node.outputs[0].export_name,
    )
    return []


@register("unit_relu_meta_only")
def unit_relu_meta_only(
    g,
    node,
    name_to_tensor,
    null_ref,
    *,
    torch_graph,
    inference_target,
    op_helper,
    **_,
):
    x = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    op_helper.add_single_output_op_from_nnef_tensors(
        node=node,
        nnef_op_type="relu",
        inputs=x,
        force_full_output_tensor_name=node.outputs[0].export_name,
    )
    return []
