from __future__ import annotations

import torch

from torch_to_nnef.op.extras import register
from torch_to_nnef.utils import torch_version

# Define t2n_extra ops only when torch exposes the Library API (>= 2.0).
has_lib = (
    torch_version() >= "2.0.0"
    and hasattr(torch, "library")
    and hasattr(torch.library, "Library")
)
has_custom_op = hasattr(torch.library, "custom_op")

if has_lib:
    lib = torch.library.Library("t2n_extra", "DEF")

    # Guard redefinition if the test module is imported twice in the same run.
    try:
        _ = torch.ops.t2n_extra.unit_relu  # type: ignore[attr-defined]
    except AttributeError:
        lib.define("unit_relu(Tensor x) -> Tensor")

        def _cpu(x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
            return torch.relu(x)

        def _meta(x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
            return torch.empty_like(x, device="meta")

        lib.impl("unit_relu", _cpu, "CPU")
        lib.impl_abstract("unit_relu", _meta)

    # Meta-only variant to exercise exporter eager→meta fallback. Only define
    # on torch versions that expose the decorator-based API.
    if has_custom_op:
        try:
            _ = torch.ops.t2n_extra.unit_relu_meta_only  # type: ignore[attr-defined]
        except AttributeError:

            @torch.library.custom_op(
                "t2n_extra::unit_relu_meta_only",
                mutates_args=(),
                schema="(Tensor x) -> Tensor",
            )
            def _unit_relu_meta_only(x: torch.Tensor) -> torch.Tensor:
                # No CPU kernel; meta-only path exercises exporter fallback.
                return torch.relu(x)  # not used; CPU impl unregistered

            @_unit_relu_meta_only.register_fake
            def _unit_relu_meta_only_meta(x: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(x, device="meta")


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
