from __future__ import annotations

import torch

from torch_to_nnef.op.extras import register
from torch_to_nnef.utils import torch_version

# 1) Declare a custom torch op under the `t2n_extra` namespace and
#    provide simple CPU + meta implementations so that `model(*args)`
#    can run during export. Only do this on torch >= 2.0 with Library.
if (
    torch_version() >= "2.0.0"
    and hasattr(torch, "library")
    and hasattr(torch.library, "Library")
):
    lib = torch.library.Library("t2n_extra", "DEF")
    try:
        _ = torch.ops.t2n_extra.my_relu  # type: ignore[attr-defined]
    except AttributeError:
        lib.define("my_relu(Tensor x) -> Tensor")

        def _my_relu_cpu(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

        def _my_relu_meta(x: torch.Tensor) -> torch.Tensor:
            # Create a meta tensor with the same dtype/shape.
            return torch.empty_like(x, device="meta")

        lib.impl("my_relu", _my_relu_cpu, "CPU")
        # `Library.impl_abstract` was renamed to `torch.library.register_fake`
        # in torch 2.4 and removed later; keep the old path for torch 2.0-2.3.
        if hasattr(torch.library, "register_fake"):
            torch.library.register_fake("t2n_extra::my_relu", _my_relu_meta)
        else:
            lib.impl_abstract("my_relu", _my_relu_meta)


# 2) Register the export handler that maps the IR node to NNEF.
@register("my_relu")
def my_relu(
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
    # Convert input to an NNEF tensor
    x = op_helper.get_or_add_tensor_variable_in_nnef(node.inputs[0])
    # Emit an NNEF relu and force the output name to match the traced tensor
    op_helper.add_single_output_op_from_nnef_tensors(
        node=node,
        nnef_op_type="relu",
        inputs=x,
        force_full_output_tensor_name=node.outputs[0].export_name,
    )
    # No custom fragment emitted
    return []
