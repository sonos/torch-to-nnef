"""Aten-level RNN op handlers and shared fragment-emission helpers.

`aten::lstm_cell` surfaces directly when an inlined JIT graph (Phase 1
selective inline of non-importable submodules) exposes calls to the
underlying `_VF.lstm_cell` primitive without a wrapping `nn.LSTMCell`
module boundary.

Both this aten handler and the module-level `LSTMCellExtractor` go
through `emit_lstm_cell_via_fragment`, which materializes the cell math
as a single `lstm_cell` NNEF fragment call (see
`op/fragment/lstm_cell.nnef`). The fragment uses the same grouped-matmul
shape the flat decomposition would have emitted (one `(B, I) @ (4H, I).T`
followed by one `(B, H) @ (4H, H).T`, then slice into the 4 gates), so
runtime cost is unchanged. The benefit is graph shape: a single node
per cell step, mirroring the existing `lstm` / `gru` / `rnn` fragment
pattern in `op/fragment/`.
"""

from __future__ import annotations

import typing as T

import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_tensor_variable_node_as_nnef_tensor,
    get_or_add_tensor_variable_in_nnef,
)
from torch_to_nnef.torch_graph.ir_data import FixedTensorList, TensorVariable

OP_REGISTRY = AtenOpRegistry()


def _add_weight_tensor(
    g, name_to_tensor, base: str, suffix: str, t: torch.Tensor
) -> NTensor:
    tv = TensorVariable(
        name=getattr(t, "nnef_name", f"{base}_{suffix}"),
        data=t.detach(),
        shape=list(t.shape),
        dtype=t.dtype,
    )
    return get_or_add_tensor_variable_in_nnef(
        g, tv, name_to_tensor, name_suffix=suffix
    )


def _multi_output_pair(
    g,
    name_to_tensor,
    base: str,
    nnef_dtype,
    batch_dim: int,
    hidden: int,
    h_new_tv: T.Optional[TensorVariable],
    c_new_tv: T.Optional[TensorVariable],
) -> T.Tuple[NTensor, NTensor]:
    """Build the two output NNEF tensors for h_new / c_new.

    Bound to the IR's TensorVariable name when one is provided (the
    extractor / aten paths both pass their `node.outputs[*]` here so the
    fragment outputs land on the names downstream consumers expect),
    otherwise an anonymous intermediate.
    """

    def make(suffix: str, tv: T.Optional[TensorVariable]) -> NTensor:
        if tv is not None:
            return add_tensor_variable_node_as_nnef_tensor(
                g, tv, name_to_tensor, prevent_variable=True
            )
        nt = NTensor(
            g,
            name=f"{base}_{suffix}",
            dtype=nnef_dtype,
            shape=(batch_dim, hidden),
        )
        name_to_tensor[nt.name] = nt
        return nt

    return make("h_new", h_new_tv), make("c_new", c_new_tv)


def emit_lstm_cell_via_fragment(
    g,
    name_to_tensor,
    base: str,
    nnef_dtype,
    batch_dim: int,
    hidden: int,
    input_ref: NTensor,
    h_prev_ref: NTensor,
    c_prev_ref: NTensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: T.Optional[torch.Tensor],
    b_hh: T.Optional[torch.Tensor],
    h_new_tv: T.Optional[TensorVariable],
    c_new_tv: T.Optional[TensorVariable],
) -> T.List[str]:
    """Emit a single `lstm_cell` NNEF fragment call.

    The fragment internally does grouped `(B, I) @ (4H, I).T` and
    `(B, H) @ (4H, H).T`, adds the unsqueezed combined bias, slices into
    the 4 gates, and computes `(h_new, c_new)`. Caller owns nothing but
    the input refs and the four weight tensors.

    `h_new_tv` and `c_new_tv` are the t2n IR `TensorVariable`s for the
    user-facing outputs; they may be None if the caller wants anonymous
    intermediates. Returns the list of fragments used so the parent
    pipeline can include them in the exported NNEF.
    """
    w_ih_ref = _add_weight_tensor(g, name_to_tensor, base, "weight_ih", w_ih)
    w_hh_ref = _add_weight_tensor(g, name_to_tensor, base, "weight_hh", w_hh)

    if b_ih is None and b_hh is None:
        bias = torch.zeros(1, 4 * hidden, dtype=w_ih.dtype)
    else:
        # PyTorch sums the two biases at runtime; pre-sum and unsqueeze
        # to (1, 4H) so the fragment's broadcast over (B, 4H) is
        # unambiguous.
        zeros = torch.zeros(4 * hidden, dtype=w_ih.dtype)
        bi = b_ih if b_ih is not None else zeros
        bh = b_hh if b_hh is not None else zeros
        bias = (bi + bh).unsqueeze(0)
    b_ref = _add_weight_tensor(g, name_to_tensor, base, "bias", bias)

    h_new_ref, c_new_ref = _multi_output_pair(
        g,
        name_to_tensor,
        base,
        nnef_dtype,
        batch_dim,
        hidden,
        h_new_tv,
        c_new_tv,
    )

    NOperation(
        graph=g,
        type="lstm_cell",
        inputs=(input_ref, h_prev_ref, c_prev_ref, w_ih_ref, w_hh_ref, b_ref),
        outputs=(h_new_ref, c_new_ref),
        attribs={
            "i_end": int(hidden),
            "f_end": int(2 * hidden),
            "g_end": int(3 * hidden),
            "o_end": int(4 * hidden),
        },
    )
    return ["lstm_cell"]


@OP_REGISTRY.register()
def lstm_cell(g, node, name_to_tensor, **kwargs):
    """Map `aten::lstm_cell(input, hx_list, w_ih, w_hh, b_ih?, b_hh?)` to NNEF.

    `hx_list` is a t2n FixedTensorList of [h_prev, c_prev]. The output is
    `(h_new, c_new)`. Emits a single `lstm_cell` fragment call via the
    shared helper.
    """
    if len(node.inputs) < 4:
        raise T2NErrorNotImplemented(
            f"aten::lstm_cell expects (input, hx_list, w_ih, w_hh[, b_ih, "
            f"b_hh]); got {len(node.inputs)} inputs"
        )
    input_node = node.inputs[0]
    hx_node = node.inputs[1]
    w_ih_node = node.inputs[2]
    w_hh_node = node.inputs[3]
    b_ih_node = node.inputs[4] if len(node.inputs) >= 5 else None
    b_hh_node = node.inputs[5] if len(node.inputs) >= 6 else None

    if not isinstance(hx_node, FixedTensorList) or len(hx_node.data) != 2:
        raise T2NErrorNotImplemented(
            "aten::lstm_cell hx must be a 2-element FixedTensorList "
            f"(h, c); got {hx_node!r}"
        )
    h_prev_node, c_prev_node = hx_node.data

    input_ref = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    h_prev_ref = get_or_add_tensor_variable_in_nnef(
        g, h_prev_node, name_to_tensor
    )
    c_prev_ref = get_or_add_tensor_variable_in_nnef(
        g, c_prev_node, name_to_tensor
    )

    def _torch_tensor_or_none(n):
        if n is None or n.data is None:
            return None
        return n.data

    w_ih = _torch_tensor_or_none(w_ih_node)
    w_hh = _torch_tensor_or_none(w_hh_node)
    if w_ih is None or w_hh is None:
        raise T2NErrorNotImplemented(
            "aten::lstm_cell requires statically-resolved weight tensors"
        )
    b_ih = _torch_tensor_or_none(b_ih_node)
    b_hh = _torch_tensor_or_none(b_hh_node)

    hidden = w_hh.shape[1]
    batch_dim = (
        input_node.shape[0]
        if input_node.shape and input_node.shape[0] is not None
        else 1
    )

    if len(node.outputs) != 2:
        raise T2NErrorNotImplemented(
            f"aten::lstm_cell expects 2 outputs; got {len(node.outputs)}"
        )
    h_new_tv, c_new_tv = node.outputs

    return emit_lstm_cell_via_fragment(
        g,
        name_to_tensor,
        base=h_new_tv.export_name,
        nnef_dtype=input_ref.dtype,
        batch_dim=batch_dim,
        hidden=hidden,
        input_ref=input_ref,
        h_prev_ref=h_prev_ref,
        c_prev_ref=c_prev_ref,
        w_ih=w_ih,
        w_hh=w_hh,
        b_ih=b_ih,
        b_hh=b_hh,
        h_new_tv=h_new_tv,
        c_new_tv=c_new_tv,
    )
