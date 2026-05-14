"""Aten-level RNN op handlers, adapters, and shared orchestration.

Canonical home for the RNN export math. Both the module-level extractors
in `op/custom_extractors/rnn.py` (`LSTMExtractor`, `GRUExtractor`,
`RNNExtractor`, `LSTMCellExtractor`) and the aten op handlers registered
below go through the same set of free functions, so exports are
byte-identical regardless of the entry point.

Layout:

  - Orchestration (multi-layer, bidirectional, batch_first, state setup):
    `emit_rnn_via_fragment` and its private helpers. Variant-agnostic;
    drives the per-layer / per-direction fragment call.
  - Per-variant params extraction: `_lstm_tensor_params`,
    `_gru_tensor_params`, `_rnn_tensor_params`. Read named weight
    attributes off a "module-like" object.
  - Aten adapters: `_LSTMAtenAdapter`, `_GRUAtenAdapter`,
    `_RNNAtenAdapter`. Build an attribute interface compatible with the
    per-variant tensor_params from the aten op's flat
    `params: Tensor[]` argument.
  - Aten op handlers: `aten::lstm`, `aten::gru`, `aten::rnn_tanh`,
    `aten::rnn_relu`, plus the single-step cell variants
    (`aten::lstm_cell`, `aten::gru_cell`, `aten::rnn_tanh_cell`,
    `aten::rnn_relu_cell`), each routed through a one-call NNEF
    fragment (`lstm_cell.nnef` / `gru_cell.nnef` /
    `rnn_tanh_cell.nnef` / `rnn_relu_cell.nnef`).
"""

from __future__ import annotations

import typing as T

import nnef
import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor

from torch_to_nnef import torch_graph as tg
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.op import helper
from torch_to_nnef.op.helper import (
    AtenOpRegistry,
    add_tensor_variable_node_as_nnef_tensor,
    get_or_add_tensor_variable_in_nnef,
)
from torch_to_nnef.tensor.named import NamedTensor
from torch_to_nnef.torch_graph.ir_data import (
    FixedTensorList,
    PythonConstant,
    TensorVariable,
)

OP_REGISTRY = AtenOpRegistry()


# -------------------------------------------------------------------------
# LSTMCell (single step, see lstm_cell.nnef fragment)
# -------------------------------------------------------------------------


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


def _single_output(
    g,
    name_to_tensor,
    base: str,
    nnef_dtype,
    batch_dim: int,
    hidden: int,
    h_new_tv: T.Optional[TensorVariable],
    suffix: str = "h_new",
) -> NTensor:
    if h_new_tv is not None:
        return add_tensor_variable_node_as_nnef_tensor(
            g, h_new_tv, name_to_tensor, prevent_variable=True
        )
    nt = NTensor(
        g,
        name=f"{base}_{suffix}",
        dtype=nnef_dtype,
        shape=(batch_dim, hidden),
    )
    name_to_tensor[nt.name] = nt
    return nt


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

    Internally does grouped `(B, I) @ (4H, I).T` plus
    `(B, H) @ (4H, H).T`, adds the unsqueezed combined bias, slices into
    the 4 gates, and computes `(h_new, c_new)`.
    """
    w_ih_ref = _add_weight_tensor(g, name_to_tensor, base, "weight_ih", w_ih)
    w_hh_ref = _add_weight_tensor(g, name_to_tensor, base, "weight_hh", w_hh)

    if b_ih is None and b_hh is None:
        bias = torch.zeros(1, 4 * hidden, dtype=w_ih.dtype)
    else:
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
    `(h_new, c_new)`.
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


# -------------------------------------------------------------------------
# GRUCell (single step, see gru_cell.nnef fragment)
# -------------------------------------------------------------------------


def emit_gru_cell_via_fragment(
    g,
    name_to_tensor,
    base: str,
    nnef_dtype,
    batch_dim: int,
    hidden: int,
    input_ref: NTensor,
    h_prev_ref: NTensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: T.Optional[torch.Tensor],
    b_hh: T.Optional[torch.Tensor],
    h_new_tv: T.Optional[TensorVariable],
) -> T.List[str]:
    """Emit a single `gru_cell` NNEF fragment call.

    Unlike `lstm_cell`, GRU keeps `b_ih` and `b_hh` separate because the
    new-gate biases are split across the reset-gated branch (see the
    fragment docstring for the equations).
    """
    w_ih_ref = _add_weight_tensor(g, name_to_tensor, base, "weight_ih", w_ih)
    w_hh_ref = _add_weight_tensor(g, name_to_tensor, base, "weight_hh", w_hh)

    zeros = torch.zeros(3 * hidden, dtype=w_ih.dtype)
    b_ih_full = (b_ih if b_ih is not None else zeros).unsqueeze(0)
    b_hh_full = (b_hh if b_hh is not None else zeros).unsqueeze(0)
    b_ih_ref = _add_weight_tensor(g, name_to_tensor, base, "bias_ih", b_ih_full)
    b_hh_ref = _add_weight_tensor(g, name_to_tensor, base, "bias_hh", b_hh_full)

    h_new_ref = _single_output(
        g, name_to_tensor, base, nnef_dtype, batch_dim, hidden, h_new_tv
    )

    NOperation(
        graph=g,
        type="gru_cell",
        inputs=(input_ref, h_prev_ref, w_ih_ref, w_hh_ref, b_ih_ref, b_hh_ref),
        outputs=(h_new_ref,),
        attribs={
            "r_end": int(hidden),
            "z_end": int(2 * hidden),
            "n_end": int(3 * hidden),
        },
    )
    return ["gru_cell"]


@OP_REGISTRY.register()
def gru_cell(g, node, name_to_tensor, **kwargs):
    """Map `aten::gru_cell(input, hx, w_ih, w_hh, b_ih?, b_hh?)` to NNEF.

    `hx` is a single 2D state tensor (unlike LSTM's 2-element list).
    """
    if len(node.inputs) < 4:
        raise T2NErrorNotImplemented(
            "aten::gru_cell expects (input, hx, w_ih, w_hh[, b_ih, b_hh]); "
            f"got {len(node.inputs)} inputs"
        )
    input_node = node.inputs[0]
    hx_node = node.inputs[1]
    w_ih_node = node.inputs[2]
    w_hh_node = node.inputs[3]
    b_ih_node = node.inputs[4] if len(node.inputs) >= 5 else None
    b_hh_node = node.inputs[5] if len(node.inputs) >= 6 else None

    input_ref = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    h_prev_ref = get_or_add_tensor_variable_in_nnef(g, hx_node, name_to_tensor)

    def _torch_tensor_or_none(n):
        if n is None or n.data is None:
            return None
        return n.data

    w_ih = _torch_tensor_or_none(w_ih_node)
    w_hh = _torch_tensor_or_none(w_hh_node)
    if w_ih is None or w_hh is None:
        raise T2NErrorNotImplemented(
            "aten::gru_cell requires statically-resolved weight tensors"
        )
    b_ih = _torch_tensor_or_none(b_ih_node)
    b_hh = _torch_tensor_or_none(b_hh_node)

    hidden = w_hh.shape[1]
    batch_dim = (
        input_node.shape[0]
        if input_node.shape and input_node.shape[0] is not None
        else 1
    )

    if len(node.outputs) != 1:
        raise T2NErrorNotImplemented(
            f"aten::gru_cell expects 1 output; got {len(node.outputs)}"
        )
    (h_new_tv,) = node.outputs

    return emit_gru_cell_via_fragment(
        g,
        name_to_tensor,
        base=h_new_tv.export_name,
        nnef_dtype=input_ref.dtype,
        batch_dim=batch_dim,
        hidden=hidden,
        input_ref=input_ref,
        h_prev_ref=h_prev_ref,
        w_ih=w_ih,
        w_hh=w_hh,
        b_ih=b_ih,
        b_hh=b_hh,
        h_new_tv=h_new_tv,
    )


# -------------------------------------------------------------------------
# RNNCell (single step Elman cell, tanh/relu variants -- see
# rnn_tanh_cell.nnef / rnn_relu_cell.nnef fragments)
# -------------------------------------------------------------------------


def emit_rnn_cell_via_fragment(
    g,
    name_to_tensor,
    base: str,
    nnef_dtype,
    batch_dim: int,
    hidden: int,
    input_ref: NTensor,
    h_prev_ref: NTensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: T.Optional[torch.Tensor],
    b_hh: T.Optional[torch.Tensor],
    h_new_tv: T.Optional[TensorVariable],
    nonlinearity: str,
) -> T.List[str]:
    """Emit a single `rnn_{tanh,relu}_cell` NNEF fragment call.

    Like `lstm_cell` the biases are pre-summed to a single `(1, H)` term
    -- the Elman cell's nonlinearity sits on the full preactivation so
    `b_ih` and `b_hh` are interchangeable in the math.
    """
    if nonlinearity not in ("tanh", "relu"):
        raise T2NErrorNotImplemented(
            f"unsupported RNN cell nonlinearity {nonlinearity!r}"
        )
    fragment = f"rnn_{nonlinearity}_cell"

    w_ih_ref = _add_weight_tensor(g, name_to_tensor, base, "weight_ih", w_ih)
    w_hh_ref = _add_weight_tensor(g, name_to_tensor, base, "weight_hh", w_hh)

    if b_ih is None and b_hh is None:
        bias = torch.zeros(1, hidden, dtype=w_ih.dtype)
    else:
        zeros = torch.zeros(hidden, dtype=w_ih.dtype)
        bi = b_ih if b_ih is not None else zeros
        bh = b_hh if b_hh is not None else zeros
        bias = (bi + bh).unsqueeze(0)
    b_ref = _add_weight_tensor(g, name_to_tensor, base, "bias", bias)

    h_new_ref = _single_output(
        g, name_to_tensor, base, nnef_dtype, batch_dim, hidden, h_new_tv
    )

    NOperation(
        graph=g,
        type=fragment,
        inputs=(input_ref, h_prev_ref, w_ih_ref, w_hh_ref, b_ref),
        outputs=(h_new_ref,),
    )
    return [fragment]


def _emit_aten_rnn_cell_simple(g, node, name_to_tensor, nonlinearity: str):
    if len(node.inputs) < 4:
        raise T2NErrorNotImplemented(
            f"aten::rnn_{nonlinearity}_cell expects (input, hx, w_ih, w_hh"
            f"[, b_ih, b_hh]); got {len(node.inputs)} inputs"
        )
    input_node = node.inputs[0]
    hx_node = node.inputs[1]
    w_ih_node = node.inputs[2]
    w_hh_node = node.inputs[3]
    b_ih_node = node.inputs[4] if len(node.inputs) >= 5 else None
    b_hh_node = node.inputs[5] if len(node.inputs) >= 6 else None

    input_ref = get_or_add_tensor_variable_in_nnef(
        g, input_node, name_to_tensor
    )
    h_prev_ref = get_or_add_tensor_variable_in_nnef(g, hx_node, name_to_tensor)

    def _torch_tensor_or_none(n):
        if n is None or n.data is None:
            return None
        return n.data

    w_ih = _torch_tensor_or_none(w_ih_node)
    w_hh = _torch_tensor_or_none(w_hh_node)
    if w_ih is None or w_hh is None:
        raise T2NErrorNotImplemented(
            f"aten::rnn_{nonlinearity}_cell requires statically-resolved "
            "weight tensors"
        )
    b_ih = _torch_tensor_or_none(b_ih_node)
    b_hh = _torch_tensor_or_none(b_hh_node)

    hidden = w_hh.shape[1]
    batch_dim = (
        input_node.shape[0]
        if input_node.shape and input_node.shape[0] is not None
        else 1
    )

    if len(node.outputs) != 1:
        raise T2NErrorNotImplemented(
            f"aten::rnn_{nonlinearity}_cell expects 1 output; "
            f"got {len(node.outputs)}"
        )
    (h_new_tv,) = node.outputs

    return emit_rnn_cell_via_fragment(
        g,
        name_to_tensor,
        base=h_new_tv.export_name,
        nnef_dtype=input_ref.dtype,
        batch_dim=batch_dim,
        hidden=hidden,
        input_ref=input_ref,
        h_prev_ref=h_prev_ref,
        w_ih=w_ih,
        w_hh=w_hh,
        b_ih=b_ih,
        b_hh=b_hh,
        h_new_tv=h_new_tv,
        nonlinearity=nonlinearity,
    )


@OP_REGISTRY.register()
def rnn_tanh_cell(g, node, name_to_tensor, **kwargs):
    """Map `aten::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih?, b_hh?)`."""
    return _emit_aten_rnn_cell_simple(g, node, name_to_tensor, "tanh")


@OP_REGISTRY.register()
def rnn_relu_cell(g, node, name_to_tensor, **kwargs):
    """Map `aten::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih?, b_hh?)`."""
    return _emit_aten_rnn_cell_simple(g, node, name_to_tensor, "relu")


# -------------------------------------------------------------------------
# Multi-step RNN orchestration (lifted from `_RNNMixin`)
# -------------------------------------------------------------------------


def _prep_states(states_0: T.Tuple, layer_index: int) -> T.Tuple:
    """Slice the initial state down to one layer.

    `states_0` is `(tensor_variable_or_None, torch_tensor_or_None)`. When
    a torch tensor is present we split along the layer axis (axis 0).
    """
    if isinstance(states_0[1], torch.Tensor):
        tensor_variable, torch_tensor = states_0
        return (
            tensor_variable,
            torch_tensor.split(1)[layer_index][:, :1, :],
        )
    return states_0


def _apply_layer_and_unsqueeze_to_params(
    params: T.Dict[str, T.Any], layer_index: int, backward: bool = False
) -> T.Dict[str, T.Any]:
    """Per-layer / direction key prefixing and unsqueeze for biases.

    Weights gain a leading layer-direction axis, biases gain that plus a
    leading singleton, and the dict keys are prefixed `l{layer}` (or
    `l{layer}_backward`).
    """
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            v_new = v.detach()
            if k.startswith("b_"):
                v_new = v_new.unsqueeze(0)
            v_new = v_new.unsqueeze(0)
            if hasattr(v, "nnef_name"):
                v_new = NamedTensor(v_new, nnef_name=v.nnef_name)
            params[k] = v_new

    linfo = str(layer_index)
    if backward:
        linfo += "_backward"
    return {f"l{linfo}_{k}": v for k, v in params.items()}


def _pre_batch_first(g, input_tensor, node, name_to_tensor) -> NTensor:
    transposed = helper.add_tensor_variable_node_as_nnef_tensor(
        g, node.inputs[0], name_to_tensor, name_suffix="transposed"
    )
    NOperation(
        g,
        type="transpose",
        inputs=input_tensor,
        outputs=transposed,
        attribs={"axes": [1, 0, 2]},
    )
    return transposed


def _post_batch_first(g, input_tensor, node, name_to_tensor) -> NTensor:
    input_tensor.name += "_batch_first"
    out = helper.add_tensor_variable_node_as_nnef_tensor(
        g, node.outputs[0], name_to_tensor
    )
    NOperation(
        g,
        type="transpose",
        inputs=input_tensor,
        outputs=out,
        attribs={"axes": [1, 0, 2]},
    )
    return out


def _multi_layers_concat(
    g, node, name_to_tensor, last_hc_at_each_layers
) -> None:
    """Concat last h_t (and c_t for LSTM) across layers."""
    for idx, out_node in enumerate(node.outputs[1:]):
        real_output = helper.add_tensor_variable_node_as_nnef_tensor(
            g, out_node, name_to_tensor
        )
        NOperation(
            graph=g,
            type="concat",
            inputs=[_[idx] for _ in last_hc_at_each_layers],
            outputs=real_output,
            attribs={"axis": 0},
        )


def _translate_state_variable_load_and_prep(
    g,
    node,
    name_to_tensor,
    var_name: str,
    tensor_variable,
    torch_tensor,
    input_tensor,
) -> NTensor:
    """Materialize a default initial state at runtime.

    Used when the user did not pass an explicit hidden state. We store
    the per-layer init tensor as a variable and tile it along the input's
    batch axis so the runtime gets a correctly-shaped state without the
    graph baking in a specific batch size.
    """
    assert tensor_variable is None, tensor_variable
    base_var_name = next(node.op_ref.parameters()).nnef_name.rsplit(".", 1)[0]
    variable_storage_id = f"{var_name}_store"
    store_tensor = helper.get_or_add_tensor_variable_in_nnef(
        name_suffix=variable_storage_id,
        node=helper.TensorVariable(
            name=node.outputs[0].name,
            data=NamedTensor(
                torch_tensor, nnef_name=f"{base_var_name}.{var_name}_init"
            )
            if not isinstance(torch_tensor, NamedTensor)
            else torch_tensor,
            shape=list(torch_tensor.shape),
            dtype=torch_tensor.dtype,
        ),
        g=g,
        name_to_tensor=name_to_tensor,
    )

    reference_rnn_input = helper.TensorVariable(
        name=input_tensor.name,
        data=None,
        shape=list(input_tensor.shape),
        dtype=node.inputs[0].dtype,
    )

    batch_size_tensor_id = f"{reference_rnn_input.export_name}_batch_size"
    if batch_size_tensor_id in name_to_tensor:
        input_batch_size_tensor = name_to_tensor[batch_size_tensor_id]
    else:
        input_shape_tensor = helper.add_tensor_variable_node_as_nnef_tensor(
            g,
            reference_rnn_input,
            name_to_tensor,
            name_suffix="shape",
            prevent_variable=True,
        )
        NOperation(
            g,
            type="tract_core_shape_of",
            inputs=name_to_tensor[reference_rnn_input.export_name],
            outputs=input_shape_tensor,
        )
        input_batch_size_slice_tensor = (
            helper.add_tensor_variable_node_as_nnef_tensor(
                g,
                reference_rnn_input,
                name_to_tensor,
                name_suffix="batch_size_sliced",
                prevent_variable=True,
            )
        )
        NOperation(
            g,
            type="slice",
            inputs=input_shape_tensor,
            outputs=input_batch_size_slice_tensor,
            attribs={
                "axes": [0],
                "begin": [1],
                "end": [2],
                "stride": [1],
            },
        )
        input_batch_size_tensor = (
            helper.add_tensor_variable_node_as_nnef_tensor(
                g,
                reference_rnn_input,
                name_to_tensor,
                name_suffix="batch_size",
                prevent_variable=True,
            )
        )
        NOperation(
            g,
            type="squeeze",
            inputs=input_batch_size_slice_tensor,
            outputs=input_batch_size_tensor,
            attribs={"axes": [0]},
        )

    initial_state_ready_tensor = helper.add_tensor_variable_node_as_nnef_tensor(
        name_suffix=var_name,
        node=helper.TensorVariable(
            name=node.outputs[0].name,
            data=torch_tensor,
            shape=list(torch_tensor.shape),
            dtype=torch_tensor.dtype,
        ),
        g=g,
        name_to_tensor=name_to_tensor,
        prevent_variable=True,
    )
    NOperation(
        g,
        type="tile",
        inputs=store_tensor,
        outputs=initial_state_ready_tensor,
        attribs={
            "repeats": [
                1,
                nnef.Identifier(input_batch_size_tensor.name),
                1,
            ]
        },
    )
    return initial_state_ready_tensor


def _translate_to_nnef_variable(
    module,
    tensor_params_kwargs,
    layer_index: int,
    node,
    g,
    name_to_tensor,
    is_backward: bool,
    input_tensor: NTensor,
    tensor_params_fn: T.Callable,
) -> T.Dict[str, NTensor]:
    """Per-layer-and-direction param materialization.

    Calls `tensor_params_fn(module, layer_index, backward, **kwargs)` to
    get a {param_name: torch.Tensor or (tensor_variable, torch_tensor)}
    dict, then converts each entry into an NNEF tensor reference.
    """
    name_to_nnef_variable: T.Dict[str, NTensor] = {}
    for var_name, item in tensor_params_fn(
        module,
        layer_index=layer_index,
        backward=is_backward,
        **tensor_params_kwargs,
    ).items():
        if isinstance(item, torch.Tensor):
            name_to_nnef_variable[var_name] = (
                helper.get_or_add_tensor_variable_in_nnef(
                    name_suffix=var_name,
                    node=helper.TensorVariable(
                        name=getattr(item, "nnef_name", node.outputs[0].name),
                        data=item,
                        shape=list(item.shape),
                        dtype=item.dtype,
                    ),
                    g=g,
                    name_to_tensor=name_to_tensor,
                )
            )
        elif isinstance(item, tuple):
            assert len(item) == 2, item
            tensor_variable, torch_tensor = item
            if torch_tensor is None:
                # User-manipulated state: slice the per-layer slab out
                # of the existing state input.
                reference = name_to_tensor[tensor_variable.export_name]
                input_layer_states_tensor = (
                    helper.add_tensor_variable_node_as_nnef_tensor(
                        g=g,
                        node=tg.TensorVariable(
                            name=node.outputs[0].name,
                            shape=[1] + list(reference.shape[1:]),
                            dtype=node.inputs[0].dtype,
                            quant=None,
                            data=None,
                        ),
                        name_to_tensor=name_to_tensor,
                        name_suffix=var_name,
                        prevent_variable=True,
                        force_full_output_tensor_name=var_name,
                    )
                )
                NOperation(
                    g,
                    type="slice",
                    inputs=reference,
                    outputs=input_layer_states_tensor,
                    attribs={
                        "axes": [0],
                        "begin": [layer_index],
                        "end": [layer_index + 1],
                        "stride": [1],
                    },
                )
                name_to_nnef_variable[var_name] = input_layer_states_tensor
            else:
                name_to_nnef_variable[var_name] = (
                    _translate_state_variable_load_and_prep(
                        g,
                        node,
                        name_to_tensor,
                        var_name,
                        tensor_variable,
                        torch_tensor,
                        input_tensor,
                    )
                )
        else:
            raise T2NErrorNotImplemented(item)

    return name_to_nnef_variable


def _translate_to_nnef_outputs(
    g, name_to_tensor, linfo: str, module, node
) -> T.List[NTensor]:
    return [
        helper.add_tensor_variable_node_as_nnef_tensor(
            g,
            out_node,
            name_to_tensor,
            name_suffix=f"l{linfo}"
            if (module.num_layers > 1 or module.bidirectional)
            else "",
        )
        for out_node in node.outputs
    ]


def _apply_rnn_bidirectional_pack_at_layer(
    g,
    node,
    name_to_tensor,
    layer_index: int,
    last_forward_h: NTensor,
    last_backward_h: NTensor,
    module,
) -> NTensor:
    out_packed_bidi = helper.add_tensor_variable_node_as_nnef_tensor(
        g,
        node.outputs[0],
        name_to_tensor,
        name_suffix=f"l{layer_index}_packed_bidi",
    )
    NOperation(
        g,
        type="rnn_bidi_pack",
        inputs=tuple([last_forward_h, last_backward_h]),
        outputs=out_packed_bidi,
        attribs={"shape": module.hidden_size * 2},
    )
    return out_packed_bidi


def emit_rnn_via_fragment(
    g,
    node,
    name_to_tensor,
    module,
    nnef_fragment_name: str,
    argument_names_order: T.Sequence[str],
    tensor_params_fn: T.Callable,
    **tensor_params_kwargs,
) -> T.List[str]:
    """Multi-layer / bidirectional RNN orchestration around a fragment call.

    Variant-agnostic: drives the per-layer-and-direction loop, calls
    `tensor_params_fn` to materialize weights / states per slice, and
    issues one `nnef_fragment_name` call per (layer, direction). Bidi
    packing and multi-layer concat are handled internally.
    """
    used_fragments = [nnef_fragment_name]
    if module.bidirectional:
        used_fragments += ["rnn_bidi_pack"]

    input_tensor = name_to_tensor[node.inputs[0].export_name]

    if module.batch_first:
        input_tensor = _pre_batch_first(g, input_tensor, node, name_to_tensor)

    last_hc_at_each_layers: T.List[T.Tuple[NTensor, ...]] = []
    passes_is_backward = [False]
    if module.bidirectional:
        passes_is_backward += [True]

    last_backward_h: T.Optional[NTensor] = None
    last_forward_h: T.Optional[NTensor] = None
    base_lstm_input = [input_tensor]
    for layer_index in range(module.num_layers):
        if last_forward_h:
            base_lstm_input = [last_forward_h]

        for is_backward in passes_is_backward:
            linfo = str(layer_index)
            name_to_nnef_variable = _translate_to_nnef_variable(
                module,
                tensor_params_kwargs,
                layer_index,
                node,
                g,
                name_to_tensor,
                is_backward,
                input_tensor=input_tensor,
                tensor_params_fn=tensor_params_fn,
            )

            if is_backward:
                linfo += "_backward"
            outputs = _translate_to_nnef_outputs(
                g, name_to_tensor, linfo, module, node
            )

            argument_order = [
                f"l{linfo}_{arg_name}" for arg_name in argument_names_order
            ]

            NOperation(
                graph=g,
                type=nnef_fragment_name,
                inputs=tuple(
                    base_lstm_input
                    + [name_to_nnef_variable[_] for _ in argument_order]
                ),
                outputs=tuple(outputs),
                attribs={"scan_pace": -1 if is_backward else 1},
            )
            if is_backward:
                last_backward_h = outputs[0]
            else:
                last_forward_h = outputs[0]

            last_hc_at_each_layers.append(outputs[1:])
        if module.bidirectional:
            last_forward_h = _apply_rnn_bidirectional_pack_at_layer(
                g,
                node,
                name_to_tensor,
                layer_index,
                last_forward_h,
                last_backward_h,
                module,
            )

    if module.batch_first:
        last_forward_h = _post_batch_first(
            g, last_forward_h, node, name_to_tensor
        )

    h_out_name = node.outputs[0].export_name
    last_forward_h.name = h_out_name
    name_to_tensor[h_out_name] = last_forward_h

    if len(last_hc_at_each_layers) > 1:
        _multi_layers_concat(g, node, name_to_tensor, last_hc_at_each_layers)
    return used_fragments


# -------------------------------------------------------------------------
# Per-variant tensor_params (read named weights off a module-like object)
# -------------------------------------------------------------------------


def _lstm_tensor_params(
    module,
    layer_index: int,
    backward: bool,
    c_0,
    h_0,
    **kwargs,
) -> T.Dict[str, T.Any]:
    h_0_layer = _prep_states(h_0, layer_index)
    c_0_layer = _prep_states(c_0, layer_index)

    suffix = str(layer_index)
    if backward:
        suffix += "_reverse"

    wi_var = getattr(module, f"weight_ih_l{suffix}")
    w_ii, w_if, w_ig, w_io = wi_var.split(int(wi_var.shape[0] / 4))
    wh_var = getattr(module, f"weight_hh_l{suffix}")
    w_hi, w_hf, w_hg, w_ho = wh_var.split(int(wh_var.shape[0] / 4))

    bias_i_name = f"bias_ih_l{suffix}"
    if (
        hasattr(module, bias_i_name)
        and getattr(module, bias_i_name) is not None
    ):
        b_var = getattr(module, bias_i_name)
        b_ii, b_if, b_ig, b_io = b_var.split(int(b_var.shape[0] / 4))
    else:
        b_ii, b_if, b_ig, b_io = (torch.tensor(0.0) for _ in range(4))

    bias_h_name = f"bias_hh_l{suffix}"
    if (
        hasattr(module, bias_h_name)
        and getattr(module, bias_h_name) is not None
    ):
        b_var = getattr(module, bias_h_name)
        b_hi, b_hf, b_hg, b_ho = b_var.split(int(b_var.shape[0] / 4))
    else:
        b_hi, b_hf, b_hg, b_ho = (torch.tensor(0.0) for _ in range(4))

    params: T.Dict[str, T.Any] = {"c_0": c_0_layer, "h_0": h_0_layer}
    base_mod_name = wi_var.nnef_name.rsplit(".", 1)[0]

    def add_param(name: str, tensor: torch.Tensor) -> None:
        lname = name.lower()
        backward_str = "back_" if backward else ""
        params[name] = NamedTensor(
            tensor,
            nnef_name=f"{base_mod_name}.l{layer_index}_{backward_str}{lname}",
        )

    add_param("W_ii", w_ii)
    add_param("W_if", w_if)
    add_param("W_ig", w_ig)
    add_param("W_io", w_io)
    add_param("W_hi", w_hi)
    add_param("W_hf", w_hf)
    add_param("W_hg", w_hg)
    add_param("W_ho", w_ho)
    add_param("b_i", b_ii + b_hi)
    add_param("b_f", b_if + b_hf)
    add_param("b_g", b_ig + b_hg)
    add_param("b_o", b_io + b_ho)
    if hasattr(module, "proj_size") and module.proj_size > 0:
        add_param("W_hr", getattr(module, f"weight_hr_l{suffix}"))

    return _apply_layer_and_unsqueeze_to_params(
        params, layer_index, backward=backward
    )


def _gru_tensor_params(
    module,
    layer_index: int,
    backward: bool,
    h_0,
    **kwargs,
) -> T.Dict[str, T.Any]:
    suffix = str(layer_index)
    if backward:
        suffix += "_reverse"

    h_0_layer = _prep_states(h_0, layer_index)
    w_var = getattr(module, f"weight_ih_l{suffix}")
    w_ir, w_iz, w_in = w_var.split(int(w_var.shape[0] / 3))
    w_var = getattr(module, f"weight_hh_l{suffix}")
    w_hr, w_hz, w_hn = w_var.split(int(w_var.shape[0] / 3))

    bias_i_name = f"bias_ih_l{suffix}"
    if (
        hasattr(module, bias_i_name)
        and getattr(module, bias_i_name) is not None
    ):
        bias_var = getattr(module, bias_i_name)
        b_ir, b_iz, b_in = bias_var.split(int(bias_var.shape[0] / 3))
    else:
        b_ir, b_iz, b_in = (torch.tensor(0.0) for _ in range(3))

    bias_h_name = f"bias_hh_l{suffix}"
    if (
        hasattr(module, bias_h_name)
        and getattr(module, bias_h_name) is not None
    ):
        bias_var = getattr(module, bias_h_name)
        b_hr, b_hz, b_hn = bias_var.split(int(bias_var.shape[0] / 3))
    else:
        b_hr, b_hz, b_hn = (torch.tensor(0.0) for _ in range(3))

    base_mod_name = w_var.nnef_name.rsplit(".", 1)[0]

    params: T.Dict[str, T.Any] = {"h_0": h_0_layer}

    def add_param(name: str, tensor: torch.Tensor) -> None:
        lname = name.lower()
        backward_str = "b" if backward else ""
        params[name] = NamedTensor(
            tensor,
            nnef_name=f"{base_mod_name}.l{layer_index}_{backward_str}{lname}",
        )

    add_param("W_ir", w_ir)
    add_param("W_iz", w_iz)
    add_param("W_in", w_in)
    add_param("W_hr", w_hr)
    add_param("W_hz", w_hz)
    add_param("W_hn", w_hn)
    add_param("b_r", b_ir + b_hr)
    add_param("b_z", b_iz + b_hz)
    add_param("b_in", b_in)
    add_param("b_hn", b_hn)
    return _apply_layer_and_unsqueeze_to_params(
        params, layer_index, backward=backward
    )


def _rnn_tensor_params(
    module,
    layer_index: int,
    backward: bool,
    h_0,
    **kwargs,
) -> T.Dict[str, T.Any]:
    suffix = str(layer_index)
    if backward:
        suffix += "_reverse"

    h_0_layer = _prep_states(h_0, layer_index)
    w_ih = getattr(module, f"weight_ih_l{suffix}")
    w_hh = getattr(module, f"weight_hh_l{suffix}")

    bias_i_name = f"bias_ih_l{suffix}"
    if (
        hasattr(module, bias_i_name)
        and getattr(module, bias_i_name) is not None
    ):
        bias_ih = getattr(module, bias_i_name)
    else:
        bias_ih = torch.tensor(0.0)

    bias_h_name = f"bias_hh_l{suffix}"
    if (
        hasattr(module, bias_h_name)
        and getattr(module, bias_h_name) is not None
    ):
        bias_hh = getattr(module, bias_h_name)
    else:
        bias_hh = torch.tensor(0.0)

    base_mod_name = w_ih.nnef_name.rsplit(".", 1)[0]

    params: T.Dict[str, T.Any] = {"h_0": h_0_layer}

    def add_param(name: str, tensor: torch.Tensor) -> None:
        lname = name.lower()
        backward_str = "b" if backward else ""
        params[name] = NamedTensor(
            tensor,
            nnef_name=f"{base_mod_name}.l{layer_index}_{backward_str}{lname}",
        )

    add_param("W_ih", w_ih)
    add_param("W_hh", w_hh)
    add_param("b_ih_hh", bias_ih + bias_hh)
    return _apply_layer_and_unsqueeze_to_params(
        params, layer_index, backward=backward
    )


# -------------------------------------------------------------------------
# Aten adapters: view a flat aten::lstm/gru/rnn `params: Tensor[]` as a
# module-like object with the named weight attributes the per-variant
# tensor_params expect.
# -------------------------------------------------------------------------


class _RNNAdapterBase:
    """Common attribute interface for aten path adapters."""

    proj_size = 0
    nonlinearity = "tanh"

    def __init__(
        self,
        params_tensors: T.Sequence[torch.Tensor],
        has_biases: bool,
        num_layers: int,
        bidirectional: bool,
        batch_first: bool,
        base_name: str,
    ):
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.batch_first = bool(batch_first)
        self.bias = bool(has_biases)
        self.base_name = base_name
        self._populate(params_tensors, has_biases, num_layers, bidirectional)

    def _set_named(self, attr: str, t: torch.Tensor) -> None:
        wrapped = NamedTensor(t, nnef_name=f"{self.base_name}.{attr}")
        setattr(self, attr, wrapped)

    def _populate(
        self,
        params_tensors: T.Sequence[torch.Tensor],
        has_biases: bool,
        num_layers: int,
        bidirectional: bool,
    ) -> None:
        per_dir = 4 if has_biases else 2
        directions = ("", "_reverse") if bidirectional else ("",)
        idx = 0
        for layer in range(num_layers):
            for dsuffix in directions:
                suffix = f"{layer}{dsuffix}"
                if idx + per_dir > len(params_tensors):
                    raise T2NErrorNotImplemented(
                        f"params list too short for layer {layer}{dsuffix}: "
                        f"got {len(params_tensors)} tensors, "
                        f"expected at least {idx + per_dir}"
                    )
                self._set_named(f"weight_ih_l{suffix}", params_tensors[idx])
                idx += 1
                self._set_named(f"weight_hh_l{suffix}", params_tensors[idx])
                idx += 1
                if has_biases:
                    self._set_named(f"bias_ih_l{suffix}", params_tensors[idx])
                    idx += 1
                    self._set_named(f"bias_hh_l{suffix}", params_tensors[idx])
                    idx += 1


class _LSTMAtenAdapter(_RNNAdapterBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wi = self.weight_ih_l0
        self.hidden_size = int(wi.shape[0]) // 4


class _GRUAtenAdapter(_RNNAdapterBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wi = self.weight_ih_l0
        self.hidden_size = int(wi.shape[0]) // 3


class _RNNAtenAdapter(_RNNAdapterBase):
    def __init__(self, *args, nonlinearity: str = "tanh", **kwargs):
        super().__init__(*args, **kwargs)
        wi = self.weight_ih_l0
        self.hidden_size = int(wi.shape[0])
        self.nonlinearity = nonlinearity


# -------------------------------------------------------------------------
# Aten op handlers
# -------------------------------------------------------------------------


def _scalar(node) -> T.Any:
    if isinstance(node, PythonConstant):
        return node.data
    if isinstance(node, TensorVariable) and node.data is not None:
        return node.data
    raise T2NErrorNotImplemented(
        f"expected statically-resolved scalar; got {node!r}"
    )


def _params_tensors(params_node) -> T.List[torch.Tensor]:
    if not isinstance(params_node, FixedTensorList):
        raise T2NErrorNotImplemented(
            "aten RNN handler expects a FixedTensorList of params; got "
            f"{params_node!r}"
        )
    out: T.List[torch.Tensor] = []
    for p in params_node.data:
        if not isinstance(p, TensorVariable) or p.data is None:
            raise T2NErrorNotImplemented(
                "aten RNN params must be statically-resolved tensors"
            )
        out.append(p.data)
    return out


def _split_aten_rnn_inputs(node) -> T.Dict[str, T.Any]:
    if len(node.inputs) != 9:
        raise T2NErrorNotImplemented(
            "aten RNN handler expects 9 inputs (input, hx, params, "
            f"has_biases, num_layers, dropout, train, bidirectional, "
            f"batch_first); got {len(node.inputs)}"
        )
    input_node, hx_node, params_node = node.inputs[:3]
    has_biases = bool(_scalar(node.inputs[3]))
    num_layers = int(_scalar(node.inputs[4]))
    # node.inputs[5] = dropout, [6] = train: ignored in inference
    bidirectional = bool(_scalar(node.inputs[7]))
    batch_first = bool(_scalar(node.inputs[8]))
    return {
        "input_node": input_node,
        "hx_node": hx_node,
        "params": _params_tensors(params_node),
        "has_biases": has_biases,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "batch_first": batch_first,
    }


def _build_state_kwargs(hx_node, single_state: bool) -> T.Dict[str, T.Tuple]:
    """Translate the aten op's `hx` into state pairs.

    Returns the (tensor_variable, torch_tensor) tuples
    `tensor_params_fn` expects.
    """
    if single_state:
        if isinstance(hx_node, FixedTensorList):
            if len(hx_node.data) != 1:
                raise T2NErrorNotImplemented(
                    "single-state RNN expects 1-element hx; "
                    f"got {len(hx_node.data)}"
                )
            (h0,) = hx_node.data
        else:
            h0 = hx_node
        return {"h_0": (h0, None)}
    if not isinstance(hx_node, FixedTensorList) or len(hx_node.data) != 2:
        raise T2NErrorNotImplemented(
            "LSTM expects a 2-element FixedTensorList for hx (h_0, c_0); "
            f"got {hx_node!r}"
        )
    h0, c0 = hx_node.data
    return {"h_0": (h0, None), "c_0": (c0, None)}


_LSTM_ARG_NAMES_ORDER = (
    "c_0",
    "h_0",
    "W_ii",
    "W_hi",
    "W_if",
    "W_hf",
    "W_ig",
    "W_hg",
    "W_io",
    "W_ho",
    "b_i",
    "b_f",
    "b_g",
    "b_o",
)

_GRU_ARG_NAMES_ORDER = (
    "h_0",
    "W_ir",
    "W_hr",
    "W_iz",
    "W_hz",
    "W_in",
    "W_hn",
    "b_r",
    "b_z",
    "b_in",
    "b_hn",
)

_RNN_ARG_NAMES_ORDER = ("h_0", "W_ih", "W_hh", "b_ih_hh")


@OP_REGISTRY.register()
def lstm(g, node, name_to_tensor, **kwargs):
    """Map `aten::lstm.input` to NNEF via the existing `lstm` fragment."""
    parsed = _split_aten_rnn_inputs(node)
    base = node.outputs[0].export_name
    adapter = _LSTMAtenAdapter(
        params_tensors=parsed["params"],
        has_biases=parsed["has_biases"],
        num_layers=parsed["num_layers"],
        bidirectional=parsed["bidirectional"],
        batch_first=parsed["batch_first"],
        base_name=f"aten_lstm.{base}",
    )
    state_kwargs = _build_state_kwargs(parsed["hx_node"], single_state=False)
    return emit_rnn_via_fragment(
        g,
        node,
        name_to_tensor,
        module=adapter,
        nnef_fragment_name="lstm",
        argument_names_order=list(_LSTM_ARG_NAMES_ORDER),
        tensor_params_fn=_lstm_tensor_params,
        **state_kwargs,
    )


@OP_REGISTRY.register()
def gru(g, node, name_to_tensor, **kwargs):
    """Map `aten::gru.input` to NNEF via the existing `gru` fragment."""
    parsed = _split_aten_rnn_inputs(node)
    base = node.outputs[0].export_name
    adapter = _GRUAtenAdapter(
        params_tensors=parsed["params"],
        has_biases=parsed["has_biases"],
        num_layers=parsed["num_layers"],
        bidirectional=parsed["bidirectional"],
        batch_first=parsed["batch_first"],
        base_name=f"aten_gru.{base}",
    )
    state_kwargs = _build_state_kwargs(parsed["hx_node"], single_state=True)
    return emit_rnn_via_fragment(
        g,
        node,
        name_to_tensor,
        module=adapter,
        nnef_fragment_name="gru",
        argument_names_order=list(_GRU_ARG_NAMES_ORDER),
        tensor_params_fn=_gru_tensor_params,
        **state_kwargs,
    )


def _emit_aten_rnn_simple(
    g, node, name_to_tensor, fragment_name: str, nonlinearity: str
):
    parsed = _split_aten_rnn_inputs(node)
    base = node.outputs[0].export_name
    adapter = _RNNAtenAdapter(
        params_tensors=parsed["params"],
        has_biases=parsed["has_biases"],
        num_layers=parsed["num_layers"],
        bidirectional=parsed["bidirectional"],
        batch_first=parsed["batch_first"],
        base_name=f"aten_{fragment_name}.{base}",
        nonlinearity=nonlinearity,
    )
    state_kwargs = _build_state_kwargs(parsed["hx_node"], single_state=True)
    return emit_rnn_via_fragment(
        g,
        node,
        name_to_tensor,
        module=adapter,
        nnef_fragment_name=fragment_name,
        argument_names_order=list(_RNN_ARG_NAMES_ORDER),
        tensor_params_fn=_rnn_tensor_params,
        **state_kwargs,
    )


@OP_REGISTRY.register()
def rnn_tanh(g, node, name_to_tensor, **kwargs):
    """Map `aten::rnn_tanh.input` to NNEF via existing `rnn_tanh` fragment."""
    return _emit_aten_rnn_simple(g, node, name_to_tensor, "rnn_tanh", "tanh")


@OP_REGISTRY.register()
def rnn_relu(g, node, name_to_tensor, **kwargs):
    """Map `aten::rnn_relu.input` to NNEF via existing `rnn_relu` fragment."""
    return _emit_aten_rnn_simple(g, node, name_to_tensor, "rnn_relu", "relu")
