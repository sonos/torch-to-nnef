import typing as T

import torch
from torch import nn

from torch_to_nnef.exceptions import (
    T2NErrorNotImplemented,
    T2NErrorStrictNNEFSpec,
)
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_ZEROS,
    LISTCONSTRUCT_KIND,
    TUPLECONSTRUCT_KIND,
)

T_RNNS = T.Union[nn.LSTM, nn.GRU, nn.RNN]


class _RNNMixin:
    """Module-extractor base for `nn.LSTM` / `nn.GRU` / `nn.RNN`.

    The actual NNEF emission lives in `op/aten/rnn.py` so the aten-op
    handlers and this module path share the same fragment-emission path
    byte-for-byte. This class only:

    - hosts `ordered_args` (JIT-arg reordering specific to module-call
      tracing),
    - hands a `tensor_params_fn` matching the variant down to
      `emit_rnn_via_fragment`.
    """

    def _tensor_params_fn(self):
        """Return the matching `_*_tensor_params` callable.

        Subclasses pull from `op/aten/rnn.py` lazily to avoid a circular
        import between the extractor module and the aten op module.
        """
        raise T2NErrorNotImplemented()

    def tensor_params(self, module, layer_index, backward, **kwargs):
        return self._tensor_params_fn()(
            module, layer_index=layer_index, backward=backward, **kwargs
        )

    def ordered_args(self, torch_graph):
        """List of args ordered to be Python call compliant.

        Sometime torch jit may reorder inputs.
        compared to targeted python ops
        in such case ordering need to be re-addressed
        """
        rnn_op = next(torch_graph.tracer.torch_graph.outputs()).node()
        if rnn_op.kind() == TUPLECONSTRUCT_KIND:
            rnn_op = next(rnn_op.inputs()).node()

        real_order = list(rnn_op.inputs())[:3]
        received_order = list(torch_graph.tracer.torch_graph.inputs())[1:]
        order = []
        for rinp in real_order[:-1]:
            try:
                order.append(received_order.index(rinp))
            except ValueError:
                node = rinp.node()
                if node.kind() == LISTCONSTRUCT_KIND:
                    for sinp in node.inputs():
                        if sinp in received_order:
                            order_idx = received_order.index(sinp)
                            order.append(order_idx)
                        else:
                            # assume default init values
                            sinp_node = sinp.node()
                            assert sinp_node.kind() == ATEN_ZEROS, (
                                sinp_node.kind()
                            )
                            continue
                    break
        new_args = [torch_graph.tracer.args[o] for o in order]
        if len(new_args) == 0:  # fallback: observed in torch==1.10
            new_args = torch_graph.tracer.args
        return new_args

    def _core_convert_to_nnef(
        self,
        module,
        node,
        g,
        name_to_tensor,
        nnef_fragment_name,
        argument_names_order,
        **tensor_params_kwargs,
    ):
        """Delegate the per-layer loop to the canonical fragment emitter.

        `emit_rnn_via_fragment` in `op/aten/rnn.py` produces the same
        NNEF output as before the lift.
        """
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.aten.rnn import emit_rnn_via_fragment

        return emit_rnn_via_fragment(
            g=g,
            node=node,
            name_to_tensor=name_to_tensor,
            module=module,
            nnef_fragment_name=nnef_fragment_name,
            argument_names_order=argument_names_order,
            tensor_params_fn=self._tensor_params_fn(),
            **tensor_params_kwargs,
        )

    # The orchestration helpers (`_pre_batch_first`, `_post_batch_first`,
    # `_translate_to_nnef_variable`, `_translate_to_nnef_outputs`,
    # `_apply_rnn_bidirectional_pack_at_layer`, `_multi_layers_concat`,
    # `_translate_state_variable_load_and_prep`,
    # `_apply_layer_and_unsqueeze_to_params`, `_prep_states`) all moved to
    # `op/aten/rnn.py` as free functions. This mixin is intentionally
    # thin now.


class LSTMExtractor(_RNNMixin, ModuleInfoExtractor):
    MODULE_CLASS = nn.LSTM

    def _tensor_params_fn(self):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.aten.rnn import _lstm_tensor_params

        return _lstm_tensor_params

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        inference_target,
        **kwargs,
    ):
        assert len(node.inputs) <= 3, node.inputs
        assert len(node.outputs) <= 3, node.outputs
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorStrictNNEFSpec(
                "Impossible to export LSTM with NNEF spec compliance activated"
            )

        lstm = node.op_ref

        nnef_fragment_selected = "lstm"

        if hasattr(lstm, "proj_size") and lstm.proj_size > 0:
            nnef_fragment_selected = "lstm_with_projection"

        layer_multiplier = 2 if lstm.bidirectional else 1

        batch_rank = 0 if lstm.batch_first else 1
        batch_dim = node.inputs[0].shape[batch_rank]
        if len(node.inputs) < 2:
            h_0_tensor_variable = None
            h_0_torch = torch.zeros(
                lstm.num_layers * layer_multiplier,
                batch_dim,
                lstm.proj_size or lstm.hidden_size,
            )
        else:
            # parameter is manipulated by user
            h_0_tensor_variable = node.inputs[1]
            h_0_torch = None

        if len(node.inputs) < 3:
            c_0_tensor_variable = None
            c_0_torch = torch.zeros(
                lstm.num_layers * layer_multiplier, batch_dim, lstm.hidden_size
            )
        else:
            # parameter is manipulated by user
            c_0_tensor_variable = node.inputs[2]
            c_0_torch = None

        tensor_params_kwargs = {
            "h_0": (h_0_tensor_variable, h_0_torch),
            "c_0": (c_0_tensor_variable, c_0_torch),
        }

        argument_names_order = [
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
            # -----
            "b_i",
            "b_f",
            "b_g",
            "b_o",
        ]
        if hasattr(lstm, "proj_size") and lstm.proj_size > 0:
            argument_names_order.append("W_hr")
        return self._core_convert_to_nnef(
            module=lstm,
            node=node,
            g=g,
            name_to_tensor=name_to_tensor,
            nnef_fragment_name=nnef_fragment_selected,
            argument_names_order=argument_names_order,
            **tensor_params_kwargs,
        )

    @staticmethod
    def _call_original_mod_with_args(mod, *args):
        """Allow to reformat args.

        In LSTM there is a difference between
            - jit lstm with flat arguments tensors
            - LSTM python interface with states tensors in a tuple
        """
        if len(args) > 1:
            args = (args[0], tuple(args[1:]))
        return mod(*args)


class GRUExtractor(_RNNMixin, ModuleInfoExtractor):
    MODULE_CLASS = nn.GRU

    def _tensor_params_fn(self):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.aten.rnn import _gru_tensor_params

        return _gru_tensor_params

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        inference_target,
        **kwargs,
    ):
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorStrictNNEFSpec(
                "Impossible to export GRU with NNEF spec compliance activated"
            )
        gru = node.op_ref

        nnef_fragment_selected = "gru"

        layer_multiplier = 2 if gru.bidirectional else 1

        if len(node.inputs) < 2:
            batch_rank = 0 if gru.batch_first else 1
            batch_dim = node.inputs[0].shape[batch_rank]
            h_0_torch = torch.zeros(
                gru.num_layers * layer_multiplier, batch_dim, gru.hidden_size
            )
            h_0_tensor_variable = None
        else:
            # parameter is manipulated by user
            h_0_tensor_variable = node.inputs[1]
            h_0_torch = None
        tensor_params_kwargs = {"h_0": (h_0_tensor_variable, h_0_torch)}
        return self._core_convert_to_nnef(
            module=gru,
            node=node,
            g=g,
            name_to_tensor=name_to_tensor,
            nnef_fragment_name=nnef_fragment_selected,
            argument_names_order=[
                "h_0",
                "W_ir",
                "W_hr",
                "W_iz",
                "W_hz",
                "W_in",
                "W_hn",
                # -----
                "b_r",
                "b_z",
                "b_in",
                "b_hn",
            ],
            **tensor_params_kwargs,
        )


class RNNExtractor(_RNNMixin, ModuleInfoExtractor):
    MODULE_CLASS = nn.RNN

    def _tensor_params_fn(self):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.aten.rnn import _rnn_tensor_params

        return _rnn_tensor_params

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        inference_target,
        **kwargs,
    ):
        if not isinstance(inference_target, TractNNEF):
            raise T2NErrorStrictNNEFSpec(
                "Impossible to export RNN with NNEF spec compliance activated"
            )

        rnn = node.op_ref

        nnef_fragment_selected = {
            "tanh": "rnn_tanh",
            "relu": "rnn_relu",
        }[rnn.nonlinearity.lower()]

        layer_multiplier = 2 if rnn.bidirectional else 1

        if len(node.inputs) < 2:
            batch_rank = 0 if rnn.batch_first else 1
            batch_dim = node.inputs[0].shape[batch_rank]
            h_0_torch = torch.zeros(
                rnn.num_layers * layer_multiplier, batch_dim, rnn.hidden_size
            )
            h_0_tensor_variable = None
        else:
            # parameter is manipulated by user
            h_0_tensor_variable = node.inputs[1]
            h_0_torch = None
        tensor_params_kwargs = {"h_0": (h_0_tensor_variable, h_0_torch)}

        return self._core_convert_to_nnef(
            module=rnn,
            node=node,
            g=g,
            name_to_tensor=name_to_tensor,
            nnef_fragment_name=nnef_fragment_selected,
            argument_names_order=[
                "h_0",
                "W_ih",
                "W_hh",
                # -----
                "b_ih_hh",
            ],
            **tensor_params_kwargs,
        )


class LSTMCellExtractor(ModuleInfoExtractor):
    """Decompose `nn.LSTMCell` into primitive NNEF ops.

    Unlike `nn.LSTM`, an LSTMCell carries a single time-step. We emit:
        preact = matmul(input, w_ih, T) + matmul(h, w_hh, T) + b_ih + b_hh
        i, f, g, o = chunk(preact, 4, axis=-1)
        c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)
        h_new = sigmoid(o) * tanh(c_new)

    Input order from the user-facing wrapper is `(input, h, c)` -- the
    internal nn.LSTMCell call expects `(input, (h, c))` which is handled by
    `_call_original_mod_with_args`.
    """

    MODULE_CLASS = nn.LSTMCell

    def ordered_args(self, torch_graph):
        """Reorder args so the first one is `input` (shape (B, input_size)).

        t2n's IR sometimes reorders the cell's inputs after FixedTensorList /
        tuple expansion, surfacing them as e.g. (h, input, c). The cell's
        `input_size` (= weight_ih.shape[1]) lets us pick the input tensor by
        shape; the relative order of (h, c) follows the JIT graph's
        prim::ListConstruct that builds hx.
        """
        cell = torch_graph.tracer.mod
        in_size = cell.input_size
        args = list(torch_graph.tracer.args)
        if len(args) != 3:
            return args
        input_idx = next(
            (
                i
                for i, a in enumerate(args)
                if hasattr(a, "shape")
                and len(a.shape) == 2
                and a.shape[1] == in_size
            ),
            None,
        )
        if input_idx is None:
            return args
        input_arg = args.pop(input_idx)
        return [input_arg, *args]

    @staticmethod
    def _reorder_cell_inputs(inputs, cell: nn.LSTMCell):
        """Identify `input` and the (h, c) state tensors among the IR inputs.

        When `input_size != hidden_size`, shape uniquely identifies `input`.
        When they are equal (e.g. Silero-VAD's 128/128 cell), we rely on
        the empirically-observed JIT trace ordering: PyTorch's tracer
        emits the cell-call args as `(h, input, c)` because the `hx`
        tuple's first element is spliced before the `input` slot during
        positional flattening.

        After picking `input`, the other two inputs are returned as (h, c)
        in their IR list order.
        """
        in_size = cell.input_size
        h_size = cell.hidden_size
        if len(inputs) != 3:
            return (
                inputs[0],
                inputs[1] if len(inputs) > 1 else None,
                (inputs[2] if len(inputs) > 2 else None),
            )

        if in_size != h_size:
            input_idx = next(
                (
                    i
                    for i, t in enumerate(inputs)
                    if t.shape and t.shape[-1] == in_size
                ),
                None,
            )
            if input_idx is None:
                input_idx = 0
        else:
            # Ambiguous shapes -- use position 1 (the empirical (h, input, c)
            # ordering of the trace).
            input_idx = 1

        rest = [t for i, t in enumerate(inputs) if i != input_idx]
        return inputs[input_idx], rest[0], rest[1]

    @staticmethod
    def _call_original_mod_with_args(mod, *args):
        return mod(args[0], (args[1], args[2]))

    def _extract_outputs(self, torch_graph, provided_outputs, results):
        """Override base behavior.

        Base `_extract_outputs` computes `used_outputs_order` from the
        cell-graph outputs' SSA offsets within their producer nodes; that
        assumes a single multi-output op (lstm/lstm_cell). nn.LSTMCell
        traces actually decompose into separate sigmoid/tanh/mul nodes, so
        offsets all collapse to 0 and the duplicated index corrupts the
        output mapping. We have a fixed signature `(h_new, c_new)`, so
        identity mapping is correct.
        """
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef import torch_graph as tg

        if (
            provided_outputs is not None
            and isinstance(provided_outputs[0], tg.ir_data.TupleTensors)
            and len(provided_outputs) == 1
        ):
            provided_outputs = provided_outputs[0].data
        expanded_results = self._expand_results(results)
        outputs = []
        for idx, result in enumerate(expanded_results):
            if provided_outputs and idx < len(provided_outputs):
                tv = provided_outputs[idx]
            else:
                tv = tg.TensorVariable(
                    name=f"{self._cname_slug}_output_{idx}",
                    shape=list(result.shape),
                    dtype=result.dtype,
                    quant=None,
                    data=None,
                )
            outputs.append(tv)
        return outputs, outputs

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        inference_target,
        **kwargs,
    ):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.aten.rnn import emit_lstm_cell_via_fragment

        cell: nn.LSTMCell = node.op_ref
        if len(node.inputs) != 3:
            raise T2NErrorNotImplemented(
                "LSTMCellExtractor requires (input, h, c); "
                f"got {len(node.inputs)} inputs"
            )
        if len(node.outputs) != 2:
            raise T2NErrorNotImplemented(
                "LSTMCellExtractor expects 2 outputs (h_new, c_new); "
                f"got {len(node.outputs)}"
            )

        # The t2n IR may surface (input, h, c) in any order after tuple
        # expansion at the call site. Identify `input` by shape and order
        # the rest positionally (`_extract_outputs` already maps
        # provided_outputs[0] = h_new, [1] = c_new).
        input_tv, h_prev_tv, c_prev_tv = self._reorder_cell_inputs(
            node.inputs, cell
        )
        h_new_tv, c_new_tv = node.outputs

        input_ref = helper.get_or_add_tensor_variable_in_nnef(
            g, input_tv, name_to_tensor
        )
        h_prev_ref = helper.get_or_add_tensor_variable_in_nnef(
            g, h_prev_tv, name_to_tensor
        )
        c_prev_ref = helper.get_or_add_tensor_variable_in_nnef(
            g, c_prev_tv, name_to_tensor
        )

        b_ih = cell.bias_ih if cell.bias else None
        b_hh = cell.bias_hh if cell.bias else None

        batch_dim = (
            input_tv.shape[0]
            if input_tv.shape and input_tv.shape[0] is not None
            else 1
        )
        return emit_lstm_cell_via_fragment(
            g,
            name_to_tensor,
            base=h_new_tv.export_name,
            nnef_dtype=input_ref.dtype,
            batch_dim=batch_dim,
            hidden=cell.hidden_size,
            input_ref=input_ref,
            h_prev_ref=h_prev_ref,
            c_prev_ref=c_prev_ref,
            w_ih=cell.weight_ih,
            w_hh=cell.weight_hh,
            b_ih=b_ih,
            b_hh=b_hh,
            h_new_tv=h_new_tv,
            c_new_tv=c_new_tv,
        )
