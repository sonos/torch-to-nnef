import typing as T

import nnef
import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor
from torch import nn

from torch_to_nnef.exceptions import (
    T2NErrorNotImplemented,
    T2NErrorStrictNNEFSpec,
)
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
from torch_to_nnef.tensor.named import NamedTensor
from torch_to_nnef.torch_graph.torch_const import (
    ATEN_ZEROS,
    LISTCONSTRUCT_KIND,
    TUPLECONSTRUCT_KIND,
)

T_RNNS = T.Union[nn.LSTM, nn.GRU, nn.RNN]


class _RNNMixin:
    def tensor_params(
        self,
        module: T_RNNS,
        layer_index: int,
        backward: bool,
        **kwargs,
    ):
        raise T2NErrorNotImplemented()

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

    def _check_rank(self, node, module):
        batch_rank = 0 if module.batch_first else 1
        assert node.inputs[0].shape[batch_rank] == 1, (
            f"should be dim=1 for rank={batch_rank} since batch_size beyond "
            f"are not supported but provided shape is {node.inputs[0].shape}"
        )

    @staticmethod
    def _prep_states(states_0, layer_index: int):
        if isinstance(states_0[1], torch.Tensor):
            states_0_tensor_variable, states_0_torch = states_0
            states_0_layer = (
                states_0_tensor_variable,
                states_0_torch.split(1)[layer_index][:, :1, :],
            )  # to be tiled
        else:
            states_0_layer = states_0
        return states_0_layer

    def _pre_batch_first(self, g, input_tensor, node, name_to_tensor):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

        transposed_input_tensor = (
            helper.add_tensor_variable_node_as_nnef_tensor(
                g, node.inputs[0], name_to_tensor, name_suffix="transposed"
            )
        )
        NOperation(
            g,
            type="transpose",
            inputs=input_tensor,
            outputs=transposed_input_tensor,
            attribs={"axes": [1, 0, 2]},
        )
        return transposed_input_tensor

    def _post_batch_first(self, g, input_tensor, node, name_to_tensor):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

        input_tensor.name += "_batch_first"
        out_transpose_tensor = helper.add_tensor_variable_node_as_nnef_tensor(
            g,
            node.outputs[0],
            name_to_tensor,
        )
        NOperation(
            g,
            type="transpose",
            inputs=input_tensor,
            outputs=out_transpose_tensor,
            attribs={"axes": [1, 0, 2]},
        )
        return out_transpose_tensor

    def _multi_layers_concat(
        self, g, node, name_to_tensor, last_hc_at_each_layers
    ):
        """Allow to concat last from each layers for h_t and and c_t."""
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

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
        self,
        g,
        node,
        name_to_tensor,
        var_name: str,
        tensor_variable,
        torch_tensor,
        input_tensor,
    ):
        """Reproduce initial states preparations before rnn layer call.

        ie:
        @code nnef
          h_0 = variable<scalar>(
            label = 'gru_output_0_l0_h_0_store',
            shape = [1, 1, 5]);
          input_shape = tract_core_shape_of(input);
          batch_size = slice(
            input_shape,
            axes=[0],
            begin=[1],
            end=[2],
            stride=[1]
          );
          h_batch_expanded_0 = tile(h_0, repeats=[1, batch_size, 1]);
        @end

        """
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

        assert tensor_variable is None, tensor_variable
        base_var_name = next(node.op_ref.parameters()).nnef_name.rsplit(".", 1)[
            0
        ]
        variable_storage_id = f"{var_name}_store"
        store_tensor = helper.get_or_add_tensor_variable_in_nnef(
            name_suffix=variable_storage_id,
            # build imaginary node to fill data correctly
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

        # NOTE: here we create a fake node so that even if rnn is 'batch_first'
        # we reference the right rnn input 'name'
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
                attribs={
                    "axes": [0],
                },
            )

        initial_state_ready_tensor = (
            helper.add_tensor_variable_node_as_nnef_tensor(
                name_suffix=var_name,
                # build imaginary node to fill data correctly
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
        self,
        module: T_RNNS,
        tensor_params_kwargs,
        layer_index: int,
        node,  # : TensorVariable
        g,
        name_to_tensor,
        is_backward: bool,
        input_tensor: NTensor,
    ) -> T.Dict[str, NTensor]:
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef import torch_graph as tg

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

        name_to_nnef_variable = {}
        for var_name, item in self.tensor_params(
            module,
            layer_index=layer_index,
            backward=is_backward,
            **tensor_params_kwargs,
        ).items():
            if isinstance(item, torch.Tensor):
                name_to_nnef_variable[var_name] = (
                    helper.get_or_add_tensor_variable_in_nnef(
                        name_suffix=var_name,
                        # build imaginary node to fill data correctly
                        node=helper.TensorVariable(
                            name=getattr(
                                item, "nnef_name", node.outputs[0].name
                            ),
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
                    # variable is manipulated by user
                    reference_state_nnef_tensor = name_to_tensor[
                        tensor_variable.export_name
                    ]
                    input_layer_states_tensor = helper.add_tensor_variable_node_as_nnef_tensor(  # noqa: E501
                        g=g,
                        node=tg.TensorVariable(
                            name=node.outputs[0].name,
                            shape=[1]
                            + list(reference_state_nnef_tensor.shape[1:]),
                            dtype=node.inputs[0].dtype,
                            quant=None,  # would probably need better handling
                            data=None,
                        ),
                        name_to_tensor=name_to_tensor,
                        name_suffix=var_name,
                        prevent_variable=True,
                        force_full_output_tensor_name=var_name,
                    )
                    NOperation(
                        g,
                        type="slice",
                        inputs=reference_state_nnef_tensor,
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
                        self._translate_state_variable_load_and_prep(
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
        self, g, name_to_tensor, linfo: str, module: T_RNNS, node
    ) -> T.List[NTensor]:
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

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
        self,
        g,
        node,
        name_to_tensor,
        layer_index: int,
        last_forward_h: NTensor,
        last_backward_h: NTensor,
        module: T_RNNS,
    ):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import helper

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
        """Core convertion to NNEF of rnn.

        Avoid repeated configuration of:
        batch_first
        multi_layers
        """
        # self._check_rank(node, module)

        used_fragments = [nnef_fragment_name]
        if module.bidirectional:
            used_fragments += ["rnn_bidi_pack"]

        input_tensor = name_to_tensor[node.inputs[0].export_name]

        if module.batch_first:
            input_tensor = self._pre_batch_first(
                g, input_tensor, node, name_to_tensor
            )

        last_hc_at_each_layers = []

        passes_is_backward = [False]
        if module.bidirectional:
            passes_is_backward += [True]

        last_backward_h = None
        last_forward_h = None
        base_lstm_input = [input_tensor]
        for layer_index in range(module.num_layers):
            if last_forward_h:
                base_lstm_input = [last_forward_h]

            for is_backward in passes_is_backward:
                linfo = str(layer_index)
                name_to_nnef_variable = self._translate_to_nnef_variable(
                    module,
                    tensor_params_kwargs,
                    layer_index,
                    node,
                    g,
                    name_to_tensor,
                    is_backward,
                    input_tensor=input_tensor,
                )

                if is_backward:
                    linfo += "_backward"
                # outputs: h_n, h_last
                outputs = self._translate_to_nnef_outputs(
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
                last_forward_h = self._apply_rnn_bidirectional_pack_at_layer(
                    g,
                    node,
                    name_to_tensor,
                    layer_index,
                    last_forward_h,
                    last_backward_h,
                    module,
                )

        if module.batch_first:
            last_forward_h = self._post_batch_first(
                g, last_forward_h, node, name_to_tensor
            )

        h_out_name = node.outputs[0].export_name
        last_forward_h.name = h_out_name
        name_to_tensor[h_out_name] = last_forward_h

        if len(last_hc_at_each_layers) > 1:
            self._multi_layers_concat(
                g, node, name_to_tensor, last_hc_at_each_layers
            )
        return used_fragments

    def _apply_layer_and_unsqueeze_to_params(
        self, params, layer_index: int, backward: bool = False
    ):
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


class LSTMExtractor(_RNNMixin, ModuleInfoExtractor):
    MODULE_CLASS = nn.LSTM

    # should be good enough for this use case
    #  liskov substitution being not used here
    #  pylint: disable-next=arguments-differ
    def tensor_params(  # type: ignore
        self,
        module: T_RNNS,
        layer_index: int,
        backward: bool,
        c_0: torch.Tensor,
        h_0: torch.Tensor,
        **kwargs,
    ):
        h_0_layer = self._prep_states(h_0, layer_index)
        c_0_layer = self._prep_states(c_0, layer_index)

        suffix = str(layer_index)
        if backward:
            suffix += "_reverse"

        # lstm weight packed in order (W_ii|W_if|W_ig|W_io)
        wi_var = getattr(module, f"weight_ih_l{suffix}")
        w_ii, w_if, w_ig, w_io = wi_var.split(int(wi_var.shape[0] / 4))
        # lstm weight packed in order (W_hi|W_hf|W_hg|W_ho)
        wh_var = getattr(module, f"weight_hh_l{suffix}")

        w_hi, w_hf, w_hg, w_ho = wh_var.split(int(wh_var.shape[0] / 4))

        bias_i_name = f"bias_ih_l{suffix}"
        if (
            hasattr(module, bias_i_name)
            and getattr(module, bias_i_name) is not None
        ):
            b_var = getattr(module, bias_i_name)
            # module packed in order (b_ii|b_if|b_ig|b_io)
            b_ii, b_if, b_ig, b_io = b_var.split(int(b_var.shape[0] / 4))
        else:
            b_ii, b_if, b_ig, b_io = (torch.tensor(0.0) for _ in range(4))

        bias_h_name = f"bias_hh_l{suffix}"
        if (
            hasattr(module, bias_h_name)
            and getattr(module, bias_h_name) is not None
        ):
            # lstm packed in order (b_hi|b_hf|b_hg|b_ho)
            b_var = getattr(module, bias_h_name)
            b_hi, b_hf, b_hg, b_ho = b_var.split(int(b_var.shape[0] / 4))
        else:
            b_hi, b_hf, b_hg, b_ho = (torch.tensor(0.0) for _ in range(4))

        params = {
            "c_0": c_0_layer,
            "h_0": h_0_layer,
        }
        base_mod_name = wi_var.nnef_name.rsplit(".", 1)[0]

        def add_param(name, tensor):
            lname = name.lower()
            backward_str = "back_" if backward else ""
            params[name] = NamedTensor(
                tensor,
                nnef_name=f"{base_mod_name}.l{layer_index}_{backward_str}{lname}",
            )

        # -----------
        add_param("W_ii", w_ii)
        add_param("W_if", w_if)
        add_param("W_ig", w_ig)
        add_param("W_io", w_io)
        # -----------
        add_param("W_hi", w_hi)
        add_param("W_hf", w_hf)
        add_param("W_hg", w_hg)
        add_param("W_ho", w_ho)
        # pre summed bias
        add_param("b_i", b_ii + b_hi)
        add_param("b_f", b_if + b_hf)
        add_param("b_g", b_ig + b_hg)
        add_param("b_o", b_io + b_ho)
        if hasattr(module, "proj_size") and module.proj_size > 0:  # type: ignore
            # LSTM.weight_hr_l[k] may be with suffix
            w_hr = getattr(module, f"weight_hr_l{suffix}")
            add_param("W_hr", w_hr)

        return self._apply_layer_and_unsqueeze_to_params(
            params, layer_index, backward=backward
        )

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

    #  pylint: disable-next=arguments-differ
    def tensor_params(  # type: ignore
        self,
        module: T_RNNS,
        layer_index: int,
        backward: bool,
        h_0: torch.Tensor,
        **kwargs,
    ):
        suffix = str(layer_index)
        if backward:
            suffix += "_reverse"

        h_0_layer = self._prep_states(h_0, layer_index)
        # module weight packed in order (W_ir|W_iz|W_in)
        w_var = getattr(module, f"weight_ih_l{suffix}")
        w_ir, w_iz, w_in = w_var.split(int(w_var.shape[0] / 3))
        # module weight packed in order (W_hr|W_hz|W_hn)
        w_var = getattr(module, f"weight_hh_l{suffix}")
        w_hr, w_hz, w_hn = w_var.split(int(w_var.shape[0] / 3))

        bias_i_name = f"bias_ih_l{suffix}"
        if (
            hasattr(module, bias_i_name)
            and getattr(module, bias_i_name) is not None
        ):
            # module packed in order (b_ir|b_iz|b_in)
            bias_var = getattr(module, bias_i_name)
            b_ir, b_iz, b_in = bias_var.split(int(bias_var.shape[0] / 3))
        else:
            b_ir, b_iz, b_in = (torch.tensor(0.0) for _ in range(3))

        bias_h_name = f"bias_hh_l{suffix}"
        if (
            hasattr(module, bias_h_name)
            and getattr(module, bias_h_name) is not None
        ):
            # module packed in order (b_hr|b_hz|b_hn)
            bias_var = getattr(module, bias_h_name)
            b_hr, b_hz, b_hn = bias_var.split(int(bias_var.shape[0] / 3))
        else:
            b_hr, b_hz, b_hn = (torch.tensor(0.0) for _ in range(3))

        base_mod_name = w_var.nnef_name.rsplit(".", 1)[0]

        def add_param(name, tensor):
            lname = name.lower()
            backward_str = "b" if backward else ""
            params[name] = NamedTensor(
                tensor,
                nnef_name=f"{base_mod_name}.l{layer_index}_{backward_str}{lname}",
            )

        params = {
            "h_0": h_0_layer,
        }
        # -----------
        add_param("W_ir", w_ir)
        add_param("W_iz", w_iz)
        add_param("W_in", w_in)
        # -----------
        add_param("W_hr", w_hr)
        add_param("W_hz", w_hz)
        add_param("W_hn", w_hn)
        # pre summed bias
        add_param("b_r", b_ir + b_hr)
        add_param("b_z", b_iz + b_hz)
        # not summable
        add_param("b_in", b_in)
        add_param("b_hn", b_hn)
        return self._apply_layer_and_unsqueeze_to_params(
            params, layer_index, backward=backward
        )

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

    #  pylint: disable-next=arguments-differ
    def tensor_params(  # type: ignore
        self,
        module: T_RNNS,
        layer_index: int,
        backward: bool,
        h_0: T.Union[torch.Tensor, str],
        **kwargs,
    ):
        suffix = str(layer_index)
        if backward:
            suffix += "_reverse"

        h_0_layer = self._prep_states(h_0, layer_index)

        w_ih = getattr(module, f"weight_ih_l{suffix}")
        w_hh = getattr(module, f"weight_hh_l{suffix}")

        bias_i_name = f"bias_ih_l{suffix}"
        if (
            hasattr(module, bias_i_name)
            and getattr(module, bias_i_name) is not None
        ):
            # module packed in order (b_ir|b_iz|b_in)
            bias_ih = getattr(module, bias_i_name)
        else:
            bias_ih = torch.tensor(0.0)

        bias_h_name = f"bias_hh_l{suffix}"
        if (
            hasattr(module, bias_h_name)
            and getattr(module, bias_h_name) is not None
        ):
            # module packed in order (b_hr|b_hz|b_hn)
            bias_hh = getattr(module, bias_h_name)
        else:
            bias_hh = torch.tensor(0.0)

        base_mod_name = w_ih.nnef_name.rsplit(".", 1)[0]

        def add_param(name, tensor):
            lname = name.lower()
            backward_str = "b" if backward else ""
            params[name] = NamedTensor(
                tensor,
                nnef_name=f"{base_mod_name}.l{layer_index}_{backward_str}{lname}",
            )

        params = {
            "h_0": h_0_layer,
        }
        add_param("W_ih", w_ih)
        add_param("W_hh", w_hh)
        # -----
        # pre summed bias
        add_param("b_ih_hh", bias_ih + bias_hh)
        return self._apply_layer_and_unsqueeze_to_params(
            params, layer_index, backward=backward
        )

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
