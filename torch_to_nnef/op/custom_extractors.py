"""
`ops.custom_extractors` provide mechanism to extract to NNEF while bypassing full expension
of `torch.Module` within `torch_graph` which by default use torch.jit.trace .

This may be for two main reasons:
    - Some layer such as LSTM/GRU have complex expension which are better
      handled by encapsulation instead of spreading high number of variable
    - Some layer might not be serializable to .jit
    - There might be some edge case where you prefer to keep full control on
      exported NNEF subgraph.

"""
import typing as T

import torch
from nnef_tools.model import Operation as NOperation
from nnef_tools.model import Tensor as NTensor
from torch import nn

CUSTOMOP_KIND = "wired_custom::"


class NotFoundModuleExtractor(KeyError):
    pass


class _ModuleInfoRegistery(type):

    """Allow extract in NNEF behavior from specific nn.Module"""

    MODULE_CLASS: T.Optional[T.Type[nn.Module]] = None

    REGISTRY: T.Dict[T.Type[nn.Module], "_ModuleInfoRegistery"] = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        if new_cls.MODULE_CLASS is not None:
            cls.REGISTRY[new_cls.MODULE_CLASS] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)


class ModuleInfoExtractor(metaclass=_ModuleInfoRegistery):
    MODULE_CLASS: T.Optional[T.Type[nn.Module]] = None

    def __init__(self):
        if self.MODULE_CLASS is None:
            raise NotImplementedError(
                f"Need to specify MODULE_CLASS in class {self.__class__}"
            )

    @classmethod
    def get_by_kind(cls, kind: str):
        classname = kind.replace(CUSTOMOP_KIND, "")
        extractor_cls = {
            str(k.__name__): v for k, v in cls.get_registry().items()
        }.get(classname)
        if extractor_cls is not None:
            return extractor_cls()
        raise NotFoundModuleExtractor(classname)

    @classmethod
    def get_by_module(cls, module: nn.Module):
        extractor_cls = cls.get_registry().get(module.__class__)
        if extractor_cls is not None:
            return extractor_cls()
        raise NotFoundModuleExtractor(module.__class__)

    def generate_in_torch_graph(self, torch_graph, *args, **kwargs):
        # ensure empty at first
        assert torch_graph.inputs == []
        assert torch_graph.data_nodes == []
        assert torch_graph.op_nodes == []
        assert torch_graph.outputs == []
        self._generate_in_torch_graph(torch_graph, *args, **kwargs)
        # ensure correctly populated graph
        assert torch_graph.inputs
        assert torch_graph.data_nodes
        assert torch_graph.op_nodes
        assert torch_graph.outputs

    @property
    def _cname_slug(self) -> str:
        if self.MODULE_CLASS:
            return self.MODULE_CLASS.__name__
        return "NotSetted"

    def _generate_in_torch_graph(
        self, torch_graph, provided_inputs, provided_outputs
    ):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef import torch_graph as tg

        inputs = []
        for idx, arg in enumerate(torch_graph._args):
            if provided_inputs:
                tensor_variable = provided_inputs[idx]
            else:
                tensor_variable = tg.TensorVariable(
                    name=f"{self._cname_slug}_input_{idx}",
                    shape=list(arg.shape),
                    dtype=arg.dtype,
                    quant=None,  # would probably need better handling
                    data=None,
                )
            inputs.append(tensor_variable)
        results = torch_graph._module(*torch_graph._args)
        if isinstance(results, torch.Tensor):
            results = (results,)

        expanded_results = []
        for result in results:
            if isinstance(result, (tuple, list)):
                for sub_result in result:
                    expanded_results.append(sub_result)
            else:
                expanded_results.append(result)

        outputs = []
        for idx, result in enumerate(expanded_results):
            if provided_outputs and idx > len(provided_outputs):
                tensor_variable = provided_outputs[idx]
            else:
                tensor_variable = tg.TensorVariable(
                    name=f"{self._cname_slug}_output_{idx}",
                    shape=list(result.shape),
                    dtype=result.dtype,
                    quant=None,  # would probably need better handling
                    data=None,
                )
            outputs.append(tensor_variable)

        torch_graph.inputs = inputs
        torch_graph.outputs = outputs
        torch_graph.data_nodes = inputs + outputs
        torch_graph.op_nodes.append(
            tg.TorchOp(
                kind=f"{CUSTOMOP_KIND}{self._cname_slug}",
                inputs=inputs,
                outputs=outputs,
                op_ref=torch_graph._module,
                call_name=self._cname_slug,
                module_path="",
                scope="",
            )
        )

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
    ):
        raise NotImplementedError()


T_RNNS = T.Union[nn.LSTM, nn.GRU, nn.RNN]


class _RNNMixin:
    def tensor_params(
        self,
        module: T_RNNS,
        layer_index: int,
        backward: bool,
        **kwargs,
    ):
        raise NotImplementedError

    def _check_rank(self, node, module):
        batch_rank = 0 if module.batch_first else 1
        assert node.inputs[0].shape[batch_rank] == 1, (
            f"should be dim=1 for rank={batch_rank} since batch_size beyond are "
            f"not supported but provided shape is {node.inputs[0].shape}"
        )

    def _pre_batch_first(self, g, input_tensor, node, name_to_tensor):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import primitive

        transposed_input_tensor = primitive.add_output_tensor(
            g, node.inputs[0], name_to_tensor, name_suffix="_transposed"
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
        from torch_to_nnef.op import primitive

        input_tensor.name += "_batch_first"
        out_transpose_tensor = primitive.add_output_tensor(
            g, node.outputs[0], name_to_tensor
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
        """allow to concat last from each layers for h_t and and c_t"""

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import primitive

        for idx, out_node in enumerate(node.outputs[1:]):
            real_output = primitive.add_output_tensor(
                g, out_node, name_to_tensor
            )
            NOperation(
                graph=g,
                type="concat",
                inputs=[_[idx] for _ in last_hc_at_each_layers],
                outputs=real_output,
                attribs={"axis": 0},
            )

    def _translate_to_nnef_variable(
        self,
        module: T_RNNS,
        tensor_params_kwargs,
        layer_index: int,
        node,  # : TensorVariable
        g,
        name_to_tensor,
        is_backward: bool,
    ) -> T.Dict[str, NTensor]:
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import primitive

        name_to_nnef_variable = {}
        for var_name, torch_tensor in self.tensor_params(
            module,
            layer_index=layer_index,
            backward=is_backward,
            **tensor_params_kwargs,
        ).items():
            name_to_nnef_variable[
                var_name
            ] = primitive.register_state_node_as_variable(
                torch_tensor,
                var_name,
                node,
                g,
                name_to_tensor,
            )
        return name_to_nnef_variable

    def _translate_to_nnef_outputs(
        self, g, name_to_tensor, linfo: str, module: T_RNNS, node
    ) -> T.List[NTensor]:
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import primitive

        return [
            primitive.add_output_tensor(
                g,
                out_node,
                name_to_tensor,
                name_suffix=f"_l{linfo}"
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
        from torch_to_nnef.op import primitive

        out_packed_bidi = primitive.add_output_tensor(
            g,
            node.outputs[0],
            name_to_tensor,
            name_suffix=f"_l{layer_index}_packed_bidi",
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
        """
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
            v = v.detach()
            if k.startswith("b_"):
                v = v.unsqueeze(0)
            params[k] = v.unsqueeze(0)

        linfo = str(layer_index)
        if backward:
            linfo += "_backward"
        return {f"l{linfo}_{k}": v for k, v in params.items()}


class LSTMExtractor(ModuleInfoExtractor, _RNNMixin):
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
        h_0_layer = h_0.split(1)[layer_index].squeeze(0)
        c_0_layer = c_0.split(1)[layer_index].squeeze(0)

        suffix = str(layer_index)
        if backward:
            suffix += "_reverse"

        # lstm weight packed in order (W_ii|W_if|W_ig|W_io)
        w_var = getattr(module, f"weight_ih_l{suffix}")
        W_ii, W_if, W_ig, W_io = w_var.split(int(w_var.shape[0] / 4))
        # lstm weight packed in order (W_hi|W_hf|W_hg|W_ho)
        w_var = getattr(module, f"weight_hh_l{suffix}")
        W_hi, W_hf, W_hg, W_ho = w_var.split(int(w_var.shape[0] / 4))

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
            # -----------
            "W_ii": W_ii,
            "W_if": W_if,
            "W_ig": W_ig,
            "W_io": W_io,
            # -----------
            "W_hi": W_hi,
            "W_hf": W_hf,
            "W_hg": W_hg,
            "W_ho": W_ho,
            # pre summed bias
            "b_i": b_ii + b_hi,
            "b_f": b_if + b_hf,
            "b_g": b_ig + b_hg,
            "b_o": b_io + b_ho,
        }
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
    ):

        lstm = node.op_ref

        nnef_fragment_selected = "lstm"

        if hasattr(lstm, "proj_size") and lstm.proj_size > 0:
            raise NotImplementedError(
                "Missing implementation NNEF LSTM with projection"
            )

        D = 2 if lstm.bidirectional else 1

        batch_rank = 0 if lstm.batch_first else 1
        batch_dim = node.inputs[0].shape[batch_rank]
        if len(node.inputs) < 2:
            h_0 = torch.zeros(
                lstm.num_layers * D,
                batch_dim,
                lstm.proj_size or lstm.hidden_size,
            )
        else:
            # might be a TensorVariable with data NOT already setted
            h_0 = node.inputs[1].data

        if len(node.inputs) < 3:
            c_0 = torch.zeros(lstm.num_layers * D, batch_dim, lstm.hidden_size)
        else:
            # might be a TensorVariable with data NOT already setted
            c_0 = node.inputs[2].data

        tensor_params_kwargs = {"h_0": h_0, "c_0": c_0}
        return self._core_convert_to_nnef(
            module=lstm,
            node=node,
            g=g,
            name_to_tensor=name_to_tensor,
            nnef_fragment_name=nnef_fragment_selected,
            argument_names_order=[
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
            ],
            **tensor_params_kwargs,
        )


class GRUExtractor(ModuleInfoExtractor, _RNNMixin):
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

        h_0_layer = h_0.split(1)[layer_index].squeeze(0)
        # module weight packed in order (W_ir|W_iz|W_in)
        w_var = getattr(module, f"weight_ih_l{suffix}")
        W_ir, W_iz, W_in = w_var.split(int(w_var.shape[0] / 3))
        # module weight packed in order (W_hr|W_hz|W_hn)
        w_var = getattr(module, f"weight_hh_l{suffix}")
        W_hr, W_hz, W_hn = w_var.split(int(w_var.shape[0] / 3))

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

        params = {
            "h_0": h_0_layer,
            # -----------
            "W_ir": W_ir,
            "W_iz": W_iz,
            "W_in": W_in,
            # -----------
            "W_hr": W_hr,
            "W_hz": W_hz,
            "W_hn": W_hn,
            # pre summed bias
            "b_r": b_ir + b_hr,
            "b_z": b_iz + b_hz,
            # not summable
            "b_in": b_in,
            "b_hn": b_hn,
        }
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
    ):

        gru = node.op_ref

        nnef_fragment_selected = "gru"

        D = 2 if gru.bidirectional else 1

        if len(node.inputs) < 2:
            batch_rank = 0 if gru.batch_first else 1
            batch_dim = node.inputs[0].shape[batch_rank]
            h_0 = torch.zeros(gru.num_layers * D, batch_dim, gru.hidden_size)
        else:
            # might be a TensorVariable with data NOT already setted
            h_0 = node.inputs[1].data
        tensor_params_kwargs = {"h_0": h_0}
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


class RNNExtractor(ModuleInfoExtractor, _RNNMixin):
    MODULE_CLASS = nn.RNN

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

        h_0_layer = h_0.split(1)[layer_index].squeeze(0)

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

        params = {
            "h_0": h_0_layer,
            "W_ih": w_ih,
            "W_hh": w_hh,
            # -----
            # pre summed bias
            "b_ih_hh": bias_ih + bias_hh,
        }
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
    ):

        rnn = node.op_ref

        nnef_fragment_selected = "rnn"

        D = 2 if rnn.bidirectional else 1

        if len(node.inputs) < 2:
            batch_rank = 0 if rnn.batch_first else 1
            batch_dim = node.inputs[0].shape[batch_rank]
            h_0 = torch.zeros(rnn.num_layers * D, batch_dim, rnn.hidden_size)
        else:
            # might be a TensorVariable with data NOT already setted
            h_0 = node.inputs[1].data
        tensor_params_kwargs = {"h_0": h_0}
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
