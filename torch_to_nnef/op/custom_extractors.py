"""
`ops.special` provide mechanism to extract to NNEF while bypassing full expension
of `torch.Module` within `torch_graph`.

This may be for two main reasons:
    - Some layer such as LSTM/GRU have complex expension which are better
      handled by encapsulation instead of spreading high number of variable
    - Some layer might not be serializable to .jit

"""
import typing as T

import torch
from nnef_tools.model import Operation as NOperation
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
        from .. import torch_graph as tg

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
            if provided_outputs and idx >= len(provided_outputs):
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


class _RNNMixin:
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
        used_fragments = [nnef_fragment_name]
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import primitive

        assert (
            node.inputs[0].shape[0 if module.batch_first else 1] == 1
        ), "first dim need to be only 1 since batch_size beyond are not supported"

        input_tensor = name_to_tensor[node.inputs[0].export_name]

        if module.batch_first:
            transposed_input_tensor = primitive.add_output_tensor(
                g, node.inputs[0], name_to_tensor, name_suffix="_transposed"
            )
            NOperation(
                g,
                type="transpose",
                inputs=input_tensor,
                outputs=transposed_input_tensor,
                attribs={"axes": [1, 0]},
            )
            input_tensor = transposed_input_tensor

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
                name_to_nnef_variable = {}
                for var_name, torch_tensor in self.tensor_params(
                    module,
                    **tensor_params_kwargs,
                    layer_index=layer_index,
                    backward=is_backward,
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

                if is_backward:
                    linfo += "_backward"
                # outputs: h_n, h_last
                outputs = [
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
                last_forward_h = out_packed_bidi

        if module.bidirectional:
            used_fragments += ["rnn_bidi_pack"]

        if module.batch_first:
            out_transpose_tensor = primitive.add_output_tensor(
                g, node.outputs[0], name_to_tensor, name_suffix="_batch_first"
            )
            NOperation(
                g,
                type="transpose",
                inputs=input_tensor,
                outputs=out_transpose_tensor,
                attribs={"axes": [1, 0]},
            )
            input_tensor = out_transpose_tensor

        h_out_name = node.outputs[0].export_name
        input_tensor.name = h_out_name
        name_to_tensor[h_out_name] = last_forward_h

        if len(last_hc_at_each_layers) > 1:
            # allow to concat last from each layers for h_t and and c_t
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
        return used_fragments

    def _apply_layer_and_unsqueeze_to_params(
        self, params, layer_index: int, backward: bool = False
    ):
        for k, v in params.items():
            v = v.detach()
            if k.startswith('b_'):
                v = v.unsqueeze(0)
            params[k] = v.unsqueeze(0)

        linfo = str(layer_index)
        if backward:
            linfo += "_backward"
        return {f"l{linfo}_{k}": v for k, v in params.items()}


class LSTMExtractor(ModuleInfoExtractor, _RNNMixin):
    MODULE_CLASS = nn.LSTM

    def tensor_params(
        self, lstm, c_0, h_0, layer_index: int = 0, backward: bool = False
    ):
        h_0_layer = h_0.split(1)[layer_index]
        c_0_layer = c_0.split(1)[layer_index]

        suffix = str(layer_index)
        if backward:
            suffix += "_reverse"

        # lstm weight packed in order (W_ii|W_if|W_ig|W_io)
        w_var = getattr(lstm, f"weight_ih_l{suffix}")
        W_ii, W_if, W_ig, W_io = w_var.split(int(w_var.shape[0] / 4))
        # lstm weight packed in order (W_hi|W_hf|W_hg|W_ho)
        w_var = getattr(lstm, f"weight_hh_l{suffix}")
        W_hi, W_hf, W_hg, W_ho = w_var.split(int(w_var.shape[0] / 4))

        bias_i_name = f"bias_ih_l{suffix}"
        if (
            hasattr(lstm, bias_i_name)
            and getattr(lstm, bias_i_name) is not None
        ):
            b_var = getattr(lstm, bias_i_name)
            # lstm packed in order (b_ii|b_if|b_ig|b_io)
            b_ii, b_if, b_ig, b_io = b_var.split(int(b_var.shape[0] / 4))
        else:
            b_ii, b_if, b_ig, b_io = (torch.tensor(0.0) for _ in range(4))

        bias_h_name = f"bias_hh_l{suffix}"
        if (
            hasattr(lstm, bias_h_name)
            and getattr(lstm, bias_h_name) is not None
        ):
            # lstm packed in order (b_hi|b_hf|b_hg|b_ho)
            b_var = getattr(lstm, bias_h_name)
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

        if len(node.inputs) < 2:
            h_0 = torch.zeros(
                lstm.num_layers * D, lstm.proj_size or lstm.hidden_size
            )
        else:
            # might be a TensorVariable with data NOT already setted
            h_0 = node.inputs[1].data

        if len(node.inputs) < 3:
            c_0 = torch.zeros(lstm.num_layers * D, lstm.hidden_size)
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

    def tensor_params(
        self, gru, h_0, layer_index: int = 0, backward: bool = False
    ):

        suffix = str(layer_index)
        if backward:
            suffix += "_reverse"

        h_0_layer = h_0.split(1)[layer_index]
        # gru weight packed in order (W_ir|W_iz|W_in)
        w_var = getattr(gru, f"weight_ih_l{suffix}")
        W_ir, W_iz, W_in = w_var.split(int(w_var.shape[0] / 3))
        # gru weight packed in order (W_hr|W_hz|W_hn)
        w_var = getattr(gru, f"weight_hh_l{suffix}")
        W_hr, W_hz, W_hn = w_var.split(int(w_var.shape[0] / 3))

        bias_i_name = f"bias_ih_l{suffix}"
        if hasattr(gru, bias_i_name) and getattr(gru, bias_i_name) is not None:
            # gru packed in order (b_ir|b_iz|b_in)
            bias_var = getattr(gru, bias_i_name)
            b_ir, b_iz, b_in = bias_var.split(int(bias_var.shape[0] / 3))
        else:
            b_ir, b_iz, b_in = (torch.tensor(0.0) for _ in range(3))

        bias_h_name = f"bias_hh_l{suffix}"
        if hasattr(gru, bias_h_name) and getattr(gru, bias_h_name) is not None:
            # gru packed in order (b_hr|b_hz|b_hn)
            bias_var = getattr(gru, bias_h_name)
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
            h_0 = torch.zeros(gru.num_layers * D, gru.hidden_size)
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
