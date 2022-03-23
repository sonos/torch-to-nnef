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
            if provided_outputs:
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


class LSTMExtractor(ModuleInfoExtractor):
    MODULE_CLASS = nn.LSTM

    def tensor_params(self, lstm, c_0, h_0):
        # lstm weight packed in order (W_ii|W_if|W_ig|W_io)
        W_ii, W_if, W_ig, W_io = lstm.weight_ih_l0.split(
            int(lstm.weight_ih_l0.shape[0] / 4)
        )
        # lstm weight packed in order (W_hi|W_hf|W_hg|W_ho)
        W_hi, W_hf, W_hg, W_ho = lstm.weight_hh_l0.split(
            int(lstm.weight_hh_l0.shape[0] / 4)
        )
        if hasattr(lstm, "bias_ih_l0") and lstm.bias_ih_l0 is not None:
            # lstm packed in order (b_ii|b_if|b_ig|b_io)
            b_ii, b_if, b_ig, b_io = lstm.bias_ih_l0.split(
                int(lstm.bias_ih_l0.shape[0] / 4)
            )
        else:
            b_ii, b_if, b_ig, b_io = (torch.tensor(0.0) for _ in range(4))
        if hasattr(lstm, "bias_hh_l0") and lstm.bias_hh_l0 is not None:
            # lstm packed in order (b_hi|b_hf|b_hg|b_ho)
            b_hi, b_hf, b_hg, b_ho = lstm.bias_hh_l0.split(
                int(lstm.bias_hh_l0.shape[0] / 4)
            )
        else:
            b_hi, b_hf, b_hg, b_ho = (torch.tensor(0.0) for _ in range(4))

        return {
            "c_0": c_0,
            "h_0": h_0,
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

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
    ):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import primitive

        lstm = node.op_ref

        nnef_fragment_selected = "lstm"
        if hasattr(lstm, "proj_size") and lstm.proj_size > 0:
            raise NotImplementedError(
                "Missing implementation NNEF LSTM with projection"
            )

        D = 2 if lstm.bidirectional else 1
        if lstm.bidirectional:
            raise NotImplementedError(
                "Missing implementation NNEF LSTM with bidirectional"
            )

        if lstm.batch_first:
            raise NotImplementedError(
                "Missing handling of proper matrix "
                "permutation of IO due to batch_first use"
            )
        if lstm.num_layers > 1:
            raise NotImplementedError("Missing handling of multi layer LSTM")
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

        name_to_nnef_variable = {}
        for var_name, torch_tensor in self.tensor_params(
            lstm, c_0, h_0
        ).items():
            torch_tensor = torch_tensor.detach()
            if var_name.startswith('b_'):
                torch_tensor = torch_tensor.unsqueeze(0)
            torch_tensor = torch_tensor.unsqueeze(0)
            name_to_nnef_variable[
                var_name
            ] = primitive.register_state_node_as_variable(
                torch_tensor,
                var_name,
                node,
                g,
                name_to_tensor,
            )

        # output, h_n, c_n
        outputs = [
            primitive.add_output_tensor(g, out_node, name_to_tensor)
            for out_node in node.outputs
        ]

        argument_order = [
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
        assert (
            node.inputs[0].shape[0] == 1
        ), "first dim need to be only 1 since batch_size beyond are not supported"

        input_tensor = name_to_tensor[node.inputs[0].export_name]

        NOperation(
            graph=g,
            type=nnef_fragment_selected,
            name=f"{node.outputs[0].export_name}_op",
            inputs=tuple(
                [input_tensor]
                + [name_to_nnef_variable[_] for _ in argument_order]
            ),
            outputs=tuple(outputs),
        )
        return [nnef_fragment_selected]


class GRUExtractor(ModuleInfoExtractor):
    MODULE_CLASS = nn.GRU

    def tensor_params(self, gru, h_0):
        # gru weight packed in order (W_ii|W_if|W_ig|W_io)
        W_ir, W_iz, W_in = gru.weight_ih_l0.split(
            int(gru.weight_ih_l0.shape[0] / 3)
        )
        # gru weight packed in order (W_hi|W_hf|W_hg|W_ho)
        W_hr, W_hz, W_hn = gru.weight_hh_l0.split(
            int(gru.weight_hh_l0.shape[0] / 3)
        )
        if hasattr(gru, "bias_ih_l0") and gru.bias_ih_l0 is not None:
            # gru packed in order (b_ii|b_if|b_ig|b_io)
            b_ir, b_iz, b_in = gru.bias_ih_l0.split(
                int(gru.bias_ih_l0.shape[0] / 3)
            )
        else:
            b_ir, b_iz, b_in = (torch.tensor(0.0) for _ in range(3))
        if hasattr(gru, "bias_hh_l0") and gru.bias_hh_l0 is not None:
            # gru packed in order (b_hi|b_hf|b_hg|b_ho)
            b_hr, b_hz, b_hn = gru.bias_hh_l0.split(
                int(gru.bias_hh_l0.shape[0] / 3)
            )
        else:
            b_hr, b_hz, b_hn = (torch.tensor(0.0) for _ in range(3))

        return {
            "h_0": h_0,
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

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
    ):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import primitive

        gru = node.op_ref

        nnef_fragment_selected = "gru"

        if gru.batch_first:
            raise NotImplementedError(
                "Missing handling of proper matrix "
                "permutation of IO due to batch_first use"
            )

        D = 2 if gru.bidirectional else 1
        if gru.bidirectional:
            raise NotImplementedError(
                "Missing implementation NNEF GRU with bidirectional"
            )

        if gru.num_layers > 1:
            raise NotImplementedError("Missing handling of multi layer GRU")

        if len(node.inputs) < 2:
            h_0 = torch.zeros(gru.num_layers * D, gru.hidden_size)
        else:
            # might be a TensorVariable with data NOT already setted
            h_0 = node.inputs[1].data

        name_to_nnef_variable = {}
        for var_name, torch_tensor in self.tensor_params(gru, h_0).items():
            torch_tensor = torch_tensor.detach()
            if var_name.startswith('b_'):
                torch_tensor = torch_tensor.unsqueeze(0)
            torch_tensor = torch_tensor.unsqueeze(0)
            name_to_nnef_variable[
                var_name
            ] = primitive.register_state_node_as_variable(
                torch_tensor,
                var_name,
                node,
                g,
                name_to_tensor,
            )

        # output, h_n, c_n
        outputs = [
            primitive.add_output_tensor(g, out_node, name_to_tensor)
            for out_node in node.outputs
        ]

        argument_order = [
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
        ]
        assert (
            node.inputs[0].shape[0] == 1
        ), "first dim need to be only 1 since batch_size beyond are not supported"

        input_tensor = name_to_tensor[node.inputs[0].export_name]

        NOperation(
            graph=g,
            type=nnef_fragment_selected,
            name=f"{node.outputs[0].export_name}_op",
            inputs=tuple(
                [input_tensor]
                + [name_to_nnef_variable[_] for _ in argument_order]
            ),
            outputs=tuple(outputs),
        )
        return [nnef_fragment_selected]
