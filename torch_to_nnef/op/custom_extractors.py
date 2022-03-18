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

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
    ):
        torch_graph.printall()
        raise NotImplementedError("Missing implementation NNEF LSTM")
