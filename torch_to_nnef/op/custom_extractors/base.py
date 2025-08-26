import typing as T

import torch
from torch import nn

from torch_to_nnef.exceptions import (
    T2NErrorNotFoundModuleExtractor,
    T2NErrorNotImplemented,
)

CUSTOMOP_KIND = "wired_custom::"


class _ModuleInfoRegistery(type):
    """Allow extract in NNEF behavior from specific nn.Module."""

    MODULE_CLASS: T.Optional[T.Type[nn.Module]] = None

    REGISTRY: T.Dict[T.Type[nn.Module], "_ModuleInfoRegistery"] = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type
        # of class being defined
        # this is currently RegisterBase but in child classes
        # will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        if new_cls.MODULE_CLASS is not None:
            cls.REGISTRY[new_cls.MODULE_CLASS] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)


class ModuleInfoExtractor(metaclass=_ModuleInfoRegistery):
    """Class to take manual control of NNEF expansion of a nn.Module.

    You need to subclass it, and set MODULE_CLASS according to your
    targeted module.

    Then write .convert_to_nnef according to your need.

    """

    MODULE_CLASS: T.Optional[T.Type[nn.Module]] = None

    def __init__(self):
        if self.MODULE_CLASS is None:
            raise T2NErrorNotImplemented(
                f"Need to specify MODULE_CLASS in class {self.__class__}"
            )

    @classmethod
    def get_by_kind(cls, kind: str):
        """Get ModuleInfoExtractor by kind in torch_to_nnef internal IR."""
        classname = kind.replace(CUSTOMOP_KIND, "")
        extractor_cls = {
            str(k.__name__): v for k, v in cls.get_registry().items()
        }.get(classname)
        if extractor_cls is not None:
            return extractor_cls()
        raise T2NErrorNotFoundModuleExtractor(classname)

    @classmethod
    def get_by_module(cls, module: nn.Module):
        """Search if the module is one of the MODULE_CLASS registered.

        return appropriate ModuleInfoExtractor subclass if found
        """
        extractor_cls = cls.get_registry().get(module.__class__)
        if extractor_cls is not None:
            return extractor_cls()
        raise T2NErrorNotFoundModuleExtractor(module.__class__)

    def generate_in_torch_graph(self, torch_graph, *args, **kwargs):
        """Internal method called by torch_to_nnef ir_graph."""
        # ensure empty at first
        assert torch_graph.inputs == []
        assert torch_graph.data_nodes.is_empty()
        assert torch_graph.op_nodes == []
        assert torch_graph.outputs == []
        self._generate_in_torch_graph(torch_graph, *args, **kwargs)
        # ensure correctly populated graph
        # assert torch_graph.inputs <== exception for QTensor
        assert torch_graph.data_nodes
        assert torch_graph.op_nodes
        assert torch_graph.outputs

    @property
    def _cname_slug(self) -> str:
        if self.MODULE_CLASS:
            return self.MODULE_CLASS.__name__
        return "NotSetted"

    @staticmethod
    def _call_original_mod_with_args(mod, *args):
        """Allow to reformat args in sub class.

        ie: in LSTM there is a difference between
            - jit lstm with flat arguments tensors
            - LSTM python interface with states tensors in a tuple
        """
        return mod(*args)

    def ordered_args(self, torch_graph):
        """Odered args for the module call.

        Sometimes torch jit may reorder inputs.
        compared to targeted python ops
        in such case ordering need to be re-addressed
        """
        return torch_graph.tracer.args

    @staticmethod
    def _expand_results(results):
        expanded_results = []
        for result in results:
            if isinstance(result, (tuple, list)):
                for sub_result in result:
                    expanded_results.append(sub_result)
            else:
                expanded_results.append(result)
        return expanded_results

    def _extract_outputs(self, torch_graph, provided_outputs, results):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef import torch_graph as tg

        expanded_results = self._expand_results(results)
        outputs = []
        # ugly hack in case provided multi output operators
        # with TupleTensors packing
        if (
            provided_outputs is not None
            and isinstance(
                provided_outputs[0],
                tg.ir_data.TupleTensors,
            )
            and len(provided_outputs) == 1
        ):
            provided_outputs = provided_outputs[0].data
        # in case of prim::CallMethod
        # it can happen than pytorch jit.trace
        # optimize graph by removing unused
        # output from a function call
        # if this function call is hooked via ModuleInfoExtractor
        # then it become the responssability of
        # .convert_to_nnef to provide right output
        gouts = list(torch_graph.tracer.traced_module.graph.outputs())
        go_kind = gouts[0].type().kind()
        if len(gouts) == 1 and go_kind == "TupleType":
            gouts = list(gouts[0].node().inputs())
        elif not all(go.type().kind() == "TensorType" for go in gouts):
            raise T2NErrorNotImplemented([go.type().kind() for go in gouts])
        used_outputs_order = [_.offset() for _ in gouts]
        if provided_outputs and len(gouts) > len(provided_outputs):
            raise T2NErrorNotImplemented(
                "Unclear how to mitigate in such case n output from "
                f"PyTorch Python: {len(results)} but "
                f"PyTorch IR graph has: {len(gouts)} "
                "(this problem was not observed in PyTorch>=2.0)"
            )
        for idx, result in enumerate(expanded_results):
            if provided_outputs and idx in used_outputs_order:
                po_ix = used_outputs_order.index(idx)
                tensor_variable = provided_outputs[po_ix]
            else:
                tensor_variable = tg.TensorVariable(
                    name=f"{self._cname_slug}_output_{idx}",
                    shape=list(result.shape),
                    dtype=result.dtype,
                    quant=None,  # would probably need better handling
                    data=None,
                )
            outputs.append(tensor_variable)
        return [outputs[uidx] for uidx in used_outputs_order], outputs

    def _extract_inputs(self, provided_inputs, o_args):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef import torch_graph as tg

        inputs = []
        for idx, arg in enumerate(o_args):
            if provided_inputs:
                tensor_variable = provided_inputs[idx]
            else:
                if arg is None:
                    continue
                tensor_variable = tg.TensorVariable(
                    name=f"{self._cname_slug}_input_{idx}",
                    shape=list(arg.shape),
                    dtype=arg.dtype,
                    quant=None,  # would probably need better handling
                    data=None,
                )
            inputs.append(tensor_variable)
        return inputs

    def _generate_in_torch_graph(
        self, torch_graph, provided_inputs, provided_outputs
    ):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef import torch_graph as tg

        o_args = self.ordered_args(torch_graph)
        inputs = self._extract_inputs(provided_inputs, o_args)
        # in case of rnn 2nd parameter is a tuple of states
        results = self._call_original_mod_with_args(
            torch_graph.tracer.mod, *o_args
        )

        if isinstance(results, torch.Tensor):
            results = (results,)

        ordered_outputs, outputs = self._extract_outputs(
            torch_graph, provided_outputs, results
        )
        torch_graph.inputs = inputs
        torch_graph.outputs = ordered_outputs
        torch_graph.data_nodes = inputs + outputs
        torch_graph.op_nodes.append(
            tg.TorchOp(
                kind=f"{CUSTOMOP_KIND}{self._cname_slug}",
                inputs=inputs,
                outputs=outputs,
                op_ref=torch_graph.tracer.mod,
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
        inference_target,
        **kwargs,
    ):
        """Control NNEF content to be written for each MODULE_CLASS.

        This happen at macro level when converting from
        internal IR to NNEF IR stage.

        This is the Core method to overwrite in subclass.

        It is no different than any op implemented in `torch_to_nnef`
        in the module
        """
        raise T2NErrorNotImplemented()
