import typing as T

import torch
from torch import jit, nn

from torch_to_nnef.dtypes import is_quantized_dtype
from torch_to_nnef.exceptions import TorchJitTraceFailed
from torch_to_nnef.utils import cache, torch_version


def _is_io_quantized_module(module):
    if isinstance(module, nn.Sequential):
        module = module[0]
    return not isinstance(module, torch.nn.quantized.Quantize) and any(
        _ in str(module.__class__)
        for _ in [
            "torch.nn.quantized",
            "torch.nn.intrinsic.quantized",
        ]
    )


def maybe_quantize_args_tensor(module, args):
    if _is_io_quantized_module(module) and args is not None:
        args = [
            # force cast in quantized form
            torch.quantize_per_tensor(
                in_item.float(),
                torch.tensor(0.1),
                torch.tensor(0),
                dtype=torch.quint8,
            )
            if isinstance(in_item, torch.Tensor)
            and not is_quantized_dtype(in_item.dtype)
            else in_item
            for in_item in args
        ]
    return args


class TorchModuleTracer:
    """Evaluate Optimized traced Function code so that signature always match

    original Module is passed to do proper un-boxing later on.
    This is needed because we have a re-routing based on actual module classtype.

    """

    def __init__(
        self,
        module: nn.Module,
        traced_module: torch.jit.TracedModule = None,
        fn_name: str = "forward",
        # likely mostly torch tensors
        args: T.Optional[T.Tuple[T.Any, ...]] = None,
    ):
        self.mod = module
        self._traced_module = traced_module
        self.fn_name = fn_name
        self.args = maybe_quantize_args_tensor(module, args)

    @property
    def traced_module(self):
        if self._traced_module is None:
            try:
                self._traced_module = jit.trace(
                    self.mod,
                    self.args,
                    check_trace=("1.8.0" <= torch_version() < "1.12.0"),
                    # since 1.12 get flaky on ViT model trace
                    strict=False,
                )
            except RuntimeError as exp:
                raise TorchJitTraceFailed(
                    "Unable to trace with jit one of following submodule:"
                    f"{[(k, v.__class__) for k,v in self.mod.named_children()]} "
                    f"with original error:\n\n'{exp}'\n\n"
                    "This maybe due to provided input dimension. "
                    "If not, you can aleviate this issue by applying a special hook"
                    "this module (explaination available in torch_to_nnef README)"
                ) from exp
        return self._traced_module

    @property  # type: ignore
    @cache
    def torch_graph(self):
        trace = self.traced_module
        if self.fn_name and self.fn_name != "forward":
            trace = getattr(trace, self.fn_name)
        return trace.graph

    def __call__(self, *args, **kwargs):
        # _actual_script_module is an implementation details
        # from torch/jit/_trace.py:l1076 in TracedModule
        if self.fn_name == "forward" and not isinstance(
            self.traced_module, torch.jit._script.RecursiveScriptModule
        ):
            traced_op_call = self.traced_module._actual_script_module.forward
        else:
            traced_op_call = getattr(self.traced_module, self.fn_name)
        return traced_op_call(*args, **kwargs)
