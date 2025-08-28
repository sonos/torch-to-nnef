"""implements :class:`TorchModuleTracer`.

It traces a ``torch.nn.Module`` with ``torch.jit.trace`` and exposes
the resulting graph.

with few related helper functions (e.g. ``_is_io_quantized_module`` and
``maybe_quantize_args_tensor``) provide small utilities used during tracing.
"""

import typing as T

import torch
from torch import jit, nn

from torch_to_nnef.dtypes import is_quantized_dtype
from torch_to_nnef.exceptions import T2NErrorTorchJitTraceFailed
from torch_to_nnef.utils import cache, torch_version


def _is_io_quantized_module(module):
    """Return ``True`` if *module* is a quantized module used in I/O.

    Args:
        module: a ``torch.nn.Module`` or ``torch.nn.Sequential``.

    The function first unwraps ``nn.Sequential`` by using its first child
    if exists, then checks the type name for the
    presence of quantization namespaces while excluding the ``Quantize``
    activation node itself.
    """
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
    """Quantize tensors in *args* if *module* expects quantized input.

    The function walks the *args* tuple, and for each tensor that is not
    already quantized, it creates a fake quantized representation using
    ``torch.quantize_per_tensor`` with a dummy scale/zero point.  The
    quantized tensor is not representative of real data; it is solely
    used to keep the tracing machinery happy when the module expects
    quantized inputs.

    Args:
        module: A ``torch.nn.Module`` instance.
        args: A tuple of inputs supplied to the module.  ``None`` is
            accepted for modules that do not require any positional
            arguments.

    Returns:
        A potentially modified tuple where tensors are quantized when
        necessary.
    """
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
    """Evaluate Optimized traced Function code so that signature always match.

    original Module is passed to do proper un-boxing later on.
    This is needed because we have a re-routing based on actual
    module classtype.

    """

    def __init__(
        self,
        module: nn.Module,
        traced_module: T.Optional[torch.jit.TracedModule] = None,
        fn_name: str = "forward",
        # likely mostly torch tensors
        args: T.Optional[T.Tuple[T.Any, ...]] = None,
    ):
        """Create a tracer for *module*.

        The tracer stores the original module, an optional pre‑traced
        ``torch.jit.TracedModule`` (which allows re‑use of a previously
        computed trace), the name of the forward method to trace, and the
        arguments used for tracing.  The arguments are post‑processed by
        :func:`maybe_quantize_args_tensor` to ensure compatibility with
        quantized modules.
        """
        self.mod = module
        self._traced_module = traced_module
        self.fn_name = fn_name
        self.args = maybe_quantize_args_tensor(module, args)

    @property
    def traced_module(self):
        """Return the traced module, computing it lazily if required.

        If ``self._traced_module`` is ``None`` the method will perform a
        ``jit.trace`` on ``self.mod`` with ``self.args`` while handling
        possible PyTorch version nuances.  Any ``RuntimeError`` raised by
        ``torch.jit.trace`` is wrapped into a
        :class:`~torch_to_nnef.exceptions.T2NErrorTorchJitTraceFailed`
        exception.
        """
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
                submod_classes = [
                    (k, v.__class__) for k, v in self.mod.named_children()
                ]
                raise T2NErrorTorchJitTraceFailed(
                    "Unable to trace with jit one of following submodule:"
                    f"{submod_classes} with original error:\n\n'{exp}'\n\n"
                    "This maybe due to provided input dimension. "
                    "If not, you can aleviate this issue by applying "
                    " a special hook to this module "
                    "(explaination available in torch_to_nnef README)"
                ) from exp
        return self._traced_module

    @property  # type: ignore
    @cache
    def torch_graph(self):
        """Return the underlying PyTorch graph object.

        The actual ``torch.Graph`` is retrieved from the traced module.
        When a different forward method is requested (``fn_name`` differs
        from "forward"), the corresponding sub‑graph is returned instead.
        """
        trace = self.traced_module
        if self.fn_name and self.fn_name != "forward":
            trace = getattr(trace, self.fn_name)
        return trace.graph

    def __call__(self, *args, **kwargs):
        """Invoke the traced forward method.

        The call path mirrors the internal logic of ``torch.jit``.
        ``RecursiveScriptModule`` exposes its ``forward`` through a public
        attribute, whereas plain ``TracedModule`` stores the actual
        implementation in ``_actual_script_module``.
        """
        # _actual_script_module is an implementation details
        # from torch/jit/_trace.py:l1076 in TracedModule
        if self.fn_name == "forward" and not isinstance(
            self.traced_module, torch.jit._script.RecursiveScriptModule
        ):
            traced_op_call = self.traced_module._actual_script_module.forward
        else:
            traced_op_call = getattr(self.traced_module, self.fn_name)
        return traced_op_call(*args, **kwargs)
