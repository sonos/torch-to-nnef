"""High-level helper that runs the JIT-only export hardening chain.

The individual passes in `jit_inline` and `jit_passes` are exposed for
fine-grained use (custom orderings, partial chains for debugging). Most
callers want the full chain; `harden_jit_for_export` wraps it with a
sensible default order and freeze step.

Each pass is a no-op on graphs that don't carry the relevant pattern,
so the chain is safe to apply unconditionally.

Supported on torch >= 1.10 (the only API requirement is
`torch._C._jit_interpret_graph`, used by `fold_data_dependent_ifs` and
probed lazily). CI gates the chain on torch 2.11.0; older versions work
but are not regression-tested.
"""

from __future__ import annotations

import logging
import typing as T

import torch

from torch_to_nnef.torch_graph.jit_inline import (
    inline_unresolvable_submodules,
)
from torch_to_nnef.torch_graph.jit_passes import (
    fold_constant_ifs,
    fold_constant_scalar_arithmetic,
    fold_data_dependent_ifs,
    fold_tuple_index_through_tuple_construct,
    fold_tuple_unpack_through_tuple_construct,
    replace_size_calls_with_constants,
    strip_assertion_ifs,
    strip_prim_data,
)

__all__ = ["harden_jit_for_export"]

LOGGER = logging.getLogger(__name__)


def harden_jit_for_export(
    model: torch.jit.ScriptModule,
    args: T.Union[T.Sequence[T.Any], torch.Tensor],
    *,
    freeze: bool = True,
    diagnostics: T.Optional[T.Dict[str, T.Any]] = None,
) -> torch.jit.ScriptModule:
    """Specialize a JIT ScriptModule's graph for the given example inputs.

    Returns the (possibly frozen) module with the chain applied in
    place. The chain has two stages.

    **Resolve CallMethod / CallFunction / GetAttr** (one of):

    - `torch.jit.freeze` (`freeze=True`, default). Requires the module
      to be in eval mode; pass a `model.eval()`-ed instance or set
      `freeze=False`. On `RuntimeError`, the helper logs a warning and
      falls back to the manual inline.
    - `inline_unresolvable_submodules`. Used when freeze was disabled
      or failed; covers CallMethods whose target class isn't on the
      import path. Skipped when freeze succeeded.

    **Specialize for the example inputs** (always, in this order):

    1. `replace_size_calls_with_constants`: forward-reach analysis
       folds `aten::dim/size/len/numel` whose values flow only into
       control flow. Tensor-shape consumers are left dynamic.
    2. `fold_constant_scalar_arithmetic`: cmps, `__not__`,
       `__contains__`, scalar casts.
    3. `fold_constant_ifs`: drop Ifs with a constant boolean condition.
    4. `fold_tuple_index_through_tuple_construct` and
       `fold_tuple_unpack_through_tuple_construct`: collapse the
       `TupleConstruct -> TupleIndex` and
       `TupleConstruct -> TupleUnpack` round-trips.
    5. `strip_prim_data`: drop `prim::data(t)` (autograd-detach no-op).
    6. `strip_assertion_ifs`: drop Ifs whose one branch is a pure
       `RaiseException`.
    7. `fold_data_dependent_ifs`: evaluate any remaining If condition
       under the example inputs and inline the chosen branch.

    `args` is the model's `forward` arguments (no implicit `self`); the
    helper prepends the (possibly frozen) `self` receiver internally
    when invoking passes that re-execute the graph through
    `torch._C._jit_interpret_graph`. A single `torch.Tensor` is accepted
    as a shorthand for `(tensor,)`, matching the permissive convention
    of `export_model_to_nnef(model, args=...)`.

    When `diagnostics` is a dict, it is populated in place with the
    count of nodes folded / stripped per pass, keyed by pass name. The
    `froze` key records whether freeze succeeded;
    `inline_unresolvable_submodules` is only present when the manual
    inline ran (i.e. freeze was disabled or failed). Useful for
    debugging unfamiliar JIT artifacts.
    """
    diag = diagnostics
    if isinstance(args, torch.Tensor):
        args = (args,)
    froze = False
    if freeze and isinstance(model, torch.jit.ScriptModule):
        # `torch.jit.freeze` reads `.training` and rejects already-frozen
        # modules with `AttributeError`; treat that as "nothing to freeze"
        # and proceed. RuntimeError covers other unfrozenable conditions
        # (e.g. modules with parametrizations) and is logged because the
        # user may want to fix those upstream.
        try:
            model = torch.jit.freeze(model)
            froze = True
        except AttributeError:
            pass
        except RuntimeError as e:
            LOGGER.warning(
                "torch.jit.freeze failed (%s); continuing with manual inline",
                e,
            )
    interp_args = [model, *args]
    graph = model.graph
    torch._C._jit_pass_dce(graph)
    if not froze:
        n_inlined = inline_unresolvable_submodules(graph, model)
        if diag is not None:
            diag["inline_unresolvable_submodules"] = n_inlined
        torch._C._jit_pass_dce(graph)
    n_size = replace_size_calls_with_constants(graph, interp_args)
    n_arith = fold_constant_scalar_arithmetic(graph)
    n_const_if = fold_constant_ifs(graph)
    n_tuple_idx = fold_tuple_index_through_tuple_construct(graph)
    n_tuple_unpack = fold_tuple_unpack_through_tuple_construct(graph)
    n_data = strip_prim_data(graph)
    n_assert = strip_assertion_ifs(graph)
    n_dd_if = fold_data_dependent_ifs(graph, interp_args)
    torch._C._jit_pass_dce(graph)
    if diag is not None:
        diag["froze"] = froze
        diag["replace_size_calls_with_constants"] = n_size
        diag["fold_constant_scalar_arithmetic"] = n_arith
        diag["fold_constant_ifs"] = n_const_if
        diag["fold_tuple_index_through_tuple_construct"] = n_tuple_idx
        diag["fold_tuple_unpack_through_tuple_construct"] = n_tuple_unpack
        diag["strip_prim_data"] = n_data
        diag["strip_assertion_ifs"] = n_assert
        diag["fold_data_dependent_ifs"] = n_dd_if
    return model
