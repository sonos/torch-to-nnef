"""Provider-agnostic helpers for dynamic-axes manipulation.

These utilities operate on the generic ``dyn`` mapping
(``{input_name: {axis_index: symbol}}``) and assertion/extension
strings.  They are used by NeMo and can be reused by any provider.
"""

import re
import typing as T

import torch

# -- symbol renames ----------------------------------------------------------


def apply_symbol_renames_to_dyn(
    dyn: T.Dict[str, T.Dict[int, str]],
    rename_map: T.Dict[str, T.List[str]],
) -> T.Dict[str, T.Dict[int, str]]:
    """Apply symbol renames directly to a dynamic axes mapping.

    This is the lightweight alternative to BoundaryAdapter when only
    symbol renames are needed (no collapse, bind, or output filtering).
    """
    if not rename_map:
        return dyn
    inv: dict[str, str] = {}
    for tgt, srcs in rename_map.items():
        t_u = str(tgt).upper()
        for s in srcs or []:
            inv[str(s).upper()] = t_u
    return {
        name: {i: inv.get(str(s).upper(), str(s)) for i, s in axes.items()}
        for name, axes in dyn.items()
    }


# -- eval symbols -----------------------------------------------------------


def apply_eval_symbols(
    test_input: list,
    input_names: list[str],
    subnet_name: str,
    dyn: T.Dict[str, T.Dict[int, str]],
    eval_symbols: T.Dict[str, T.Dict[str, int]],
) -> list:
    """Resize test_input tensors according to eval_symbols."""
    result = list(test_input)
    for i, name in enumerate(input_names):
        if i >= len(result):
            break
        qname = f"{subnet_name}.{name}"
        evals = eval_symbols.get(qname)
        if not evals:
            continue
        t = result[i]
        if not torch.is_tensor(t):
            continue
        axes = dyn.get(name, {})
        for ax_idx, sym in axes.items():
            target = evals.get(str(sym).upper())
            if target is not None and 0 <= ax_idx < t.dim():
                current = t.size(ax_idx)
                if target < current:
                    t = t.narrow(ax_idx, 0, target)
                elif target > current:
                    new_shape = list(t.shape)
                    new_shape[ax_idx] = target
                    new_t = t.new_zeros(new_shape)
                    slices = [slice(None)] * t.dim()
                    slices[ax_idx] = slice(0, current)
                    new_t[tuple(slices)] = t
                    t = new_t
        result[i] = t
    return result


def remove_eval_symbols_from_dyn(
    input_names: list[str],
    subnet_name: str,
    dyn: T.Dict[str, T.Dict[int, str]],
    eval_symbols: T.Dict[str, T.Dict[str, int]],
) -> None:
    """Remove pinned axes from *dyn* so the backend treats them as constant.

    Must be called **after** the BoundaryAdapter is built, because the
    adapter needs the symbols to resolve bindings.
    """
    for name in input_names:
        qname = f"{subnet_name}.{name}"
        evals = eval_symbols.get(qname)
        if not evals:
            continue
        axes = dyn.get(name)
        if axes is None:
            continue
        dyn[name] = {
            ax: sym
            for ax, sym in axes.items()
            if evals.get(str(sym).upper()) is None
        }


# -- assertion rewriting ----------------------------------------------------


def rewrite_assertions_with_renames(
    assertions: list[str],
    rename_map: T.Optional[dict[str, list[str]]],
) -> list[str]:
    """Rewrite assertion symbol names based on a rename mapping.

    Args:
        assertions: List of assertion strings,
            e.g. "tract_assert U = BATCH".
        rename_map: Mapping of target symbol to list of source symbols
            that should be rewritten to the target.  Comparison is
            case-insensitive; rewritten symbols are emitted uppercased.

    Returns:
        A list of assertions with symbols rewritten according to
        the provided mapping.  Unknown tokens are left unchanged.
    """
    if not rename_map:
        return list(assertions)

    inv: dict[str, str] = {}
    for tgt, srcs in (rename_map or {}).items():
        t_u = str(tgt).upper()
        for s in srcs or []:
            inv[str(s).upper()] = t_u

    ident = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

    def _sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        return inv.get(tok.upper(), tok)

    return [ident.sub(_sub, str(a)) for a in assertions]


def filter_assertions_present_in_dyn(
    assertions: list[str],
    dyn: T.Optional[dict[str, dict[int, str]]],
) -> list[str]:
    """Drop assertions that reference a symbol absent from every axis.

    An assertion whose symbol(s) never appear in ``dyn`` (renamed away,
    collapsed, bound, or pinned static via ``eval_symbols``) can never be
    evaluated against a real dimension: it is dead weight in the exported
    NNEF that tract itself flags as a "mislabeled symbol name" warning.
    Applies to both auto-generated and user/slug/derived-declared
    assertions alike -- callers should apply this *after* any such symbol
    has already been removed from ``dyn``.

    Returns de-duplicated assertions, order preserved.
    """
    present: set[str] = set()
    for axes in (dyn or {}).values():
        for s in (axes or {}).values():
            present.add(str(s).upper())

    ident = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    filtered = [
        a
        for a in assertions
        if all(
            t.upper() in present
            for t in ident.findall(a)
            if t.upper() not in {"TRACT_ASSERT"}
        )
    ]
    return list(dict.fromkeys(filtered))


def rewrite_and_filter_assertions(
    assertions: list[str],
    rename_map: T.Optional[dict[str, list[str]]],
    dyn: T.Optional[dict[str, dict[int, str]]],
) -> list[str]:
    """Rewrite assertions and drop those referencing removed symbols.

    - Applies symbol renames so source symbols map to their target alias.
    - Computes the set of present symbols from the current dynamic axes
      and discards any assertion that mentions a symbol not present after
      rewriting.
    - Returns de-duplicated assertions.
    """
    rewritten = rewrite_assertions_with_renames(assertions, rename_map)
    return filter_assertions_present_in_dyn(rewritten, dyn)
