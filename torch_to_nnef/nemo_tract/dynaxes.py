"""Dynamic-axes helpers shared by wrappers and export code.

This module centralizes dynamic-axis symbol generation, input-name expansion,
normalization of NeMo mappings, and rank-based filtering to avoid duplication.
"""

from __future__ import annotations

import typing as T

import torch

from torch_to_nnef.nemo_tract.constants import make_axis_symbol


def symbols_from_input_types(input_types) -> T.Dict[str, T.Dict[int, str]]:
    """Build symbols per input name from a module's ``input_types``.

    Each input maps to ``{axis_index: symbol}`` via ``make_axis_symbol``.
    """
    result: T.Dict[str, T.Dict[int, str]] = {}
    for name, ntype in input_types.items():
        axes = getattr(ntype, "axes", ())
        result[name] = {
            i: make_axis_symbol(name, ax, i) for i, ax in enumerate(axes)
        }
    return result


def expand_input_names(
    input_names: T.Sequence[str],
    input_example: T.Optional[T.Sequence[object]],
) -> T.Tuple[T.Dict[str, T.List[str]], T.Dict[str, int]]:
    """Expand tuple/list inputs and infer ranks for each expanded name.

    Returns (expand_map, ranks_by_name).
    """
    expand_map: T.Dict[str, T.List[str]] = {}
    ranks: T.Dict[str, int] = {}
    if isinstance(input_example, (list, tuple)):
        for name, val in zip(input_names, input_example):
            if isinstance(val, (list, tuple)) and len(val) > 0:
                tnames: T.List[str] = []
                for i, elem in enumerate(val):
                    tname = f"{name}_{i}"
                    tnames.append(tname)
                    ranks[tname] = (
                        int(elem.dim()) if torch.is_tensor(elem) else 0
                    )
                expand_map[name] = tnames
            else:
                expand_map[name] = [name]
                ranks[name] = int(val.dim()) if torch.is_tensor(val) else 0
    else:
        for name in input_names:
            expand_map[name] = [name]
            ranks[name] = 0
    return expand_map, ranks


def normalize_dynamic_indices(
    nemo_dynamic_axes: T.Mapping[str, T.Any], input_names: T.Sequence[str]
) -> T.Dict[str, T.Set[int]]:
    """Normalize NeMo dynamic mapping to base-name -> set(axis indices)."""
    base_to_idxs: T.Dict[str, T.Set[int]] = {n: set() for n in input_names}

    def to_base(k: str) -> T.Optional[str]:
        if k in base_to_idxs:
            return k
        if "_" in k:
            cand = k.split("_", 1)[0]
            if cand in base_to_idxs:
                return cand
        return None

    for k, v in nemo_dynamic_axes.items():
        base = to_base(str(k))
        if base is None:
            continue
        if hasattr(v, "keys") or isinstance(v, (list, tuple, set)):
            idxs = {int(i) for i in v}
        else:
            continue
        base_to_idxs[base].update(idxs)
    return base_to_idxs


def filter_dynamic_axes_by_ranks(
    dynamic_axes: T.Dict[str, T.Dict[int, str]],
    ranks_by_name: T.Mapping[str, int],
) -> T.Dict[str, T.Dict[int, str]]:
    """Drop axis indices that are >= rank for each input tensor name."""
    filtered: T.Dict[str, T.Dict[int, str]] = {}
    for name, axes in (dynamic_axes or {}).items():
        rank = int(ranks_by_name.get(name, 0))
        keep = {i: s for i, s in axes.items() if i < rank}
        if keep:
            filtered[name] = keep
    return filtered


def build_dynamic_axes(
    subnet,
    nemo_dynamic_axes: T.Mapping[str, T.Any],
    input_example: T.Optional[T.Sequence[object]] = None,
) -> T.Tuple[T.Dict[str, T.Dict[int, str]], T.Set[str]]:
    """Build dynamic_axes mapping for a subnet using its example.

    - Expand tuple/list inputs based on ``input_example``.
    - Normalize NeMo mapping to ``base-name -> set(indices)``.
    - Emit symbols via ``make_axis_symbol`` for each expanded name and index.
    - Return ``(dynamic_axes, custom_extensions_set)``.
    """
    expand_map, ranks = expand_input_names(subnet.input_names, input_example)
    base_to_idxs = normalize_dynamic_indices(
        nemo_dynamic_axes, subnet.input_names
    )

    def sym_for(tname: str, base: str, axis_idx: int) -> str:
        try:
            axes = subnet.input_types[base].axes  # type: ignore[index]
            return make_axis_symbol(tname, axes[axis_idx], axis_idx)
        except AttributeError:  # pragma: no cover - tolerant to NeMo variations
            return make_axis_symbol(tname, "DIM", axis_idx)

    dynamic_axes: T.Dict[str, T.Dict[int, str]] = {}
    used: T.Set[str] = set()
    for base, idxs in base_to_idxs.items():
        if not idxs:
            continue
        for tname in expand_map.get(base, [base]):
            valid = [i for i in sorted(idxs) if i < int(ranks.get(tname, 0))]
            if not valid:
                continue
            dynamic_axes[tname] = {i: sym_for(tname, base, i) for i in valid}
            used.update(dynamic_axes[tname].values())

    custom_ext = {f"tract_assert {s} >= 1" for s in used}
    return dynamic_axes, custom_ext
