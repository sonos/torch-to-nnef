"""Dynamic axes helpers for NeMo Tract wrappers."""

import typing as T


def _is_batch_symbol(sym: str) -> bool:
    """Return True when symbol denotes a batch axis by name.

    Only names ending with '__BATCH' are considered batch symbols.
    """
    s = str(sym).upper()
    return s.endswith("__BATCH")


def _collapse_axes_for_input(
    orig_axes: T.Dict[int, str],
    *,
    full_axes_spec: T.Optional[T.Sequence[str]] = None,
    assume_batch_at0: bool = True,
) -> T.Dict[int, str]:
    """Collapse a single input axes mapping.

    Removes batch-like dims and reindexes remaining axes.
    """
    if full_axes_spec is not None:
        b_positions = [
            ix for ix, sym in enumerate(full_axes_spec) if _is_batch_symbol(sym)
        ]
    else:
        pairs_for_b = sorted(orig_axes.items(), key=lambda kv: kv[0])
        b_positions = [ix for ix, sym in pairs_for_b if _is_batch_symbol(sym)]
        if assume_batch_at0 and 0 not in b_positions:
            b_positions = [0] + b_positions

    if full_axes_spec is None:
        max_idx = max(orig_axes) if orig_axes else -1
        full_axes = [None] * (max_idx + 1)
        for i, s in orig_axes.items():
            full_axes[i] = s
        full_axes_spec = [s if s is not None else "?" for s in full_axes]

    new_map: T.Dict[int, int] = {}
    new_i = 0
    for i, sym in enumerate(full_axes_spec):
        if _is_batch_symbol(sym):
            continue
        new_map[i] = new_i
        new_i += 1

    collapsed: T.Dict[int, str] = {}
    for i, sym in orig_axes.items():
        if _is_batch_symbol(full_axes_spec[i]):
            continue
        collapsed[new_map[i]] = sym
    return collapsed


def collapse_dynamic_axes_mapping(
    nemo_dynamic_axes: T.Dict[str, T.Dict[int, str]],
    input_names: T.Sequence[str],
) -> T.Dict[str, T.Dict[int, str]]:
    """Collapse mapping for all inputs keeping only the exposed inputs."""
    full_axes_by_name: T.Dict[str, T.Sequence[str]] = {}
    for name, axes in nemo_dynamic_axes.items():
        max_idx = max(axes) if axes else -1
        spec = ["?"] * (max_idx + 1)
        for i, s in axes.items():
            spec[i] = s
        full_axes_by_name[name] = tuple(spec)

    collapsed: T.Dict[str, T.Dict[int, str]] = {}
    for name in input_names:
        axes = nemo_dynamic_axes.get(name)
        if not axes:
            continue
        collapsed[name] = _collapse_axes_for_input(
            axes,
            full_axes_spec=(full_axes_by_name or {}).get(name),
        )
    return collapsed
