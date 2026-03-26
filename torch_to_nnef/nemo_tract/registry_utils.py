import typing as T

from torch_to_nnef.exceptions import T2NErrorInvalidArgument
from torch_to_nnef.nemo_tract.axis_registry import (
    AxisSymbolRegistry,
)
from torch_to_nnef.remodeler import SubnetSignature


def _build_discovery_maps(
    signatures: T.List[SubnetSignature],
) -> tuple[
    set[str],
    set[str],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    disc_subnets: set[str] = {ss.name for ss in signatures}
    disc_inputs: set[str] = set()
    disc_outputs: dict[str, set[str]] = {}
    disc_input_dyn_syms: dict[str, set[str]] = {}
    subnet_dyn_syms: dict[str, set[str]] = {}
    for ss in signatures:
        for i in ss.inputs:
            disc_inputs.add(f"{ss.name}.{i.name}")
            axes = (ss.symbol_axes or {}).get(i.name, {}) or {}
            syms = {str(s).upper() for s in axes.values()}
            disc_input_dyn_syms[f"{ss.name}.{i.name}"] = syms
            subnet_dyn_syms.setdefault(ss.name, set()).update(syms)
        disc_outputs.setdefault(ss.name, set()).update(
            o.name for o in ss.outputs
        )
    return (
        disc_subnets,
        disc_inputs,
        disc_outputs,
        disc_input_dyn_syms,
        subnet_dyn_syms,
    )


def _collect_registry_sets(
    registry: AxisSymbolRegistry,
) -> tuple[set[str], set[str], set[str], set[str]]:
    reg_inputs = set((registry.symbols_per_input or {}).keys())
    reg_binds = set((registry.bind_to_dim or {}).keys())
    reg_collapse = set((registry.input_collapse_dims or {}).keys())
    reg_subnets: set[str] = set()
    for q in reg_inputs | reg_binds | reg_collapse:
        if "." in q:
            reg_subnets.add(q.split(".", 1)[0])
    reg_subnets |= set((registry.renamed_symbols_per_subnet or {}).keys())
    reg_subnets |= set((registry.outputs_keep_per_subnet or {}).keys())
    return reg_inputs, reg_binds, reg_collapse, reg_subnets


def _append_unknowns(
    problems: list[str],
    *,
    reg_subnets: set[str],
    disc_subnets: set[str],
    reg_inputs: set[str],
    reg_binds: set[str],
    reg_collapse: set[str],
    disc_inputs: set[str],
) -> None:
    unknown_subnets = sorted(reg_subnets - disc_subnets)
    if unknown_subnets:
        problems.append("unknown subnets: " + ", ".join(unknown_subnets))
    unknown_inputs = sorted(
        (reg_inputs | reg_binds | reg_collapse) - disc_inputs
    )
    if unknown_inputs:
        problems.append("unknown inputs: " + ", ".join(unknown_inputs))


def _check_outputs_keep(
    problems: list[str],
    *,
    registry: AxisSymbolRegistry,
    disc_outputs: dict[str, set[str]],
) -> None:
    for subnet, keep in (registry.outputs_keep_per_subnet or {}).items():
        disc = disc_outputs.get(subnet, set())
        extra = sorted(set(keep or []) - disc)
        if extra:
            problems.append(
                f"outputs_keep for '{subnet}' has unknown: " + ", ".join(extra)
            )


def _check_bind_refs(
    problems: list[str],
    *,
    registry: AxisSymbolRegistry,
    disc_inputs: set[str],
    disc_input_dyn_syms: dict[str, set[str]],
) -> None:
    for tgt_q, src in (registry.bind_to_dim or {}).items():
        if not isinstance(src, str) or "." not in src:
            problems.append(f"bind_to_dim for '{tgt_q}' is invalid: {src!r}")
            continue
        src_q, _, sym = src.rpartition(".")
        if src_q not in disc_inputs:
            problems.append(
                f"bind_to_dim references unknown source '{src_q}' for '{tgt_q}'"
            )
        if not sym:
            problems.append(
                f"bind_to_dim for '{tgt_q}' missing terminal symbol in '{src}'"
            )
        if sym:
            known = disc_input_dyn_syms.get(src_q, set())
            if str(sym).upper() not in known:
                problems.append(
                    "bind_to_dim symbol '"
                    + str(sym)
                    + "' not in dynamic axes of '"
                    + src_q
                    + "'"
                )


def _check_collapse_and_renames(
    problems: list[str],
    *,
    registry: AxisSymbolRegistry,
    disc_input_dyn_syms: dict[str, set[str]],
    subnet_dyn_syms: dict[str, set[str]],
) -> None:
    for q, syms in (registry.input_collapse_dims or {}).items():
        known = disc_input_dyn_syms.get(q, set())
        for s in syms or []:
            if str(s).upper() not in known:
                problems.append(
                    f"collapse_dims for '{q}' contains unknown symbol '{s}'"
                )
    for subnet, mapping in (registry.renamed_symbols_per_subnet or {}).items():
        known = subnet_dyn_syms.get(subnet, set())
        for _t, srcs in (mapping or {}).items():
            for s in srcs or []:
                if str(s).upper() not in known:
                    problems.append(
                        "renamed_symbols for '"
                        + subnet
                        + "' has unknown source '"
                        + str(s)
                        + "'"
                    )


def dump_registry_from_signatures(
    signatures: T.List[SubnetSignature],
) -> AxisSymbolRegistry:
    symbols: dict[str, dict[int, str]] = {}
    ranks: dict[str, int] = {}
    outputs_keep_per_subnet: dict[str, list[str]] = {}

    for ss in signatures:
        outputs_keep_per_subnet[ss.name] = [o.name for o in (ss.outputs or [])]
        axes_map = ss.symbol_axes or {}
        for io in ss.inputs:
            q = f"{ss.name}.{io.name}"
            if io.shape:
                ranks[q] = len(io.shape)
            if io.name in axes_map and axes_map[io.name]:
                symbols[q] = {
                    int(ax): str(sym) for ax, sym in axes_map[io.name].items()
                }
            else:
                symbols.setdefault(q, {})

    return AxisSymbolRegistry(
        symbols_per_input=symbols,
        rank_per_input=ranks,
        bind_to_dim={},
        input_collapse_dims={},
        renamed_symbols_per_subnet={},
        outputs_keep_per_subnet=outputs_keep_per_subnet,
    )


def validate_registry_against_signatures(
    signatures: T.List[SubnetSignature], registry: AxisSymbolRegistry
) -> None:
    (
        disc_subnets,
        disc_inputs,
        disc_outputs,
        disc_input_dyn_syms,
        subnet_dyn_syms,
    ) = _build_discovery_maps(signatures)
    reg_inputs, reg_binds, reg_collapse, reg_subnets = _collect_registry_sets(
        registry
    )
    problems: list[str] = []
    _append_unknowns(
        problems,
        reg_subnets=reg_subnets,
        disc_subnets=disc_subnets,
        reg_inputs=reg_inputs,
        reg_binds=reg_binds,
        reg_collapse=reg_collapse,
        disc_inputs=disc_inputs,
    )
    _check_outputs_keep(problems, registry=registry, disc_outputs=disc_outputs)
    _check_bind_refs(
        problems,
        registry=registry,
        disc_inputs=disc_inputs,
        disc_input_dyn_syms=disc_input_dyn_syms,
    )
    _check_collapse_and_renames(
        problems,
        registry=registry,
        disc_input_dyn_syms=disc_input_dyn_syms,
        subnet_dyn_syms=subnet_dyn_syms,
    )
    if problems:
        details = "; ".join(problems)
        raise T2NErrorInvalidArgument(
            f"shape-config validation failed: {details}"
        )
