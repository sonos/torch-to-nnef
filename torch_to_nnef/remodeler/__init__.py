"""Provider-agnostic boundary remodeler scaffold.

This module defines small, typed building blocks to describe IO signatures
and boundary-only transforms (collapse, bind, and backend-facing symbol
renames), plus helpers to load/save a strict nested config.

Notes:
- The concrete YAML/JSON schema is identical to the NeMo remodeler and is
  parsed by the shared AxisSymbolRegistry loader used in the NeMo path.
- Providers are expected to discover per-subnet signatures, and to apply a
  remodel plan by wrapping inner modules with an adapter that enforces the
  external boundary while preserving the internal contract.
"""

from __future__ import annotations

import json
import typing as T
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
import yaml

# Reuse the validated nested schema and data container
from torch_to_nnef.nemo_tract.axis_registry import (
    AxisSymbolRegistry,
    load_axis_symbol_registry,
)
from torch_to_nnef.remodeler.adapter import BoundaryAdapter, RenameOutputs

__all__ = [
    "Stage",
    "IODescriptor",
    "SubnetSignature",
    "RemodelPlan",
    "Provider",
    "dump_registry_from_signatures",
    "plan_from_registry",
    "load_config",
    "save_config",
    "validate_registry_against_signatures",
    "BoundaryAdapter",
    "RenameOutputs",
]


class Stage(Enum):
    """Logical stages for signature inspection."""

    RAW = "raw"
    COLLAPSED = "collapsed"
    BOUND = "bound"
    FINAL = "final"


@dataclass(frozen=True)
class IODescriptor:
    """Description of a single input or output."""

    name: str
    shape: list[T.Union[int, str]]
    dtype: str | None = None
    notes: list[str] | None = None


@dataclass(frozen=True)
class SubnetSignature:
    """Per-subnet signature snapshot at a given stage."""

    name: str
    stage: Stage
    inputs: list[IODescriptor]
    outputs: list[IODescriptor]
    # Optional: dynamic axes as name -> {axis -> symbol}
    symbol_axes: dict[str, dict[int, str]] | None = None


@dataclass(frozen=True)
class RemodelPlan:
    """A remodel plan built from a validated AxisSymbolRegistry.

    Attributes:
    - registry: The validated, parsed axis registry (nested schema).
    """

    registry: AxisSymbolRegistry


class Provider(T.Protocol):
    """Provider SPI to discover signatures and apply remodel plans.

    Implementations should be small adapters around an existing provider
    (e.g., NeMo, plain PyTorch) that can:
    - discover raw and post-processed signatures for inspection
    - apply the boundary remodel plan to return wrapped modules ready for export
    """

    def discover_signatures(
        self, model: torch.nn.Module, stage: Stage
    ) -> list[SubnetSignature]:  # pragma: no cover - interface only
        raise NotImplementedError

    def apply(
        self, model: torch.nn.Module, plan: RemodelPlan
    ) -> dict[str, torch.nn.Module]:  # pragma: no cover - interface only
        raise NotImplementedError


def dump_registry_from_signatures(
    signatures: list[SubnetSignature],
) -> AxisSymbolRegistry:
    """Infer a starter AxisSymbolRegistry from signatures.

    - Uses provided input shapes and symbol axes to build per-input maps.
    - Always pre-fills per-subnet outputs_keep from discovered outputs.
    """
    symbols: dict[str, dict[int, str]] = {}
    ranks: dict[str, int] = {}
    outputs_keep_per_subnet: dict[str, list[str]] = {}

    for ss in signatures:
        # Pre-fill outputs_keep in the template for convenience
        outputs_keep_per_subnet[ss.name] = [o.name for o in (ss.outputs or [])]
        axes_map = ss.symbol_axes or {}
        for io in ss.inputs:
            q = f"{ss.name}.{io.name}"
            if io.shape:
                ranks[q] = len(io.shape)
            # Map existing dynamic axes; keep stable ints as concrete dims
            if io.name in axes_map and axes_map[io.name]:
                # Normalize symbols to strings
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


def plan_from_registry(registry: AxisSymbolRegistry) -> RemodelPlan:
    """Build a remodel plan from a validated axis registry."""
    if not isinstance(registry, AxisSymbolRegistry):
        raise TypeError("registry must be an AxisSymbolRegistry")
    return RemodelPlan(registry=registry)


def load_config(path: Path | str) -> AxisSymbolRegistry:
    """Load a YAML/JSON remodel config into an AxisSymbolRegistry."""
    return load_axis_symbol_registry(Path(path))


def save_config(
    path: Path | str | None,
    registry: AxisSymbolRegistry,
    *,
    flow_seq: bool = True,
    stream: T.TextIO | None = None,
) -> None:
    """Save an AxisSymbolRegistry to YAML or JSON.

    Args:
        path: Output file path (.yml/.yaml/.json) or None when using stream.
        registry: Parsed registry to serialize.
        flow_seq: When YAML, render short lists in flow style.
        stream: Optional text stream to write into (YAML or JSON). When
            provided, `path` can be None and the function infers YAML vs JSON
            based on the filename if available; defaults to YAML behavior.
    """
    if path is None and stream is None:
        raise ValueError("either 'path' or 'stream' must be provided")
    p = Path(path) if path is not None else None
    is_yaml = True if p is None else p.suffix.lower() in (".yml", ".yaml")
    if is_yaml:

        class _FlowSeqDumper(yaml.SafeDumper):
            pass

        def _repr_seq(dumper, data):  # type: ignore[no-untyped-def]
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=flow_seq
            )

        _FlowSeqDumper.add_representer(list, _repr_seq)  # type: ignore[arg-type]

        payload = _registry_to_nested_mapping(registry)
        if stream is not None:
            yaml.dump(
                payload,
                stream,
                Dumper=_FlowSeqDumper,
                sort_keys=False,
                default_flow_style=None,
            )
            return
        assert p is not None
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf8") as fh:
            yaml.dump(
                payload,
                fh,
                Dumper=_FlowSeqDumper,
                sort_keys=False,
                default_flow_style=None,
            )
        return

    # JSON
    payload = _registry_to_nested_mapping(registry)
    txt = json.dumps(payload, indent=2) + "\n"
    if stream is not None:
        stream.write(txt)
        return
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf8")


def _registry_to_nested_mapping(reg: AxisSymbolRegistry) -> dict[str, dict]:
    """Render AxisSymbolRegistry back to the nested mapping schema."""
    # Prepare nested layout: { subnet: { inputs: { name: {...} }, ... } }
    nested: dict[str, dict] = {}

    # Invert inputs like "subnet.input" back into nested keys
    for qname, axis_map in (reg.symbols_per_input or {}).items():
        if "." not in qname:
            # Keep defensive, but the loader always yields qualified keys
            subnet, inp = "", qname
        else:
            subnet, inp = qname.split(".", 1)
        bucket = nested.setdefault(subnet, {})
        inputs_map: dict = bucket.setdefault("inputs", {})
        entry: dict = {}
        # Reconstruct original_shape from rank map using axis positions
        rank = (reg.rank_per_input or {}).get(qname)
        if isinstance(rank, int) and rank >= 0:
            dims: list[T.Union[int, str]] = []
            for i in range(rank):
                sym = (axis_map or {}).get(i)
                dims.append(str(sym) if sym is not None else 1)
            entry["original_shape"] = dims
        else:
            entry["original_shape"] = []
        entry["collapse_dims"] = list(
            (reg.input_collapse_dims or {}).get(qname, [])
        )
        b = (reg.bind_to_dim or {}).get(qname)
        if isinstance(b, str) and b:
            entry["bind_scalar_to_dim_size"] = b
        inputs_map[inp] = entry

    # Copy optional renamed_symbols and outputs_keep per subnet
    for subnet, mapping in (reg.renamed_symbols_per_subnet or {}).items():
        bucket = nested.setdefault(subnet, {})
        if mapping:
            bucket["renamed_symbols"] = {
                str(t): [str(s) for s in (srcs or [])]
                for t, srcs in mapping.items()
            }
    for subnet, keep in (reg.outputs_keep_per_subnet or {}).items():
        bucket = nested.setdefault(subnet, {})
        bucket["outputs_keep"] = list(keep or [])

    return nested


def validate_registry_against_signatures(
    signatures: list[SubnetSignature], registry: AxisSymbolRegistry
) -> None:
    """Validate a registry structurally against discovered signatures.

    Checks
    - Unknown subnets in registry vs discovered
    - Unknown qualified inputs in registry vs discovered
    - outputs_keep entries are subsets of discovered outputs
    - bind_to_dim references point to known qualified inputs

    Raises ValueError with a concise aggregated message when mismatches exist.
    """
    # Build discovered sets
    disc_subnets: set[str] = {ss.name for ss in signatures}
    disc_inputs: set[str] = set()
    disc_outputs: dict[str, set[str]] = {}
    disc_input_dyn_syms: dict[str, set[str]] = {}
    subnet_dyn_syms: dict[str, set[str]] = {}
    for ss in signatures:
        for i in ss.inputs:
            disc_inputs.add(f"{ss.name}.{i.name}")
            # dynamic symbols known for this input at discovery time
            axes = (ss.symbol_axes or {}).get(i.name, {}) or {}
            syms = {str(s).upper() for s in axes.values()}
            disc_input_dyn_syms[f"{ss.name}.{i.name}"] = syms
            subnet_dyn_syms.setdefault(ss.name, set()).update(syms)
        disc_outputs.setdefault(ss.name, set()).update(
            o.name for o in ss.outputs
        )

    # Registry-derived subnets and inputs
    reg_inputs = set((registry.symbols_per_input or {}).keys())
    reg_binds = set((registry.bind_to_dim or {}).keys())
    reg_collapse = set((registry.input_collapse_dims or {}).keys())
    reg_subnets = set()
    for q in reg_inputs | reg_binds | reg_collapse:
        if "." in q:
            reg_subnets.add(q.split(".", 1)[0])
    reg_subnets |= set((registry.renamed_symbols_per_subnet or {}).keys())
    reg_subnets |= set((registry.outputs_keep_per_subnet or {}).keys())

    problems: list[str] = []

    # Unknown subnets
    unknown_subnets = sorted(reg_subnets - disc_subnets)
    if unknown_subnets:
        problems.append(
            "unknown subnets: " + ", ".join(unknown_subnets)
        )

    # Unknown inputs
    unknown_inputs = sorted(
        (reg_inputs | reg_binds | reg_collapse) - disc_inputs
    )
    if unknown_inputs:
        problems.append(
            "unknown inputs: " + ", ".join(unknown_inputs)
        )

    # outputs_keep subset
    for subnet, keep in (registry.outputs_keep_per_subnet or {}).items():
        disc = disc_outputs.get(subnet, set())
        extra = sorted(set(keep or []) - disc)
        if extra:
            problems.append(
                f"outputs_keep for '{subnet}' has unknown: " + ", ".join(extra)
            )

    # bind_to_dim source references
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
        # Check symbol present among dynamic axes of the source input
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

    # collapse_dims symbols known at input level
    for q, syms in (registry.input_collapse_dims or {}).items():
        known = disc_input_dyn_syms.get(q, set())
        for s in syms or []:
            if str(s).upper() not in known:
                problems.append(
                    f"collapse_dims for '{q}' contains unknown symbol '{s}'"
                )

    # renamed_symbols sources known at subnet level
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

    if problems:
        raise ValueError(
            "shape-config validation failed: " + "; ".join(problems)
        )
