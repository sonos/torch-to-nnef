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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

if TYPE_CHECKING:  # only for type checkers; avoids import-time cycles
    # Reuse the validated nested schema and data container
    from torch_to_nnef.nemo_tract.axis_registry import AxisSymbolRegistry
from torch_to_nnef.remodeler.adapter import BoundaryAdapter, RenameOutputs
from torch_to_nnef.remodeler.schema import (
    INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE,
    INPUT_FIELD_COLLAPSE_DIMS,
    INPUT_FIELD_ORIGINAL_SHAPE,
    SHAPE_KEY_INPUTS,
    SHAPE_KEY_OUTPUTS_KEEP,
    SHAPE_KEY_RENAMED,
)

__all__ = [
    "Stage",
    "IODescriptor",
    "SubnetSignature",
    "RemodelPlan",
    "Provider",
    "plan_from_registry",
    "save_config",
    "BoundaryAdapter",
    "RenameOutputs",
]


class Stage(Enum):
    """Logical stages for signature inspection."""

    RAW = "raw"
    COLLAPSED = "collapsed"
    BOUND = "bound"
    FINAL = "final"

    @property
    def order(self) -> int:
        """Stable sort order (RAW < COLLAPSED < BOUND < FINAL)."""
        return {
            Stage.RAW: 0,
            Stage.COLLAPSED: 1,
            Stage.BOUND: 2,
            Stage.FINAL: 3,
        }[self]


@dataclass(frozen=True)
class IODescriptor:
    """Description of a single input or output."""

    name: str
    shape: list[T.Union[int, str]]
    dtype: T.Optional[str] = None
    notes: T.Optional[list[str]] = None


@dataclass(frozen=True)
class SubnetSignature:
    """Per-subnet signature snapshot at a given stage."""

    name: str
    stage: Stage
    inputs: list[IODescriptor]
    outputs: list[IODescriptor]
    # Optional: dynamic axes as name -> {axis -> symbol}
    symbol_axes: T.Optional[dict[str, dict[int, str]]] = None
    # Optional: flags applied during transforms/inspection (provider-specific)
    applied_flags: T.List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RemodelPlan:
    """A remodel plan built from a validated AxisSymbolRegistry.

    Attributes:
    - registry: The validated, parsed axis registry (nested schema).
    """

    registry: "AxisSymbolRegistry"


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


def plan_from_registry(registry: T.Any) -> RemodelPlan:
    """Build a remodel plan from a validated registry (provider-specific)."""
    return RemodelPlan(registry=registry)


def save_config(
    path: T.Union[Path, str, None],
    registry: "AxisSymbolRegistry",
    *,
    flow_seq: bool = True,
    stream: T.Optional[T.TextIO] = None,
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
        raw_yaml = yaml.dump(
            payload,
            Dumper=_FlowSeqDumper,
            sort_keys=False,
            default_flow_style=None,
        )
        output_notes = getattr(registry, "_output_notes", {}) or {}
        raw_yaml = _inject_outputs_keep_comment(raw_yaml, output_notes)
        if stream is not None:
            stream.write(raw_yaml)
            return
        assert p is not None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(raw_yaml, encoding="utf8")
        return

    # JSON
    payload = (
        registry
        if isinstance(registry, dict)
        else _registry_to_nested_mapping(registry)
    )
    txt = json.dumps(payload, indent=2) + "\n"
    if stream is not None:
        stream.write(txt)
        return
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf8")


def _inject_outputs_keep_comment(
    raw_yaml: str,
    output_notes: dict[str, dict[str, str]],
) -> str:
    """Insert per-output unfold comments on outputs_keep lines.

    Produces e.g.::

        outputs_keep: [outputs, states]  # states unfolds to: states_0, states_1
    """
    if not output_notes:
        return raw_yaml
    lines = raw_yaml.splitlines(keepends=True)
    out: list[str] = []
    current_subnet: T.Optional[str] = None
    for line in lines:
        stripped = line.lstrip()
        # Track current top-level subnet key (no leading whitespace)
        if not line[0:1].isspace() and ":" in line:
            current_subnet = line.split(":", 1)[0].strip()
        if stripped.startswith(f"{SHAPE_KEY_OUTPUTS_KEEP}:"):
            notes = output_notes.get(current_subnet or "", {})
            if notes:
                parts = [f"{n} {v}" for n, v in notes.items()]
                line = line.rstrip("\n") + "  # " + "; ".join(parts) + "\n"
        out.append(line)
    return "".join(out)


def _registry_to_nested_mapping(reg: T.Any) -> dict[str, dict]:
    """Render AxisSymbolRegistry back to the nested mapping schema."""
    # Prepare nested layout: { subnet: { inputs: { name: {...} }, ... } }
    nested: dict[str, dict] = {}

    # Invert inputs like "subnet.input" back into nested keys
    for qname, axis_map in (
        getattr(reg, "symbols_per_input", None) or {}
    ).items():
        if "." not in qname:
            # Keep defensive, but the loader always yields qualified keys
            subnet, inp = "", qname
        else:
            subnet, inp = qname.split(".", 1)
        bucket = nested.setdefault(subnet, {})
        inputs_map: dict = bucket.setdefault(SHAPE_KEY_INPUTS, {})
        entry: dict = {}
        # Reconstruct original_shape from rank map using axis positions
        rank = (getattr(reg, "rank_per_input", None) or {}).get(qname)
        # Prefer captured original shapes (ints/strings) when available
        orig_map = getattr(reg, "original_shape_per_input", None) or {}
        orig_dims: list[T.Union[int, str]] = list(orig_map.get(qname, []))
        if isinstance(rank, int) and rank >= 0:
            dims: list[T.Union[int, str]] = []
            for i in range(rank):
                sym = axis_map.get(i)
                if sym is not None:
                    dims.append(str(sym))
                elif i < len(orig_dims):
                    dims.append(orig_dims[i])
                else:
                    dims.append(1)
            entry[INPUT_FIELD_ORIGINAL_SHAPE] = dims
        else:
            entry[INPUT_FIELD_ORIGINAL_SHAPE] = []
        entry[INPUT_FIELD_COLLAPSE_DIMS] = list(
            (getattr(reg, "input_collapse_dims", None) or {}).get(qname, [])
        )
        b = (getattr(reg, "bind_to_dim", None) or {}).get(qname)
        if isinstance(b, str) and b:
            entry[INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE] = b
        inputs_map[inp] = entry

    # Copy optional renamed_symbols and outputs_keep per subnet
    for subnet, mapping in (
        getattr(reg, "renamed_symbols_per_subnet", None) or {}
    ).items():
        bucket = nested.setdefault(subnet, {})
        if mapping:
            bucket[SHAPE_KEY_RENAMED] = {
                str(t): [str(s) for s in srcs] for t, srcs in mapping.items()
            }
    for subnet, keep in (
        getattr(reg, "outputs_keep_per_subnet", None) or {}
    ).items():
        bucket = nested.setdefault(subnet, {})
        bucket[SHAPE_KEY_OUTPUTS_KEEP] = list(keep)

    return nested
