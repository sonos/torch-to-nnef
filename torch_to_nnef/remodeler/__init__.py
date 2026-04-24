"""Provider-agnostic boundary remodeler scaffold.

This module defines small, typed building blocks to describe IO signatures
and boundary-only transforms (collapse, bind, and backend-facing symbol
renames), plus helpers to load/save a strict nested config.

Notes:
- The concrete YAML/JSON schema is parsed by domain-specific loaders
  (e.g. AxisSymbolRegistry in the NeMo package).
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

import torch
import yaml

from torch_to_nnef.remodeler.adapter import BoundaryAdapter, RenameOutputs
from torch_to_nnef.remodeler.dyn_axes import (
    apply_eval_symbols,
    apply_symbol_renames_to_dyn,
    remove_eval_symbols_from_dyn,
    rewrite_and_filter_assertions,
    rewrite_assertions_with_renames,
)
from torch_to_nnef.remodeler.schema import (
    INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE,
    INPUT_FIELD_COLLAPSE_DIMS,
    INPUT_FIELD_ORIGINAL_SHAPE,
    SHAPE_KEY_EXTENSIONS,
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
    "apply_eval_symbols",
    "apply_symbol_renames_to_dyn",
    "remove_eval_symbols_from_dyn",
    "rewrite_and_filter_assertions",
    "rewrite_assertions_with_renames",
    "PreparedSubnet",
    "prepare_subnet_export",
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
    """A remodel plan built from a validated axis-symbol registry.

    Attributes:
    - registry: The validated, parsed axis registry (nested schema).
        Typically an ``AxisSymbolRegistry`` instance provided by a
        domain package (e.g. ``torch_to_nnef_nemo``).
    """

    registry: T.Any


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


@dataclass(frozen=True)
class PreparedSubnet:
    """Result of applying registry-driven transforms to a raw subnet."""

    model: torch.nn.Module
    test_input: list
    input_names: list[str]
    output_names: list[str]
    dyn: dict[str, dict[int, str]]
    custom_extensions: list[str]


def prepare_subnet_export(
    model: torch.nn.Module,
    test_input: list,
    input_names: list[str],
    output_names: list[str],
    subnet_name: str,
    dyn: dict[str, dict[int, str]],
    custom_extensions: list[str],
    axis_registry: T.Optional[T.Any] = None,
) -> PreparedSubnet:
    """Apply all registry-driven transforms and return export-ready data.

    This consolidates eval-symbol resizing, boundary adaptation (collapse,
    bind, rename, output filtering), assertion rewriting, extension
    merging, and eval-symbol pinning into a single call.

    Providers feed in raw subnet data; the remodeler returns everything
    needed to build final export parameters.
    """
    if axis_registry is not None and axis_registry.eval_symbols_per_input:
        test_input = apply_eval_symbols(
            test_input,
            input_names,
            subnet_name,
            dyn,
            axis_registry.eval_symbols_per_input,
        )

    rename_map: dict[str, list[str]] = {}
    if axis_registry is not None:
        rename_map = axis_registry.renamed_symbols_per_subnet.get(
            subnet_name, {}
        )
        outputs_keep = axis_registry.outputs_keep_per_subnet.get(
            subnet_name, []
        )

        has_collapse = any(
            q.startswith(f"{subnet_name}.")
            for q in axis_registry.input_collapse_dims
        )
        has_bind = any(
            q.startswith(f"{subnet_name}.") for q in axis_registry.bind_to_dim
        )
        has_outputs_filter = bool(outputs_keep) and set(outputs_keep) != set(
            output_names
        )

        out_collapse = {
            qout.split(".", 1)[1]: axes
            for qout, axes in axis_registry.output_collapse_dims.items()
            if qout.startswith(f"{subnet_name}.")
        }
        has_out_collapse = bool(out_collapse)

        if has_collapse or has_bind or has_outputs_filter or has_out_collapse:
            model = BoundaryAdapter(
                model,
                subnet_name,
                test_input,
                dyn,
                {
                    k: set(v)
                    for k, v in axis_registry.input_collapse_dims.items()
                },
                axis_registry.bind_to_dim,
                rename_map,
                outputs_keep=outputs_keep,
                output_collapse_dims=out_collapse,
            )
            input_names = model.input_names
            output_names = model.output_names
            test_input = list(model.input_example())
            dyn = model.dynamic_shapes_for_export()
        elif rename_map:
            dyn = apply_symbol_renames_to_dyn(dyn, rename_map)

    custom_ext = set(
        rewrite_and_filter_assertions(
            list(custom_extensions),
            (
                axis_registry.renamed_symbols_per_subnet
                if axis_registry is not None
                else {}
            ).get(subnet_name, {}),
            dyn,
        )
    )
    if axis_registry is not None:
        custom_ext.update(
            axis_registry.extensions_per_subnet.get(subnet_name, [])
        )

    if axis_registry is not None and axis_registry.eval_symbols_per_input:
        remove_eval_symbols_from_dyn(
            input_names,
            subnet_name,
            dyn,
            axis_registry.eval_symbols_per_input,
        )

    return PreparedSubnet(
        model=model,
        test_input=test_input,
        input_names=input_names,
        output_names=output_names,
        dyn=dyn,
        custom_extensions=list(custom_ext),
    )


def save_config(
    path: T.Union[Path, str, None],
    registry: T.Any,
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

        def _repr_quoted(dumper, data):  # type: ignore[no-untyped-def]
            return dumper.represent_scalar(
                "tag:yaml.org,2002:str", str(data), style='"'
            )

        _FlowSeqDumper.add_representer(list, _repr_seq)  # type: ignore[arg-type]
        _FlowSeqDumper.add_representer(_QuotedStr, _repr_quoted)  # type: ignore[arg-type]

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


class _QuotedStr(str):
    """Marker for strings that must be double-quoted in YAML output."""


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

    for subnet, exts in (
        getattr(reg, "extensions_per_subnet", None) or {}
    ).items():
        if exts:
            bucket = nested.setdefault(subnet, {})
            bucket[SHAPE_KEY_EXTENSIONS] = [_QuotedStr(e) for e in exts]

    return nested
