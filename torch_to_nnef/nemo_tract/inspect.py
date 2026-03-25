import json
import logging
import typing as T
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch

from torch_to_nnef.nemo_tract.export import (
    iter_export_params_for_generic_nemo_asr_model,
)
from torch_to_nnef.utils import INJECTED, T2NExtra, require_extra_decorator

LOGGER = logging.getLogger(__name__)


class InspectStage(Enum):
    """Stages for signature inspection.

    Attributes:
        RAW: Model IO before any collapsing/binding/removal.
        COLLAPSED: After collapse requests are applied to IO boundaries.
        BOUND: After binding or removals (e.g., length from shape()).
        FINAL: Final export-visible IO after adapters and normalization.
    """

    RAW = "raw"
    COLLAPSED = "collapsed"
    BOUND = "bound"
    FINAL = "final"

    def order(self) -> int:
        """Return a stable order rank for human rendering.

        Returns:
            An integer rank where lower means earlier in display order.
        """
        return {
            InspectStage.RAW: 0,
            InspectStage.COLLAPSED: 1,
            InspectStage.BOUND: 2,
            InspectStage.FINAL: 3,
        }[self]


class InspectFormat(Enum):
    """Output formats for inspection results."""

    HUMAN = "human"
    JSON = "json"
    HUMAN_RICH = "human-rich"


@dataclass
class IODescriptor:
    """Input/Output descriptor for inspection rendering.

    Attributes:
        name: Tensor name in the subnet signature.
        shape: Symbolic or concrete shape per dimension.
        dtype: Stringified dtype (e.g., "float32"), if known.
        notes: Annotations about transformations (e.g., "collapsed:B").
    """

    name: str
    shape: T.List[T.Union[int, str]]
    dtype: str | None
    notes: T.List[str]


# AxisSymbolMap is a simple type alias for axis-index→symbol mapping.
AxisSymbolMap = T.Dict[int, str]


@dataclass
class SubnetSignature:
    """Per-subnet signature snapshot at a given stage."""

    name: str
    stage: InspectStage
    inputs: T.List[IODescriptor]
    outputs: T.List[IODescriptor]
    applied_flags: T.List[str]
    symbol_axes: T.Dict[str, T.Dict[int, str]]


def group_by_subnet(
    snapshots: T.List[SubnetSignature],
) -> dict[str, list[SubnetSignature]]:
    """Group signature snapshots by subnet name preserving order."""
    per: dict[str, list[SubnetSignature]] = {}
    for s in snapshots:
        per.setdefault(s.name, []).append(s)
    return per


def sig_key(e: SubnetSignature) -> tuple:
    in_key = tuple(
        (i.name, tuple(map(str, i.shape)), i.dtype or "", tuple(i.notes))
        for i in e.inputs
    )
    out_key = tuple(
        (o.name, tuple(map(str, o.shape)), o.dtype or "", tuple(o.notes))
        for o in e.outputs
    )
    return (in_key, out_key)


def group_consecutive(
    entries: list[SubnetSignature],
) -> list[tuple[list[InspectStage], SubnetSignature]]:
    """Group consecutive entries with identical IO signatures."""
    groups: list[tuple[list[InspectStage], SubnetSignature]] = []
    for e in entries:
        k = sig_key(e)
        if groups and sig_key(groups[-1][1]) == k:
            groups[-1][0].append(e.stage)
        else:
            groups.append(([e.stage], e))
    return groups


def render_groups_plain(
    subnet_name: str,
    groups: list[tuple[list[InspectStage], SubnetSignature]],
) -> list[str]:
    """Render grouped signatures for one subnet into plain-text lines."""
    lines: list[str] = [f"Subnet: {subnet_name}"]
    for stages, rep in groups:
        stages_txt = ", ".join(s.value for s in stages)
        lines.append(f"  Stages: {stages_txt}")
        lines.append("    Inputs:")
        for i in rep.inputs:
            shp = ", ".join(str(d) for d in i.shape) if i.shape else ""
            ann = f" [{' '.join(i.notes)}]" if i.notes else ""
            dt = f" ({i.dtype})" if i.dtype else ""
            shape_txt = f" [{shp}]" if shp else ""
            lines.append(f"      - {i.name}:{shape_txt}{dt}{ann}")
        lines.append("    Outputs:")
        for o in rep.outputs:
            lines.append(f"      - {o.name}")
    return lines


def render_diffs_plain(
    groups: list[tuple[list[InspectStage], SubnetSignature]],
) -> list[str]:
    """Render diffs between successive unique signature groups."""
    lines: list[str] = []
    for gi in range(len(groups) - 1):
        stages_a, a = groups[gi]
        stages_b, b = groups[gi + 1]
        left = ",".join(s.value for s in stages_a)
        right = ",".join(s.value for s in stages_b)
        lines.append(f"  Diff: {left} -> {right}")
        a_map = {i.name: i for i in a.inputs}
        b_map = {i.name: i for i in b.inputs}
        all_names = sorted(set(a_map.keys()) | set(b_map.keys()))
        for nm in all_names:
            ai, bi = a_map.get(nm), b_map.get(nm)
            if ai and bi:
                if (
                    ai.shape != bi.shape
                    or ai.dtype != bi.dtype
                    or ai.notes != bi.notes
                ):
                    lines.append(
                        f"    - {nm}: {ai.shape or []} -> {bi.shape or []}"
                    )
            elif ai and not bi:
                lines.append(f"    - {nm}: present -> removed")
            elif bi and not ai:
                lines.append(f"    - {nm}: absent -> present")
    return lines


def _tensor_shape_with_symbols(
    t: object, symbols: AxisSymbolMap | None
) -> T.List[T.Union[int, str]]:
    """Build a list for the tensor shape using provided symbols.

    Args:
        t: A candidate tensor or non-tensor object.
        symbols: Optional axis-index to symbol mapping.

    Returns:
        A list mixing ints and symbols representing each dimension.
        Returns an empty list for non-tensor inputs.
    """
    if not torch.is_tensor(t):
        return []
    shp: T.List[T.Union[int, str]] = []
    syms = symbols or {}
    for i, dim in enumerate(list(t.shape)):
        shp.append(syms.get(i, int(dim)))
    return shp


def _dtype_of(t: object) -> str | None:
    """Return the tensor dtype as a short string, if available."""
    if torch.is_tensor(t):
        return str(t.dtype).replace("torch.", "")
    return None


def _collect_signatures_for_stage(
    *,
    asr_model,
    inference_target,
    stage: InspectStage,
    skip_preprocessor: bool,
    split_joint_decoder: bool,
    float_dtype: torch.dtype,
    only_subnets: T.Optional[T.Collection[str]],
    collapse_batch_dim_default: bool,
) -> T.List[SubnetSignature]:
    """Collect per-subnet signatures for a specific stage.

    Args:
        asr_model: NeMo model instance.
        inference_target: Export target wrapper.
        stage: Logical inspection stage.
        skip_preprocessor: Whether to skip preprocessor subnet.
        split_joint_decoder: Whether to split joint/decoder.
        float_dtype: Preferred float dtype for examples.
        only_subnets: Optional subset filter.
        collapse_batch_dim_default: Collapse flag for non-RAW stages.

    Returns:
        A list of SubnetSignature snapshots in discovery order.
    """
    # Phase 0 mapping:
    # - raw => no collapse
    # - collapsed/bound/final => follow provided flag
    if stage == InspectStage.RAW:
        collapse = False
    else:
        collapse = collapse_batch_dim_default

    sigs: T.List[SubnetSignature] = []
    for ep in iter_export_params_for_generic_nemo_asr_model(
        asr_model,
        inference_target,
        skip_preprocessor=skip_preprocessor,
        split_joint_decoder=split_joint_decoder,
        float_dtype=float_dtype,
        remove_unused_inputs=True,
        collapse_batch_dim=collapse,
        only_subnets=only_subnets,
    ):
        # Inputs: names, shapes, dtypes, notes
        inputs: T.List[IODescriptor] = []
        # ep.test_input can be a list/tuple aligned with ep.input_names
        test_in = (
            list(ep.test_input)
            if isinstance(ep.test_input, (list, tuple))
            else []
        )
        dyn_axes = ep.inference_target.dynamic_axes or {}
        for idx, name in enumerate(ep.input_names):
            t = test_in[idx] if idx < len(test_in) else None
            sym_map = (
                (dyn_axes.get(name) or {}) if isinstance(dyn_axes, dict) else {}
            )
            shp = _tensor_shape_with_symbols(t, sym_map)
            dt = _dtype_of(t)
            notes: T.List[str] = []
            # Basic annotation in Phase 0:
            # note collapse when a B/BATCH symbol existed at any axis index
            if collapse and any(
                (str(s).upper() == "B" or "BATCH" in str(s).upper())
                for s in sym_map.values()
            ):
                notes.append("collapsed:B")
            inputs.append(
                IODescriptor(name=name, shape=shp, dtype=dt, notes=notes)
            )

        # Outputs: list names only at this phase (no extra forward pass)
        outputs: T.List[IODescriptor] = [
            IODescriptor(name=nm, shape=[], dtype=None, notes=[])
            for nm in (ep.output_names or [])
        ]

        applied_flags: T.List[str] = []
        if collapse:
            applied_flags.append("--collapse-batch-dim")

        sigs.append(
            SubnetSignature(
                name=ep.name,
                stage=stage,
                inputs=inputs,
                outputs=outputs,
                applied_flags=applied_flags,
                symbol_axes={
                    k: v
                    for k, v in (dyn_axes or {}).items()
                    if k in ep.input_names
                },
            )
        )

    return sigs


def _print_human(
    sigs: T.List[SubnetSignature], *, to_path: Path | None, diff: bool
) -> None:
    """Render signatures as human-readable text to stdout or a file."""
    # Use module-level helpers to keep complexity low

    lines: list[str] = []
    for subnet_name, entries in group_by_subnet(sigs).items():
        entries.sort(key=lambda e: e.stage.order())
        groups = group_consecutive(entries)
        lines.extend(render_groups_plain(subnet_name, groups))
        if diff and len(groups) >= 2:
            lines.extend(render_diffs_plain(groups))

    out_txt = "\n".join(lines) + ("\n" if lines else "")
    if to_path is None:
        print(out_txt, end="")
    else:
        to_path.parent.mkdir(parents=True, exist_ok=True)
        to_path.write_text(out_txt, encoding="utf8")


def _print_json(sigs: T.List[SubnetSignature], *, to_path: Path | None) -> None:
    """Render signatures as JSON for tooling and CI.

    Args:
        sigs: Signatures to serialize.
        to_path: Optional output file path; stdout if None.
    """
    payload = {
        "subnets": [
            {
                "name": s.name,
                "stage": s.stage.value,
                "inputs": [
                    {
                        "name": i.name,
                        "shape": [str(d) for d in i.shape],
                        "dtype": i.dtype,
                        "notes": i.notes,
                    }
                    for i in s.inputs
                ],
                "outputs": [
                    {
                        "name": o.name,
                        "shape": [str(d) for d in o.shape],
                        "dtype": o.dtype,
                        "notes": o.notes,
                    }
                    for o in s.outputs
                ],
                "applied_flags": s.applied_flags,
                "symbol_axes": {
                    k: {int(ax): str(sym) for ax, sym in v.items()}
                    for k, v in s.symbol_axes.items()
                },
            }
            for s in sigs
        ]
    }
    txt = json.dumps(payload, indent=2)
    if to_path is None:
        print(txt)
    else:
        to_path.parent.mkdir(parents=True, exist_ok=True)
        to_path.write_text(txt + "\n", encoding="utf8")


def _rich_make_tables(rich, rep: SubnetSignature):
    """Build Rich tables for inputs/outputs and return with counts.

    Returns:
        has_notes, tin, tout, in_count, out_count
    """
    table_cls = rich.table.Table
    has_notes = any(bool(i.notes) for i in rep.inputs)
    tin = table_cls(show_header=True, header_style="bold", pad_edge=False)
    tin.add_column("Input")
    tin.add_column("Shape")
    tin.add_column("Dtype", style="dim")
    if has_notes:
        tin.add_column("Notes", style="yellow")
    in_count = 0
    for i in rep.inputs:
        shp = ", ".join(str(d) for d in i.shape) if i.shape else ""
        row = [i.name, f"[{shp}]" if shp else "", i.dtype or ""]
        if has_notes:
            row.append(" ".join(i.notes) if i.notes else "")
        tin.add_row(*row)
        in_count += 1
    tout = table_cls(show_header=True, header_style="bold", pad_edge=False)
    tout.add_column("Output")
    out_count = 0
    for o in rep.outputs:
        tout.add_row(o.name)
        out_count += 1
    return has_notes, tin, tout, in_count, out_count


def _rich_balance_tables(
    tin, tout, in_count: int, out_count: int, has_notes: bool
):
    if in_count > out_count:
        for _ in range(in_count - out_count):
            tout.add_row("")
    elif out_count > in_count:
        pad_cols = 3 + (1 if has_notes else 0)
        empty_row = [""] * pad_cols
        for _ in range(out_count - in_count):
            tin.add_row(*empty_row)


def _rich_print_diffs(rich, console, groups):
    text_cls = rich.text.Text
    table_cls = rich.table.Table
    for gi in range(len(groups) - 1):
        stages_a, a = groups[gi]
        stages_b, b = groups[gi + 1]
        left = ",".join(s.value for s in stages_a)
        right = ",".join(s.value for s in stages_b)
        console.print(text_cls(f"Diff: {left} → {right}", style="bold yellow"))
        a_map = {i.name: i for i in a.inputs}
        b_map = {i.name: i for i in b.inputs}
        all_names = sorted(set(a_map.keys()) | set(b_map.keys()))
        td = table_cls(show_header=True, header_style="bold")
        td.add_column("Input")
        td.add_column(",".join(s.value for s in stages_a))
        td.add_column("")
        td.add_column(",".join(s.value for s in stages_b))
        for nm in all_names:
            ai, bi = a_map.get(nm), b_map.get(nm)
            if ai and bi:
                changed = (
                    ai.shape != bi.shape
                    or ai.dtype != bi.dtype
                    or ai.notes != bi.notes
                )
                if changed:
                    td.add_row(
                        nm, str(ai.shape or []), "→", str(bi.shape or [])
                    )
            elif ai and not bi:
                td.add_row(nm, "present", "→", "removed")
            elif bi and not ai:
                td.add_row(nm, "absent", "→", "present")
        console.print(td)


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="rich")
def _print_human_rich(
    sigs: T.List[SubnetSignature],
    *,
    to_path: Path | None,
    diff: bool,
    rich=INJECTED,
) -> None:
    """Render signatures using Rich tables/colors (requires extra)."""
    if to_path is not None:
        _print_human(sigs, to_path=to_path, diff=diff)
        return
    console_cls = rich.console.Console
    columns_cls = rich.columns.Columns
    text_cls = rich.text.Text

    console = console_cls()
    for subnet_name, entries in group_by_subnet(sigs).items():
        entries.sort(key=lambda e: e.stage.order())
        groups = group_consecutive(entries)

        console.print(text_cls(f"Subnet: {subnet_name}", style="bold cyan"))
        for stages, rep in groups:
            stages_txt = ", ".join(s.value for s in stages)
            console.print(
                text_cls(f"Stages: {stages_txt}", style="bold magenta")
            )
            has_notes, tin, tout, in_count, out_count = _rich_make_tables(
                rich, rep
            )
            _rich_balance_tables(tin, tout, in_count, out_count, has_notes)
            try:
                term_width = console.size.width  # type: ignore[attr-defined]
            except AttributeError:
                term_width = 80
            if term_width >= 100:
                cols = columns_cls(
                    [tin, tout], equal=False, expand=False, padding=1
                )
                console.print(cols)
            else:
                console.print(tin)
                console.print(tout)

        if diff and len(groups) >= 2:
            _rich_print_diffs(rich, console, groups)


def run_inspection(
    *,
    asr_model,
    inference_target,
    collapse_batch_dim: bool,
    skip_preprocessor: bool,
    split_joint_decoder: bool,
    float_dtype: torch.dtype,
    only_subnets: T.Optional[T.Collection[str]],
    stages: T.Optional[T.List[InspectStage]],
    fmt: InspectFormat,
    to_path: Path | None,
    diff: bool,
) -> None:
    """Run the inspection pipeline and print results.

    Args:
        asr_model: NeMo model instance.
        inference_target: Export target wrapper.
        collapse_batch_dim: Whether to drop batch in boundary shapes.
        skip_preprocessor: Whether to skip preprocessor subnet.
        split_joint_decoder: Whether to split joint/decoder.
        float_dtype: Preferred float dtype for examples.
        only_subnets: Optional subset filter.
        stages: Stages to display; defaults to FINAL.
        fmt: Output format (human or json).
        to_path: Optional file path to write output.
        diff: Whether to display diffs between unique signature groups.
    """
    # Normalize stages selection
    chosen: list[InspectStage] = stages if stages else [InspectStage.FINAL]

    LOGGER.info(
        "running inspection for stages: %s",
        ", ".join(st.value for st in chosen),
    )

    all_sigs: list[SubnetSignature] = []
    for st in chosen:
        all_sigs.extend(
            _collect_signatures_for_stage(
                asr_model=asr_model,
                inference_target=inference_target,
                stage=st,
                skip_preprocessor=skip_preprocessor,
                split_joint_decoder=split_joint_decoder,
                float_dtype=float_dtype,
                only_subnets=only_subnets,
                collapse_batch_dim_default=collapse_batch_dim,
            )
        )

    if fmt == InspectFormat.HUMAN:
        _print_human(all_sigs, to_path=to_path, diff=diff)
    elif fmt == InspectFormat.HUMAN_RICH:
        _print_human_rich(all_sigs, to_path=to_path, diff=diff)
    else:
        _print_json(all_sigs, to_path=to_path)
