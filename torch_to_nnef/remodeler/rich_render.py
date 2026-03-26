import typing as T
from importlib import import_module

from torch_to_nnef.remodeler import SubnetSignature
from torch_to_nnef.remodeler.inspect_utils import (
    group_by_subnet,
    group_consecutive,
    render_diffs_plain,
    render_groups_plain,
)


def _rich_classes(rich):
    """Resolve rich classes using the provided module object.

    Imports submodules from the module name of the provided `rich` to avoid
    hard import-time dependencies.
    """
    base = rich.__name__
    Table = import_module(f"{base}.table").Table
    Text = import_module(f"{base}.text").Text
    Console = import_module(f"{base}.console").Console
    Columns = import_module(f"{base}.columns").Columns
    Rule = import_module(f"{base}.rule").Rule
    return Table, Text, Console, Columns, Rule


def _make_tables(rich, rep: SubnetSignature):
    Table, _, _, _, _ = _rich_classes(rich)

    has_notes = any(bool(i.notes) for i in rep.inputs)
    tin = Table(show_header=True, header_style="bold", pad_edge=False)
    tin.add_column("Input")
    tin.add_column("Shape")
    tin.add_column("Dtype", style="dim")
    if has_notes:
        tin.add_column("Notes", style="yellow")
    in_count = 0
    for i in rep.inputs:
        shp = ", ".join(str(d) for d in (i.shape or []))
        row = [i.name, f"[{shp}]" if shp else "", i.dtype or ""]
        if has_notes:
            row.append(" ".join(i.notes or []))
        tin.add_row(*row)
        in_count += 1
    tout = Table(show_header=True, header_style="bold", pad_edge=False)
    tout.add_column("Output")
    out_count = 0
    for o in rep.outputs:
        tout.add_row(o.name)
        out_count += 1
    return has_notes, tin, tout, in_count, out_count


def _balance_tables(tin, tout, in_count: int, out_count: int, has_notes: bool):
    if in_count > out_count:
        for _ in range(in_count - out_count):
            tout.add_row("")
    elif out_count > in_count:
        pad_cols = 3 + (1 if has_notes else 0)
        empty_row = [""] * pad_cols
        for _ in range(out_count - in_count):
            tin.add_row(*empty_row)


def _print_diffs(rich, console, groups):
    Table, Text, _, _, _ = _rich_classes(rich)

    for gi in range(len(groups) - 1):
        stages_a, a = groups[gi]
        stages_b, b = groups[gi + 1]
        left = ",".join(s.value for s in stages_a)
        right = ",".join(s.value for s in stages_b)
        console.print(Text(f"Diff: {left} → {right}", style="bold yellow"))
        a_map = {i.name: i for i in a.inputs}
        b_map = {i.name: i for i in b.inputs}
        all_names = sorted(set(a_map.keys()) | set(b_map.keys()))
        td = Table(show_header=True, header_style="bold")
        td.add_column("Input")
        td.add_column(",".join(s.value for s in stages_a))
        td.add_column("")
        td.add_column(",".join(s.value for s in stages_b))
        for nm in all_names:
            ai, bi = a_map.get(nm), b_map.get(nm)
            if ai and bi:
                changed = (
                    (ai.shape or []) != (bi.shape or [])
                    or (ai.dtype or "") != (bi.dtype or "")
                    or (ai.notes or []) != (bi.notes or [])
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


def print_signatures_rich(
    sigs: T.List[SubnetSignature],
    *,
    to_path=None,
    diff: bool = False,
    rich=None,
    model_label: T.Optional[str] = None,
) -> None:
    """Render signatures using Rich tables/colors (requires `rich`).

    If `to_path` is provided, falls back to plain-text rendering since rich
    console layout is typically for terminals only.
    """
    if to_path is not None:
        # Fallback to plain text when a file path is requested
        lines: list[str] = []
        printed_header = False
        for subnet_name, entries in group_by_subnet(sigs).items():
            entries.sort(key=lambda e: e.stage.order)
            groups = group_consecutive(entries)
            if model_label and not printed_header:
                lines.append(f"Model: {model_label}")
                printed_header = True
            lines.extend(render_groups_plain(subnet_name, groups))
            if diff and len(groups) >= 2:
                lines.extend(render_diffs_plain(groups))
        out_txt = "\n".join(lines) + ("\n" if lines else "")
        to_path.parent.mkdir(parents=True, exist_ok=True)
        to_path.write_text(out_txt, encoding="utf8")
        return

    if rich is None:
        raise RuntimeError(
            "rich module must be provided to print_signatures_rich"
        )
    Table, Text, Console, Columns, Rule = _rich_classes(rich)
    console = Console()
    if model_label:
        console.print(Rule(Text(f"Model: {model_label}", style="bold")))

    for subnet_name, entries in group_by_subnet(sigs).items():
        entries.sort(key=lambda e: e.stage.order)
        groups = group_consecutive(entries)
        console.print(Text(f"Subnet: {subnet_name}", style="bold"))
        for _, rep in groups:
            has_notes, tin, tout, in_count, out_count = _make_tables(rich, rep)
            _balance_tables(tin, tout, in_count, out_count, has_notes)
            try:
                term_width = console.size.width  # type: ignore[attr-defined]
            except AttributeError:
                term_width = 80
            if term_width >= 100:
                cols = Columns(
                    [tin, tout], equal=False, expand=False, padding=1
                )
                console.print(cols)
            else:
                console.print(tin)
                console.print(tout)
        if diff and len(groups) >= 2:
            _print_diffs(rich, console, groups)
