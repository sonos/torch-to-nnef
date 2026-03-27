import typing as T

from torch_to_nnef.remodeler import Stage, SubnetSignature


def group_by_subnet(
    snapshots: T.List[SubnetSignature],
) -> dict[str, list[SubnetSignature]]:
    """Group signature snapshots by subnet name preserving order."""
    per: dict[str, list[SubnetSignature]] = {}
    for s in snapshots:
        per.setdefault(s.name, []).append(s)
    return per


def _sig_key(e: SubnetSignature) -> tuple:
    in_key = tuple(
        (i.name, tuple(map(str, i.shape)), i.dtype or "", tuple(i.notes or []))
        for i in e.inputs
    )
    out_key = tuple(
        (o.name, tuple(map(str, o.shape)), o.dtype or "", tuple(o.notes or []))
        for o in e.outputs
    )
    return (in_key, out_key)


def group_consecutive(
    entries: list[SubnetSignature],
) -> list[tuple[list[Stage], SubnetSignature]]:
    """Group consecutive entries with identical IO signatures."""
    groups: list[tuple[list[Stage], SubnetSignature]] = []
    for e in entries:
        k = _sig_key(e)
        if groups and _sig_key(groups[-1][1]) == k:
            groups[-1][0].append(e.stage)
        else:
            groups.append(([e.stage], e))
    return groups


def render_groups_plain(
    subnet_name: str,
    groups: list[tuple[list[Stage], SubnetSignature]],
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
    groups: list[tuple[list[Stage], SubnetSignature]],
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
                    or (ai.notes or []) != (bi.notes or [])
                ):
                    lines.append(f"    - {nm}: {ai.shape} -> {bi.shape}")
            elif ai and not bi:
                lines.append(f"    - {nm}: present -> removed")
            elif bi and not ai:
                lines.append(f"    - {nm}: absent -> present")
    return lines
