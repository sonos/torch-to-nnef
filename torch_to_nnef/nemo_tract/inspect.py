import logging
import typing as T
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch

from torch_to_nnef.nemo_tract.export import (
    iter_export_params_for_generic_nemo_asr_model,
)
from torch_to_nnef.remodeler import (
    IODescriptor,
    SubnetSignature,
)
from torch_to_nnef.remodeler import (
    Stage as InspectStage,
)
from torch_to_nnef.remodeler.inspect_utils import (
    group_by_subnet,
    group_consecutive,
    render_diffs_plain,
    render_groups_plain,
)
from torch_to_nnef.remodeler.rich_render import print_signatures_rich
from torch_to_nnef.remodeler.serialize import write_signatures_json
from torch_to_nnef.utils import INJECTED, T2NExtra, require_extra_decorator

LOGGER = logging.getLogger(__name__)


def _stage_order(stage: InspectStage) -> int:
    return {
        InspectStage.RAW: 0,
        InspectStage.COLLAPSED: 1,
        InspectStage.BOUND: 2,
        InspectStage.FINAL: 3,
    }[stage]


@dataclass(frozen=True)
class StageInputTransform:
    """Result of applying stage transforms to a single input."""

    skip: bool
    new_shape: T.List[T.Union[int, str]]
    remap: dict[int, str]
    notes: T.List[str]
    bind_flag: T.Optional[str]

    # No ordering logic here; ordering is defined on InspectStage


class InspectFormat(Enum):
    """Output formats for inspection results."""

    HUMAN = "human"
    JSON = "json"
    HUMAN_RICH = "human-rich"


AxisSymbolMap = T.Dict[int, str]


def _order_stage_in_place(entries: list[SubnetSignature]) -> None:
    entries.sort(key=lambda e: _stage_order(e.stage))


def _tensor_shape_with_symbols(
    t: object, symbols: T.Optional[AxisSymbolMap]
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


def _dtype_of(t: object) -> T.Optional[str]:
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


    Returns:
        A list of SubnetSignature snapshots in discovery order.
    """
    sigs: T.List[SubnetSignature] = []
    for ep in iter_export_params_for_generic_nemo_asr_model(
        asr_model,
        inference_target,
        skip_preprocessor=skip_preprocessor,
        split_joint_decoder=split_joint_decoder,
        float_dtype=float_dtype,
        remove_unused_inputs=True,
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
            # Handle tuple/list inputs by expanding into
            # name_0, name_1, ...
            if isinstance(t, (list, tuple)) and len(t) > 0:
                for k, tk in enumerate(t):
                    ename = f"{name}_{k}"
                    sym_map = (
                        (dyn_axes.get(ename) or {})
                        if isinstance(dyn_axes, dict)
                        else {}
                    )
                    shp = _tensor_shape_with_symbols(tk, sym_map)
                    dt = _dtype_of(tk)
                    notes: T.List[str] = []
                    inputs.append(
                        IODescriptor(
                            name=ename, shape=shp, dtype=dt, notes=notes
                        )
                    )
                continue

            sym_map = (
                (dyn_axes.get(name) or {}) if isinstance(dyn_axes, dict) else {}
            )
            shp = _tensor_shape_with_symbols(t, sym_map)
            dt = _dtype_of(t)
            notes: T.List[str] = []
            inputs.append(
                IODescriptor(name=name, shape=shp, dtype=dt, notes=notes)
            )

        # Outputs: list names only at this phase (no extra forward pass)
        outputs: T.List[IODescriptor] = [
            IODescriptor(name=nm, shape=[], dtype=None, notes=[])
            for nm in (ep.output_names or [])
        ]

        applied_flags: T.List[str] = []

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
    sigs: T.List[SubnetSignature],
    *,
    to_path: T.Optional[Path],
    diff: bool,
    model_label: T.Optional[str] = None,
) -> None:
    """Render signatures as human-readable text to stdout or a file."""
    # Use module-level helpers to keep complexity low

    lines: list[str] = []
    printed_header = False
    for subnet_name, entries in group_by_subnet(sigs).items():
        _order_stage_in_place(entries)
        groups = group_consecutive(entries)
        if model_label and not printed_header:
            lines.append(f"Model: {model_label}")
            printed_header = True
        lines.extend(render_groups_plain(subnet_name, groups))
        if diff and len(groups) >= 2:
            lines.extend(render_diffs_plain(groups))

    out_txt = "\n".join(lines) + ("\n" if lines else "")
    if to_path is None:
        print(out_txt, end="")
    else:
        to_path.parent.mkdir(parents=True, exist_ok=True)
        to_path.write_text(out_txt, encoding="utf8")


def _print_json(
    sigs: T.List[SubnetSignature],
    *,
    to_path: T.Optional[Path],
    model_label: T.Optional[str] = None,
) -> None:
    """Render signatures as JSON for tooling and CI."""
    write_signatures_json(sigs, to_path=to_path, model_label=model_label)

    # end loop


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="rich")
def _print_human_rich(
    sigs: T.List[SubnetSignature],
    *,
    to_path: T.Optional[Path],
    diff: bool,
    rich=INJECTED,
    model_label: T.Optional[str] = None,
) -> None:
    """Render signatures using Rich via remodeler (requires extra)."""
    print_signatures_rich(
        sigs, to_path=to_path, diff=diff, rich=rich, model_label=model_label
    )


def run_inspection(
    *,
    asr_model,
    inference_target,
    skip_preprocessor: bool,
    split_joint_decoder: bool,
    float_dtype: torch.dtype,
    only_subnets: T.Optional[T.Collection[str]],
    stages: T.Optional[T.List[InspectStage]],
    fmt: InspectFormat,
    to_path: T.Optional[Path],
    diff: bool,
    axis_registry=None,
    model_label: T.Optional[str] = None,
) -> None:
    """Run the inspection pipeline and print results.

    Args:
        asr_model: NeMo model instance.
        inference_target: Export target wrapper.
        skip_preprocessor: Whether to skip preprocessor subnet.
        split_joint_decoder: Whether to split joint/decoder.
        float_dtype: Preferred float dtype for examples.
        only_subnets: Optional subset filter.
        stages: Stages to display; defaults to FINAL.
        fmt: Output format (human or json).
        to_path: Optional file path to write output.
        diff: Whether to display diffs between unique signature groups.
        axis_registry: Optional registry to overlay symbolic axes for inputs.
        model_label: Optional model slug or local path for header/JSON.
    """
    chosen: list[InspectStage] = stages if stages else [InspectStage.FINAL]
    all_sigs = _collect_for_stages(
        asr_model=asr_model,
        inference_target=inference_target,
        chosen=chosen,
        skip_preprocessor=skip_preprocessor,
        split_joint_decoder=split_joint_decoder,
        float_dtype=float_dtype,
        only_subnets=only_subnets,
    )

    # Strict validation and overlay from shape-config, then transforms
    if axis_registry and getattr(axis_registry, "symbols_per_input", None):
        (
            qualified,
            q_ranks,
            q_shapes,
            subnets,
            bare_to_qualified,
        ) = _build_signature_maps(all_sigs)
        reg = axis_registry.symbols_per_input
        resolved = _resolve_config_keys(
            reg, qualified, subnets, bare_to_qualified
        )
        _validate_ranks(
            resolved,
            getattr(axis_registry, "rank_per_input", {}) or {},
            q_ranks,
            q_shapes,
        )
        all_sigs, q_to_axes = _overlay_symbols(all_sigs, resolved, reg)
        cfg_collapse, cfg_binds_src, batch_alias = _derive_cfg_transforms(
            axis_registry, resolved, qualified, q_to_axes
        )

        all_sigs = _apply_stage_transforms(
            all_sigs, cfg_collapse, cfg_binds_src, batch_alias
        )

        # Warnings when non-batch collapse is configured
        non_batch = []
        for q, syms in cfg_collapse.items():
            for s in syms:
                if s not in {"BATCH", "B"}:
                    non_batch.append((q, s))
        if non_batch:
            LOGGER.warning(
                "Non-batch collapse configured for %s; consider probing sizes "
                "with representative inputs to validate correctness.",
                ", ".join(f"{q}:{s}" for q, s in non_batch),
            )

    _emit_output(fmt, all_sigs, to_path, diff, model_label)


def _collect_for_stages(
    *,
    asr_model,
    inference_target,
    chosen: list[InspectStage],
    skip_preprocessor: bool,
    split_joint_decoder: bool,
    float_dtype: torch.dtype,
    only_subnets: T.Optional[T.Collection[str]],
) -> list[SubnetSignature]:
    LOGGER.info(
        "running inspection for stages: %s",
        ", ".join(st.value for st in chosen),
    )
    all_sigs: list[SubnetSignature] = []
    for st in chosen:
        snaps = _collect_signatures_for_stage(
            asr_model=asr_model,
            inference_target=inference_target,
            stage=st,
            skip_preprocessor=skip_preprocessor,
            split_joint_decoder=split_joint_decoder,
            float_dtype=float_dtype,
            only_subnets=only_subnets,
        )
        all_sigs.extend(snaps)
    return all_sigs


def _emit_output(
    fmt: InspectFormat,
    all_sigs: list[SubnetSignature],
    to_path: T.Optional[Path],
    diff: bool,
    model_label: T.Optional[str],
) -> None:
    if fmt == InspectFormat.HUMAN:
        _print_human(
            all_sigs, to_path=to_path, diff=diff, model_label=model_label
        )
    elif fmt == InspectFormat.HUMAN_RICH:
        _print_human_rich(
            all_sigs, to_path=to_path, diff=diff, model_label=model_label
        )
    else:
        _print_json(all_sigs, to_path=to_path, model_label=model_label)


def _apply_stage_transforms(
    all_sigs: list[SubnetSignature],
    cfg_collapse: dict[str, list[str]],
    cfg_binds_src: dict[str, tuple[str, str]],
    batch_alias: dict[str, str],
) -> list[SubnetSignature]:
    transformed: list[SubnetSignature] = []
    for ss in all_sigs:
        if ss.stage == InspectStage.RAW:
            transformed.append(ss)
            continue
        new_inputs: list[IODescriptor] = []
        new_axes: dict[str, dict[int, str]] = {}
        applied = list(ss.applied_flags)
        for i in ss.inputs:
            tr = _compute_input_transform(
                ss, i, cfg_collapse, cfg_binds_src, batch_alias
            )
            if tr.bind_flag:
                applied.append(tr.bind_flag)
            if tr.skip:
                continue
            new_axes[i.name] = tr.remap
            new_inputs.append(
                IODescriptor(
                    name=i.name,
                    shape=tr.new_shape,
                    dtype=i.dtype,
                    notes=tr.notes,
                )
            )
        transformed.append(
            SubnetSignature(
                name=ss.name,
                stage=ss.stage,
                inputs=new_inputs,
                outputs=ss.outputs,
                applied_flags=applied,
                symbol_axes=new_axes if new_axes else ss.symbol_axes,
            )
        )
    return transformed


def _compute_input_transform(
    ss: SubnetSignature,
    i: IODescriptor,
    cfg_collapse: dict[str, list[str]],
    cfg_binds_src: dict[str, tuple[str, str]],
    batch_alias: dict[str, str],
) -> StageInputTransform:
    qname = f"{ss.name}.{i.name}"
    sym_map = ss.symbol_axes.get(i.name, {}) or {}
    remove_syms: set[str] = set(cfg_collapse.get(qname, []))
    known_syms = {str(sym).upper() for sym in sym_map.values()}
    extra = [s for s in remove_syms if s not in known_syms]
    if remove_syms and extra and ss.stage != InspectStage.RAW:
        raise ValueError(
            f"Cannot collapse non-dynamic dims for {qname} at "
            f"stage {ss.stage.value}: requested {sorted(remove_syms)} "
            f"but dynamic symbols are {sorted(known_syms)}"
        )
    drop_idx = sorted(
        [
            ax
            for ax, sym in (sym_map.items() if sym_map else [])
            if str(sym).upper() in remove_syms
        ]
    )
    is_bound_here = (
        ss.stage in (InspectStage.BOUND, InspectStage.FINAL)
        and qname in cfg_binds_src
    )
    if is_bound_here:
        src_q, src_sym = cfg_binds_src[qname]
        bind_flag = f"--bind {i.name}={src_q}.{src_sym}"
        if (
            i.shape
            and len([d for j, d in enumerate(i.shape) if j not in drop_idx]) > 0
        ):
            msg = (
                f"Binding requires scalar input for {qname}; "
                f"after collapse kept rank>0: {i.shape}"
            )
            raise ValueError(msg)
        return StageInputTransform(
            skip=True,
            new_shape=[],
            remap={},
            notes=list(i.notes),
            bind_flag=bind_flag,
        )
    if i.shape:
        new_shape = [d for j, d in enumerate(i.shape) if j not in drop_idx]
    else:
        new_shape = list(i.shape)
    if sym_map:
        remap: dict[int, str] = {}
        shift = 0
        for ax in sorted(sym_map.keys()):
            if ax in drop_idx:
                shift += 1
                continue
            remap[ax - shift] = sym_map[ax]
    else:
        remap = {}
    q_alias = batch_alias.get(qname)
    if q_alias:
        for pos, sym in list(remap.items()):
            if str(sym).upper() == "BATCH":
                remap[pos] = q_alias
                if (
                    pos < len(new_shape)
                    and str(new_shape[pos]).upper() == "BATCH"
                ):
                    new_shape[pos] = q_alias
    notes = list(i.notes)
    if remove_syms:
        notes.append("collapsed:" + ",".join(sorted(remove_syms)))
    return StageInputTransform(
        skip=False,
        new_shape=new_shape,
        remap=remap,
        notes=notes,
        bind_flag=None,
    )


def _build_signature_maps(all_sigs: list[SubnetSignature]):
    qualified: set[str] = set()
    q_ranks: dict[str, int] = {}
    q_shapes: dict[str, T.List[T.Union[int, str]]] = {}
    subnets: set[str] = set()
    bare_to_qualified: dict[str, list[str]] = {}
    for ss in all_sigs:
        subnets.add(ss.name)
        for i in ss.inputs:
            q = f"{ss.name}.{i.name}"
            qualified.add(q)
            if i.shape:
                q_ranks.setdefault(q, len(i.shape))
                q_shapes.setdefault(q, list(i.shape))
            bare_to_qualified.setdefault(i.name, []).append(q)
    return qualified, q_ranks, q_shapes, subnets, bare_to_qualified


def _resolve_config_keys(
    reg: dict, qualified: set[str], subnets: set[str], bare_to_qualified: dict
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key in reg:
        if "." in key:
            if key not in qualified:
                subnet, _, _ = key.partition(".")
                if subnet not in subnets:
                    raise ValueError(
                        "Unknown subnet in --shape-config: '"
                        + subnet
                        + "'. Known subnets: "
                        + ", ".join(sorted(subnets))
                    )
                inputs_in_subnet = sorted(
                    {
                        q.split(".", 1)[1]
                        for q in qualified
                        if q.startswith(subnet + ".")
                    }
                )
                raise ValueError(
                    "Unknown qualified input '"
                    + key
                    + "' in --shape-config. Known inputs for subnet '"
                    + subnet
                    + "': "
                    + ", ".join(inputs_in_subnet)
                )
            resolved[key] = key
        else:
            qlist = bare_to_qualified.get(key, [])
            if not qlist:
                known_bare = sorted(set(bare_to_qualified.keys()))
                raise ValueError(
                    "Unknown input name in --shape-config: '"
                    + key
                    + "'. Known input names: "
                    + ", ".join(known_bare)
                )
            if len(qlist) > 1:
                raise ValueError(
                    "Ambiguous input name in --shape-config '"
                    + key
                    + "' appears in multiple subnets: "
                    + ", ".join(sorted(qlist))
                    + ". Qualify as 'subnet.input'"
                )
            resolved[key] = qlist[0]
    return resolved


def _validate_ranks(
    resolved: dict[str, str],
    cfg_ranks: dict[str, int],
    q_ranks: dict[str, int],
    q_shapes: dict[str, T.List[T.Union[int, str]]],
) -> None:
    mismatches: list[str] = []
    for key, q in resolved.items():
        if (
            (q in q_ranks)
            and (key in cfg_ranks)
            and (int(cfg_ranks[key]) != int(q_ranks[q]))
        ):
            hint = ", ".join(str(d) for d in q_shapes.get(q, []))
            hint_shape = f"[{hint}]"
            msg = (
                f"{key}: config rank {int(cfg_ranks[key])} vs discovered "
                f"rank {int(q_ranks[q])} {hint_shape}"
            )
            mismatches.append(msg)
    if mismatches:
        raise ValueError(
            "Rank mismatch in --shape-config: " + "; ".join(mismatches)
        )


def _overlay_symbols(
    all_sigs: list[SubnetSignature], resolved: dict[str, str], reg: dict
) -> tuple[list[SubnetSignature], dict[str, dict[int, str]]]:
    q_to_axes: dict[str, dict[int, str]] = {}
    for key, amap in reg.items():
        q = resolved.get(key)
        if q and isinstance(amap, dict):
            q_to_axes[q] = {int(k): str(v) for k, v in amap.items()}

    merged_all: list[SubnetSignature] = []
    for ss in all_sigs:
        merged_axes: dict[str, dict[int, str]] = dict(ss.symbol_axes)
        for i in ss.inputs:
            q = f"{ss.name}.{i.name}"
            if q in q_to_axes:
                merged_axes[i.name] = dict(q_to_axes[q])
        new_inputs: list[IODescriptor] = []
        for i in ss.inputs:
            amap = merged_axes.get(i.name, {}) or {}
            if i.shape:
                new_shape: list[T.Union[int, str]] = []
                for ax, d in enumerate(i.shape):
                    new_shape.append(str(amap[ax]) if ax in amap else d)
            else:
                new_shape = list(i.shape)
            new_inputs.append(
                IODescriptor(
                    name=i.name, shape=new_shape, dtype=i.dtype, notes=i.notes
                )
            )
        merged_all.append(
            SubnetSignature(
                name=ss.name,
                stage=ss.stage,
                inputs=new_inputs,
                outputs=ss.outputs,
                applied_flags=ss.applied_flags,
                symbol_axes=merged_axes,
            )
        )
    return merged_all, q_to_axes


def _derive_cfg_transforms(
    axis_registry,
    resolved: dict[str, str],
    qualified: set[str],
    q_to_axes: dict[str, dict[int, str]],
) -> tuple[dict[str, list[str]], dict[str, tuple[str, str]], dict[str, str]]:
    cfg_collapse: dict[str, list[str]] = {}
    raw_collapse = getattr(axis_registry, "input_collapse_dims", {}) or {}
    for key, seq in raw_collapse.items():
        q = resolved.get(key, key)
        syms = [str(s).strip().upper() for s in (seq or [])]
        cfg_collapse[q] = syms

    cfg_binds_src: dict[str, tuple[str, str]] = {}
    raw_binds = getattr(axis_registry, "bind_to_dim", {}) or {}
    for key, val in raw_binds.items():
        qkey = resolved.get(key, key)
        if not isinstance(val, str) or "." not in val:
            raise ValueError(
                "Invalid bind_scalar_to_dim_size for '"
                + key
                + "': expected 'subnet.input.SYMBOL'"
            )
        src, _, sym = val.rpartition(".")
        if src not in qualified:
            raise ValueError(
                "Unknown source input in bind '"
                + val
                + "'. Known qualified inputs: "
                + ", ".join(sorted(qualified))
            )
        cfg_binds_src[qkey] = (src, str(sym).strip().upper())

    batch_alias: dict[str, str] = {}
    for q, amap in q_to_axes.items():
        for _, sym in amap.items():
            s = str(sym).upper()
            if s != "BATCH" and (s.endswith("__BATCH") or "BATCH" in s):
                batch_alias[q] = str(sym)
    return cfg_collapse, cfg_binds_src, batch_alias


def collect_signatures(
    *,
    asr_model,
    inference_target,
    stage: InspectStage = InspectStage.RAW,
    skip_preprocessor: bool = False,
    split_joint_decoder: bool = False,
    float_dtype: T.Optional[torch.dtype] = None,
    only_subnets: T.Optional[T.Collection[str]] = None,
) -> T.List[SubnetSignature]:
    """Collect per-subnet signatures without printing.

    Args:
        asr_model: NeMo model instance.
        inference_target: Export target wrapper.
        stage: Stage to collect (RAW, COLLAPSED, BOUND, FINAL semantics).
        skip_preprocessor: Whether to skip preprocessor subnet.
        split_joint_decoder: Whether to split joint/decoder.
        float_dtype: Preferred float dtype for examples.
        only_subnets: Optional subset filter.

    Returns:
        List of SubnetSignature snapshots.
    """
    return _collect_signatures_for_stage(
        asr_model=asr_model,
        inference_target=inference_target,
        stage=stage,
        skip_preprocessor=skip_preprocessor,
        split_joint_decoder=split_joint_decoder,
        float_dtype=float_dtype or torch.float32,
        only_subnets=only_subnets,
    )
