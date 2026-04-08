import json
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from torch_to_nnef.exceptions import (
    T2NErrorInvalidArgument,
    T2NErrorNotFoundFile,
)
from torch_to_nnef.remodeler.schema import (
    INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE,
    INPUT_FIELD_COLLAPSE_DIMS,
    INPUT_FIELD_EVAL_SYMBOLS,
    INPUT_FIELD_ORIGINAL_SHAPE,
    OUTPUT_FIELD_COLLAPSE_DIMS,
    SHAPE_KEY_EXTENSIONS,
    SHAPE_KEY_INPUTS,
    SHAPE_KEY_OUTPUTS,
    SHAPE_KEY_OUTPUTS_KEEP,
    SHAPE_KEY_RENAMED,
)

AxisSymbolMap = T.Dict[int, str]


@dataclass
class AxisSymbolRegistry:
    """Registry mapping input names to symbolic axis annotations.

    Attributes:
        symbols_per_input: map from fully qualified input name
            (e.g., "encoder.audio_signal") to axis-index→symbol map.
    """

    symbols_per_input: T.Dict[str, AxisSymbolMap]
    rank_per_input: T.Dict[str, int]
    # Optional binding: qualified input -> qualified source.axis (scalar-only)
    bind_to_dim: T.Dict[str, str]
    # Collapse dims (dynamic-only), per input only
    input_collapse_dims: T.Dict[str, T.List[str]]
    # Optional per-subnet renames:
    # subnet -> { target_symbol: [source_symbols...] }
    renamed_symbols_per_subnet: T.Dict[str, T.Dict[str, T.List[str]]]
    # Optional per-subnet output selection: keep only these outputs
    outputs_keep_per_subnet: T.Dict[str, T.List[str]]
    # Per-output collapse dims: qualified output name -> list of axis indices
    # to squeeze (e.g., {"preprocessor.processed_signal": [0]})
    output_collapse_dims: T.Dict[str, T.List[int]] = field(default_factory=dict)
    # Optional: pin dynamic symbols to concrete values in test_input tensors
    # qualified input -> { SYMBOL: int_value }
    eval_symbols_per_input: T.Dict[str, T.Dict[str, int]] = field(
        default_factory=dict
    )
    # Optional: discovered shapes with mixed ints/symbols per qualified input
    # (used for template serialization; not required when loading config)
    original_shape_per_input: T.Dict[str, T.List[T.Union[int, str]]] = field(
        default_factory=dict
    )
    # Optional: per-subnet custom extensions (e.g., tract_assert constraints)
    # subnet -> list of extension strings
    extensions_per_subnet: T.Dict[str, T.List[str]] = field(
        default_factory=dict
    )

    @staticmethod
    def empty() -> "AxisSymbolRegistry":
        return AxisSymbolRegistry(
            symbols_per_input={},
            rank_per_input={},
            bind_to_dim={},
            input_collapse_dims={},
            renamed_symbols_per_subnet={},
            outputs_keep_per_subnet={},
            output_collapse_dims={},
            eval_symbols_per_input={},
            original_shape_per_input={},
            extensions_per_subnet={},
        )


def _list_to_axis_map(
    shape_list: T.Sequence[T.Union[str, int]],
) -> AxisSymbolMap:
    """Convert dims into an axis-index→symbol map (uppercased)."""
    return {
        idx: str(v).strip().upper()
        for idx, v in enumerate(shape_list)
        if isinstance(v, str) and v
    }


def _validate_key(key: str) -> None:
    """Validate a non-empty mapping key.

    Args:
        key: Key string from the config.
    """
    if not isinstance(key, str) or not key:
        raise T2NErrorInvalidArgument(
            f"Invalid key in shape-config (expected non-empty string): {key!r}"
        )


def _validate_and_record(
    key: str,
    shape_val: T.Sequence[T.Union[str, int]],
    symbols: T.Dict[str, AxisSymbolMap],
    ranks: T.Dict[str, int],
    orig_shapes: T.Dict[str, T.List[T.Union[int, str]]],
) -> None:
    """Validate a shape list and record into outputs.

    Args:
        key: Qualified input key.
        shape_val: Sequence of dim entries.
        symbols: Output map to axis symbols.
        ranks: Output map to ranks.
        orig_shapes: Output map to original dims (ints/strings).
    """
    _validate_key(key)
    for i, v in enumerate(shape_val):
        if not isinstance(v, (str, int)):
            raise T2NErrorInvalidArgument(
                "Invalid dim at "
                f"{key}[{i}]: {type(v).__name__}; expected str or int"
            )
        if isinstance(v, str) and not v.strip():
            raise T2NErrorInvalidArgument(
                f"Empty dim symbol at {key}[{i}] is not allowed"
            )
    symbols[key] = _list_to_axis_map(shape_val)
    ranks[key] = len(shape_val)
    orig_shapes[key] = [
        int(v) if isinstance(v, int) else str(v) for v in shape_val
    ]


def _normalize_syms(seq: T.Sequence[str]) -> T.List[str]:
    """Uppercase a list of symbols without applying aliases."""
    return [str(s).strip().upper() for s in seq]


def _parse_renamed_symbols(
    top_key: str, raw_val: object
) -> dict[str, list[str]]:
    """Parse an optional `renamed_symbols` block for a subnet.

    Args:
        top_key: Subnet name.
        raw_val: Raw value from the config.

    Returns:
        Mapping of TARGET -> [SOURCES...], all uppercased.
    """
    if raw_val is None:
        return {}
    mapping: dict[str, list[str]] = {}
    if isinstance(raw_val, dict):
        items_iter = raw_val.items()
    elif isinstance(raw_val, (list, tuple)):
        items_iter = []
        for elem in raw_val:
            if isinstance(elem, dict) and len(elem) == 1:
                items_iter += list(elem.items())
            else:
                msg = (
                    f"Invalid {SHAPE_KEY_RENAMED} entry for subnet '{top_key}'"
                )
                raise T2NErrorInvalidArgument(msg)
    else:
        msg = (
            f"{SHAPE_KEY_RENAMED} for subnet '{top_key}' must be mapping or "
            "list"
        )
        raise T2NErrorInvalidArgument(msg)
    for tgt, srcs in items_iter:  # type: ignore[attr-defined]
        if not isinstance(tgt, str) or not isinstance(srcs, (list, tuple)):
            msg = (
                f"Invalid {SHAPE_KEY_RENAMED} target/sources for subnet "
                f"'{top_key}'"
            )
            raise T2NErrorInvalidArgument(msg)
        tnorm = str(tgt).strip().upper()
        snorm = [str(s).strip().upper() for s in srcs]
        if tnorm in snorm:
            msg = (
                f"renamed_symbols for subnet {top_key}: target '{tnorm}' "
                "cannot include itself"
            )
            raise T2NErrorInvalidArgument(msg)
        mapping[tnorm] = snorm
    return mapping


def _nts_handle_tuple_group(
    top_key: str,
    inp_name: str,
    group: dict,
    symbols: T.Dict[str, AxisSymbolMap],
    ranks: T.Dict[str, int],
    binds: T.Dict[str, str],
    input_dims: T.Dict[str, T.List[str]],
    orig_shapes: T.Dict[str, T.List[T.Union[int, str]]],
    eval_syms: T.Dict[str, T.Dict[str, int]],
) -> None:
    """Handle tuple-group style input mapping (index -> sub-mapping)."""
    for idx_str, inner in group.items():
        qname = f"{top_key}.{inp_name}_{idx_str}"
        if not isinstance(inner, dict):
            raise T2NErrorInvalidArgument(
                f"Invalid tuple entry for '{qname}': expected mapping"
            )
        if INPUT_FIELD_ORIGINAL_SHAPE in inner:
            oshp = inner.get(INPUT_FIELD_ORIGINAL_SHAPE)
            if not isinstance(oshp, (list, tuple)):
                raise T2NErrorInvalidArgument(
                    f"Invalid original_shape for '{qname}'"
                )
            _validate_and_record(qname, oshp, symbols, ranks, orig_shapes)
        if INPUT_FIELD_COLLAPSE_DIMS in inner:
            cdv = inner.get(INPUT_FIELD_COLLAPSE_DIMS)
            if not isinstance(cdv, (list, tuple)):
                raise T2NErrorInvalidArgument(
                    f"Invalid collapse_dims for '{qname}'"
                )
            input_dims[qname] = _normalize_syms([str(x) for x in cdv])
        if INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE in inner:
            b = inner.get(INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE)
            if isinstance(b, str) and b:
                binds[qname] = b
        if INPUT_FIELD_EVAL_SYMBOLS in inner:
            es = inner.get(INPUT_FIELD_EVAL_SYMBOLS)
            if not isinstance(es, dict):
                raise T2NErrorInvalidArgument(
                    f"eval_symbols for '{qname}' must be a mapping "
                    "{{SYMBOL: int_value}}"
                )
            eval_syms[qname] = {
                str(k).strip().upper(): int(v) for k, v in es.items()
            }


def _nts_handle_single_mapping(
    top_key: str,
    inp_name: str,
    mapping: dict,
    symbols: T.Dict[str, AxisSymbolMap],
    ranks: T.Dict[str, int],
    binds: T.Dict[str, str],
    input_dims: T.Dict[str, T.List[str]],
    orig_shapes: T.Dict[str, T.List[T.Union[int, str]]],
    eval_syms: T.Dict[str, T.Dict[str, int]],
) -> None:
    """Handle single input mapping with optional shape/collapse/bind fields."""
    qbase = f"{top_key}.{inp_name}"
    if INPUT_FIELD_ORIGINAL_SHAPE in mapping:
        oshp = mapping.get(INPUT_FIELD_ORIGINAL_SHAPE)
        if not isinstance(oshp, (list, tuple)):
            raise T2NErrorInvalidArgument(
                f"Invalid original_shape for '{qbase}'"
            )
        _validate_and_record(qbase, oshp, symbols, ranks, orig_shapes)
    if INPUT_FIELD_COLLAPSE_DIMS in mapping:
        cdv = mapping.get(INPUT_FIELD_COLLAPSE_DIMS)
        if not isinstance(cdv, (list, tuple)):
            raise T2NErrorInvalidArgument(
                f"Invalid collapse_dims for '{qbase}'"
            )
        input_dims[qbase] = _normalize_syms([str(x) for x in cdv])
    if INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE in mapping:
        b = mapping.get(INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE)
        if isinstance(b, str) and b:
            binds[qbase] = b
    if INPUT_FIELD_EVAL_SYMBOLS in mapping:
        es = mapping.get(INPUT_FIELD_EVAL_SYMBOLS)
        if not isinstance(es, dict):
            raise T2NErrorInvalidArgument(
                f"eval_symbols for '{qbase}' must be a mapping "
                "{{SYMBOL: int_value}}"
            )
        eval_syms[qbase] = {
            str(k).strip().upper(): int(v) for k, v in es.items()
        }


def _parse_top_level(
    raw: dict,
) -> tuple[
    T.Dict[str, AxisSymbolMap],
    T.Dict[str, int],
    T.Dict[str, str],
    T.Dict[str, T.List[str]],
    T.Dict[str, T.List[str]],
    T.Dict[str, T.List[int]],
    T.Dict[str, T.List[T.Union[int, str]]],
    T.Dict[str, T.Dict[str, int]],
    T.Dict[str, T.List[str]],
]:
    """Parse top-level entries into symbols/ranks/binds/input_dims."""
    symbols: T.Dict[str, AxisSymbolMap] = {}
    ranks: T.Dict[str, int] = {}
    binds: T.Dict[str, str] = {}
    input_dims: T.Dict[str, T.List[str]] = {}
    outputs_keep: T.Dict[str, T.List[str]] = {}
    output_collapse: T.Dict[str, T.List[int]] = {}
    orig_shapes: T.Dict[str, T.List[T.Union[int, str]]] = {}
    eval_syms: T.Dict[str, T.Dict[str, int]] = {}
    extensions: T.Dict[str, T.List[str]] = {}
    for top_key, val in dict(raw or {}).items():
        _validate_key(top_key)
        if isinstance(val, dict):
            _ = _parse_renamed_symbols(top_key, val.get(SHAPE_KEY_RENAMED))
            _parse_nested_subnet(
                top_key,
                val,
                symbols,
                ranks,
                binds,
                input_dims,
                output_collapse,
                orig_shapes,
                eval_syms,
            )
            if SHAPE_KEY_OUTPUTS_KEEP in val:
                oks = val.get(SHAPE_KEY_OUTPUTS_KEEP)
                if not isinstance(oks, (list, tuple)) or not all(
                    isinstance(x, str) and x for x in oks
                ):
                    msg = (
                        f"Invalid outputs_keep for subnet '{top_key}' "
                        "(list[str] expected)"
                    )
                    raise T2NErrorInvalidArgument(msg)
                outputs_keep[top_key] = [str(x) for x in oks]
            if SHAPE_KEY_EXTENSIONS in val:
                exts = val.get(SHAPE_KEY_EXTENSIONS)
                if not isinstance(exts, (list, tuple)) or not all(
                    isinstance(x, str) and x for x in exts
                ):
                    msg = (
                        f"Invalid extensions for subnet '{top_key}' "
                        "(list[str] expected)"
                    )
                    raise T2NErrorInvalidArgument(msg)
                extensions[top_key] = [str(x) for x in exts]
        elif isinstance(val, (list, tuple)):
            _validate_and_record(top_key, val, symbols, ranks, orig_shapes)
        else:
            msg = (
                f"Invalid value for '{top_key}': expected list/tuple or nested "
                "mapping"
            )
            raise T2NErrorInvalidArgument(msg)
    return (
        symbols,
        ranks,
        binds,
        input_dims,
        outputs_keep,
        output_collapse,
        orig_shapes,
        eval_syms,
        extensions,
    )


def _build_renamed_map(raw: dict) -> dict[str, dict[str, list[str]]]:
    """Build per-subnet renamed_symbols mapping from raw config."""
    out: dict[str, dict[str, list[str]]] = {}
    for top_key, val in dict(raw or {}).items():
        if isinstance(val, dict) and SHAPE_KEY_RENAMED in val:
            mapping = _parse_renamed_symbols(
                top_key, val.get(SHAPE_KEY_RENAMED)
            )
            if mapping:
                out[top_key] = mapping
    return out


def _parse_output_collapse_section(
    top_key: str,
    outputs_section: dict,
    output_collapse: T.Dict[str, T.List[int]],
) -> None:
    """Parse the ``outputs`` section of a subnet config.

    Each entry maps an output name to ``{collapse_dims: [axis_indices]}``.
    """
    for out_name, out_cfg in outputs_section.items():
        qname = f"{top_key}.{out_name}"
        if not isinstance(out_cfg, dict):
            raise T2NErrorInvalidArgument(
                f"Invalid output config for '{qname}': expected mapping"
            )
        if OUTPUT_FIELD_COLLAPSE_DIMS in out_cfg:
            cdv = out_cfg[OUTPUT_FIELD_COLLAPSE_DIMS]
            if not isinstance(cdv, (list, tuple)) or not all(
                isinstance(x, int) for x in cdv
            ):
                raise T2NErrorInvalidArgument(
                    f"output collapse_dims for '{qname}' must be a list "
                    "of int axis indices"
                )
            output_collapse[qname] = list(cdv)
        unknown_keys = set(out_cfg.keys()) - {OUTPUT_FIELD_COLLAPSE_DIMS}
        if unknown_keys:
            raise T2NErrorInvalidArgument(
                f"Unknown keys in output config for '{qname}': "
                + ", ".join(sorted(unknown_keys))
            )


def _parse_nested_subnet(
    top_key: str,
    val: dict,
    symbols: T.Dict[str, AxisSymbolMap],
    ranks: T.Dict[str, int],
    binds: T.Dict[str, str],
    input_dims: T.Dict[str, T.List[str]],
    output_collapse: T.Dict[str, T.List[int]],
    orig_shapes: T.Dict[str, T.List[T.Union[int, str]]],
    eval_syms: T.Dict[str, T.Dict[str, int]],
) -> None:
    """Parse a nested subnet mapping into outputs.

    Args:
        top_key: Subnet name.
        val: Mapping for the subnet.
        symbols: Output symbols map.
        ranks: Output ranks map.
        binds: Output binds mapping.
        input_dims: Output collapse-dims mapping.
        output_collapse: Output per-output collapse axes.
        orig_shapes: Output map to original dims (ints/strings).
        eval_syms: Output eval-symbols mapping.
    """
    if INPUT_FIELD_COLLAPSE_DIMS in val:
        raise T2NErrorInvalidArgument(
            f"Do not set collapse_dims at subnet '{top_key}'. "
            "Define per-input collapse_dims instead."
        )

    # Require new-style nested inputs mapping under key 'inputs'.
    input_section = val.get(SHAPE_KEY_INPUTS) if isinstance(val, dict) else None
    if not isinstance(input_section, dict):
        raise T2NErrorInvalidArgument(
            "Each subnet must declare an 'inputs' mapping. Flat per-input "
            "keys at the subnet level are no longer supported."
        )
    # Reject unknown top-level keys besides the allowed ones
    allowed = {
        SHAPE_KEY_INPUTS,
        SHAPE_KEY_OUTPUTS,
        SHAPE_KEY_RENAMED,
        SHAPE_KEY_OUTPUTS_KEEP,
        SHAPE_KEY_EXTENSIONS,
    }
    unknown = {k for k in val if k not in allowed}
    if unknown:
        raise T2NErrorInvalidArgument(
            "Unknown keys in subnet config: " + ", ".join(sorted(unknown))
        )

    # Parse outputs section (per-output collapse_dims)
    outputs_section = val.get(SHAPE_KEY_OUTPUTS)
    if outputs_section is not None:
        if not isinstance(outputs_section, dict):
            raise T2NErrorInvalidArgument(
                f"'outputs' for subnet '{top_key}' must be a mapping"
            )
        _parse_output_collapse_section(
            top_key,
            outputs_section,
            output_collapse,
        )

    items = input_section.items()

    for inp_name, shape in items:
        if isinstance(shape, dict):
            is_tuple_group = "original_shape" not in shape and all(
                isinstance(k, str) and k.isdigit() for k in shape
            )
            if is_tuple_group:
                _nts_handle_tuple_group(
                    top_key,
                    inp_name,
                    shape,
                    symbols,
                    ranks,
                    binds,
                    input_dims,
                    orig_shapes,
                    eval_syms,
                )
            else:
                _nts_handle_single_mapping(
                    top_key,
                    inp_name,
                    shape,
                    symbols,
                    ranks,
                    binds,
                    input_dims,
                    orig_shapes,
                    eval_syms,
                )
        elif isinstance(shape, (list, tuple)):
            _validate_and_record(
                f"{top_key}.{inp_name}", shape, symbols, ranks, orig_shapes
            )
        else:
            raise T2NErrorInvalidArgument(
                f"Invalid value for '{top_key}.{inp_name}': "
                "expected list/tuple or mapping"
            )


def load_axis_symbol_registry(config_path: Path) -> AxisSymbolRegistry:
    """Load a YAML/JSON shape config into an AxisSymbolRegistry.

    The expected structure is a mapping of input-name → list of dims, e.g.:
        encoder.audio_signal: [B, 128, S]
        encoder.length: [B]
        joiner.encoder_outputs: [B, 1024, R]
        joiner.decoder_outputs: [B, 640, U]
    """
    if not config_path.exists() or not config_path.is_file():
        raise T2NErrorNotFoundFile(
            "--shape-config path does not exist or is not a file: "
            f"{config_path}"
        )
    text = config_path.read_text(encoding="utf8")
    if config_path.suffix.lower() in (".yml", ".yaml"):
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text or "{}")
    # Validate structure upfront for clear, early feedback
    if not isinstance(raw, dict):
        raise T2NErrorInvalidArgument(
            "shape-config must be a mapping (optionally nested) of "
            "input-name -> list of dims"
        )
    # Delegate detailed parsing to helpers to keep complexity low
    (
        symbols,
        ranks,
        binds,
        input_dims,
        outputs_keep,
        output_collapse,
        orig_shapes,
        eval_syms,
        extensions,
    ) = _parse_top_level(raw)
    if not symbols:
        raise T2NErrorInvalidArgument(
            "shape-config did not define any input shapes"
        )
    renamed_per_subnet = _build_renamed_map(raw)

    return AxisSymbolRegistry(
        symbols_per_input=symbols,
        rank_per_input=ranks,
        bind_to_dim=binds,
        input_collapse_dims=input_dims,
        renamed_symbols_per_subnet=renamed_per_subnet,
        outputs_keep_per_subnet=outputs_keep,
        output_collapse_dims=output_collapse,
        eval_symbols_per_input=eval_syms,
        original_shape_per_input=orig_shapes,
        extensions_per_subnet=extensions,
    )
