import json
import typing as T
from dataclasses import dataclass
from pathlib import Path

import yaml

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

    @staticmethod
    def empty() -> "AxisSymbolRegistry":
        return AxisSymbolRegistry(
            symbols_per_input={},
            rank_per_input={},
            bind_to_dim={},
            input_collapse_dims={},
            renamed_symbols_per_subnet={},
            outputs_keep_per_subnet={},
        )


def _list_to_axis_map(
    shape_list: T.Sequence[T.Union[str, int]],
) -> AxisSymbolMap:
    axis: AxisSymbolMap = {}
    for idx, v in enumerate(shape_list):
        if isinstance(v, str) and v:
            # Normalize to canonical forms: batch synonyms -> BATCH; else UPPER
            s = v.strip()
            s_lower = s.lower()
            if s_lower in ("b", "batch"):
                axis[idx] = "BATCH"
            else:
                axis[idx] = s.upper()
    return axis


def _validate_key(key: str) -> None:
    """Validate a non-empty mapping key.

    Args:
        key: Key string from the config.
    """
    if not isinstance(key, str) or not key:
        raise ValueError(
            f"Invalid key in shape-config (expected non-empty string): {key!r}"
        )


def _validate_and_record(
    key: str,
    shape_val: T.Sequence[T.Union[str, int]],
    symbols: T.Dict[str, AxisSymbolMap],
    ranks: T.Dict[str, int],
) -> None:
    """Validate a shape list and record into outputs.

    Args:
        key: Qualified input key.
        shape_val: Sequence of dim entries.
        symbols: Output map to axis symbols.
        ranks: Output map to ranks.
    """
    _validate_key(key)
    for i, v in enumerate(shape_val):
        if not isinstance(v, (str, int)):
            raise ValueError(
                "Invalid dim at "
                f"{key}[{i}]: {type(v).__name__}; expected str or int"
            )
        if isinstance(v, str) and not v.strip():
            raise ValueError(f"Empty dim symbol at {key}[{i}] is not allowed")
    symbols[key] = _list_to_axis_map(shape_val)
    ranks[key] = len(shape_val)


def _normalize_syms(seq: T.Sequence[str]) -> T.List[str]:
    """Normalize a list of symbol strings to canonical uppercase tokens."""
    out: T.List[str] = []
    for s in seq:
        ss = s.strip()
        if ss.lower() in ("b", "batch"):
            out.append("BATCH")
        else:
            out.append(ss.upper())
    return out


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
                msg = f"Invalid renamed_symbols entry for subnet '{top_key}'"
                raise ValueError(msg)
    else:
        msg = (
            "renamed_symbols for subnet '"
            + top_key
            + "' must be mapping or list"
        )
        raise ValueError(msg)
    for tgt, srcs in items_iter:  # type: ignore[attr-defined]
        if not isinstance(tgt, str) or not isinstance(srcs, (list, tuple)):
            msg = (
                f"Invalid renamed_symbols target/sources for subnet '{top_key}'"
            )
            raise ValueError(msg)
        tnorm = str(tgt).strip().upper()
        snorm = [str(s).strip().upper() for s in srcs]
        if tnorm in snorm:
            msg = (
                "renamed_symbols for subnet "
                + top_key
                + ": target '"
                + tnorm
                + "' cannot include itself"
            )
            raise ValueError(msg)
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
) -> None:
    """Handle tuple-group style input mapping (index -> sub-mapping)."""
    for idx_str, inner in group.items():
        qname = f"{top_key}.{inp_name}_{idx_str}"
        if not isinstance(inner, dict):
            raise ValueError(
                f"Invalid tuple entry for '{qname}': expected mapping"
            )
        if "original_shape" in inner:
            oshp = inner.get("original_shape")
            if not isinstance(oshp, (list, tuple)):
                raise ValueError(f"Invalid original_shape for '{qname}'")
            _validate_and_record(qname, oshp, symbols, ranks)
        if "collapse_dims" in inner:
            cdv = inner.get("collapse_dims")
            if not isinstance(cdv, (list, tuple)):
                raise ValueError(f"Invalid collapse_dims for '{qname}'")
            input_dims[qname] = _normalize_syms([str(x) for x in cdv])
        b = None
        if "bind_scalar_to_dim_size" in inner:
            b = inner.get("bind_scalar_to_dim_size")
        elif "bind_to_dim" in inner:
            b = inner.get("bind_to_dim")
        if isinstance(b, str) and b:
            binds[qname] = b


def _nts_handle_single_mapping(
    top_key: str,
    inp_name: str,
    mapping: dict,
    symbols: T.Dict[str, AxisSymbolMap],
    ranks: T.Dict[str, int],
    binds: T.Dict[str, str],
    input_dims: T.Dict[str, T.List[str]],
) -> None:
    """Handle single input mapping with optional shape/collapse/bind fields."""
    qbase = f"{top_key}.{inp_name}"
    if "original_shape" in mapping:
        oshp = mapping.get("original_shape")
        if not isinstance(oshp, (list, tuple)):
            raise ValueError(f"Invalid original_shape for '{qbase}'")
        _validate_and_record(qbase, oshp, symbols, ranks)
    if "collapse_dims" in mapping:
        cdv = mapping.get("collapse_dims")
        if not isinstance(cdv, (list, tuple)):
            raise ValueError(f"Invalid collapse_dims for '{qbase}'")
        input_dims[qbase] = _normalize_syms([str(x) for x in cdv])
    b = None
    if "bind_scalar_to_dim_size" in mapping:
        b = mapping.get("bind_scalar_to_dim_size")
    elif "bind_to_dim" in mapping:
        b = mapping.get("bind_to_dim")
    if isinstance(b, str) and b:
        binds[qbase] = b


def _parse_top_level(
    raw: dict,
) -> tuple[
    T.Dict[str, AxisSymbolMap],
    T.Dict[str, int],
    T.Dict[str, str],
    T.Dict[str, T.List[str]],
    T.Dict[str, T.List[str]],
]:
    """Parse top-level entries into symbols/ranks/binds/input_dims."""
    symbols: T.Dict[str, AxisSymbolMap] = {}
    ranks: T.Dict[str, int] = {}
    binds: T.Dict[str, str] = {}
    input_dims: T.Dict[str, T.List[str]] = {}
    outputs_keep: T.Dict[str, T.List[str]] = {}
    for top_key, val in dict(raw or {}).items():
        _validate_key(top_key)
        if isinstance(val, dict):
            _ = _parse_renamed_symbols(top_key, val.get("renamed_symbols"))
            _parse_nested_subnet(
                top_key, val, symbols, ranks, binds, input_dims
            )
            if "outputs_keep" in val:
                oks = val.get("outputs_keep")
                if not isinstance(oks, (list, tuple)) or not all(
                    isinstance(x, str) and x for x in oks
                ):
                    msg = (
                        "Invalid outputs_keep for subnet '"
                        + top_key
                        + "' (list[str] expected)"
                    )
                    raise ValueError(msg)
                outputs_keep[top_key] = [str(x) for x in oks]
        elif isinstance(val, (list, tuple)):
            _validate_and_record(top_key, val, symbols, ranks)
        else:
            msg = (
                "Invalid value for '"
                + top_key
                + "': expected list/tuple or nested mapping"
            )
            raise ValueError(msg)
    return symbols, ranks, binds, input_dims, outputs_keep


def _build_renamed_map(raw: dict) -> dict[str, dict[str, list[str]]]:
    """Build per-subnet renamed_symbols mapping from raw config."""
    out: dict[str, dict[str, list[str]]] = {}
    for top_key, val in dict(raw or {}).items():
        if isinstance(val, dict) and "renamed_symbols" in val:
            mapping = _parse_renamed_symbols(
                top_key, val.get("renamed_symbols")
            )
            if mapping:
                out[top_key] = mapping
    return out


def _parse_nested_subnet(
    top_key: str,
    val: dict,
    symbols: T.Dict[str, AxisSymbolMap],
    ranks: T.Dict[str, int],
    binds: T.Dict[str, str],
    input_dims: T.Dict[str, T.List[str]],
) -> None:
    """Parse a nested subnet mapping into outputs.

    Args:
        top_key: Subnet name.
        val: Mapping for the subnet.
        symbols: Output symbols map.
        ranks: Output ranks map.
        binds: Output binds mapping.
        input_dims: Output collapse-dims mapping.
    """
    if "collapse_dims" in val:
        raise ValueError(
            f"Do not set collapse_dims at subnet '{top_key}'. "
            "Define per-input collapse_dims instead."
        )
    if "collapse_batch_dim" in val:
        raise ValueError(
            f"Legacy collapse_batch_dim found at subnet '{top_key}'. "
            "Use per-input collapse_dims only."
        )

    # Require new-style nested inputs mapping under key 'inputs'.
    input_section = val.get("inputs") if isinstance(val, dict) else None
    if not isinstance(input_section, dict):
        raise ValueError(
            "Each subnet must declare an 'inputs' mapping. Flat per-input "
            "keys at the subnet level are no longer supported."
        )
    # Reject unknown top-level keys besides the allowed ones
    allowed = {"inputs", "renamed_symbols", "outputs_keep"}
    unknown = {k for k in val if k not in allowed}
    if unknown:
        raise ValueError(
            "Unknown keys in subnet config: " + ", ".join(sorted(unknown))
        )
    items = input_section.items()

    for inp_name, shape in items:
        if isinstance(shape, dict):
            is_tuple_group = "original_shape" not in shape and all(
                isinstance(k, str) and k.isdigit() for k in shape
            )
            if is_tuple_group:
                _nts_handle_tuple_group(
                    top_key, inp_name, shape, symbols, ranks, binds, input_dims
                )
            else:
                _nts_handle_single_mapping(
                    top_key, inp_name, shape, symbols, ranks, binds, input_dims
                )
        elif isinstance(shape, (list, tuple)):
            _validate_and_record(f"{top_key}.{inp_name}", shape, symbols, ranks)
        else:
            raise ValueError(
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
        raise FileNotFoundError(
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
        raise ValueError(
            "shape-config must be a mapping (optionally nested) of "
            "input-name -> list of dims"
        )
    # Delegate detailed parsing to helpers to keep complexity low
    symbols, ranks, binds, input_dims, outputs_keep = _parse_top_level(raw)
    if not symbols:
        raise ValueError("shape-config did not define any input shapes")
    renamed_per_subnet = _build_renamed_map(raw)

    return AxisSymbolRegistry(
        symbols_per_input=symbols,
        rank_per_input=ranks,
        bind_to_dim=binds,
        input_collapse_dims=input_dims,
        renamed_symbols_per_subnet=renamed_per_subnet,
        outputs_keep_per_subnet=outputs_keep,
    )
