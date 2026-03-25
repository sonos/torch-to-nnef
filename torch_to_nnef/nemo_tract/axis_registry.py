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

    @staticmethod
    def empty() -> "AxisSymbolRegistry":
        return AxisSymbolRegistry(
            symbols_per_input={},
            rank_per_input={},
            bind_to_dim={},
            input_collapse_dims={},
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
    symbols: T.Dict[str, AxisSymbolMap] = {}
    ranks: T.Dict[str, int] = {}
    binds: T.Dict[str, str] = {}
    input_dims: T.Dict[str, T.List[str]] = {}

    def _validate_and_set(key: str, shape_val: T.Sequence[T.Union[str, int]]):
        if not isinstance(key, str) or not key:
            raise ValueError(
                "Invalid key in shape-config (expected non-empty string): "
                f"{key!r}"
            )
        for i, v in enumerate(shape_val):
            if not isinstance(v, (str, int)):
                raise ValueError(
                    "Invalid dim at "
                    f"{key}[{i}]: {type(v).__name__}; expected str or int"
                )
            if isinstance(v, str) and not v.strip():
                raise ValueError(
                    f"Empty dim symbol at {key}[{i}] is not allowed"
                )
        symbols[key] = _list_to_axis_map(shape_val)
        ranks[key] = len(shape_val)

    def _normalize_syms(seq: T.Sequence[str]) -> T.List[str]:
        out: T.List[str] = []
        for s in seq:
            ss = s.strip()
            if ss.lower() in ("b", "batch"):
                out.append("BATCH")
            else:
                out.append(ss.upper())
        return out

    # Support both flat and nested (subnet -> inputs -> mapping) forms
    for top_key, val in dict(raw or {}).items():
        if not isinstance(top_key, str) or not top_key:
            raise ValueError(
                "Invalid key in shape-config (expected non-empty string): "
                f"{top_key!r}"
            )
        if isinstance(val, dict):
            # Nested context: top_key is subnet, keys inside are input names
            # Reject subnet-level collapse_dims / collapse_batch_dim
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
            for inp_name, shape in val.items():
                if inp_name == "collapse_batch_dim":
                    continue
                if inp_name == "collapse_dims":
                    continue
                # Structured input mapping or legacy list/tuple
                if isinstance(shape, dict):
                    # Tuple grouping support: keys are indices 0,1,...
                    is_tuple_group = (
                        "original_shape" not in shape
                        and all(
                            isinstance(k, str) and k.isdigit() for k in shape
                        )
                    )
                    if is_tuple_group:
                        for idx_str, inner in shape.items():
                            qname = f"{top_key}.{inp_name}_{idx_str}"
                            if not isinstance(inner, dict):
                                raise ValueError(
                                    "Invalid tuple entry for '"
                                    f"{qname}': expected mapping"
                                )
                            if "original_shape" in inner:
                                oshp = inner.get("original_shape")
                                if not isinstance(oshp, (list, tuple)):
                                    raise ValueError(
                                        f"Invalid original_shape for '{qname}'"
                                    )
                                _validate_and_set(qname, oshp)
                            if "collapse_dims" in inner:
                                cdv = inner.get("collapse_dims")
                                if not isinstance(cdv, (list, tuple)):
                                    raise ValueError(
                                        f"Invalid collapse_dims for '{qname}'"
                                    )
                                input_dims[qname] = _normalize_syms(
                                    [str(x) for x in cdv]
                                )
                            b = None
                            if "bind_scalar_to_dim_size" in inner:
                                b = inner.get("bind_scalar_to_dim_size")
                            elif "bind_to_dim" in inner:
                                b = inner.get("bind_to_dim")
                            if isinstance(b, str) and b:
                                binds[qname] = b
                    else:
                        # Optional fields per input:
                        # - original_shape
                        # - collapse_dims
                        # - bind_scalar_to_dim_size (preferred), or legacy bind_to_dim
                        if "original_shape" in shape:
                            oshp = shape.get("original_shape")
                            if not isinstance(oshp, (list, tuple)):
                                raise ValueError(
                                    "Invalid original_shape for '"
                                    f"{top_key}.{inp_name}'"
                                )
                            _validate_and_set(f"{top_key}.{inp_name}", oshp)
                        if "collapse_dims" in shape:
                            cdv = shape.get("collapse_dims")
                            if not isinstance(cdv, (list, tuple)):
                                raise ValueError(
                                    "Invalid collapse_dims for '"
                                    f"{top_key}.{inp_name}'"
                                )
                            input_dims[f"{top_key}.{inp_name}"] = _normalize_syms(
                                [str(x) for x in cdv]
                            )
                        # Prefer new key name; fall back to legacy if present
                        b = None
                        if "bind_scalar_to_dim_size" in shape:
                            b = shape.get("bind_scalar_to_dim_size")
                        elif "bind_to_dim" in shape:
                            b = shape.get("bind_to_dim")
                        if isinstance(b, str) and b:
                            binds[f"{top_key}.{inp_name}"] = b
                elif isinstance(shape, (list, tuple)):
                    _validate_and_set(f"{top_key}.{inp_name}", shape)
                else:
                    raise ValueError(
                        f"Invalid value for '{top_key}.{inp_name}': "
                        "expected list/tuple or mapping"
                    )
        elif isinstance(val, (list, tuple)):
            # Flat qualified or bare name at top-level
            _validate_and_set(top_key, val)
        else:
            raise ValueError(
                f"Invalid value for '{top_key}': expected list/tuple or "
                "nested mapping"
            )
    if not symbols:
        raise ValueError("shape-config did not define any input shapes")
    return AxisSymbolRegistry(
        symbols_per_input=symbols,
        rank_per_input=ranks,
        bind_to_dim=binds,
        input_collapse_dims=input_dims,
    )
