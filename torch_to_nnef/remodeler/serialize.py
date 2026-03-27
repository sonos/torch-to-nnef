import json
import typing as T
from pathlib import Path

from torch_to_nnef.remodeler import IODescriptor, SubnetSignature


def _io_to_json(io: IODescriptor) -> dict:
    return {
        "name": io.name,
        "shape": [str(d) for d in io.shape],
        "dtype": io.dtype,
        "notes": list(io.notes or []),
    }


def signatures_to_json_payload(
    sigs: T.List[SubnetSignature], *, model_label: T.Optional[str] = None
) -> dict:
    return {
        "model": model_label,
        "subnets": [
            {
                "name": s.name,
                "stage": s.stage.value,
                "inputs": [_io_to_json(i) for i in s.inputs],
                "outputs": [_io_to_json(o) for o in s.outputs],
                "applied_flags": list(s.applied_flags),
                "symbol_axes": {
                    k: {int(ax): str(sym) for ax, sym in v.items()}
                    for k, v in (s.symbol_axes or {}).items()
                },
            }
            for s in sigs
        ],
    }


def signatures_to_json_text(
    sigs: T.List[SubnetSignature],
    *,
    model_label: T.Optional[str] = None,
    indent: int = 2,
) -> str:
    payload = signatures_to_json_payload(sigs, model_label=model_label)
    return json.dumps(payload, indent=indent)


def write_signatures_json(
    sigs: T.List[SubnetSignature],
    *,
    to_path: T.Optional[Path] = None,
    stream=None,
    model_label: T.Optional[str] = None,
    indent: int = 2,
) -> None:
    txt = signatures_to_json_text(sigs, model_label=model_label, indent=indent)
    if stream is not None:
        stream.write(txt + "\n")
        return
    if to_path is None:
        print(txt)
        return
    to_path.parent.mkdir(parents=True, exist_ok=True)
    to_path.write_text(txt + "\n", encoding="utf8")
