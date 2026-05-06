"""Extract Pocket-TTS' SentencePiece tokenizer to a local ``tokenizer.model``.

Pocket-TTS bundles its tokenizer inside ``LUTConditioner`` -- this script
loads the model (HF download on first call) and serialises the underlying
SentencePiece protobuf to disk so the Rust runtime can ``mmap`` it via the
``sentencepiece`` crate.

Run:
    python extract_tokenizer.py --out cli/tokenizer.model
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pocket_tts import TTSModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tokenizer.model"))
    args = parser.parse_args()

    print("loading Pocket-TTS to read its bundled tokenizer")
    model = TTSModel.load_model()
    sp = model.flow_lm.conditioner.tokenizer.sp
    proto = sp.serialized_model_proto()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(proto)
    print(
        f"wrote {args.out} ({len(proto)} bytes, vocab_size={sp.vocab_size()})"
    )


if __name__ == "__main__":
    main()
