from pathlib import Path

EXTENSION = ".nnef"
FRAGMENTS = {}
for path in Path(__file__).parent.glob(f"**/*{EXTENSION}"):
    if path.is_file():
        if path.name in FRAGMENTS:
            raise KeyError(f"This fragments name: {path.name} is already used")
        with path.open("r", encoding="utf8") as fh:
            FRAGMENTS[path.with_suffix("").name] = fh.read().strip() + "\n"
