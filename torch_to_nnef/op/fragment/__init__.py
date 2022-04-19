import typing as T
from dataclasses import dataclass
from pathlib import Path

EXTENSION = ".nnef"
NNEF_EXTENSION_KEYWORD = "extension"


@dataclass
class Fragment:
    """Extract definitions and extensions from our custom nnef files"""

    name: str
    extensions: T.List[str]
    definition: str

    @classmethod
    def load_file(cls, path: Path) -> "Fragment":
        extensions = []
        with path.open("r", encoding="utf8") as fh:
            name = path.with_suffix("").name
            definition = fh.read().strip() + "\n" * 2
            filtered_definition = ""
            for defline in definition.split("\n"):
                if defline.strip().startswith(NNEF_EXTENSION_KEYWORD):
                    extensions.append(
                        defline.replace(NNEF_EXTENSION_KEYWORD, "")
                        .replace(";", "")
                        .strip()
                    )
                    continue
                filtered_definition += f"{defline}\n"
        return cls(
            name=name, extensions=extensions, definition=filtered_definition
        )

    @classmethod
    def load_all_from_dir(cls, base_path: Path):
        names_loaded = set()
        for path in base_path.glob(f"**/*{EXTENSION}"):
            if path.is_file():
                if path.name in names_loaded:
                    raise KeyError(
                        f"This fragments name: {path.name} is already used"
                    )
                names_loaded.add(path.name)
                yield Fragment.load_file(path)


FRAGMENTS: T.Dict[str, Fragment] = {
    f.name: f for f in Fragment.load_all_from_dir(Path(__file__).parent)
}
