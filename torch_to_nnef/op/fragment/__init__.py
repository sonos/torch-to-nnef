"""List of fragment directly written in NNEF to allow composition.

Most reimplement aten:: operators.
"""

import typing as T
from dataclasses import dataclass
from pathlib import Path

from mako.template import Template

from torch_to_nnef.exceptions import T2NError, T2NErrorFragmentFile

EXTENSION = ".nnef"
TMPL_EXTENSION = ".tmpl"
NNEF_EXTENSION_KEYWORD = "extension"


@dataclass(frozen=True, eq=True)
class Fragment:
    """Extract definitions and extensions from our custom NNEF files."""

    name: str
    extensions: T.Tuple[str, ...]
    definition: str

    @classmethod
    def load_file(cls, path: Path) -> "Fragment":
        extensions = []
        with path.open("r", encoding="utf8") as fh:
            while path.suffix in {EXTENSION, TMPL_EXTENSION}:
                path = path.with_suffix("")
            name = path.name
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
                if defline.strip().startswith("#"):  # avoid to export comments
                    continue
                filtered_definition += f"{defline}\n"
        return cls(
            name=name,
            extensions=tuple(extensions),
            definition=filtered_definition,
        )

    @classmethod
    def load_all_from_dir(cls, base_path: Path):
        names_loaded = set()
        for path in base_path.glob(f"**/*{EXTENSION}"):
            if path.is_file():
                if ".tmpl" in path.suffixes:
                    continue
                if path.name in names_loaded:
                    raise T2NErrorFragmentFile(
                        f"This fragment name: {path.name} is already used"
                    )
                names_loaded.add(path.name)
                yield Fragment.load_file(path)


class TmplFragment(Fragment):
    def into_concrete_fragment(self, **kwargs):
        concrete_definition = Template(self.definition).render(**kwargs)
        main_entry_found = False
        main_fragment_str = "!main fragment"
        name = ""
        for cd_line in concrete_definition.split("\n"):
            if cd_line.startswith(main_fragment_str):
                main_entry_found = True
                name = (
                    cd_line.replace(main_fragment_str, "").split("(")[0].strip()
                )
                break
        if not main_entry_found or not name:
            raise T2NError(
                f"Missing '{main_fragment_str}' "
                f"in fragment template: {self.name}"
            )
        concrete_definition = concrete_definition.replace(
            main_fragment_str, "fragment"
        )
        return Fragment(
            name=name,
            definition=concrete_definition,
            extensions=self.extensions,
        )

    @classmethod
    def load_all_from_dir(cls, base_path: Path):
        names_loaded = set()
        for path in base_path.glob(f"**/*{EXTENSION}"):
            if path.is_file():
                if ".tmpl" not in path.suffixes:
                    continue
                if path.name in names_loaded:
                    raise T2NErrorFragmentFile(
                        f"This TmplFragment name: {path.name} is already used"
                    )
                names_loaded.add(path.name)
                yield TmplFragment.load_file(path)


FRAGMENTS: T.Dict[str, Fragment] = {
    f.name: f for f in Fragment.load_all_from_dir(Path(__file__).parent)
}

TMPL_FRAGMENTS: T.Dict[str, TmplFragment] = {
    f.name: f for f in TmplFragment.load_all_from_dir(Path(__file__).parent)
}
