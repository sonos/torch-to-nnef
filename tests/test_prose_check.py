"""Tests for the prose gate in ``.github/scripts/check_prose.py``.

Without these, a typo in one of the regexes would silently stop the gate from
firing and every pull request would read as green. Each banned character is
asserted individually for that reason.

The checker lives outside the importable packages (it is stdlib-only tooling
invoked by CI, pre-commit and ``make prose``), so it is loaded by path.

This file is listed in the checker's own ``WORDS_EXEMPT``: the fixtures below
have to spell out banned wording as data. Characters are still checked here,
which is why the fixtures use unicode escapes rather than literal glyphs.
"""

import importlib.util
import pathlib
import sys

import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "check_prose.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("check_prose", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_prose"] = module
    spec.loader.exec_module(module)
    return module


check_prose = _load()


@pytest.mark.parametrize("char", sorted(check_prose.BANNED_CHARS))
def test_every_banned_char_is_detected(char, tmp_path):
    """Each entry of BANNED_CHARS must actually be caught, not just listed."""
    target = tmp_path / "sample.md"
    target.write_text(f"a{char}b\n", encoding="utf-8")

    found = check_prose.scan(str(target))

    assert len(found) == 1, f"U+{ord(char):04X} not detected"
    assert found[0].line == 1
    assert found[0].col == 2
    assert f"U+{ord(char):04X}" in found[0].what


@pytest.mark.parametrize(
    "text",
    [
        "a genuine constant",
        "we should delve into this",
        "it works seamlessly",
        "a pivotal moment",
        "that said, it holds",
        "in essence, it holds",
        "at its core it is a graph",
        "it is worth noting that it holds",
    ],
)
def test_banned_wording_is_detected(text, tmp_path):
    target = tmp_path / "sample.md"
    target.write_text(text + "\n", encoding="utf-8")

    assert check_prose.scan(str(target)), f"missed: {text!r}"


def test_char_and_word_report_line_and_column(tmp_path):
    target = tmp_path / "sample.md"
    target.write_text("clean line\nok\u2014a genuine one\n", encoding="utf-8")

    found = check_prose.scan(str(target))

    assert [(v.line, v.col) for v in found] == [(2, 3), (2, 6)]


def test_clean_ascii_passes(tmp_path):
    target = tmp_path / "sample.md"
    target.write_text(
        "A plain ASCII line with a hyphen-joined word and 'quotes'.\n"
        "Emoji, box-drawing and arrows stay allowed: OK 100% -> done.\n",
        encoding="utf-8",
    )

    assert check_prose.scan(str(target)) == []


def test_deliberately_allowed_chars_pass(tmp_path):
    """Emoji, box-drawing, arrows, ellipsis and units are not banned."""
    target = tmp_path / "sample.md"
    target.write_text("✅ ❌ → ─ … × µ\n", encoding="utf-8")

    assert check_prose.scan(str(target)) == []


def test_underscored_identifier_is_not_flagged(tmp_path):
    """`_` is a word character, so identifiers are not prose."""
    target = tmp_path / "sample.py"
    target.write_text("genuine_hit = 1\n", encoding="utf-8")

    assert check_prose.scan(str(target)) == []


def test_check_words_false_still_reports_chars(tmp_path):
    """Characters have no exemption, even where wording is exempt."""
    target = tmp_path / "sample.py"
    target.write_text("x = 1  # a genuine note\u2014here\n", encoding="utf-8")

    found = check_prose.scan(str(target), check_words=False)

    assert len(found) == 1
    assert "U+2014" in found[0].what


def test_binary_file_is_skipped(tmp_path):
    target = tmp_path / "blob.bin"
    target.write_bytes(b"\x00\x01" + "\u2014".encode())

    assert check_prose.scan(str(target)) == []


def test_symlink_is_skipped(tmp_path):
    """A symlink's hits belong to its target, reported once at the real path."""
    real = tmp_path / "real.md"
    real.write_text("bad\u2014here\n", encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(real)

    assert check_prose.scan(str(link)) == []
    assert len(check_prose.scan(str(real))) == 1


def test_not_ours_paths_are_out_of_scope():
    assert check_prose.matches(
        "docs/html/uPlot.iife.min.js", check_prose.NOT_OURS
    )
    assert check_prose.matches(
        "examples/nemo_asr/src/x/normalizer/abbr.py", check_prose.NOT_OURS
    )
    assert not check_prose.matches(
        "docs/html/vad/plot.js", check_prose.NOT_OURS
    )
    assert not check_prose.matches(
        "torch_to_nnef/export.py", check_prose.NOT_OURS
    )


def test_main_reports_and_exits_nonzero(tmp_path, capsys):
    """`main` must annotate and fail, which is what CI relies on.

    Deliberately scoped to an explicit path list rather than a whole-repo scan:
    running the real gate in here would fail the unit suite over an unrelated
    in-progress edit elsewhere in the tree, and would error outright in a
    non-git checkout (`main([])` shells out to git).
    """
    target = tmp_path / "sample.md"
    target.write_text("bad\u2014here\n", encoding="utf-8")

    rc = check_prose.main([str(target)])
    out = capsys.readouterr().out

    assert rc == 1
    assert "::error file=" in out
    assert "U+2014" in out


def test_main_exits_zero_on_clean_input(tmp_path):
    target = tmp_path / "sample.md"
    target.write_text("plain ASCII line\n", encoding="utf-8")

    assert check_prose.main([str(target)]) == 0
