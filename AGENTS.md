# AGENTS.md

## Code style

1. Follow PEP8. Keep function complexity small -- each function should do one thing.
2. Use Google-style docstrings for code comments.
3. Do not capitalize words in the middle of a sentence in docstrings or docs.
4. Use list comprehensions instead of for-loops for simple structure building.
5. Prefer clear, unmangled names for variables and classes (e.g., `SubnetSignature` not `SubnetSig`).
6. Never use non-ascii characters in code, docs, markdown, or comments. No emoji, no em-dash, no unicode arrows. Use ascii equivalents (`--`, `->`, etc.).

## String formatting

7. Use implicit string concatenation, not `+`, for multi-part strings:
    ```python
    # good
    a = (
        "prefix "
        f"middle {value} "
        "suffix"
    )

    # bad
    a = (
        "prefix "
        + f"middle {value} "
        + "suffix"
    )
    ```
8. For multi-line strings in handlers, prefer triple-quoted strings with `textwrap.dedent`.

## Error handling

9. Never use bare `except Exception:` -- no pokemon catching.
10. Never use raw exceptions (`raise Exception(...)`). Use one of the exceptions defined in `torch_to_nnef/exceptions.py`, or create a new one there if no existing class matches.

## Imports and dependencies

11. All imports at the top of the file except for optional extras. For optional dependencies, use the `@require_extra_decorator(extra=T2NExtra...., module="..")` pattern instead of try/except ImportError blocks.
12. Never use inline imports like `__import__("datetime")`.

## Data structures

13. Prefer dataclasses over dicts for anything with more than 3 packed primitives. For simple mapping tables, define a type alias (e.g., `AxisSymbolMap = T.Dict[int, str]`).
14. Functions must never accept or return a plain dict unless it comes from outside library control (external API, JSON, etc.).

## Constants and references

15. Never inline magic constants in code. Define a named constant at module level and reference it.

## Code documentation

16. Do not reference implementation plans in code comments (e.g., "Stage 1", "Step 2"). No leftover refactoring notes like "moved from X".

## Quality checks

17. Before pushing code, run `tox run -e format` and `tox run -e static_check`. Fix all errors before pushing. Repeat until clean.
18. Run `python -m py_compile <file>` on changed files to verify syntax.

## Git

19. Never include Co-Authored-By lines or any co-author attribution in commit messages.
