#!/usr/bin/env python3
"""Fail if any GitHub Action is not pinned to a full commit SHA.

Enforces the supply-chain policy established when the workflows were first
pinned: third-party (and first-party) actions must reference a 40-char commit
SHA, not a floating tag. Validates that Dependabot's action bumps keep that
form. Local actions (./ or ../) are exempt; docker:// refs must use @sha256:.
"""

from __future__ import annotations

import pathlib
import re
import sys

WORKFLOWS = pathlib.Path(".github/workflows")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
USES = re.compile(r"""uses:\s*['"]?([^'"\s#]+)""")


def main() -> int:
    # (workflow, line, action ref, why it failed)
    errors: list[tuple[pathlib.Path, int, str, str]] = []
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    for path in files:
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            match = USES.search(line)
            if not match:
                continue
            ref = match.group(1)
            if ref.startswith("./") or ref.startswith("../"):
                continue  # local action
            if ref.startswith("docker://"):
                if "@sha256:" not in ref:
                    errors.append(
                        (path, lineno, ref, "docker ref not pinned by @sha256:")
                    )
                continue
            if "@" not in ref:
                errors.append(
                    (path, lineno, ref, "no @ref (must be a commit SHA)")
                )
                continue
            tag = ref.split("@", 1)[1]
            if not SHA40.match(tag):
                errors.append(
                    (path, lineno, ref, "not pinned to a 40-char commit SHA")
                )

    for path, lineno, ref, why in errors:
        print(f"::error file={path},line={lineno}::{ref} -> {why}")
    if errors:
        print(f"\n{len(errors)} unpinned action reference(s).")
        return 1
    print(f"All GitHub Actions across {len(files)} workflow(s) are SHA-pinned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
