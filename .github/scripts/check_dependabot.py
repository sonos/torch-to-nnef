#!/usr/bin/env python3
"""Validate .github/dependabot.yml.

Checks that the config is schema version 2 and that every directory it
references actually contains the manifest its ecosystem expects. Keeps
Dependabot from silently skipping a path that was moved or mistyped.
"""

from __future__ import annotations

import os
import sys

import yaml

MANIFEST = {
    "uv": "uv.lock",
    "cargo": "Cargo.toml",
    "pip": "requirements.txt",
    "github-actions": ".github/workflows",
}


def main() -> int:
    with open(".github/dependabot.yml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    errors: list[str] = []
    if cfg.get("version") != 2:
        errors.append("top-level `version` must be 2")

    for upd in cfg.get("updates", []):
        eco = upd.get("package-ecosystem")
        manifest = MANIFEST.get(eco)
        if manifest is None:
            errors.append(f"unhandled package-ecosystem: {eco!r}")
            continue
        dirs = upd.get("directories") or [upd.get("directory")]
        for d in dirs:
            if d is None:
                errors.append(
                    f"{eco}: entry has neither `directory` nor `directories`"
                )
                continue
            path = os.path.join("." + d, manifest)
            if not os.path.exists(path):
                errors.append(f"{eco}: {d} has no {manifest}")

    for err in errors:
        print(f"::error::{err}")
    if errors:
        print(f"\n{len(errors)} dependabot.yml problem(s).")
        return 1
    print(
        "dependabot.yml OK: "
        "version 2 and all referenced directories have manifests."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
