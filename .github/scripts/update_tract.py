"""Prepare a supported-tract bump from GitHub's paginated release response."""

import argparse
import json
import os
import re
from pathlib import Path

VERSION = re.compile(r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
SUPPORTED = re.compile(
    r"(OFFICIAL_SUPPORTED_VERSIONS\s*=\s*\[\s*"
    r'SemanticVersion\.from_str\(version\)\s*for version in\s*\[\s*")'
    r'(?P<version>[0-9]+\.[0-9]+\.[0-9]+)(")'
)


def newest_stable(pages):
    versions = []
    for page in pages:
        for release in page:
            match = VERSION.fullmatch(release["tag_name"])
            if match and not release["draft"] and not release["prerelease"]:
                versions.append(tuple(map(int, match.groups())))
    if not versions:
        raise ValueError("No stable tract releases found")
    return ".".join(map(str, max(versions)))


def prepare(source, changelog, latest):
    match = SUPPORTED.search(source)
    if not match:
        raise ValueError(
            "Cannot locate latest officially supported tract version"
        )
    current = match["version"]
    if tuple(map(int, latest.split("."))) <= tuple(
        map(int, current.split("."))
    ):
        return source, changelog, current
    heading = "## Unreleased\n"
    if changelog.count(heading) != 1:
        raise ValueError("Expected exactly one Unreleased changelog section")
    start, end = match.span("version")
    source = source[:start] + latest + source[end:]
    section_start = changelog.index(heading) + len(heading)
    section_end = changelog.find("\n## ", section_start)
    if section_end == -1:
        section_end = len(changelog)
    section = changelog[section_start:section_end]
    entry = (
        f"- **Latest officially supported tract version is now {latest}**, "
        f"up from {current}. Default exports target {latest}.\n"
    )
    if "### Changed\n" in section:
        section = section.replace("### Changed\n", "### Changed\n" + entry, 1)
    else:
        section = "\n### Changed\n" + entry + section
    changelog = changelog[:section_start] + section + changelog[section_end:]
    return source, changelog, current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("releases", type=Path)
    args = parser.parse_args()
    latest = newest_stable(json.loads(args.releases.read_text()))
    target = Path("torch_to_nnef/inference_target/tract.py")
    changelog = Path("CHANGELOG.md")
    source, notes, current = prepare(
        target.read_text(), changelog.read_text(), latest
    )
    target.write_text(source)
    changelog.write_text(notes)
    print(f"Supported tract: {current}; latest stable release: {latest}")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"latest={latest}\ncurrent={current}\n")


if __name__ == "__main__":
    main()
