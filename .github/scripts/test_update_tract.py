"""Release selection and file-editing regression tests; no network needed."""

import unittest

from update_tract import newest_stable, prepare

SOURCE = """OFFICIAL_SUPPORTED_VERSIONS = [
    SemanticVersion.from_str(version)
    for version in [
        "0.23.8",
        "0.22.1",
    ]
]
"""
CHANGELOG = (
    "# Changelog\n\n## Unreleased\n\n### Added\n- Existing entry.\n\n## 0.1\n"
)


def release(tag, **kwargs):
    return {"tag_name": tag, "draft": False, "prerelease": False, **kwargs}


class UpdateTractTests(unittest.TestCase):
    def test_numeric_order_across_pages_and_ignore_unstable(self):
        self.assertEqual(
            newest_stable(
                [
                    [release("v0.23.9"), release("v1.0.0", draft=True)],
                    [
                        release("v0.23.10"),
                        release("v2.0.0", prerelease=True),
                        release("v3.0.0-rc.1"),
                        release("not-a-version"),
                    ],
                ]
            ),
            "0.23.10",
        )

    def test_missing_stable_release_fails(self):
        with self.assertRaises(ValueError):
            newest_stable([[release("v1.0.0-rc.1")]])

    def test_upgrade_preserves_older_support_and_history(self):
        source, notes, current = prepare(SOURCE, CHANGELOG, "0.24.0")
        self.assertEqual(current, "0.23.8")
        self.assertEqual(source, SOURCE.replace("0.23.8", "0.24.0"))
        self.assertIn("### Changed\n- **Latest officially", notes)
        self.assertIn("### Added\n- Existing entry.", notes)
        self.assertTrue(notes.endswith("\n## 0.1\n"))

    def test_existing_changed_section(self):
        notes = CHANGELOG.replace("### Added", "### Changed")
        _, updated, _ = prepare(SOURCE, notes, "0.23.9")
        self.assertEqual(updated.count("### Changed"), 1)
        self.assertIn("- Existing entry.", updated)

    def test_equal_or_older_is_noop(self):
        for version in ["0.23.8", "0.23.7", "0.22.10"]:
            self.assertEqual(
                prepare(SOURCE, CHANGELOG, version),
                (SOURCE, CHANGELOG, "0.23.8"),
            )

    def test_rerun_is_idempotent(self):
        source, notes, _ = prepare(SOURCE, CHANGELOG, "0.23.9")
        self.assertEqual(
            prepare(source, notes, "0.23.9"), (source, notes, "0.23.9")
        )

    def test_format_changes_fail_loudly(self):
        with self.assertRaises(ValueError):
            prepare("unexpected source", CHANGELOG, "0.23.9")
        with self.assertRaises(ValueError):
            prepare(SOURCE, "missing unreleased section", "0.23.9")


if __name__ == "__main__":
    unittest.main()
