# LLM tests

The newest Transformers CI environment runs this entire directory recursively.
Add new `test_*.py` files here and they are included automatically; no tox file
list needs updating. Tests requiring unavailable model families should use
`pytest.importorskip`. GPU-only tests use `ci_skip`, and real-checkpoint tests
can use `experimental` (enabled locally with `--run-experimental`).

`compat/` contains the shared compatibility suite, which also runs against the
older Transformers versions in CI. Add tests there when they should run across
all supported Transformers environments.

For tract runtime tests, parametrize over `TRACT_INFERENCES_TO_TESTS_APPROX`
(or `TRACT_INFERENCES_TO_TESTS_EXACT`) from the root `tests.utils`. This tests
all entries in `TractNNEF.OFFICIAL_SUPPORTED_VERSIONS`, including the oldest
and newest, unless an explicit test override selects a single target. Gate
features against the selected target's minimum required version. Constructing
only `TractNNEF.latest()` does not provide compatibility coverage.

Directory discovery does not make an unsupported feature executable: tests
whose minimum tract version is newer than every supported target will skip.
A compatible tract target must be available for runtime coverage.
