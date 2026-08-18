# 🤗 Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of contributions

### Report bugs

Report bugs at <https://github.com/sonos/torch-to-nnef/issues>.

The most useful thing you can include is a **debug bundle** generated from:

```python
export_model_to_nnef(..., debug_bundle_path="bundle.zip")
```

Please also include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write documentation

`torch_to_nnef` could always use more documentation, whether as part of the
official docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit feedback

The best way to send feedback is to file an issue at <https://github.com/sonos/torch-to-nnef/issues>.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

---

## Get started

Ready to contribute? Here's how to set up `torch_to_nnef` for local development.

1. Fork the `torch_to_nnef` repo on GitHub.

2. Clone your fork locally:

   ```
   git clone git@github.com:your-username/torch-to-nnef.git
   ```

3. Ensure [uv](https://github.com/astral-sh/uv) is installed.

4. Install dependencies:

   ```
   uv pip install --extra test --extra doc --extra dev
   ```

5. Install the pre-commit hooks:

   ```
   pre-commit install
   ```

6. Create a branch for local development, following the naming convention below:

   ```
   git checkout -b feat/my-new-operator
   ```

   Branch prefixes:
   * `feat/`: new features (new operators, handlers, …)
   * `fix/`: bugfixes and hotfixes
   * `chore/`: version bumps, CI/CD, packaging
   * `docs/`: documentation only changes
   * `test/`: test additions or fixes

7. Make your changes locally.

8. When you're done, run the full test suite across Python versions:

   ```
   uv run tox
   ```

9. Commit your changes and push:

   ```
   git add .
   git commit -m "feat: add support for aten::my_op operator"
   git push origin feat/my-new-operator
   ```

10. Submit a pull request through the GitHub website.

---

## Commit messages

Use the format `type: short description` in the imperative mood, matching your branch prefix:

| prefix | when to use |
|--------|-------------|
| `feat:` | new operator, new export feature |
| `fix:` | bug fix |
| `chore:` | CI, packaging, version bump |
| `docs:` | documentation only |
| `test:` | adding or fixing tests |

Examples:
```
feat: add support for aten::pixel_shuffle
fix: handle empty tensor in reshape export
chore: bump version to 0.22.0
docs: clarify debug_bundle_path usage in README
```

The subject line should be ≤72 characters. Reference related issues where relevant (`Fixes #42`).

---

## Code style

Formatting and linting are handled automatically by pre-commit (installed in step 5 above).
On every commit, the following run automatically:

* **ruff format**: auto-formats code (line length: 80)
* **ruff check --fix**: lints and auto-fixes where possible
* **mypy**: type-checks library code (excluding tests)
* **pyupgrade**: modernises syntax to Python 3.7+
* **prose check**: rejects banned characters and wording (see [Prose](#prose))
* Standard file hygiene: trailing whitespace, CRLF, missing newlines, YAML/JSON/TOML validity

Naming conventions follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### Docstrings

Docstrings are written in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
They are **encouraged but not required on every item**: the linter will not fail on missing docstrings.
Do add them for any public function or class that isn't self-explanatory, especially if you are adding new functionality.

### AI-assisted contributions

Using an AI coding assistant is allowed, and needs no disclaimer. Plenty of
good patches start that way. A patch is judged on its merits, never on how it
was produced.

What we do ask is that you own what you send. Read every line as if you had
typed it, and check that it says something true about *this* codebase rather
than something plausible about codebases in general. If a comment explains a
constraint, verify the constraint still holds.

The one thing we consistently send back is the register these tools default to.
It usually shows up as:

* padding that restates the code (`# increment the counter` above `i += 1`)
* comments describing *what* a line does, where the useful comment says *why*
  it does it, or why the obvious alternative was rejected
* marketing adjectives in technical prose (`powerful`, `elegant`, `blazing`)
* hedging that commits to nothing, and closing paragraphs that summarise the
  section directly above them
* filler transitions and symmetrical flourishes that add length, not meaning
* typographic lookalikes: em dashes, curly quotes, non-breaking hyphens

Only the last of those is machine-checkable, and the [Prose](#prose) check below
enforces it. The rest is a review conversation: expect a reviewer to ask you to
cut it. Shorter is nearly always better, and one comment recording a decision is
worth ten that describe syntax.

And that conversation is the part we most want a human in. Reviews are done by
people, in their own time, and they go much better when someone on the other
side has the whole change in their head. So when a reviewer asks why something
is shaped the way it is, answer in your own words: what you tried, what broke,
what you decided against. A reply that paraphrases the diff back, or that
reopens a point the thread already settled, spends review attention that is in
short supply. Staying available to talk a change through counts for more than
sending a large one.

That check is a floor, not a proxy for having read your own text. Passing it
means you avoided the characters and words we reject, not that the writing is
any good.

### Prose

`.github/scripts/check_prose.py` scans every tracked text file, on each commit
and on each pull request. It rejects two things.

**Characters.** Typographic lookalikes: em dash (U+2014), en dash (U+2013),
non-breaking hyphen (U+2011), curly quotes and apostrophes (U+2018, U+2019,
U+201C, U+201D), non-breaking space (U+00A0), zero-width space (U+200B) and
byte-order mark (U+FEFF). Write ASCII `-`, `'`, `"` and a plain space instead.
Where one of these was separating clauses, rephrase with `:` or `,`, or split
the sentence.

This half is a correctness rule rather than a preference: U+2011 renders exactly
like `-` but breaks `grep`, in-repo search and anchor links, and 25 of them had
reached words such as `provider-agnostic` before the check existed. **It has no
escape hatch: no pragma and no per-file exemption.**

Emoji, box-drawing, arrows and mathematical symbols are all fine. Only the
lookalikes above are rejected.

**Wording.** A short list of words and phrases that read as machine-generated
rather than written. `BUZZWORDS` and `HEDGING` in the script are the single
source of truth, so run the check instead of memorising them.

Vendored bundles, wasm-pack output and ASR data tables are out of scope, since
we do not author them (see `NOT_OURS`).

```bash
make prose                                       # every tracked file
python3 .github/scripts/check_prose.py FILE ...  # only these paths
```

---

## Pull request guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request includes tests covering the new behaviour or bug fix.
2. If the pull request adds functionality, update the docs and add a docstring to new public functions. Add the feature to `README.md` if user-facing.
3. The pull request passes CI for Python 3.9, 3.10, 3.11, and 3.12. Check <https://github.com/sonos/torch-to-nnef/actions>.

### Slow tests

If your test is slow (e.g. downloads a model or runs a large export), mark it with `@pytest.mark.ci_skip`
so it is excluded from the fast CI run but can still be run locally with `uv run pytest -m ci_skip`.

---

## Tips

Run a specific test file:

```
uv run pytest tests/test_torch_to_nnef.py
```

Run only fast tests (CI mode):

```
uv run pytest -m "not ci_skip"
```

Run the deeper static analysis:

```
uv run tox -e static_check
```

---

## Deploying (maintainers only)

Make sure all changes are committed, including an entry in `CHANGELOG.md`. Then:

```
uv run bump2version patch  # or: major / minor
git push
git push --tags
```

GitHub Actions will deploy to PyPI automatically if tests pass.
