# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/sonos/torch-to-nnef/issues>.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.
* Ideally share the debug bundle generated from `export_model_to_nnef(..., debug_bundle_path=)`

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

torch_to_nnef could always use more documentation, whether as part of the
official torch_to_nnef docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at <https://github.com/sonos/torch-to-nnef/issues>.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started

Ready to contribute? Here's how to set up `torch_to_nnef` for local development.

1. Fork the `torch_to_nnef` repo on GitHub.
2. Clone your fork locally

    ```bash
    git clone git@github.com:sonos/torch_to_nnef.git
    ```

3. Ensure [uv](https://github.com/astral-sh/uv) is installed.
4. Install dependencies and start your virtualenv:

    ```bash
    uv pip install --extra test --extra doc --extra dev
    ```

5. Create a branch for local development:

    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

    ```bash
    uv run tox
    ```

7. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.9, 3.10, 3.11 and 3.12. Check
   <https://github.com/sonos/torch-to-nnef/actions>
   and make sure that the tests pass for all supported Python versions.

## Tips

```bash
uv run pytest tests/test_torch_to_nnef.py
```

To run a subset of tests.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in CHANGELOG.md).
Then run:

```bash
uv run bump2version patch # possible: major / minor / patch
git push
git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.
