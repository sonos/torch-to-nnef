# Security Policy

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.**

Instead, report them privately through GitHub's private vulnerability reporting:

> Repository **Security** tab → **Report a vulnerability**

This opens a private advisory visible only to you and the maintainers, where we
can discuss, coordinate a fix, and (if warranted) request a CVE.

When reporting, please include as much of the following as you can:

- the affected package(s) and version(s) (see *Scope* below);
- a description of the issue and its impact;
- steps to reproduce, or a proof-of-concept;
- any suggested remediation.

We will acknowledge your report as soon as we can, keep you informed of
progress, and coordinate the timing of any public disclosure with you.

## Scope

This policy covers the packages published from this repository:

- `torch_to_nnef` (core)
- `torch_to_nnef_llm`
- `torch_to_nnef_nemo_asr`

The `examples/` directory is illustrative and not part of the supported,
published surface.

## Supported Versions

Security fixes target the **latest released version**. Please confirm an issue
reproduces on the latest release before reporting.

## A Note on Untrusted Models

`torch-to-nnef` converts PyTorch models, which means it **loads and executes
model code and weights**. Treat a model you did not produce as untrusted input:
loading an untrusted checkpoint or repository can execute arbitrary code (this
is inherent to the PyTorch / Hugging Face loading paths, not specific to this
project). Only export models you trust.
