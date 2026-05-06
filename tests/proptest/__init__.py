"""Property-based tests for torch-to-nnef primitives.

This package wires `hypothesis` to the export pipeline so we can sweep input
shapes, dtypes, and op kwargs while still comparing PyTorch reference outputs
against tract with NaN/Inf-aware semantics. See the design at
`docs/contributing/internal_design.md` (proptest section) for rationale.
"""
