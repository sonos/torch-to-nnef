import logging
from pathlib import Path

import pytest

from tests.proptest.conftest_helpers import register_profiles


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "experimental: mark test as experimental to run"
    )
    logging.getLogger("torch_to_nnef").setLevel(logging.WARNING)
    register_profiles()


def pytest_addoption(parser):
    parser.addoption(
        "--run-experimental",
        action="store_true",
        default=False,
        help="run experimental tests",
    )
    # ONNX support-measurement sweep. See
    # `tests/test_primitive_proptest_onnx.py` and
    # `docs/contributing/onnx_support_page.md`.
    parser.addoption(
        "--onnx-report",
        action="store",
        default=None,
        help=(
            "write the graded ONNX support artifact to this path "
            "(no path: the sweep runs but nothing is persisted)"
        ),
    )
    parser.addoption(
        "--onnx-reuse",
        action="store",
        default=None,
        help=(
            "prior artifact to carry `full` grades over from, when the "
            "recorded environment fingerprint matches this one. Defaults "
            "to --onnx-report's path when that file already exists."
        ),
    )
    parser.addoption(
        "--onnx-no-reuse",
        action="store_true",
        default=False,
        help="re-measure every spec, ignoring any prior artifact",
    )
    parser.addoption(
        "--onnx-opset",
        action="store",
        type=int,
        default=None,
        help="ONNX opset to export against (default: DEFAULT_OPSET)",
    )
    parser.addoption(
        "--onnx-skip-numerics",
        action="store_true",
        default=False,
        help=(
            "only measure export and onnxruntime load/run, skipping the "
            "numeric comparison (roughly halves the sweep cost)"
        ),
    )


@pytest.fixture(scope="session")
def onnx_sweep(pytestconfig):
    """Session-wide collector for the ONNX support-measurement sweep.

    Session-scoped because the artifact is an aggregate over every spec:
    each test contributes its examples, and the report is written once at
    teardown. Importing the ONNX layer lazily keeps `onnx` / `onnxruntime`
    / `onnxscript` out of the fast suite's dependency set.
    """
    from hypothesis import settings

    from tests.proptest.conftest_helpers import selected_profile
    from tests.proptest.onnx_backend import DEFAULT_OPSET, OnnxRunConfig
    from tests.proptest.onnx_report import (
        OnnxReport,
        ReuseIndex,
        environment_fingerprint,
        load_prior,
    )

    report_path = pytestconfig.getoption("--onnx-report")
    reuse_path = pytestconfig.getoption("--onnx-reuse")
    config = OnnxRunConfig(
        opset=pytestconfig.getoption("--onnx-opset") or DEFAULT_OPSET,
        check_numerics=not pytestconfig.getoption("--onnx-skip-numerics"),
    )

    # Reusing from the report we are about to overwrite is the common
    # case: regenerating in place should not have to name the file twice.
    prior_path = None
    if not pytestconfig.getoption("--onnx-no-reuse"):
        prior_path = Path(reuse_path) if reuse_path else None
        if prior_path is None and report_path:
            candidate = Path(report_path)
            prior_path = candidate if candidate.exists() else None

    max_examples = settings.default.max_examples
    reuse = ReuseIndex(
        prior=load_prior(prior_path),
        fingerprint=environment_fingerprint(config),
        current_examples=max_examples,
        enabled=not pytestconfig.getoption("--onnx-no-reuse"),
    )
    report = OnnxReport(
        config=config,
        max_examples=max_examples,
        profile=selected_profile(),
        reuse=reuse,
    )
    yield report
    if report_path:
        report.write(Path(report_path))
    pytestconfig.stash[_ONNX_SUMMARY] = report.summary_lines() + (
        [f"ONNX support artifact written to {report_path}"]
        if report_path
        else ["no --onnx-report path given: results not persisted"]
    )


_ONNX_SUMMARY = pytest.StashKey[list]()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Surface the sweep's grade distribution after the test counts."""
    for line in config.stash.get(_ONNX_SUMMARY, []):
        terminalreporter.write_line(line)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-experimental"):
        return
    skip_experimental = pytest.mark.skip(
        reason="need --run-experimental option to run"
    )
    for item in items:
        if "experimental" in item.keywords:
            item.add_marker(skip_experimental)
