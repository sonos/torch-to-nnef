"""LLM test configuration."""

import logging
import sys
from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    """Walk up from this file to find the repo root (contains tests/utils.py)."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "tests" / "utils.py").exists():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root with tests/utils.py")


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "experimental: mark test as experimental to run"
    )
    logging.getLogger("torch_to_nnef").setLevel(logging.WARNING)


def pytest_addoption(parser):
    parser.addoption(
        "--run-experimental",
        action="store_true",
        default=False,
        help="run experimental tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-experimental"):
        return
    skip_experimental = pytest.mark.skip(
        reason="need --run-experimental option to run"
    )
    for item in items:
        if "experimental" in item.keywords:
            item.add_marker(skip_experimental)
