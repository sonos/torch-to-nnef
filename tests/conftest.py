import logging
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "experimental: mark test as experimental to run"
    )
    # Pre-set log levels; NeMo's singleton Logger may add handlers later,
    # so we also use an autouse fixture (_silence_nemo_logs) to catch those.
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


@pytest.fixture(autouse=True, scope="session")
def _silence_nemo_logs():
    """Mute NeMo's chatty logging during the entire test session.

    NeMo's singleton Logger adds StreamHandlers lazily (including
    during ``from_pretrained``).  We raise the level on the underlying
    Python logger to ERROR **and** install a filter that blocks
    everything below ERROR on all current and future handlers.

    Override with ``NEMO_LOG_LEVEL`` env var.
    """
    if "NEMO_LOG_LEVEL" in os.environ:
        yield
        return

    class _ErrorOnly(logging.Filter):
        def filter(self, record):
            return record.levelno >= logging.ERROR

    nemo_logger = logging.getLogger("nemo_logger")
    filt = _ErrorOnly()
    nemo_logger.addFilter(filt)
    nemo_logger.setLevel(logging.ERROR)
    yield
    nemo_logger.removeFilter(filt)
