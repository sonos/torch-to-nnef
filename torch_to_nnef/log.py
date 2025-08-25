"""Simple helper to manage torch_to_nnef logging."""

import logging as log


def set_lib_log_level(log_level: int):
    """Set torch_to_nnef log_level."""
    logger = log.getLogger("torch_to_nnef")
    logger.setLevel(log_level)


def init_log():
    """Default init log handlers for torch_to_nnef clis."""
    _stream_log = log.StreamHandler()
    try:
        # use rich handler if availlable
        # pylint: disable-next=import-outside-toplevel
        from rich.logging import RichHandler

        _stream_log = RichHandler()
    except ImportError:
        pass

    log.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s "
        "[%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=log.INFO,
        handlers=[_stream_log],
    )
    return log
