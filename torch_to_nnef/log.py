import logging as log


def init_log():
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
