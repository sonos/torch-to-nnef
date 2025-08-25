"""Helper to display in console nicely."""

import re
import typing as T


def striptags(data):
    p = re.compile(r"\[.*?\]")
    return p.sub("", data)


class Console:
    """Inspired by rich.console without the deps on rich.

    Only support linux/Mac TTY

    """

    COLORMAP = {"blue": "34", "grey82": "82", "yellow": "33", "red": "31"}

    def __init__(self, theme: T.Optional[T.Dict[str, str]]):
        """Initialize the Console with an optional theme.

        This constructor sets up the underlying printer. If the ``rich``
        library is available, it will be used; otherwise a simple fallback
        printing routine is installed.

        Args:
            theme: Optional mapping of color names to ANSI codes for custom
                styling. If ``None`` the default ``rich`` theme is used.

        """
        self.theme = theme
        try:
            # pylint: disable-next=import-outside-toplevel
            from rich.console import Console as rConsole

            # pylint: disable-next=import-outside-toplevel
            from rich.theme import Theme

            self.print = rConsole(theme=Theme(self.theme)).print  # type: ignore
        except ImportError:
            self.print = self._degraded_print  # type: ignore

    @staticmethod
    def _degraded_print(*args):
        print(striptags(" ".join(args)))

    def print(self, *args):
        text = " ".join(args)
        self.print(text)
