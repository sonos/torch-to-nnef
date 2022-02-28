import typing as T

from rich.console import Console as rConsole
from rich.theme import Theme


class Console:
    """
    Inspired by rich.console without the deps on rich

    Only support linux/Mac TTY

    """

    COLORMAP = {"blue": "34", "grey82": "82", "yellow": "33", "red": "31"}

    def __init__(self, theme: T.Optional[T.Dict[str, str]]):
        self.theme = theme
        self._c = rConsole(theme=Theme(self.theme))

    def print(self, *args):
        text = " ".join(args)
        self._c.print(text)
