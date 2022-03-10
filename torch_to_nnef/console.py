import re
import typing as T


def striptags(data):
    p = re.compile(r'\[.*?\]')
    return p.sub('', data)


class Console:
    """
    Inspired by rich.console without the deps on rich

    Only support linux/Mac TTY

    """

    COLORMAP = {"blue": "34", "grey82": "82", "yellow": "33", "red": "31"}

    def __init__(self, theme: T.Optional[T.Dict[str, str]]):
        self.theme = theme
        try:
            from rich.console import Console as rConsole
            from rich.theme import Theme

            self.print = rConsole(theme=Theme(self.theme)).print
        except ImportError:
            self.print = self._degraded_print

    @staticmethod
    def _degraded_print(*args):
        print(striptags(" ".join(args)))

    def print(self, *args):
        text = " ".join(args)
        self.print(text)
