"""Small utilities shared by wrappers."""

import typing as T


def map_args_to_kwargs_by_names(
    args: T.Sequence[object], kwargs: T.Mapping[str, object], names: T.Sequence[str]
) -> dict:
    """Merge positional args into kwargs using a known parameter name order.

    Existing keys in kwargs take precedence over positional args.
    """
    call_kwargs = dict(kwargs)
    ai = 0
    for name in names:
        if name in call_kwargs:
            continue
        if ai < len(args):
            call_kwargs[name] = args[ai]
            ai += 1
    return call_kwargs

