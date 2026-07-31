"""How a declared operator name relates to the name torch really emits.

Shared, because two checks need it and they used to disagree. The
attribution test asks "does this spec trace the operator it claims", and
the gap guard asks "is that operator registered yet". Both have to know
that `conv2d` arrives as `_convolution`, and a guard that only knows the
declared spelling silently never fires.

The two relations are kept apart on purpose, because only one of them
supports a conclusion about the registry:

- a **dispatch rename** is the same operator under another name, so an
  emitter for it is an emitter for the declared operator;
- a **decomposition** is several *different* operators standing in for
  the declared one, and we translate most of them already. Reading those
  as evidence would report a composite gap as closed the moment anyone
  looked, since `sub` and `tanh` have been supported for years.

Attribution accepts either (both prove the spec exercises the operator).
The registry guard accepts only the first.
"""

import typing as T

#: Declared name -> the name torch dispatches it to. Same operator,
#: different spelling, so an emitter registers under the right-hand name
#: and *is* a translation of the declared one.
DISPATCH_RENAMES: T.Dict[str, T.Tuple[str, ...]] = {
    "conv1d": ("_convolution",),
    "conv2d": ("_convolution",),
    "conv3d": ("_convolution",),
    "convolution": ("_convolution",),
    # `torch.unique_consecutive(x, dim=...)` dispatches to the
    # dim-specific C++ op but keeps the generic name in the trace.
    "unique_dim_consecutive": ("unique_consecutive",),
    # There is no public `torch.gamma`: the sampler is
    # `_standard_gamma`, which the page's source grep drops for being
    # `_`-prefixed, leaving the bare row name behind.
    "gamma": ("_standard_gamma",),
    # Same page-normalisation shape: the dispatcher spelling is private,
    # but the row survived without the leading underscore.
    "index_put_impl_": ("_index_put_impl_",),
}

#: Declared name -> the constituents torch decomposes it into before
#: tracing. These are *other* operators, so they say nothing about
#: whether the declared one is translated.
DECOMPOSITIONS: T.Dict[str, T.Tuple[str, ...]] = {
    # tanhshrink(x) == x - tanh(x), decomposed at trace time.
    "tanhshrink": ("sub", "tanh"),
    # `Tensor.fill_` on a traced tensor lands as a `full_like` + copy.
    "fill": ("full_like",),
    # conj on a complex tensor goes through the lazy-conjugate path.
    "conj": ("_conj", "resolve_conj", "view_as_real", "view_as_complex"),
}

#: What the attribution test accepts: either relation proves the spec
#: really exercises the operator it declares.
KNOWN_TRACE_RENAMES: T.Dict[str, T.Tuple[str, ...]] = {
    name: DISPATCH_RENAMES.get(name, ()) + DECOMPOSITIONS.get(name, ())
    for name in {*DISPATCH_RENAMES, *DECOMPOSITIONS}
}


def _lookup_key(name: str) -> str:
    """The registry key `name` resolves to.

    `aten_to_nnef_tensor_and_ops` strips a single trailing underscore
    before the lookup, which is how the page's merged in-place rows
    (`uniform` for `aten::uniform_`) resolve. A dunder suffix is left
    alone, mirroring that same condition.
    """
    if name.endswith("_") and not name.endswith("__"):
        return name[:-1]
    return name


def registry_lookup_names(declared: str) -> T.Tuple[str, ...]:
    """Every registry key an emitter for `declared` could land under.

    Dispatch renames only. A decomposition's constituents are different
    operators that we may well translate already, so including them
    would turn "we support `sub`" into "the `tanhshrink` gap is closed".
    """
    names = {declared, *DISPATCH_RENAMES.get(declared, ())}
    return tuple(sorted(_lookup_key(n) for n in names))
