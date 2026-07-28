"""How a declared operator name relates to the name torch really emits.

Shared, because two checks need it and they used to disagree. The
attribution test asks "does this spec trace the operator it claims", and
the gap guard asks "is that operator registered yet". Both have to know
that `conv2d` arrives as `_convolution` and `gamma` as `_standard_gamma`,
and a guard that only knows the declared spelling silently never fires.
"""

import typing as T

#: Declared name -> the name(s) torch actually emits in the trace.
#:
#: Two reasons a declared name legitimately never appears:
#:   - torch renames the op on the way into the graph (`conv2d` is
#:     dispatched through `aten::_convolution`),
#:   - the op is a composite that PyTorch decomposes before tracing, so
#:     only its constituents survive.
#: Either way the spec is still testing the declared op, which is the
#: name the support page lists, so the declaration is the useful one.
KNOWN_TRACE_RENAMES: T.Dict[str, T.Tuple[str, ...]] = {
    "conv1d": ("_convolution",),
    "conv2d": ("_convolution",),
    "conv3d": ("_convolution",),
    # tanhshrink(x) == x - tanh(x), decomposed at trace time.
    "tanhshrink": ("sub", "tanh"),
    # `Tensor.fill_` on a traced tensor lands as a `full_like` + copy.
    "fill": ("full_like",),
    # `torch.unique_consecutive(x, dim=...)` dispatches to the
    # dim-specific C++ op but keeps the generic name in the trace.
    "unique_dim_consecutive": ("unique_consecutive",),
    # There is no public `torch.gamma`: the sampler is
    # `_standard_gamma`, which the page's source grep drops for being
    # `_`-prefixed, leaving the bare row name behind.
    "gamma": ("_standard_gamma",),
    # conj on a complex tensor goes through the lazy-conjugate path.
    "conj": ("_conj", "resolve_conj", "view_as_real", "view_as_complex"),
}


def registry_lookup_names(declared: str) -> T.Tuple[str, ...]:
    """Every registry key an emitter for `declared` could land under.

    `aten_to_nnef_tensor_and_ops` strips a single trailing underscore
    before the lookup, which is how the page's merged in-place rows
    (`uniform` for `aten::uniform_`) resolve. Renamed operators keep the
    name torch dispatches, underscore prefix and all.
    """
    names = {declared, *KNOWN_TRACE_RENAMES.get(declared, ())}
    return tuple(
        sorted(
            n[:-1] if n.endswith("_") and not n.endswith("__") else n
            for n in names
        )
    )
