"""Spec builders for the activation op group."""

import typing as T
from functools import partial

import torch
import torch.nn.functional as F
from hypothesis import strategies as st

from torch_to_nnef.inference_target.tract import TractCheckTolerance

from ...wrapper import (
    TensorFnPrimitive,
    UnaryPrimitive,
)
from ..inputs import Interval, tensor_st
from ..joint import (
    reduction_dim_st,
)
from ..shapes import (
    shape_st,
)
from ._common import (
    OpSample,
    OpSpec,
    _unary_sample_st,
)

# nn.functional activations are mostly unary on a tensor with a fixed
# bounded output (sigmoid, hardsigmoid, hardtanh) or saturating output
# (gelu, silu, mish). Domain bounded to keep results numerically stable.
_ACT_DOMAIN = Interval(-30.0, 30.0)


def _activation_specs() -> T.List[OpSpec]:

    EXACT = TractCheckTolerance.EXACT
    VERY = TractCheckTolerance.VERY
    SUPER = TractCheckTolerance.SUPER

    # Pure unary activations -- no kwargs, just elementwise.
    pure_unary: T.List[T.Tuple[str, T.Callable, TractCheckTolerance]] = [
        ("relu", F.relu, EXACT),
        ("sigmoid", F.sigmoid, VERY),
        ("gelu", F.gelu, VERY),
        ("silu", F.silu, VERY),
        # hardswish = x * relu6(x+3) / 6 -- multi-op chain, ULP-level
        # divergence between PyTorch and tract is normal, EXACT is too
        # strict.
        ("hardswish", F.hardswish, TractCheckTolerance.APPROXIMATE),
        # hardsigmoid = clamp((x+3)/6, 0, 1) -- one mul + one add +
        # min/max chain; small ULP drift, APPROXIMATE matches the
        # tolerance picked for hardswish.
        ("hardsigmoid", F.hardsigmoid, TractCheckTolerance.APPROXIMATE),
        # mish = x * tanh(softplus(x)) -- saturating, slow tails.
        ("mish", F.mish, VERY),
        ("selu", F.selu, VERY),
        ("relu6", F.relu6, EXACT),
        ("erf", torch.erf, VERY),
    ]
    specs: T.List[OpSpec] = [
        OpSpec(
            name=name,
            sample_st=_unary_sample_st(op, domain=_ACT_DOMAIN),
            tolerance=tol,
        )
        for name, op, tol in pure_unary
    ]

    # Activations with a single optional kwarg pinned to its default; the
    # kwarg-sweep variants (`-broad`) are added separately below.
    leaky_relu = partial(F.leaky_relu, negative_slope=0.01)
    elu_default = partial(F.elu, alpha=1.0)
    hardtanh_default = partial(F.hardtanh, min_val=-1.0, max_val=1.0)
    softplus_default = partial(F.softplus, beta=1.0, threshold=20.0)

    specs.extend(
        [
            OpSpec(
                name="leaky_relu",
                sample_st=_unary_sample_st(leaky_relu, domain=_ACT_DOMAIN),
                tolerance=EXACT,
            ),
            OpSpec(
                name="elu",
                sample_st=_unary_sample_st(elu_default, domain=_ACT_DOMAIN),
                tolerance=VERY,
            ),
            OpSpec(
                name="hardtanh",
                sample_st=_unary_sample_st(
                    hardtanh_default, domain=_ACT_DOMAIN
                ),
                tolerance=EXACT,
            ),
            OpSpec(
                name="softplus",
                sample_st=_unary_sample_st(
                    softplus_default, domain=_ACT_DOMAIN
                ),
                tolerance=SUPER,
            ),
        ]
    )

    # threshold(input, threshold, value): elementwise gating with two
    # scalar args. Sweep both inside the input domain so we get a healthy
    # mix of below-threshold and above-threshold positions.
    @st.composite
    def _threshold_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        thresh = draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        value = draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                partial(F.threshold, threshold=thresh, value=value)
            ),
        )

    specs.append(
        OpSpec(
            name="threshold",
            sample_st=_threshold_sample(),
            tolerance=EXACT,
        )
    )

    # ---- kwarg-broad variants ----
    # gelu has an `approximate` kwarg (`"none"` (default) or
    # `"tanh"`). Per the PyTorch doc, "tanh" uses an approximate formula
    # that often matches different cuda kernels.
    @st.composite
    def _gelu_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        approximate = draw(st.sampled_from(["none", "tanh"]))
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(partial(F.gelu, approximate=approximate)),
        )

    @st.composite
    def _leaky_relu_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        # Negative slopes from PyTorch examples: 0.01 (default), 0.1, 0.2.
        slope = draw(
            st.floats(
                min_value=0.001,
                max_value=0.5,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(partial(F.leaky_relu, negative_slope=slope)),
        )

    @st.composite
    def _elu_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        # alpha controls the negative-side saturation; 1.0 is default but
        # other values are common in tuned models.
        alpha = draw(
            st.floats(
                min_value=0.1,
                max_value=3.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(partial(F.elu, alpha=alpha)),
        )

    @st.composite
    def _hardtanh_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        # min_val < max_val by construction.
        a = draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        b = draw(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        if a > b:
            a, b = b, a
        if b - a < 1e-2:
            b = a + 1.0
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(partial(F.hardtanh, min_val=a, max_val=b)),
        )

    @st.composite
    def _softplus_kwarg_sample(draw) -> OpSample:
        shape = draw(shape_st(min_rank=0, max_rank=4))
        x = draw(
            tensor_st(shape, torch.float32, finite=True, domain=_ACT_DOMAIN)
        )
        # softplus has beta and threshold; t2n's softplus emitter only
        # supports beta=1 (raises NotImplemented otherwise -- see
        # `torch_to_nnef/op/aten/activation.py`). We sweep
        # threshold (default 20) within a safe range; beta stays at 1.
        threshold = draw(
            st.floats(
                min_value=5.0,
                max_value=50.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        return OpSample(
            inputs=(x,),
            module=UnaryPrimitive(
                partial(F.softplus, beta=1.0, threshold=threshold)
            ),
        )

    specs.extend(
        [
            OpSpec(
                name="gelu-broad",
                sample_st=_gelu_kwarg_sample(),
                tolerance=VERY,
            ),
            OpSpec(
                name="leaky_relu-broad",
                sample_st=_leaky_relu_kwarg_sample(),
                tolerance=EXACT,
            ),
            OpSpec(
                name="elu-alpha",
                sample_st=_elu_kwarg_sample(),
                tolerance=VERY,
            ),
            OpSpec(
                name="hardtanh-broad",
                sample_st=_hardtanh_kwarg_sample(),
                tolerance=EXACT,
            ),
            OpSpec(
                name="softplus-broad",
                sample_st=_softplus_kwarg_sample(),
                tolerance=SUPER,
            ),
        ]
    )

    return specs


def _softmax_dim_sample_st(op_name: str) -> st.SearchStrategy[OpSample]:
    """Softmax / log_softmax with a random valid dim."""

    @st.composite
    def _draw(draw) -> OpSample:
        rank = draw(st.integers(min_value=1, max_value=4))
        shape = tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=6),
                    min_size=rank,
                    max_size=rank,
                )
            )
        )
        dim = draw(reduction_dim_st(rank))
        # Bound inputs to keep softmax outputs stable; large positives
        # all cluster near 1.0 and large negatives near 0.0, which is
        # numerically unstable for both PyTorch and tract.
        x = draw(
            tensor_st(
                shape,
                torch.float32,
                finite=True,
                domain=Interval(-10.0, 10.0),
            )
        )
        return OpSample(
            inputs=(x,),
            module=TensorFnPrimitive(op_name, kwargs={"dim": dim}),
        )

    return _draw()


def _softmax_specs() -> T.List[OpSpec]:
    return [
        OpSpec(
            name="softmax",
            sample_st=_softmax_dim_sample_st("softmax"),
            tolerance=TractCheckTolerance.VERY,
        ),
        OpSpec(
            name="log_softmax",
            sample_st=_softmax_dim_sample_st("log_softmax"),
            tolerance=TractCheckTolerance.VERY,
        ),
    ]


# Selector / indexing specs

SPECS = (
    *_activation_specs(),
    *_softmax_specs(),
)
