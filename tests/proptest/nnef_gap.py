"""Assert that a declared t2n translation gap is still a gap.

A spec carrying `OpSpec.nnef_gap` exists to measure other exporters on an
operator we cannot translate. The risk with any such marker is rot: the
day someone registers the emitter, a spec that merely *skipped* would go
on reporting the operator as unsupported, and the support page with it.

So the marker is verified rather than trusted. This module attempts the
real export (and, if that succeeds, the real tract run) and reports which
stage actually failed, so the driver can compare it against the declared
one. Both directions are errors:

  - the gap closed (nothing failed): the spec must become a normal one
  - it failed somewhere else: the declared stage is describing the wrong
    thing, which would mislead anyone using it to plan work

The stage vocabulary earns its keep on both counts. Some operators never
reach the emitter lookup at all (the constant-folding pass runs them
first), and a couple crash with a bare `TypeError` instead of a
`T2NError`, which is a bug in our error handling regardless of whether we
ever translate the operator.

Cost is low. `no-emitter`, the common case, raises during graph
translation, long before tract is invoked.
"""

import tempfile
import typing as T
from pathlib import Path

import torch
from torch import nn

from torch_to_nnef.exceptions import T2NError, T2NErrorMissingOpEmitter
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.inference_target.tract import build_io
from torch_to_nnef.log import log

from .comparator import ProptestComparatorError, run_tract
from .op_specs import NnefGap, NnefGapStage


class NnefGapMismatch(AssertionError):
    """Raised when a spec's declared gap is not what actually happens."""


def observe_nnef_gap(
    model: nn.Module,
    inputs: T.Tuple[torch.Tensor, ...],
    inference_target: TractNNEF,
) -> T.Optional[NnefGapStage]:
    """Export and run one example, returning the stage that failed.

    Returns `None` when the whole pipeline succeeds, which for a spec
    declaring a gap means the gap is closed.

    Numeric agreement is deliberately *not* checked: this answers "can
    t2n produce a graph tract will run", and a spec that gets that far
    has outgrown this path and belongs on the normal driver (with
    `xfail_reason` if its values still disagree).
    """
    model = model.eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        nnef_path = tmp / "model.nnef"
        inputs_npz = tmp / "inputs.npz"
        outputs_ref_npz = tmp / "outputs_ref.npz"
        outputs_act_npz = tmp / "outputs_act.npz"
        # Outside the try on purpose: this only runs the module under
        # torch. A failure here is a bad draw in the spec's own
        # strategy, and swallowing it as `raw-error` would turn a broken
        # spec into a passing one.
        input_names, output_names = build_io(
            model,
            inputs,
            input_bundle_path=inputs_npz,
            output_bundle_path=outputs_ref_npz,
        )
        try:
            exported = export_model_to_nnef(
                model=model,
                args=inputs,
                file_path_export=nnef_path,
                compression_level=0,
                input_names=input_names,
                output_names=output_names,
                log_level=log.ERROR,
                inference_target=inference_target,
                allow_same_io_names=True,
            )
        except T2NErrorMissingOpEmitter:
            return NnefGapStage.NO_EMITTER
        except T2NError:
            return NnefGapStage.EXPORT_ERROR
        except Exception:  # noqa: BLE001  pylint: disable=broad-except
            # Not a refusal but a crash: the exporter was supposed to
            # turn every failure into a `T2NError` naming the operator.
            return NnefGapStage.RAW_ERROR
        try:
            run_tract(
                inference_target.tract_cli.run_save_outputs_cmd_str(
                    exported, inputs_npz, outputs_act_npz
                )
            )
        except ProptestComparatorError:
            return NnefGapStage.TRACT_ERROR
    return None


def assert_nnef_gap(
    gap: NnefGap,
    spec_name: str,
    model: nn.Module,
    inputs: T.Tuple[torch.Tensor, ...],
    inference_target: TractNNEF,
) -> None:
    """Assert one drawn example fails exactly where the spec says it does.

    Raises:
        NnefGapMismatch: when the pipeline succeeds, or fails at a stage
            other than the declared one.
    """
    observed = observe_nnef_gap(model, inputs, inference_target)
    if observed == gap.stage:
        return
    tracked = f" (tracked by {gap.tracked_by})" if gap.tracked_by else ""
    if observed is None:
        raise NnefGapMismatch(
            f"spec {spec_name!r} declares a `{gap.stage.value}` t2n gap, "
            "but the model exported and ran under tract. The gap is "
            "closed: drop `nnef_gap` so the spec starts guarding the "
            "translation, and regenerate the support page.\n"
            f"declared reason: {gap.reason}{tracked}"
        )
    raise NnefGapMismatch(
        f"spec {spec_name!r} declares a `{gap.stage.value}` t2n gap but "
        f"failed at `{observed.value}` instead. Update `nnef_gap.stage`, "
        "or fix whatever moved the failure.\n"
        f"declared reason: {gap.reason}{tracked}"
    )
