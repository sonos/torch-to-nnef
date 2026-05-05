"""Hypothesis profiles and helpers for the proptest layer.

Three profiles:
  - dev (default for local): 25 examples, normal shrinking, default DB.
  - ci: 25 examples, derandomize=True, deadline disabled, shrink phase capped.
  - nightly: 200 examples, no shrink cap.

Profile selection (in priority order):
  1. ``T2N_HYP_PROFILE`` env var.
  2. ``HYPOTHESIS_PROFILE`` env var.
  3. ``ci`` if ``CI`` is truthy (GitHub Actions / most CI runners set this).
  4. ``dev``.

The CI fallback is belt-and-suspenders for direct ``pytest`` invocations on
CI runners that don't go through tox (which sets ``T2N_HYP_PROFILE`` itself).
"""

import os

from hypothesis import HealthCheck, Phase, settings

_COMMON_SUPPRESSIONS = (
    HealthCheck.too_slow,
    HealthCheck.data_too_large,
    HealthCheck.large_base_example,
)


def _truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def _select_profile() -> str:
    explicit = os.environ.get("T2N_HYP_PROFILE") or os.environ.get(
        "HYPOTHESIS_PROFILE"
    )
    if explicit:
        return explicit
    ci_flag = os.environ.get("CI", "")
    if ci_flag and _truthy(ci_flag):
        return "ci"
    return "dev"


def register_profiles() -> None:
    """Register the dev/ci/nightly profiles. Idempotent."""
    settings.register_profile(
        "dev",
        max_examples=25,
        deadline=None,
        suppress_health_check=_COMMON_SUPPRESSIONS,
    )
    settings.register_profile(
        "ci",
        max_examples=25,
        deadline=None,
        derandomize=True,
        phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink),
        suppress_health_check=_COMMON_SUPPRESSIONS,
    )
    settings.register_profile(
        "nightly",
        max_examples=200,
        deadline=None,
        suppress_health_check=_COMMON_SUPPRESSIONS,
    )
    settings.load_profile(_select_profile())
