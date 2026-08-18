"""Pre-fetch the small Hugging Face models the LLM test suite downloads.

Run before pytest in CI to warm the shared HF cache (``~/.cache/huggingface``),
so the collection-time ``from_pretrained`` calls hit the cache instead of
racing Hugging Face's per-IP rate limit (HTTP 429) on shared CI runners.

Only the model list lives here. The download policy (retry with backoff, skip
permanent failures, fall back to plain LFS when the Xet backend is out) is
shared with the other callers in ``scripts/hf_pull.py``. This file used to
carry its own near-identical copy, which is how it missed the Xet fallback
added to the examples path and sat one outage away from breaking CI.

The slugs mirror the *real-download* entries in ``torch_to_nnef_llm.config``;
the ``*_debug`` slugs there are synthetic local configs and are intentionally
absent. Keep this list in sync if the suite starts exercising a new hub model.
"""

import sys
from pathlib import Path

# Resolved from this file, not the cwd: CI invokes it from the repository root
# while the package's own tooling runs from `packages/llm`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from hf_pull import prefetch  # noqa: E402  (depends on the sys.path above)

# Real HF repos pulled by tests/ (see torch_to_nnef_llm.config slug enums).
MODELS = [
    "yujiepan/llama-2-tiny-random",  # LlamaSlugs.DUMMY (the usual 429 victim)
    "HuggingFaceTB/SmolLM-135M",  # SmolSlugs.TINY
    "Qwen/Qwen3-0.6B",  # Qwen3Slugs.TINY
    "google/gemma-3-270m",  # Gemma3Slugs.TINY
]


def main() -> int:
    for repo_id in MODELS:
        prefetch(repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
