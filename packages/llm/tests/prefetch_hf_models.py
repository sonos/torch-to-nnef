"""Pre-fetch the small Hugging Face models the LLM test suite downloads.

Run before pytest in CI to warm the shared HF cache (``~/.cache/huggingface``)
with a few retries and backoff, so the collection-time ``from_pretrained`` calls
hit the cache instead of racing Hugging Face's per-IP rate limit (HTTP 429) on
shared CI runners.

Best-effort by design: a model that cannot be fetched is logged and skipped,
never failing the step (the test will still attempt its own download). This can
only help, never introduce a failure.

The slugs mirror the *real-download* entries in ``torch_to_nnef_llm.config``;
the ``*_debug`` slugs there are synthetic local configs and are intentionally
absent. Keep this list in sync if the suite starts exercising a new hub model.
"""

import sys
import time

# Real HF repos pulled by tests/ (see torch_to_nnef_llm.config slug enums).
MODELS = [
    "yujiepan/llama-2-tiny-random",  # LlamaSlugs.DUMMY (the usual 429 victim)
    "HuggingFaceTB/SmolLM-135M",  # SmolSlugs.TINY
    "Qwen/Qwen3-0.6B",  # Qwen3Slugs.TINY
    "google/gemma-3-270m",  # Gemma3Slugs.TINY
]


# Only these are worth retrying: 429 (rate limit) + transient 5xx. An auth or
# missing error (401/403/404, e.g. a gated repo without a token) will never
# succeed on retry, so skip it immediately rather than burning the backoff.
TRANSIENT = {429, 500, 502, 503, 504}


def prefetch(repo_id: str, attempts: int = 5, base_delay: float = 3.0) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    for i in range(1, attempts + 1):
        try:
            snapshot_download(repo_id)
            print(f"[prefetch] ok: {repo_id}")
            return
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status not in TRANSIENT:
                print(f"[prefetch] skip {repo_id}: HTTP {status} (permanent)")
                return
            print(
                f"[prefetch] attempt {i}/{attempts} for {repo_id} "
                f"failed: HTTP {status}"
            )
            if i == attempts:
                print(f"[prefetch] giving up on {repo_id} (best-effort)")
                return
            time.sleep(base_delay * 2 ** (i - 1))
        except Exception as e:  # offline/unknown: skip, test decides
            print(f"[prefetch] skip {repo_id}: {type(e).__name__}: {e}")
            return


def main() -> int:
    for repo_id in MODELS:
        prefetch(repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
