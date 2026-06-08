"""Pre-fetch the small Hugging Face models the LLM test suite downloads.

Run before pytest in CI to warm the shared HF cache (``~/.cache/huggingface``)
with a few retries and backoff, so the collection-time ``from_pretrained`` calls
hit the cache instead of racing Hugging Face's per-IP rate limit (HTTP 429) on
shared CI runners.

Best-effort for HF download failures: a model that cannot be fetched after
retries is logged and skipped (the test will still attempt its own download),
so this only helps. Only HF request errors are handled; an unexpected
(non-request) error is left to propagate rather than be masked.

The slugs mirror the *real-download* entries in ``torch_to_nnef_llm.config``;
the ``*_debug`` slugs there are synthetic local configs and are intentionally
absent. Keep this list in sync if the suite starts exercising a new hub model.
"""

import sys
import time

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

# Real HF repos pulled by tests/ (see torch_to_nnef_llm.config slug enums).
MODELS = [
    "yujiepan/llama-2-tiny-random",  # LlamaSlugs.DUMMY (the usual 429 victim)
    "HuggingFaceTB/SmolLM-135M",  # SmolSlugs.TINY
    "Qwen/Qwen3-0.6B",  # Qwen3Slugs.TINY
    "google/gemma-3-270m",  # Gemma3Slugs.TINY
]


# Auth/missing (401/403/404, e.g. a gated repo without a token) will never
# succeed on retry, so skip those immediately. Everything else (429 rate limit,
# transient 5xx, or a connection error with no status) is worth retrying.
PERMANENT = {401, 403, 404}


def _http_status(exc: BaseException) -> "int | None":
    """Find an HTTP status anywhere in the exception's cause chain.

    snapshot_download wraps a rate-limited metadata fetch in a
    LocalEntryNotFoundError whose cause is the real HfHubHTTPError, so the
    status we must branch on (e.g. 429) is not on the outermost exception.
    """
    seen: set = set()
    cur: "BaseException | None" = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status = getattr(getattr(cur, "response", None), "status_code", None)
        if status is not None:
            return status
        cur = cur.__cause__ or cur.__context__
    return None


def prefetch(repo_id: str, attempts: int = 5, base_delay: float = 3.0) -> None:
    for i in range(1, attempts + 1):
        try:
            snapshot_download(repo_id)
            print(f"[prefetch] ok: {repo_id}")
            return
        # HfHubHTTPError is the base of every HF HTTP error (429/5xx + auth/404,
        # and the LocalEntryNotFoundError that wraps a rate-limited fetch), and
        # imports without the requests/httpx backend (which the minimal prefetch
        # env lacks). A non-HF error is unexpected and is left to propagate.
        except HfHubHTTPError as e:
            status = _http_status(e)
            if status in PERMANENT:
                print(f"[prefetch] skip {repo_id}: HTTP {status} (permanent)")
                return
            label = f"HTTP {status}" if status else type(e).__name__
            print(f"[prefetch] attempt {i}/{attempts} {repo_id}: {label}")
            if i == attempts:
                print(f"[prefetch] giving up on {repo_id} (best-effort)")
                return
            time.sleep(base_delay * 2 ** (i - 1))


def main() -> int:
    for repo_id in MODELS:
        prefetch(repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
