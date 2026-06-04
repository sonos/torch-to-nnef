"""Best-effort prefetch of Hugging Face models to warm the CI cache.

Usage: python .github/scripts/hf_prefetch.py <repo_id> [<repo_id> ...]

``snapshot_download``s each repo with backoff, retrying only transient errors
(429 rate limit + 5xx) and skipping permanent ones (401/403/404) immediately.
Best-effort by design: a model it cannot fetch is logged and skipped, never
failing the step, so warming the cache can only help, never introduce a
failure. CI runs this before pytest to dodge Hugging Face's per-IP 429 on
shared runners (the suites download small models at collection time).
"""

import sys
import time

# Auth/missing (401/403/404, e.g. a gated repo without a token) will never
# succeed on retry, so skip those immediately. Everything else (429 rate limit,
# transient 5xx, or a connection error with no status) is worth retrying.
PERMANENT = {401, 403, 404}


def _http_status(exc: BaseException) -> int | None:
    """Find an HTTP status anywhere in the exception's cause chain.

    snapshot_download wraps a rate-limited metadata fetch in a
    LocalEntryNotFoundError whose cause is the real HfHubHTTPError, so the
    status we must branch on (e.g. 429) is not on the outermost exception.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status = getattr(getattr(cur, "response", None), "status_code", None)
        if status is not None:
            return status
        cur = cur.__cause__ or cur.__context__
    return None


def prefetch(repo_id: str, attempts: int = 5, base_delay: float = 3.0) -> None:
    from huggingface_hub import snapshot_download

    for i in range(1, attempts + 1):
        try:
            snapshot_download(repo_id)
            print(f"[prefetch] ok: {repo_id}")
            return
        except Exception as e:
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


def main(argv: list[str]) -> int:
    for repo_id in argv:
        prefetch(repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
