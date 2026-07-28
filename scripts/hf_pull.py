"""Best-effort pre-download of Hugging Face repos, shared by every caller.

Usage: python scripts/hf_pull.py <repo_id> [<repo_id> ...]

``snapshot_download``s each repo with backoff, retrying only transient errors
(429 rate limit + 5xx) and skipping permanent ones (401/403/404) immediately.
Best-effort in the strong sense: nothing that happens here fails the build. A
repo that cannot be fetched after retries is logged and skipped, so warming the
cache only ever helps and the caller that actually needs the model still fails
on its own terms.

The first attempt uses the default download path, so a healthy Xet backend is
used and stays exercised; only if that fails does the retry loop switch to
plain LFS. Xet's ``xet-read-token`` endpoint intermittently 404s, which is not
on the HF status page and is not auto-retried, so an outage degrades to LFS
instead of failing CI, rather than giving up the fast path for everyone.

There is one copy of this policy on purpose. It used to live in three places
(a CI-only script, ``packages/llm/tests/prefetch_hf_models.py``, and ``hf_pull``
in ``examples/bootstrap-uv.sh``); the Xet fallback was added to one of them, and
the next Xet outage took the other two down. Callers:

- ``.github/workflows/ci-core.yml`` (the zoo suite's albert)
- ``packages/llm/tests/prefetch_hf_models.py`` (the LLM suite's tiny models)
- ``hf_pull`` in ``examples/bootstrap-uv.sh`` (each example's run.sh)

CI runs this before pytest to dodge Hugging Face's per-IP 429 on shared
runners, since the suites download models at collection time.

Annotations are quoted so this stays importable on the oldest interpreter any
caller uses; the examples pin python 3.11 but are not guaranteed to.
"""

import sys
import time

from huggingface_hub import constants, snapshot_download

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
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status = getattr(getattr(cur, "response", None), "status_code", None)
        if status is not None:
            return status
        cur = cur.__cause__ or cur.__context__
    return None


def _fall_back_to_lfs(repo_id: str, exc: BaseException) -> None:
    """Switch this process off Xet after a failed first attempt.

    `file_download` reads `constants.HF_HUB_DISABLE_XET` per call, so
    setting it here is enough; no subprocess or re-import is needed, unlike
    the shell caller this replaced.
    """
    constants.HF_HUB_DISABLE_XET = True
    print(
        f"[hf_pull] default (Xet) download failed for {repo_id}: "
        f"{type(exc).__name__}; falling back to the plain-LFS path"
    )


def prefetch(repo_id: str, attempts: int = 5, base_delay: float = 3.0) -> None:
    """Pull one repo into the local cache. Never raises on an HF failure."""
    # Try the default path first, so a working Xet is used and keeps being
    # exercised. The catch is broad only to decide whether to fall back: the
    # Xet backend is a Rust extension raising its own types (a ConnectionError
    # here, a RuntimeError on the next call), so it cannot be enumerated.
    # Nothing is swallowed: a failure that is not specific to Xet recurs on
    # the LFS path below, and every attempt prints what it saw.
    if not constants.HF_HUB_DISABLE_XET:
        try:
            snapshot_download(repo_id)
            print(f"[hf_pull] ok: {repo_id}")
            return
        except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
            _fall_back_to_lfs(repo_id, exc)

    for i in range(1, attempts + 1):
        try:
            snapshot_download(repo_id)
            print(f"[hf_pull] ok: {repo_id}")
            return
        # Deliberately broad, for the same reason as the Xet attempt
        # above: the failures that reach here are transport-level and
        # come from whichever backend huggingface_hub is built on, so
        # enumerating their types is whack-a-mole. Observed so far: the
        # Xet extension's ConnectionError/RuntimeError, and httpx's
        # RemoteProtocolError when the CDN truncates a response
        # mid-body. None is an HfHubHTTPError, all are retryable, and
        # each one that escapes fails a step whose entire contract is to
        # warm a cache without ever failing the build.
        #
        # Nothing is hidden: `_http_status` still short-circuits the
        # permanent statuses, and anything it cannot classify is printed
        # with its type name and retried, so a genuine bug shows up in
        # the log and then loses the cache warm rather than the job.
        except Exception as e:  # noqa: BLE001  pylint: disable=broad-except
            status = _http_status(e)
            if status in PERMANENT:
                print(f"[hf_pull] skip {repo_id}: HTTP {status} (permanent)")
                return
            label = f"HTTP {status}" if status else type(e).__name__
            print(f"[hf_pull] attempt {i}/{attempts} {repo_id}: {label}")
            if i == attempts:
                print(f"[hf_pull] giving up on {repo_id} (best-effort)")
                return
            time.sleep(base_delay * 2 ** (i - 1))


def main(argv: "list[str]") -> int:
    if not argv:
        print("usage: python scripts/hf_pull.py <repo_id> [<repo_id> ...]")
        return 2
    for repo_id in argv:
        prefetch(repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
