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

# Worth retrying: 429 (rate limit) + transient 5xx. Auth/missing (401/403/404,
# e.g. a gated repo without a token) will never succeed on retry, so skip those
# immediately rather than burning the backoff.
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
            print(f"[prefetch] attempt {i}/{attempts} {repo_id}: HTTP {status}")
            if i == attempts:
                print(f"[prefetch] giving up on {repo_id} (best-effort)")
                return
            time.sleep(base_delay * 2 ** (i - 1))
        except Exception as e:  # offline/unknown: skip, test decides
            print(f"[prefetch] skip {repo_id}: {type(e).__name__}: {e}")
            return


def main(argv: list[str]) -> int:
    for repo_id in argv:
        prefetch(repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
