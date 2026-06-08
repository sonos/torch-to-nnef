"""Unit tests for the LLM loader's HF-rate-limit resilience.

Network-free: the hub interactions are monkeypatched/faked, so these exercise
only the retry/backoff branching in LLMExporter.load and the cache-first
snapshot resolution used by the device_map path, not any real download.
"""

import pytest

from torch_to_nnef.exceptions import T2NErrorRuntime
from torch_to_nnef_llm import exporter as exp_mod
from torch_to_nnef_llm.exporter import LLMExporter, _hf_http_status
from torch_to_nnef_llm.loader import _resolve_snapshot_dir


def _oserror_wrapping_status(status: int) -> OSError:
    """An OSError whose cause chain carries an HfHubHTTPError-like status.

    Mirrors how transformers re-raises a rate-limited HF fetch: the HTTP error
    is the __cause__, not the outermost exception.
    """

    class _Resp:
        status_code = status

    inner = Exception("http error")
    inner.response = _Resp()  # type: ignore[attr-defined]
    outer = OSError("We couldn't connect to huggingface.co")
    outer.__cause__ = inner  # same chain as `raise outer from inner`
    return outer


def test_hf_http_status_walks_cause_chain():
    assert _hf_http_status(_oserror_wrapping_status(429)) == 429
    assert _hf_http_status(_oserror_wrapping_status(404)) == 404
    assert _hf_http_status(OSError("no status here")) is None


class _FakeHub:
    """snapshot_download that records calls and can simulate a cache miss."""

    def __init__(self, cached: bool):
        self.cached = cached
        self.calls: list = []

    def snapshot_download(self, slug, local_files_only=False):
        from huggingface_hub.errors import LocalEntryNotFoundError

        self.calls.append(local_files_only)
        if local_files_only and not self.cached:
            raise LocalEntryNotFoundError("not cached")
        return f"/cache/{slug}"


def test_resolve_snapshot_cache_hit_makes_no_network_call():
    hub = _FakeHub(cached=True)
    assert _resolve_snapshot_dir("some/model", hub) == "/cache/some/model"
    assert hub.calls == [True]  # only the cache-only attempt, no /tree call


def test_resolve_snapshot_cache_miss_falls_back_to_network():
    hub = _FakeHub(cached=False)
    assert _resolve_snapshot_dir("some/model", hub) == "/cache/some/model"
    assert hub.calls == [True, False]  # cache-only attempt then network


def test_resolve_snapshot_propagates_non_cache_miss_error():
    """A genuine error (not a cache miss) must not be swallowed."""

    class _BoomHub:
        def snapshot_download(self, slug, local_files_only=False):
            raise PermissionError("cache dir not readable")

    with pytest.raises(PermissionError):
        _resolve_snapshot_dir("some/model", _BoomHub())


def test_load_retries_transient_then_succeeds(monkeypatch):
    sleeps: list = []
    monkeypatch.setattr(exp_mod.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}
    sentinel = object()

    def fake_load(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _oserror_wrapping_status(429)
        return sentinel

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake_load)
    out = LLMExporter.load("dummy/slug", hf_download_n_retries=5)
    assert out is sentinel
    assert calls["n"] == 3
    assert len(sleeps) == 2  # two retries before the success


def test_load_gives_up_after_n_retries(monkeypatch):
    monkeypatch.setattr(exp_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_load(**kwargs):
        calls["n"] += 1
        raise _oserror_wrapping_status(429)

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake_load)
    with pytest.raises(T2NErrorRuntime):
        LLMExporter.load("dummy/slug", hf_download_n_retries=2)
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_load_does_not_retry_permanent_error(monkeypatch):
    monkeypatch.setattr(exp_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_load(**kwargs):
        calls["n"] += 1
        raise _oserror_wrapping_status(404)

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake_load)
    with pytest.raises(T2NErrorRuntime):
        LLMExporter.load("dummy/slug", hf_download_n_retries=5)
    assert calls["n"] == 1  # 404 is not retried


def test_load_disabled_when_zero_retries(monkeypatch):
    monkeypatch.setattr(exp_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_load(**kwargs):
        calls["n"] += 1
        raise _oserror_wrapping_status(429)

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake_load)
    with pytest.raises(T2NErrorRuntime):
        LLMExporter.load("dummy/slug", hf_download_n_retries=0)
    assert calls["n"] == 1  # no retries when disabled


@pytest.mark.parametrize("trust", [True, False])
def test_load_forwards_trust_remote_code(monkeypatch, trust):
    """LLMExporter.load threads trust_remote_code down to the loader."""
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake)
    LLMExporter.load("dummy/slug", trust_remote_code=trust)
    assert captured["trust_remote_code"] is trust
