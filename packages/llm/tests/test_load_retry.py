"""Unit tests for LLMExporter.load's local-first + retry behavior.

Network-free: `_load_exporter_from` is monkeypatched, so these exercise only
the local-first short-circuit and the transient-failure retry/backoff branching,
not any Hugging Face download. The fakes branch on ``local_files_only`` to tell
the cache-only attempt apart from the network attempts.
"""

import pytest

from torch_to_nnef.exceptions import T2NErrorRuntime
from torch_to_nnef_llm import exporter as exp_mod
from torch_to_nnef_llm.exporter import LLMExporter, _hf_http_status


def _oserror_wrapping_status(status: int) -> OSError:
    """An OSError whose cause chain carries an HfHubHTTPError-like status.

    Mirrors how transformers re-raises a rate-limited HF fetch: the HTTP error
    is the __cause__, not the outermost exception.
    """

    class _Resp:
        status_code = status

    inner = Exception("http error")
    inner.response = _Resp()  # type: ignore[attr-defined]
    try:
        try:
            raise inner
        except Exception as e:
            raise OSError("We couldn't connect to huggingface.co") from e
    except OSError as oserr:
        return oserr


def test_hf_http_status_walks_cause_chain():
    assert _hf_http_status(_oserror_wrapping_status(429)) == 429
    assert _hf_http_status(_oserror_wrapping_status(404)) == 404
    assert _hf_http_status(OSError("no status here")) is None


def test_load_uses_local_cache_without_network(monkeypatch):
    """A cached model loads via the local-first attempt, with no network."""
    monkeypatch.setattr(
        exp_mod.time, "sleep", lambda s: pytest.fail("should not sleep")
    )
    sentinel = object()
    calls = {"local": 0, "net": 0}

    def fake_load(**kwargs):
        if kwargs.get("local_files_only"):
            calls["local"] += 1
            return sentinel  # cache hit
        calls["net"] += 1
        pytest.fail("network load must not run when the model is cached")

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake_load)
    out = LLMExporter.load("dummy/slug")
    assert out is sentinel
    assert calls == {"local": 1, "net": 0}


def test_load_retries_transient_then_succeeds(monkeypatch):
    sleeps: list = []
    monkeypatch.setattr(exp_mod.time, "sleep", lambda s: sleeps.append(s))
    net = {"n": 0}
    sentinel = object()

    def fake_load(**kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")  # local-first miss
        net["n"] += 1
        if net["n"] < 3:
            raise _oserror_wrapping_status(429)
        return sentinel

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake_load)
    out = LLMExporter.load("dummy/slug", hf_download_n_retries=5)
    assert out is sentinel
    assert net["n"] == 3  # network: 2 failures + 1 success
    assert len(sleeps) == 2  # two retries before the success


def test_load_gives_up_after_n_retries(monkeypatch):
    monkeypatch.setattr(exp_mod.time, "sleep", lambda s: None)
    net = {"n": 0}

    def fake_load(**kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        net["n"] += 1
        raise _oserror_wrapping_status(429)

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake_load)
    with pytest.raises(T2NErrorRuntime):
        LLMExporter.load("dummy/slug", hf_download_n_retries=2)
    assert net["n"] == 3  # network: initial attempt + 2 retries


def test_load_does_not_retry_permanent_error(monkeypatch):
    monkeypatch.setattr(exp_mod.time, "sleep", lambda s: None)
    net = {"n": 0}

    def fake_load(**kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        net["n"] += 1
        raise _oserror_wrapping_status(404)

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake_load)
    with pytest.raises(T2NErrorRuntime):
        LLMExporter.load("dummy/slug", hf_download_n_retries=5)
    assert net["n"] == 1  # 404 is not retried


def test_load_disabled_when_zero_retries(monkeypatch):
    monkeypatch.setattr(exp_mod.time, "sleep", lambda s: None)
    net = {"n": 0}

    def fake_load(**kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        net["n"] += 1
        raise _oserror_wrapping_status(429)

    monkeypatch.setattr(exp_mod, "_load_exporter_from", fake_load)
    with pytest.raises(T2NErrorRuntime):
        LLMExporter.load("dummy/slug", hf_download_n_retries=0)
    assert net["n"] == 1  # no retries when disabled
