"""Unit tests for LLMExporter.load's transient-failure retry logic.

Network-free: `_load_exporter_from` is monkeypatched, so these exercise only
the retry/backoff branching, not any Hugging Face download.
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
