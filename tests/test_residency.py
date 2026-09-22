import threading

import pytest
import torch

from tests.utils import skipif_limited_offload_support
from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef.tensor import OffloadedTensor, TensorResidencyPool


@skipif_limited_offload_support
def test_residency_lease_reuses_loaded_value(monkeypatch, tmp_path):
    source = OffloadedTensor.from_original_tensor(
        torch.arange(8), "resident", offload_dir=tmp_path
    )
    original_reload = source._reload_unmanaged
    reload_count = 0

    def counted_reload(*args, **kwargs):
        nonlocal reload_count
        reload_count += 1
        return original_reload(*args, **kwargs)

    monkeypatch.setattr(source, "_reload_unmanaged", counted_reload)
    with TensorResidencyPool() as pool:
        with pool.acquire(source) as first:
            assert torch.equal(first, torch.arange(8))
        with pool.acquire(source) as second:
            assert second is first
    assert reload_count == 1


@skipif_limited_offload_support
def test_read_write_lease_flushes_changes(tmp_path):
    source = OffloadedTensor.from_original_tensor(
        torch.zeros(4), "dirty", offload_dir=tmp_path
    )
    with TensorResidencyPool() as pool:
        with pool.acquire(source, mode="read_write") as value:
            value.add_(3)
        pool.flush(source)
    assert torch.equal(source.reload(), torch.full((4,), 3.0))


@skipif_limited_offload_support
def test_prefetch_overlaps_work(monkeypatch, tmp_path):
    source = OffloadedTensor.from_original_tensor(
        torch.ones(4), "prefetch", offload_dir=tmp_path
    )
    original_reload = source._reload_unmanaged
    load_started = threading.Event()
    allow_load = threading.Event()

    def controlled_reload(*args, **kwargs):
        load_started.set()
        assert allow_load.wait(timeout=5)
        return original_reload(*args, **kwargs)

    monkeypatch.setattr(source, "_reload_unmanaged", controlled_reload)
    with TensorResidencyPool() as pool:
        handle = pool.prefetch(source)
        assert load_started.wait(timeout=5)
        assert not handle.done()
        allow_load.set()
        assert torch.equal(handle.wait(), torch.ones(4))


@skipif_limited_offload_support
def test_completed_prefetches_obey_budget(tmp_path):
    first = OffloadedTensor.from_original_tensor(
        torch.ones(4), "prefetched_first", offload_dir=tmp_path
    )
    second = OffloadedTensor.from_original_tensor(
        torch.ones(4), "prefetched_second", offload_dir=tmp_path
    )
    with TensorResidencyPool(max_cached_bytes=16) as pool:
        pool.prefetch(first).wait()
        assert pool.is_resident(first)
        pool.prefetch(second).wait()
        assert pool.resident_bytes <= 16
        assert not pool.is_resident(first)
        assert pool.is_resident(second)


@skipif_limited_offload_support
def test_oversized_prefetch_delivers_value_without_caching_it(
    caplog, monkeypatch, tmp_path
):
    source = OffloadedTensor.from_original_tensor(
        torch.ones(8), "oversized", offload_dir=tmp_path
    )
    with (
        caplog.at_level("DEBUG", logger="torch_to_nnef.tensor.residency"),
        TensorResidencyPool(max_cached_bytes=16) as pool,
    ):
        completion_observed = threading.Event()
        original_complete = pool._complete_prefetch

        def observed_complete(*args, **kwargs):
            original_complete(*args, **kwargs)
            completion_observed.set()

        monkeypatch.setattr(pool, "_complete_prefetch", observed_complete)
        handle = pool.prefetch(source)
        assert completion_observed.wait(timeout=5)
        assert pool.resident_bytes == 0
        assert not pool.is_resident(source)
        value = handle.wait()
        assert torch.equal(value, torch.ones(8))
        assert pool.resident_bytes == 0
        assert not pool.is_resident(source)
    assert "delivering it without cache retention" in caplog.text


@skipif_limited_offload_support
def test_oversized_lease_is_pinned_only_for_lease_lifetime(caplog, tmp_path):
    source = OffloadedTensor.from_original_tensor(
        torch.ones(8), "oversized_lease", offload_dir=tmp_path
    )
    with (
        caplog.at_level("DEBUG", logger="torch_to_nnef.tensor.residency"),
        TensorResidencyPool(max_cached_bytes=16) as pool,
    ):
        with pool.acquire(source) as value:
            assert torch.equal(value, torch.ones(8))
            assert pool.is_resident(source)
        assert pool.resident_bytes == 0
        assert not pool.is_resident(source)
    assert "keeping it until its final lease ends" in caplog.text


@skipif_limited_offload_support
def test_close_rejects_new_work_once_shutdown_starts(monkeypatch, tmp_path):
    source = OffloadedTensor.from_original_tensor(
        torch.ones(4), "closing", offload_dir=tmp_path
    )
    pool = TensorResidencyPool()
    flush_started = threading.Event()
    allow_flush = threading.Event()
    original_flush = pool.flush

    def controlled_flush(*args, **kwargs):
        flush_started.set()
        assert allow_flush.wait(timeout=5)
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(pool, "flush", controlled_flush)
    close_thread = threading.Thread(target=pool.close)
    close_thread.start()
    assert flush_started.wait(timeout=5)
    with pytest.raises(T2NErrorMisuse, match="closing"):
        pool.prefetch(source)
    with (
        pytest.raises(T2NErrorMisuse, match="closing"),
        pool.acquire(source),
    ):
        pass
    allow_flush.set()
    close_thread.join(timeout=5)
    assert not close_thread.is_alive()


@skipif_limited_offload_support
def test_scope_transparently_reuses_offloaded_value(monkeypatch, tmp_path):
    source = OffloadedTensor.from_original_tensor(
        torch.arange(4), "transparent", offload_dir=tmp_path
    )
    original_reload = source._reload_unmanaged
    reload_count = 0

    def counted_reload(*args, **kwargs):
        nonlocal reload_count
        reload_count += 1
        return original_reload(*args, **kwargs)

    monkeypatch.setattr(source, "_reload_unmanaged", counted_reload)
    with TensorResidencyPool() as pool, pool.scope():
        torch.testing.assert_close(source + 1, torch.arange(4) + 1)
        torch.testing.assert_close(source * 2, torch.arange(4) * 2)
    assert reload_count == 1


@skipif_limited_offload_support
def test_scope_device_does_not_mutate_tensor_target(tmp_path):
    source = OffloadedTensor.from_original_tensor(
        torch.arange(4), "scope_device", offload_dir=tmp_path
    )
    assert source.target_device.type == "cpu"
    with TensorResidencyPool() as pool, pool.scope(device="meta"):
        assert source.reload().device.type == "meta"
    assert source.target_device.type == "cpu"


@skipif_limited_offload_support
def test_failed_prefetch_still_closes_pool(monkeypatch, tmp_path):
    source = OffloadedTensor.from_original_tensor(
        torch.ones(4), "failed", offload_dir=tmp_path
    )

    def fail_reload(*args, **kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(source, "_reload_unmanaged", fail_reload)
    pool = TensorResidencyPool()
    with pytest.raises(RuntimeError, match="load failed"), pool:
        pool.prefetch(source)
    with pytest.raises(T2NErrorMisuse, match="closed"):
        pool.prefetch(source)


@skipif_limited_offload_support
def test_budget_evicts_least_recently_used_value(tmp_path):
    first = OffloadedTensor.from_original_tensor(
        torch.ones(4), "first", offload_dir=tmp_path
    )
    second = OffloadedTensor.from_original_tensor(
        torch.ones(4), "second", offload_dir=tmp_path
    )
    with TensorResidencyPool(max_cached_bytes=16) as pool:
        with pool.acquire(first):
            pass
        with pool.acquire(second):
            assert not pool.is_resident(first)
            assert pool.is_resident(second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_cached_bytes": -1}, "non-negative"),
        ({"max_workers": 0}, "at least one"),
    ],
)
def test_invalid_pool_configuration(kwargs, message):
    with pytest.raises(T2NErrorMisuse, match=message):
        TensorResidencyPool(**kwargs)


@pytest.mark.parametrize(
    "operation",
    [
        lambda pool, tensor: pool.acquire(tensor),
        lambda pool, tensor: pool.prefetch(tensor),
        lambda pool, tensor: pool.resolve(tensor),
        lambda pool, tensor: pool.flush(tensor),
        lambda pool, tensor: pool.evict(tensor),
    ],
)
def test_pool_rejects_regular_tensors_with_clear_error(operation):
    tensor = torch.ones(4)
    with (
        TensorResidencyPool() as pool,
        pytest.raises(
            T2NErrorMisuse,
            match="only manages OffloadedTensor values; got Tensor",
        ),
    ):
        operation(pool, tensor)
