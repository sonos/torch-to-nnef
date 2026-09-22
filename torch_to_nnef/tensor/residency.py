"""Scoped residency and asynchronous prefetch for offloaded tensors."""

import concurrent.futures
import contextlib
import contextvars
import dataclasses
import logging
import threading
import typing as T

import torch

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef.tensor.offload import TDEVICE, OffloadedTensor

LOGGER = logging.getLogger(__name__)

_ACTIVE_RESIDENCY: contextvars.ContextVar[
    T.Optional[T.Tuple["TensorResidencyPool", T.Optional[torch.device]]]
] = contextvars.ContextVar("t2n_active_residency", default=None)


def active_tensor_residency() -> T.Optional[
    T.Tuple["TensorResidencyPool", T.Optional[torch.device]]
]:
    """Return the pool and default device active in this context."""
    return _ACTIVE_RESIDENCY.get()


@dataclasses.dataclass
class _ResidentEntry:
    source: OffloadedTensor
    device: torch.device
    value: T.Optional[torch.Tensor] = None
    future: T.Optional[concurrent.futures.Future] = None
    leases: int = 0
    dirty: bool = False
    access_order: int = 0
    nbytes: int = 0


class TensorPrefetch:
    """Handle for an asynchronous tensor prefetch."""

    def __init__(
        self,
        pool: "TensorResidencyPool",
        source: OffloadedTensor,
        device: torch.device,
        future: T.Optional[concurrent.futures.Future],
        value: T.Optional[torch.Tensor] = None,
    ):
        self._pool = pool
        self._source = source
        self._device = device
        self._future = future
        self._value = value

    def wait(self) -> torch.Tensor:
        """Wait for prefetch completion and return the resident value."""
        if self._value is not None:
            self._pool._enforce_budget()
            return self._value
        if self._future is None:
            return self._pool._wait(id(self._source))
        value = self._future.result()
        self._pool._complete_prefetch(id(self._source), self._future)
        self._pool._enforce_budget()
        return value

    def done(self) -> bool:
        """Return whether the prefetch has finished."""
        if self._value is not None:
            return True
        if self._future is not None:
            return self._future.done()
        return self._pool._done(id(self._source))


class TensorResidencyLease:
    """Scoped access to a value managed by a tensor residency pool."""

    def __init__(
        self,
        pool: "TensorResidencyPool",
        source: OffloadedTensor,
        device: torch.device,
        mode: str,
    ):
        self._pool = pool
        self._source = source
        self._device = device
        self._key: T.Optional[int] = None
        self._mode = mode
        self._value: T.Optional[torch.Tensor] = None

    def __enter__(self) -> torch.Tensor:
        self._key, self._value = self._pool._enter_lease(
            self._source, self._device
        )
        return self._value

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._key is not None:
            self._pool._exit_lease(self._key, dirty=self._mode == "read_write")
        self._key = None
        self._value = None


class TensorResidencyPool:
    """Keep offloaded tensors resident across a bounded operation scope.

    Values can be prefetched by background workers and acquired through a
    lease. A lease is a scoped claim that a materialized value is in active
    use, so its value remains resident until the lease ends. Unleased values
    are evicted in least-recently-used order when the optional cache budget is
    exceeded. A ``read_write`` lease writes the value back before eviction.

    ``max_cached_bytes`` is a soft cache-retention limit, not a hard bound on
    process memory. Leased values remain available even when they exceed it.
    An oversized value can therefore be materialized for an active caller but
    is evicted as soon as its final lease ends. Prefetched oversized values are
    returned to their waiting caller without being retained by the pool.
    """

    def __init__(
        self,
        max_cached_bytes: T.Optional[int] = None,
        max_workers: int = 1,
    ):
        if max_cached_bytes is not None and max_cached_bytes < 0:
            raise T2NErrorMisuse("max_cached_bytes must be non-negative")
        if max_workers < 1:
            raise T2NErrorMisuse("max_workers must be at least one")
        self.max_cached_bytes = max_cached_bytes
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="t2n-residency",
        )
        self._entries: T.Dict[int, _ResidentEntry] = {}
        self._lock = threading.RLock()
        self._access_order = 0
        self._closed = False
        self._closing = False

    @property
    def resident_bytes(self) -> int:
        """Return the approximate bytes held by resident values."""
        with self._lock:
            return sum(entry.nbytes for entry in self._entries.values())

    def is_resident(self, source: OffloadedTensor) -> bool:
        """Return whether ``source`` has completed loading into the pool."""
        key = id(source)
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and entry.value is not None

    def prefetch(
        self,
        source: OffloadedTensor,
        device: T.Optional[TDEVICE] = None,
    ) -> TensorPrefetch:
        """Begin loading ``source`` and return a completion handle."""
        target_device = torch.device(
            source.target_device if device is None else device
        )
        return self._prefetch(source, target_device, lease=False)

    def _prefetch(
        self,
        source: OffloadedTensor,
        target_device: torch.device,
        *,
        lease: bool,
    ) -> TensorPrefetch:
        """Start or reuse a load, optionally reserving it for a lease."""
        key = id(source)
        with self._lock:
            self._ensure_open()
            entry = self._entries.get(key)
            if entry is not None:
                if entry.device != target_device:
                    raise T2NErrorMisuse(
                        "a tensor cannot be resident on two devices"
                    )
                if lease:
                    entry.leases += 1
                self._touch(entry)
                return TensorPrefetch(
                    self,
                    source,
                    target_device,
                    entry.future,
                    entry.value,
                )
            entry = _ResidentEntry(
                source=source,
                device=target_device,
                leases=int(lease),
            )
            self._touch(entry)
            self._entries[key] = entry
            future = self._executor.submit(
                source._reload_unmanaged, device=target_device
            )
            entry.future = future

            def complete_prefetch(
                future: concurrent.futures.Future,
            ) -> None:
                self._complete_prefetch(key, future)

            future.add_done_callback(complete_prefetch)
        return TensorPrefetch(self, source, target_device, future)

    def acquire(
        self,
        source: OffloadedTensor,
        device: T.Optional[TDEVICE] = None,
        mode: str = "read",
    ) -> TensorResidencyLease:
        """Return a scoped lease for ``source``.

        Use ``mode="read_write"`` when the returned value may be modified.
        The pool writes dirty values back before eviction or flushing.
        """
        if mode not in ("read", "read_write"):
            raise T2NErrorMisuse("mode must be 'read' or 'read_write'")
        target_device = torch.device(
            source.target_device if device is None else device
        )
        return TensorResidencyLease(self, source, target_device, mode)

    def resolve(
        self,
        source: OffloadedTensor,
        device: T.Optional[TDEVICE] = None,
    ) -> torch.Tensor:
        """Return a resident value, loading it when necessary.

        This is the read-only path used automatically by ``OffloadedTensor``
        inside :meth:`scope`. Use :meth:`acquire` for explicit pinning or
        mutation tracking.
        """
        return self.prefetch(source, device=device).wait()

    @contextlib.contextmanager
    def scope(self, device: T.Optional[TDEVICE] = None):
        """Make this pool transparently serve offloaded tensor reloads."""
        self._ensure_open()
        default_device = None if device is None else torch.device(device)
        token = _ACTIVE_RESIDENCY.set((self, default_device))
        try:
            yield self
        finally:
            _ACTIVE_RESIDENCY.reset(token)

    def flush(self, source: T.Optional[OffloadedTensor] = None) -> None:
        """Wait for loads and write dirty resident values back to storage."""
        with self._lock:
            keys = list(self._entries)
            if source is not None:
                keys = [id(source)] if id(source) in self._entries else []
        for key in keys:
            self._wait(key)
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    self._write_back(entry)

    def evict(self, source: OffloadedTensor) -> None:
        """Write back and remove an unleased value from the pool."""
        key = id(source)
        self._wait(key)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.leases:
                raise T2NErrorMisuse("cannot evict a tensor with active leases")
            self._write_back(entry)
            del self._entries[key]

    def close(self) -> None:
        """Flush all values, stop workers, and release resident memory."""
        with self._lock:
            if self._closed:
                return
            if self._closing:
                raise T2NErrorMisuse("tensor residency pool is closing")
            active = [entry for entry in self._entries.values() if entry.leases]
            if active:
                raise T2NErrorMisuse("cannot close a pool with active leases")
            # Prevent a new prefetch or lease from entering after the active
            # lease check and before the entries are flushed and cleared.
            self._closing = True
        try:
            self.flush()
        finally:
            with self._lock:
                self._entries.clear()
                self._closed = True
                self._closing = False
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "TensorResidencyPool":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception:  # pylint: disable=broad-exception-caught
            if exc_value is None:
                raise
            LOGGER.exception("failed to close tensor residency pool")

    def _complete_prefetch(
        self,
        key: int,
        future: concurrent.futures.Future,
    ) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        value = future.result()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.future is not future:
                return
            entry.value = value
            entry.future = None
            entry.nbytes = self._value_nbytes(value)
            self._touch(entry)
            if (
                self.max_cached_bytes is not None
                and entry.nbytes > self.max_cached_bytes
                and not entry.leases
            ):
                del self._entries[key]
            else:
                self._evict_to_budget(exclude={key})

    def _wait(self, key: int) -> torch.Tensor:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise T2NErrorMisuse("tensor is not managed by this pool")
            future = entry.future
            value = entry.value
        if value is None:
            assert future is not None
            value = future.result()
            with self._lock:
                entry = self._entries.get(key)
                if entry is None:
                    raise T2NErrorMisuse("tensor left the pool while loading")
                if entry.value is None:
                    entry.value = value
                    entry.nbytes = self._value_nbytes(value)
                    entry.future = None
                    self._touch(entry)
                    self._evict_to_budget(exclude={key})
                value = entry.value
        # The returned local reference keeps the value alive even when the
        # pool must immediately evict an oversized, unleased entry.
        self._enforce_budget()
        return value

    def _enforce_budget(self) -> None:
        """Evict unleased entries after a caller has captured its value."""
        with self._lock:
            self._evict_to_budget()

    def _done(self, key: int) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            return entry.value is not None or (
                entry.future is not None and entry.future.done()
            )

    def _enter_lease(
        self, source: OffloadedTensor, device: torch.device
    ) -> T.Tuple[int, torch.Tensor]:
        handle = self._prefetch(source, device, lease=True)
        key = id(handle._source)
        try:
            value = handle.wait()
        except BaseException:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.leases -= 1
            raise
        return key, value

    def _exit_lease(self, key: int, dirty: bool) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.leases < 1:
                raise T2NErrorMisuse("lease is not active")
            entry.leases -= 1
            entry.dirty = entry.dirty or dirty
            self._touch(entry)
            self._evict_to_budget()

    def _evict_to_budget(self, exclude: T.Optional[T.Set[int]] = None) -> None:
        if self.max_cached_bytes is None:
            return
        excluded = set() if exclude is None else exclude
        while self.resident_bytes > self.max_cached_bytes:
            candidates = [
                (key, entry)
                for key, entry in self._entries.items()
                if key not in excluded
                and entry.value is not None
                and not entry.leases
            ]
            if not candidates:
                return
            key, entry = min(candidates, key=lambda item: item[1].access_order)
            self._write_back(entry)
            del self._entries[key]

    @staticmethod
    def _value_nbytes(value: torch.Tensor) -> int:
        return value.numel() * value.element_size()

    @staticmethod
    def _write_back(entry: _ResidentEntry) -> None:
        if entry.dirty:
            assert entry.value is not None
            entry.source.update_values(entry.value)
            entry.dirty = False

    def _touch(self, entry: _ResidentEntry) -> None:
        self._access_order += 1
        entry.access_order = self._access_order

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise T2NErrorMisuse("tensor residency pool is closed")
            if self._closing:
                raise T2NErrorMisuse("tensor residency pool is closing")
