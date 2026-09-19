"""Shared helpers for deterministic episode/trial batch scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class IndexRange:
    """A half-open range of logical benchmark episode indices."""

    start: int
    count: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("range start must be non-negative")
        if self.count <= 0:
            raise ValueError("range count must be positive")

    @property
    def stop(self) -> int:
        return self.start + self.count


def split_range(start: int, count: int, batch_size: int) -> list[IndexRange]:
    """Split one logical range into stable contiguous batches."""

    requested = IndexRange(start, count)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return [
        IndexRange(offset, min(batch_size, requested.stop - offset))
        for offset in range(requested.start, requested.stop, batch_size)
    ]
