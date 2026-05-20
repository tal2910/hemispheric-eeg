"""Tests for StreamStats accounting under failure modes.

The provider must report accurate partial counts even when the stream is
interrupted by resource exhaustion (MemoryError) or I/O failures on the
mmap'd .npy (OSError) — not just on consumer disconnect. The orchestrator's
integrity summary depends on these counts being right.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from hemispheric.chunking import ChunkRef
from hemispheric.provider import StreamStats, _stream_shard_to_consumer


class _MemoryWriter:
    """Stand-in for asyncio.StreamWriter that records bytes and never blocks."""

    def __init__(self):
        self.buffer = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        return

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return

    def get_extra_info(self, key: str):
        return ("127.0.0.1", 0) if key == "peername" else None


def _make_npy(tmp_path: Path, n_samples: int) -> Path:
    npy_path = tmp_path / "visit.npy"
    arr = np.random.randn(n_samples, 4).astype(np.float32)
    np.save(npy_path, arr)
    return npy_path


def _refs_for(npy_path: Path, n_chunks: int) -> list[ChunkRef]:
    from hemispheric import SAMPLES_PER_CHUNK
    vid = b"\x00" * 16
    return [
        ChunkRef(visit_id_bytes=vid, npy_path=npy_path,
                 sample_start=i * SAMPLES_PER_CHUNK)
        for i in range(n_chunks)
    ]


def test_clean_stream_reports_full_count(tmp_path):
    """Sanity check: a complete run reports all chunks streamed."""
    npy = _make_npy(tmp_path, n_samples=10 * 10_000)  # 10 chunks worth
    chunks = _refs_for(npy, n_chunks=10)
    writer = _MemoryWriter()

    stats = asyncio.run(
        _stream_shard_to_consumer(writer, chunks, port=5000)
    )

    assert stats.chunks_planned == 10
    assert stats.chunks_streamed == 10
    assert stats.clean_close is True
    assert stats.is_complete
    assert stats.chunks_lost == 0


def test_broken_pipe_reports_partial_count(tmp_path):
    """Consumer disconnect mid-stream reports partial chunks_streamed."""
    npy = _make_npy(tmp_path, n_samples=10 * 10_000)
    chunks = _refs_for(npy, n_chunks=10)
    writer = _MemoryWriter()

    call_count = {"n": 0}
    real_drain = writer.drain

    async def failing_drain():
        call_count["n"] += 1
        if call_count["n"] > 3:
            raise BrokenPipeError("consumer went away")
        await real_drain()

    writer.drain = failing_drain
    stats = asyncio.run(
        _stream_shard_to_consumer(writer, chunks, port=5000)
    )

    # 3 chunks drained successfully; 4th drain raised before sent was incremented
    assert stats.chunks_streamed == 3
    assert stats.chunks_planned == 10
    assert stats.chunks_lost == 7
    assert stats.clean_close is False
    assert not stats.is_complete


def test_memory_error_preserves_partial_stats(tmp_path):
    """OOM mid-stream still records what was successfully streamed."""
    npy = _make_npy(tmp_path, n_samples=10 * 10_000)
    chunks = _refs_for(npy, n_chunks=10)
    writer = _MemoryWriter()

    call_count = {"n": 0}
    real_drain = writer.drain

    async def oom_drain():
        call_count["n"] += 1
        if call_count["n"] > 5:
            raise MemoryError("simulated allocator failure")
        await real_drain()

    writer.drain = oom_drain
    stats = asyncio.run(
        _stream_shard_to_consumer(writer, chunks, port=5001)
    )

    # 5 chunks drained successfully before OOM. Stats must NOT show 0/10.
    assert stats.chunks_streamed == 5, (
        "OOM lost the partial count; orchestrator would report data loss inaccurately"
    )
    assert stats.chunks_planned == 10
    assert stats.chunks_lost == 5
    assert stats.clean_close is False


def test_os_error_preserves_partial_stats(tmp_path):
    """Disk-side I/O failure (corrupt .npy, ENOSPC) still records partial."""
    npy = _make_npy(tmp_path, n_samples=10 * 10_000)
    chunks = _refs_for(npy, n_chunks=10)
    writer = _MemoryWriter()

    call_count = {"n": 0}
    real_drain = writer.drain

    async def io_drain():
        call_count["n"] += 1
        if call_count["n"] > 2:
            raise OSError("Input/output error")
        await real_drain()

    writer.drain = io_drain
    stats = asyncio.run(
        _stream_shard_to_consumer(writer, chunks, port=5002)
    )

    assert stats.chunks_streamed == 2
    assert stats.chunks_planned == 10
    assert stats.clean_close is False


def test_immediate_failure_on_first_chunk(tmp_path):
    """If file #1's very first chunk fails, we still get accurate stats (0/N)."""
    npy = _make_npy(tmp_path, n_samples=10 * 10_000)
    chunks = _refs_for(npy, n_chunks=10)
    writer = _MemoryWriter()

    async def immediate_oom():
        raise MemoryError("simulated allocator failure at startup")

    writer.drain = immediate_oom
    stats = asyncio.run(
        _stream_shard_to_consumer(writer, chunks, port=5003)
    )

    assert stats.chunks_streamed == 0
    assert stats.chunks_planned == 10
    assert stats.chunks_lost == 10
    assert stats.clean_close is False
    assert not stats.is_complete
