"""TCP data provider.

------------------------------------------------------------------------------
Configuration this module uses (indirectly, via serialize_chunk):
------------------------------------------------------------------------------
  SAMPLES_PER_CHUNK, CHUNK_PAYLOAD_BYTES, UUID_BYTES, CHUNK_WIRE_BYTES
  (all from config.yaml's dataset + wire_protocol sections)

  This module itself takes host/ports as function arguments rather than
  reading config directly; the orchestrator passes whatever the CLI parsed
  (which defaulted to config.yaml's runtime.host / runtime.ports).
------------------------------------------------------------------------------

Binds one asyncio TCP server per port. Each server waits for a single consumer
connection, streams its assigned shard of chunks to that consumer in the
pre-shuffled order, then closes. run_provider() returns when every shard has
drained.

Sequencing: the orchestrator must not launch the consumer until every port is
bound, otherwise the consumer's connect() may race ahead and get ECONNREFUSED.
We expose this via the optional `ready` event in run_provider.

Concurrency model: a single event loop, N concurrent serve() coroutines, one
per port. Numpy mmap reads are technically blocking syscalls, but the OS page
cache makes them effectively free after the first touch. For very large
datasets or cold caches, consider routing reads through a thread pool.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import CHUNK_WIRE_BYTES
from .chunking import ChunkRef
from .protocol import serialize_chunk
from .timing import timed

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamStats:
    """Per-port accounting of what the provider streamed.

    Used by the orchestrator to verify no chunks were lost (chunks_streamed
    equals chunks_planned, and clean_close is True). If a consumer drops
    mid-stream the provider catches the broken pipe and reports the partial
    count — the orchestrator's post-run summary surfaces the loss.
    """

    port: int
    chunks_planned: int
    chunks_streamed: int
    bytes_streamed: int
    clean_close: bool

    @property
    def chunks_lost(self) -> int:
        return self.chunks_planned - self.chunks_streamed

    @property
    def is_complete(self) -> bool:
        return self.clean_close and self.chunks_streamed == self.chunks_planned


class _ArrayPool:
    """Cache of memory-mapped numpy arrays, scoped to one shard.

    Each chunk read needs the source visit's array. Re-mapping per chunk is
    wasteful when a file produces many chunks (one of our visits holds 120).
    We keep one mmap per file for the lifetime of the shard's stream. Numpy
    drops the mmap when the last reference goes away, which is when we clear
    the dict in close_all().
    """

    def __init__(self) -> None:
        self._arrays: dict[Path, np.ndarray] = {}

    def get(self, path: Path) -> np.ndarray:
        arr = self._arrays.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            self._arrays[path] = arr
        return arr

    def close_all(self) -> None:
        # mmap arrays are released when no Python reference remains. Dropping
        # the dict is enough; we don't need to call any explicit close.
        self._arrays.clear()


async def _stream_shard_to_consumer(
    writer: asyncio.StreamWriter,
    chunks: list[ChunkRef],
    *,
    port: int,
) -> StreamStats:
    """Send every chunk in `chunks` over `writer`, then close.

    Returns a StreamStats describing exactly what got across the wire. The
    caller uses this to verify no data loss at end-of-run.
    """
    pool = _ArrayPool()
    sent = 0
    bytes_sent = 0
    clean_close = False
    try:
        peer = writer.get_extra_info("peername")
        log.info("port %d: consumer connected from %s, streaming %d chunks",
                 port, peer, len(chunks))
        for ref in chunks:
            arr = pool.get(ref.npy_path)
            buffer = serialize_chunk(ref, arr)
            writer.write(buffer)
            # drain() applies backpressure: it suspends until the OS send
            # buffer has room. Without it the producer can outrun a slow
            # consumer (the real consumer sleeps 0.1s per chunk) and explode
            # memory.
            await writer.drain()
            sent += 1
            bytes_sent += len(buffer)
        clean_close = True
        log.info("port %d: streamed all %d chunks (%.1f MB), closing",
                 port, sent, bytes_sent / (1024 * 1024))
    except (ConnectionResetError, BrokenPipeError) as exc:
        log.warning("port %d: consumer dropped after %d/%d chunks: %s",
                    port, sent, len(chunks), exc)
    finally:
        pool.close_all()
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
    return StreamStats(
        port=port,
        chunks_planned=len(chunks),
        chunks_streamed=sent,
        bytes_streamed=bytes_sent,
        clean_close=clean_close,
    )


@timed
async def run_provider(
    shards: list[list[ChunkRef]],
    ports: list[int],
    *,
    host: str = "0.0.0.0",
    ready: asyncio.Event | None = None,
) -> list[StreamStats]:
    """Run one server per (shard, port) pair until all shards drain.

    Stages:
      1. Bind every port. Errors here fail fast.
      2. Signal `ready` so the orchestrator can launch the consumer.
      3. Each server accepts exactly one connection, streams, then exits.

    Returns one StreamStats per port (in port-input order) so the orchestrator
    can verify no data was lost.
    """
    if len(shards) != len(ports):
        raise ValueError(
            f"Got {len(shards)} shards for {len(ports)} ports; must match"
        )

    servers: list[asyncio.base_events.Server] = []
    done_events: list[asyncio.Event] = []
    # Per-port stats are populated by the handler before it signals done.
    stats_by_port: dict[int, StreamStats] = {}

    # Stage 1: bind all ports.
    for shard, port in zip(shards, ports):
        done = asyncio.Event()
        done_events.append(done)

        # Bind shard, port, done, stats into the closure so each handler sees its own.
        def make_handler(s=shard, p=port, d=done, st=stats_by_port):
            async def handle(reader, writer):
                try:
                    result = await _stream_shard_to_consumer(writer, s, port=p)
                    st[p] = result
                finally:
                    d.set()
            return handle

        server = await asyncio.start_server(make_handler(), host=host, port=port)
        servers.append(server)
        log.info("port %d: listening on %s, %d chunks queued",
                 port, host, len(shard))

    log.info("all %d ports listening", len(ports))
    if ready is not None:
        ready.set()

    # Stage 2 + 3: serve concurrently until each shard's done event fires.
    async def run_until_done(server: asyncio.base_events.Server, done: asyncio.Event):
        async with server:
            await done.wait()

    try:
        await asyncio.gather(*[
            run_until_done(s, d) for s, d in zip(servers, done_events)
        ])
    except asyncio.CancelledError:
        for s in servers:
            s.close()
        raise

    # Stats are populated by each handler before signaling done; a missing
    # entry would mean a port handler crashed before recording its result.
    # Synthesize a zero-stats entry so the orchestrator's summary still
    # reflects every planned port.
    return [
        stats_by_port.get(
            p,
            StreamStats(
                port=p,
                chunks_planned=len(shard),
                chunks_streamed=0,
                bytes_streamed=0,
                clean_close=False,
            ),
        )
        for p, shard in zip(ports, shards)
    ]
