"""Convert visits into a shuffled, shardable list of chunk references.

------------------------------------------------------------------------------
Configuration this module uses (from config.yaml):
------------------------------------------------------------------------------
  SAMPLES_PER_CHUNK   ← sample_rate_hz × chunk_duration_sec
                        This sets how many timesteps go into one chunk.
                        At 1000 Hz × 10 s = 10,000 samples = one chunk.
------------------------------------------------------------------------------

A ChunkRef is a lightweight pointer (visit UUID, .npy path, sample offset).
The provider reads chunk bytes on demand from memory-mapped numpy arrays.

We use a sample (timestep) offset rather than a byte offset because .npy files
have a variable-length header; once we have an mmap'd array, indexing by
timestep is what numpy expects and what tobytes() needs.

This split (plan = cheap pointers, bytes = on-demand) means we can plan tens of
thousands of chunks in milliseconds and shard with a single pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import SAMPLES_PER_CHUNK
from .metadata import Visit
from .timing import timed


@dataclass(frozen=True)
class ChunkRef:
    """One 10-second chunk addressed by file path and starting sample index."""

    visit_id_bytes: bytes
    npy_path: Path
    sample_start: int

    def __post_init__(self) -> None:
        if len(self.visit_id_bytes) != 16:
            raise ValueError("visit_id_bytes must be exactly 16 bytes")


def chunks_for_visit(visit: Visit) -> list[ChunkRef]:
    """Slice a visit's EEG file into back-to-back 10-second chunks.

    A trailing partial chunk (fewer than SAMPLES_PER_CHUNK timesteps) is
    dropped, since the consumer expects fixed-size chunks. We mmap rather than
    fully read; only the array header is touched here.
    """
    arr = np.load(visit.npy_path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"{visit.npy_path}: expected shape (N, 4) of float32, got "
            f"shape={arr.shape}, dtype={arr.dtype}"
        )
    if arr.dtype != np.float32:
        raise ValueError(
            f"{visit.npy_path}: expected float32, got {arr.dtype}"
        )
    num_timesteps = arr.shape[0]
    num_chunks = num_timesteps // SAMPLES_PER_CHUNK
    return [
        ChunkRef(
            visit_id_bytes=visit.visit_id_bytes,
            npy_path=visit.npy_path,
            sample_start=i * SAMPLES_PER_CHUNK,
        )
        for i in range(num_chunks)
    ]


@timed
def plan_chunks(
    visits: list[Visit],
    *,
    rng: random.Random | None = None,
) -> list[ChunkRef]:
    """Build the global shuffled chunk list across all matching visits."""
    rng = rng or random.Random()
    all_chunks: list[ChunkRef] = []
    for visit in visits:
        all_chunks.extend(chunks_for_visit(visit))
    rng.shuffle(all_chunks)
    return all_chunks


def shard_round_robin(chunks: list[ChunkRef], num_shards: int) -> list[list[ChunkRef]]:
    """Distribute chunks across N shards round-robin.

    Round-robin (vs contiguous slicing) keeps shards balanced even when the
    global count isn't divisible by num_shards, and keeps the random ordering
    intact within each shard.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    shards: list[list[ChunkRef]] = [[] for _ in range(num_shards)]
    for i, chunk in enumerate(chunks):
        shards[i % num_shards].append(chunk)
    return shards
