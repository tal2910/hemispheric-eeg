"""Wire protocol for one chunk on the TCP socket.

------------------------------------------------------------------------------
Configuration this module uses (from config.yaml):
------------------------------------------------------------------------------
  SAMPLES_PER_CHUNK    ← sample_rate_hz × chunk_duration_sec
  CHUNK_PAYLOAD_BYTES  ← samples_per_chunk × num_channels × bytes_per_float
  CHUNK_WIRE_BYTES     ← chunk_payload_bytes + uuid_bytes
  UUID_BYTES           ← wire_protocol.uuid_bytes (16, the UUIDv4 size)

  With the shipped defaults this gives a 160,016-byte chunk on the wire.
  Change chunk_duration_sec in config.yaml from 10 to e.g. 5 and the wire
  size automatically becomes 80,016 bytes — coordinate with the training
  team before doing so since it changes their batch shape.
------------------------------------------------------------------------------

Layout (little-endian, fixed size, no framing):

    offset    size           contents
    0         UUID_BYTES     visit UUID (raw bytes, not the hyphenated string)
    16        PAYLOAD_BYTES  float32 samples, channel-interleaved
    total:    CHUNK_WIRE_BYTES

The float32 section is row-major (samples × channels), matching numpy's
C-contiguous order on the source .npy files. On the wire it reads as
[t0_c0, t0_c1, t0_c2, t0_c3, t1_c0, t1_c1, ...]. The bundled consumer
reshapes to (samples_per_chunk, num_channels), which only works for this layout.

Chunks are concatenated back-to-back on the socket. The consumer reads exactly
CHUNK_WIRE_BYTES per chunk and parses; there is no length prefix or magic
because every chunk is the same size.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from . import (
    CHUNK_PAYLOAD_BYTES,
    CHUNK_WIRE_BYTES,
    SAMPLES_PER_CHUNK,
    UUID_BYTES,
)
from .chunking import ChunkRef


@dataclass(frozen=True)
class ParsedChunk:
    """Result of parsing a chunk off the wire. Used by the consumer side."""

    visit_id: uuid.UUID
    payload: bytes  # raw float32 bytes; len == CHUNK_PAYLOAD_BYTES


def serialize_chunk(ref: ChunkRef, arr: np.ndarray) -> bytes:
    """Build the wire bytes for one chunk from a memory-mapped numpy array.

    `arr` is the source visit's array, shape (N, 4), dtype float32. We slice
    out [sample_start:sample_start+SAMPLES_PER_CHUNK] and copy to bytes via
    tobytes(), which produces the C-contiguous row-major layout the consumer
    expects. The slice itself is an mmap view; .tobytes() is the one copy.
    """
    end = ref.sample_start + SAMPLES_PER_CHUNK
    chunk = arr[ref.sample_start:end]
    if chunk.shape != (SAMPLES_PER_CHUNK, 4):
        raise IOError(
            f"Short read at {ref.npy_path}[{ref.sample_start}:{end}]: "
            f"got shape {chunk.shape}, expected ({SAMPLES_PER_CHUNK}, 4)"
        )
    payload = chunk.tobytes(order="C")
    if len(payload) != CHUNK_PAYLOAD_BYTES:
        raise IOError(
            f"Serialized chunk wrong size: got {len(payload)} bytes, "
            f"expected {CHUNK_PAYLOAD_BYTES}"
        )
    return ref.visit_id_bytes + payload


def parse_chunk(buffer: bytes) -> ParsedChunk:
    """Parse a complete chunk buffer of length CHUNK_WIRE_BYTES."""
    if len(buffer) != CHUNK_WIRE_BYTES:
        raise ValueError(
            f"Expected {CHUNK_WIRE_BYTES} bytes, got {len(buffer)}"
        )
    return ParsedChunk(
        visit_id=uuid.UUID(bytes=buffer[:UUID_BYTES]),
        payload=buffer[UUID_BYTES:],
    )
