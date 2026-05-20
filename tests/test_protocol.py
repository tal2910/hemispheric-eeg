"""Tests for chunk serialization and parsing."""

import struct
import uuid
from pathlib import Path

import numpy as np
import pytest

from hemispheric import (
    CHUNK_PAYLOAD_BYTES,
    CHUNK_WIRE_BYTES,
    NUM_CHANNELS,
    SAMPLES_PER_CHUNK,
)
from hemispheric.chunking import ChunkRef
from hemispheric.protocol import parse_chunk, serialize_chunk


def test_constants_sanity():
    assert CHUNK_WIRE_BYTES == 16 + 160_000
    assert CHUNK_PAYLOAD_BYTES == 160_000


def test_serialize_chunk_returns_uuid_then_payload():
    vid = uuid.uuid4()
    arr = np.full((SAMPLES_PER_CHUNK, NUM_CHANNELS), 1.5, dtype=np.float32)
    ref = ChunkRef(visit_id_bytes=vid.bytes, npy_path=Path("x"), sample_start=0)
    out = serialize_chunk(ref, arr)
    assert len(out) == CHUNK_WIRE_BYTES
    assert out[:16] == vid.bytes
    # Should round-trip through numpy.
    back = np.frombuffer(out[16:], dtype=np.float32).reshape(SAMPLES_PER_CHUNK, NUM_CHANNELS)
    assert np.allclose(back, 1.5)


def test_serialize_slices_at_correct_sample_start():
    vid = uuid.uuid4()
    # Two distinct chunks in a single array so we can verify the slice
    # boundary by checking which one comes out.
    arr = np.zeros((2 * SAMPLES_PER_CHUNK, NUM_CHANNELS), dtype=np.float32)
    arr[:SAMPLES_PER_CHUNK] = 7.0
    arr[SAMPLES_PER_CHUNK:] = 9.0
    ref = ChunkRef(visit_id_bytes=vid.bytes, npy_path=Path("x"),
                   sample_start=SAMPLES_PER_CHUNK)
    out = serialize_chunk(ref, arr)
    back = np.frombuffer(out[16:], dtype=np.float32)
    assert np.allclose(back, 9.0)


def test_serialize_short_slice_raises():
    vid = uuid.uuid4()
    # Array too small for a full chunk at the requested offset.
    arr = np.zeros((SAMPLES_PER_CHUNK - 100, NUM_CHANNELS), dtype=np.float32)
    ref = ChunkRef(visit_id_bytes=vid.bytes, npy_path=Path("x"), sample_start=0)
    with pytest.raises(IOError):
        serialize_chunk(ref, arr)


def test_parse_chunk_round_trip():
    vid = uuid.uuid4()
    payload = struct.pack(f"<{40_000}f", *([0.5] * 40_000))
    buffer = vid.bytes + payload
    parsed = parse_chunk(buffer)
    assert parsed.visit_id == vid
    assert parsed.payload == payload


def test_parse_chunk_wrong_size():
    with pytest.raises(ValueError):
        parse_chunk(b"\x00" * 100)


def test_chunk_ref_rejects_bad_uuid_bytes():
    with pytest.raises(ValueError):
        ChunkRef(visit_id_bytes=b"\x00" * 8, npy_path=Path("x"), sample_start=0)


def test_serialize_preserves_interleaved_layout():
    """Verify on-the-wire bytes match the consumer's reshape(10000, 4)."""
    vid = uuid.uuid4()
    arr = np.zeros((SAMPLES_PER_CHUNK, NUM_CHANNELS), dtype=np.float32)
    # Mark each (timestep, channel) with a unique value so we can verify order.
    for t in range(SAMPLES_PER_CHUNK):
        for c in range(NUM_CHANNELS):
            arr[t, c] = t * NUM_CHANNELS + c
    ref = ChunkRef(visit_id_bytes=vid.bytes, npy_path=Path("x"), sample_start=0)
    out = serialize_chunk(ref, arr)
    # Consumer reshapes (10000, 4); we should get the same array back.
    back = np.frombuffer(out[16:], dtype=np.float32).reshape(SAMPLES_PER_CHUNK, NUM_CHANNELS)
    assert np.array_equal(back, arr)
