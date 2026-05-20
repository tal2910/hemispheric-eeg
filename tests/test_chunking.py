"""Tests for chunk planning and sharding."""

import json
import random
import uuid
from pathlib import Path

import numpy as np
import pytest

from hemispheric import NUM_CHANNELS, SAMPLES_PER_CHUNK
from hemispheric.chunking import (
    ChunkRef,
    chunks_for_visit,
    plan_chunks,
    shard_round_robin,
)
from hemispheric.metadata import Visit


@pytest.fixture
def write_visit(tmp_path):
    """Factory: writes a fake .npy of `seconds` duration and returns a Visit."""
    def _make(seconds: int, vid: uuid.UUID | None = None) -> Visit:
        vid = vid or uuid.uuid4()
        npy = tmp_path / f"{vid}.npy"
        meta = tmp_path / f"{vid}.json"
        arr = np.zeros((seconds * 1000, NUM_CHANNELS), dtype=np.float32)
        np.save(npy, arr)
        meta.write_text(json.dumps({
            "visit_id": str(vid),
            "person_id": str(uuid.uuid4()),
            "person_name": "T",
            "age": 30,
            "gender": "female",
            "wears_glasses": False,
            "date_of_visit": "2024-01-01",
            "dominant_hand": "right",
        }))
        return Visit(
            visit_id=vid, person_id=uuid.uuid4(), person_name="T", age=30,
            gender="female", wears_glasses=False, date_of_visit="2024-01-01",
            dominant_hand="right", npy_path=npy, metadata_path=meta, extra={},
        )
    return _make


class TestChunksForVisit:
    def test_60_seconds_yields_6_chunks(self, write_visit):
        visit = write_visit(60)
        chunks = chunks_for_visit(visit)
        assert len(chunks) == 6
        assert all(isinstance(c, ChunkRef) for c in chunks)

    def test_sample_starts_are_sequential(self, write_visit):
        visit = write_visit(30)
        chunks = chunks_for_visit(visit)
        starts = [c.sample_start for c in chunks]
        assert starts == [0, SAMPLES_PER_CHUNK, 2 * SAMPLES_PER_CHUNK]

    def test_partial_trailing_dropped(self, write_visit):
        # 15 seconds = 1 full 10s chunk + 5s trailing => only 1 chunk emitted.
        visit = write_visit(15)
        chunks = chunks_for_visit(visit)
        assert len(chunks) == 1

    def test_under_10s_yields_zero_chunks(self, write_visit):
        visit = write_visit(5)
        chunks = chunks_for_visit(visit)
        assert chunks == []

    def test_visit_id_propagated(self, write_visit):
        vid = uuid.uuid4()
        visit = write_visit(10, vid=vid)
        chunks = chunks_for_visit(visit)
        assert chunks[0].visit_id_bytes == vid.bytes

    def test_rejects_wrong_shape(self, tmp_path, write_visit):
        # Create a 1D npy to verify validation rejects bad shapes.
        bad = tmp_path / "bad.npy"
        np.save(bad, np.zeros(40_000, dtype=np.float32))
        meta = tmp_path / "bad.json"
        meta.write_text("{}")
        vid = uuid.uuid4()
        visit = Visit(
            visit_id=vid, person_id=uuid.uuid4(), person_name="X", age=20,
            gender="other", wears_glasses=False, date_of_visit="2024-01-01",
            dominant_hand="right", npy_path=bad, metadata_path=meta, extra={},
        )
        with pytest.raises(ValueError):
            chunks_for_visit(visit)


class TestPlanChunks:
    def test_combines_across_visits(self, write_visit):
        v1 = write_visit(20)  # 2 chunks
        v2 = write_visit(30)  # 3 chunks
        chunks = plan_chunks([v1, v2], rng=random.Random(0))
        assert len(chunks) == 5

    def test_shuffle_is_deterministic_with_seed(self, write_visit):
        visits = [write_visit(40) for _ in range(3)]
        a = plan_chunks(visits, rng=random.Random(123))
        b = plan_chunks(visits, rng=random.Random(123))
        assert [c.sample_start for c in a] == [c.sample_start for c in b]

    def test_shuffle_actually_reorders(self, write_visit):
        visits = [write_visit(100) for _ in range(5)]
        chunks = plan_chunks(visits, rng=random.Random(7))
        ids = [c.visit_id_bytes for c in chunks]
        same_neighbor = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
        assert same_neighbor < len(chunks) * 0.5


class TestShardRoundRobin:
    def test_distributes_evenly(self):
        chunks = [
            ChunkRef(visit_id_bytes=b"\x00" * 16, npy_path=Path("x"), sample_start=i)
            for i in range(10)
        ]
        shards = shard_round_robin(chunks, num_shards=2)
        assert len(shards) == 2
        assert len(shards[0]) == 5 and len(shards[1]) == 5

    def test_uneven_count_distributes_one_per_shard(self):
        chunks = [
            ChunkRef(visit_id_bytes=b"\x00" * 16, npy_path=Path("x"), sample_start=i)
            for i in range(7)
        ]
        shards = shard_round_robin(chunks, num_shards=3)
        sizes = sorted(len(s) for s in shards)
        assert sizes == [2, 2, 3]

    def test_invalid_shard_count(self):
        with pytest.raises(ValueError):
            shard_round_robin([], num_shards=0)
