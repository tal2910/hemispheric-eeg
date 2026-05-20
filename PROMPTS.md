# Coding Agent Prompts

This project was built collaboratively with Claude as a coding agent. Rather than listing every prompt verbatim, this document groups the work into the four phases that actually shaped the structure of the codebase. Each phase summarizes the directives given and what they produced.

The verbatim text of the original exercise brief is preserved at the bottom for reference.

---

## Phase 1 — Initial design and implementation

**Directive:** Build the orchestrator + TCP data provider described in the exercise brief. Filter visits by metadata, stream matching EEG data as randomly-ordered 10-second chunks over multiple TCP ports, define the on-wire payload (16-byte visit UUID + float32 array), and document a cluster design for the architecture even though the implementation is single-machine.

**What got built:**

- `run_training.py` entrypoint and `hemispheric/` package with separate modules for config, metadata, filtering, chunking, the wire protocol, and the asyncio TCP provider.
- Filter language: repeatable CLI flags with categorical OR semantics (`--gender female --gender other`), numeric comparators with AND semantics for ranges (`--age ">=18" --age "<65"`), and boolean toggles (`--wears-glasses` / `--no-glasses`).
- Chunk plan as cheap pointers: `ChunkRef` (visit_id_bytes, npy_path, sample_start) at ~80 bytes each, shuffled once globally and sharded round-robin across ports. Actual EEG data stays on disk; mmap is used at stream time only.
- One-connection-per-port provider model: each port serves exactly one consumer connection, drains its shard, then closes. Backpressure via `writer.drain()` so a slow consumer can't blow up provider memory.
- Race-free startup: an `asyncio.Event` (`ready`) fires once every port has bound, and the orchestrator awaits it before launching the consumer subprocess — eliminates the ECONNREFUSED race that surfaced during early testing.
- `ARCHITECTURE.md` documenting how this would scale to a Kubernetes cluster: stateless provider pods reading from object storage, a Redis Streams (or equivalent) work queue, K8s Job orchestration. Implementation stays single-machine for this exercise.

---

## Phase 2 — Real data integration

**Directive:** Replace the placeholder synthetic data path with the actual data team's deliverables: real `.npy` files, real per-visit JSON sidecars with the proper schema, the data team's unmodified `consumer.py`, and their `Dockerfile`. Make the orchestrator work end-to-end against the real artifacts.

**What got changed:**

- File extension and IO model: `.eeg` raw binary → `.npy` numpy native. Reading via `np.load(path, mmap_mode='r')` and slicing on sample indices, rather than `seek + read` on byte offsets. `ChunkRef.byte_offset` (bytes) became `ChunkRef.sample_start` (samples) to match how numpy addresses arrays.
- Metadata schema aligned with the real sidecars: `subject_name` → `person_name`; added `person_id`, `wears_glasses`, `date_of_visit`, `dominant_hand`.
- Filter language extended to cover the new fields: `--wears-glasses` / `--no-glasses` (boolean), `--dominant-hand` (categorical), `--person-id` (UUID).
- `_FileHandlePool` (one open file per .eeg) replaced with `_ArrayPool` (one mmap'd numpy array per .npy), so the OS page cache handles paging and the resident set stays tiny even for GB-scale datasets.
- The data team's `consumer.py` is used unmodified. Their `Dockerfile`, `requirements.txt`, and the consumer-side validation file (`visit_db.json`) are kept exactly as provided.
- Wire protocol locked to 160,016 bytes per chunk (16 UUID + 10s × 1000 Hz × 4 channels × 4 bytes), time-major channel-interleaved float32, little-endian — chosen as the documented default since the spec left both layout and endianness implicit.

---

## Phase 3 — Cleanup and simplification

**Directive:** Trim the project down to what's actually needed. Remove development scaffolding that no longer pulls its weight. Consolidate configuration. Stop shipping artifacts that can be derived from the source of truth.

**What got removed or simplified:**

- Dev-only files dropped: `hemispheric/mock_consumer.py`, `hemispheric/samplegen.py`, `tests/test_end_to_end.py`, `run.sh`, `Makefile`, `BUILD_PROMPTS.md`. The orchestrator now has two consumer modes (real subprocess by default, or `--no-consumer`); the mock variant is gone.
- Bundled sample data (5 visits, ~88 MB) removed. The repo ships at ~470 KB; reviewers drop their own `.npy` + `.json` pairs into `./data/` and re-run. A clear preflight error spells out what to add if the folder is empty.
- `visit_db.json` no longer ships as a static file. It's auto-generated at orchestrator startup from the per-visit JSON sidecars in `./data/` (the single source of truth), with hex UUID keys to match what `consumer.py` looks up. The file is gitignored. The same builder is exposed as `python -m hemispheric.preflight` for cases where you need the file to exist before running the orchestrator (for example, building the consumer Docker image).
- All tunable parameters consolidated into `config.yaml`: dataset shape, wire-protocol chunk duration, runtime ports/host/consumer_mode, and the default filter cohort. CLI flags remain for per-run overrides. Config search order: `$HEMISPHERIC_CONFIG` env var → `./config.yaml` → package-relative default → built-in fallback.
- Repository layout reorganized so the data team's files (`Dockerfile`, `consumer.py`, `requirements.txt`) sit at the root unmodified, and everything new is under `hemispheric/`.

---

## Phase 4 — Observability and reliability

**Directive:** Make the system honest about what it did. Time the hot paths. Catch and quarantine bad input rather than crashing the run. Verify no data was lost on the wire, and on failures report which visits were affected so the operator can act.

**What got added:**

- `@timed` decorator (`hemispheric/timing.py`) applied to the three main orchestration phases — `load_all_visits`, `plan_chunks`, `run_provider` — emitting elapsed-time logs at INFO level. Works on sync and async functions.
- Quarantine for corrupt input (`hemispheric/preflight.py`): visits that fail any check (malformed JSON, missing required fields, missing `.npy`, unreadable `.npy` header) are moved to `<data_dir>/.quarantine/` with a sibling `.reason.txt` explaining why. The run continues with the surviving visits, and a final count of quarantined files is logged.
- `StreamStats` dataclass in `hemispheric/provider.py`: per-port accounting of `chunks_planned`, `chunks_streamed`, `bytes_streamed`, `clean_close`. The provider catches not just `ConnectionResetError`/`BrokenPipeError` but also `MemoryError` and `OSError`, always returning accurate partial counts even when streaming aborts mid-shard.
- Integrity summary at end of run: orchestrator aggregates the per-port stats and emits either an `integrity OK: N/N chunks streamed` line or an `integrity FAIL: ...` block with per-port and per-visit breakdowns (top 5 most-affected visits per port shown in logs). Exit code 0 on clean runs, 2 on integrity failure, 1 on startup errors.
- Per-run audit log: every run writes `./run-reports/<timestamp>.json` containing the filter, visits loaded/matched, total chunks planned/streamed/bytes, and per-port stats including the full list of affected visit_ids and their per-visit chunk-loss counts when a failure occurred. The folder is gitignored.
- Test coverage extended to 71 tests, including a dedicated `tests/test_provider_failures.py` that asserts partial stats survive `ConnectionResetError`, `MemoryError`, `OSError`, and the chunk-zero immediate-failure case.

---

## Original exercise brief (verbatim, for reference)

> AI Platform Engineer - Exercise 1 Data Infrastructure for AI Training [...]
> Write a tool that orchestrates the training process. The orchestrator gets command line arguments in order to train the model based on a certain filter of the data. For instance: `python run_training.py --gender female --age >20`. [...] Write the data provider process. It should read the data that matches the filters, start a TCP server on the desired ports, and send all of the data as random chunks of 10 seconds. Each chunk should start with the visit ID (it's a UUIDv4, so 128bit) followed by the float32 array. [...] Design your architecture (but don't implement) such that the producers and consumers could run in a cluster.

Everything else followed from interpreting this brief and the data team's supplied files.
