# Coding Agent Prompts

This project was built with the help of Claude as a coding agent. Per the exercise's request, the most important prompts that shaped the final code are recorded below. The conversation also contained back-and-forth clarification on EEG fundamentals and the wire payload layout, but those were context-setting rather than design directives.

The original 100-file dataset and consumer source were not provided, so the agent worked from the spec text alone, generated a synthetic dataset, and wrote a mock consumer that satisfies the documented protocol. Any design choice not pinned by the spec (channel layout, endianness, sharding strategy, sequencing) is documented in `README.md` and `ARCHITECTURE.md`.

## Prompts

### Prompt 1: the exercise itself (verbatim)

> AI Platform Engineer - Exercise 1 Data Infrastructure for AI Training [...]
> Write a tool that orchestrates the training process. The orchestrator gets command line arguments in order to train the model based on a certain filter of the data. For instance: python run_training.py --gender female --age >20. [...] Write the data provider process. It should read the data that matches the filters, start a TCP server on the desired ports, and send all of the data as random chunks of 10 seconds. Each chunk should start with the visit ID (it's a UUIDv4, so 128bit) followed by the float32 array. [...] Design your architecture (but don't implement) such that the producers and consumers could run in a cluster.

This was the design brief. Everything else followed from interpreting it.

### Prompt 2: clarification on the data and payload

> what is EEG data?
> how the payload looks like?

These shaped the protocol module and the constants in `__init__.py`. The agent worked out the per-chunk size of 160,016 bytes from the spec and surfaced the two ambiguities that the consumer code would normally settle (channel layout and endianness), then chose interleaved + little-endian as the documented assumption.

### Prompt 3: proceed without the dataset

> i dont have the zip file, lets assume that we have the data

This was the directive to build a self-contained system that includes a synthetic dataset generator (`samplegen.py`) and a mock consumer (`consumer.py`) so the deliverable is runnable end-to-end without the original files.

### Prompt 4: actually use the real data

> all the part of the consumer and the data: [...] is in the attached zip file

After the user supplied the real consumer source, the real metadata schema, and a sample of real `.npy` files, the codebase was refactored end to end:

- File extension changed from `.eeg` (raw binary) to `.npy` (numpy native), which means reading via `np.load(path, mmap_mode='r')` and slicing the array rather than `seek + read` on byte offsets.
- `ChunkRef.byte_offset` (bytes) became `ChunkRef.sample_start` (timesteps), since the `.npy` header is variable-length and sample indexing is what numpy expects.
- Metadata schema updated: `subject_name` to `person_name`, added `person_id`, `wears_glasses`, `date_of_visit`, `dominant_hand`.
- Filter language extended with `--wears-glasses` / `--no-glasses` (boolean), `--dominant-hand` (categorical set), `--person-id` (UUID).
- `_FileHandlePool` replaced with `_ArrayPool` (mmap'd numpy arrays).
- The `samplegen` module rewritten to emit `.npy` files with the real field schema, including a `visit_db.json` matching the format of the supplied one.

End-to-end validation: ran the orchestrator with `--consumer-cmd "python consumer.py 5000 5001"` against five real visits, filtered to females over 20, streamed 240 chunks across two ports. Consumer received every chunk with the correct UUID and reshape-compatible payload.

## Design decisions the agent owned

These were not specified in the prompts and were chosen and documented by the agent:

- **Filter syntax.** Repeatable flags with comparator-aware values for numeric fields (`--age ">20"`), simple equality with OR-on-repeat for categorical fields (`--gender female`).
- **Chunk plan as cheap pointers.** Plan as a list of `ChunkRef` (50 bytes each), shuffle once globally, shard round-robin across ports. The actual 160 KB payloads are read on demand.
- **One-connection-per-port provider model.** Each port serves exactly one consumer connection, streams its shard, then closes.
- **Sequencing via ready event.** Provider exposes an `asyncio.Event` so the orchestrator can defer launching the consumer until every port is bound.
- **Cluster design.** Coordinator-as-work-queue (Redis Streams or similar), stateless provider pods with object-storage or RWX-volume backing, K8s Job orchestrator. Documented in `ARCHITECTURE.md`.
- **Tests.** 43 unit and integration tests including two real-socket end-to-end tests in `tests/test_end_to_end.py`.

## What the agent got wrong on first pass

A small race in the orchestrator: both the provider task and the consumer task were created with `asyncio.create_task` and run concurrently. Without sequencing, the consumer's `connect()` could fire before the provider's servers had finished binding, producing intermittent `ECONNREFUSED`. The fix was to expose an `asyncio.Event` from `run_provider` that fires once every port has bound; the orchestrator awaits it before starting the consumer.

Also the synthetic-data generator initially asserted that durations had to be multiples of 10 seconds. That assertion was wrong; the chunker already drops trailing partial chunks, and forcing aligned durations would have hidden the partial-chunk test case.

Both of these surfaced when running the orchestrator end-to-end and when running the test suite, respectively, and were fixed in the same session.

## What the agent noticed but didn't fix

The provided `consumer.py` validates against `visit_db.json` using `uuid_bytes.hex()` (no hyphens), but the supplied `visit_db.json` keys are hyphenated. The lookup will always miss and print a warning. This was identified but left alone because the consumer is provided as-is; the README documents the quirk and offers two workarounds (no `visit_db.json` in the working directory, or pre-transform the keys).
