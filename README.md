# Hemispheric EEG Data Infrastructure

A Python orchestrator and TCP data provider for EEG-based deep-learning training. The orchestrator filters subject visits by metadata, the provider serves matching EEG data over TCP as randomly-ordered 10-second chunks, and the consumer (bundled mock or the provided real one) reads those chunks and trains.

## Quick start

```bash
git clone https://github.com/YOUR_ORG/hemispheric-eeg.git
cd hemispheric-eeg
pip install numpy pyyaml
python run_training.py --gender female --age ">20"
```

The repository ships with 5 sample visits in `./data/` so you can run end-to-end immediately. The orchestrator spawns the bundled consumer as a subprocess, streams ~240 chunks over two TCP ports, and exits in ~12 seconds.

For platform-specific installation (Windows, macOS, Linux), Docker deployment, and remote server setup, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

## Features

- **Filter language**: `--gender female --age ">20" --dominant-hand right` style flags with OR/AND semantics.
- **Configurable chunking**: 10-second windows by default, change `chunk_duration_sec` in `config.yaml` and every derived value updates.
- **Streaming over TCP**: fixed-size 160,016-byte chunks; backpressure via `asyncio.StreamWriter.drain()`.
- **Memory-mapped reads**: 1 GB datasets stream without loading into RAM.
- **Two consumer modes**: real subprocess (default) or provider-only (`--no-consumer`).
- **Cluster-ready design**: same code scales to Kubernetes + Redis Streams; see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Documentation

| File | What it covers |
|---|---|
| [DEPLOYMENT.md](DEPLOYMENT.md) | Step-by-step installation per platform, troubleshooting, remote deployment |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Cluster deployment design (Kubernetes + Redis Streams + S3) |
| [config.yaml](config.yaml) | All tunable parameters with comments |
| [PROMPTS.md](PROMPTS.md) | Prompts used during development |

## Run it (pick the path that matches your setup)

### The exercise's example command, exactly as written

```bash
python run_training.py --gender female --age ">20"
```

Works out of the box. `--data-dir ./data` and `--ports 5000 5001` are defaults; the real `consumer.py` is launched automatically as a subprocess. The only requirements are `numpy` and `pyyaml`.

### Containerized consumer

The provided `Dockerfile` (used as-is, unmodified from the data team) builds the consumer into a container. It only includes the consumer, `requirements.txt`, and `visit_db.json` because that's what the data team shipped.

```bash
docker build -t hemispheric-consumer .
docker run --rm --network host hemispheric-consumer \
    uv run python -u consumer.py 5000 5001
```

The consumer hardcodes `localhost` as the provider host, so the container needs to share the host network namespace (`--network host`, Linux only) to reach the provider. On macOS or Windows Docker Desktop, run the consumer as local Python (`python consumer.py 5000 5001`) instead.

Typical two-process flow with the consumer in Docker:

```bash
# Terminal 1: provider running locally
python run_training.py --gender female --age ">20" --no-consumer

# Terminal 2: consumer in Docker
docker run --rm --network host hemispheric-consumer \
    uv run python -u consumer.py 5000 5001
```

The orchestrator and provider are not containerized in this repo; per the original deployment design, only the consumer is. For a cluster deployment, ARCHITECTURE.md describes how the provider would also become its own image.

### Bring your own training process

The default is to launch the provided `consumer.py` as a subprocess. To wire in your real training job instead:

```bash
python run_training.py \
    --gender female --age ">20" \
    --consumer-cmd "./train_model --data-ports 5000 5001"
```

Or run the provider only and connect from a different machine or process:

```bash
# host A
python run_training.py --gender female --no-consumer

# host B
python consumer.py 5000 5001    # or your training process
```

## For reviewers: testing with your full ~1 GB dataset

The bundled `./data/` folder has 5 sample visits so the project runs out of the box. To test with your full 100-visit (~1 GB) dataset, pick one of three options. **You do not need to load anything into RAM.** See "Scaling and memory" below for the why.

**Option 1: Replace the bundled data in place.**

```bash
rm -f ./data/*.npy ./data/*.json
cp /path/to/your/visits/*.npy ./data/
cp /path/to/your/visits/*.json ./data/
python run_training.py
```

**Option 2: Point at your data without touching `./data/`.**

```bash
python run_training.py --data-dir /path/to/your/visits
```

**Option 3: Provider locally, consumer in the provided Docker image.**

```bash
docker build -t hemispheric-consumer .                                    # build the consumer image
python run_training.py --no-consumer &                                    # start provider locally
docker run --rm --network host hemispheric-consumer \
    uv run python -u consumer.py 5000 5001                                # run consumer container
```

This matches the data team's original deployment intent: their consumer ships as a Docker image, the orchestrator and provider run on the host.

## Scaling and memory

The provider doesn't load EEG data into memory. It uses `np.load(path, mmap_mode='r')`, so the OS page cache handles paging, and each 10-second chunk is a single 160 KB copy from a memory-mapped view to a socket buffer. With 100 files of 19 MB each (1.9 GB of raw data), the resident set stays in the low tens of megabytes for the whole run.

What scales with dataset size:

- **Plan-time work.** Opens each `.npy` to read its shape header (a few bytes per file). 100 file opens, sub-second total.
- **Chunk pointer list.** Around 80 bytes per `ChunkRef` × ~12,000 chunks for 100 visits = ~1 MB.
- **Per-shard mmap pool.** Each port holds one mmap view per source file it serves. These are virtual memory mappings, not RAM commitments; the kernel pages them in and out as needed.

Expected runtime with the real consumer: it sleeps 0.1 seconds per chunk to simulate training, so a full 12,000-chunk run across 2 ports takes about 10 minutes (`12000 / 2 ports × 0.1s`). Two ways to make a reviewer run faster:

```bash
# More parallelism via more ports
python run_training.py --ports 5000 5001 5002 5003

# Smaller subset via filters
python run_training.py --age ">=30" --age "<=60"
```

## The bundled starter dataset

`./data/` contains five subject visits (paired `.npy` + `.json`) plus the team's `visit_db.json`, included so the project runs end-to-end with no setup. For testing at full scale, see "For reviewers" above.

## Repository layout

```
hemispheric-eeg/
├── README.md
├── ARCHITECTURE.md          cluster design (not implemented)
├── PROMPTS.md               coding-agent prompts used
├── Dockerfile               consumer-only image, from the data team (unmodified)
├── .dockerignore
├── pyproject.toml
├── config.yaml                 all tunable parameters
├── run_training.py             orchestrator entrypoint
├── consumer.py                 the real consumer (provided, unmodified)
├── requirements.txt            consumer runtime deps (numpy)
├── visit_db.json               global visit registry (provided, used by consumer)
├── data/                       bundled sample dataset (5 visits)
├── hemispheric/
│   ├── __init__.py             re-exports constants from config.py
│   ├── config.py               YAML loader + validation dataclasses
│   ├── metadata.py             Visit dataclass + JSON loading
│   ├── filters.py              filter parsing and predicates
│   ├── chunking.py             ChunkRef, plan_chunks, shard_round_robin
│   ├── protocol.py             wire serialize/parse
│   ├── provider.py             async TCP servers + numpy mmap pool
│   └── cli.py                  orchestrator implementation
└── tests/                      4 unit test modules, 66 tests
```

The provided files (`Dockerfile`, `consumer.py`, `requirements.txt`, `visit_db.json`) are kept exactly as the data team shipped them. The new code we wrote is everything else: the orchestrator, the provider, the filter language, the chunking and sharding logic, the wire protocol module, the bundled mock consumer, and the tests.

## Configuration

All tunable parameters live in `config.yaml` at the project root, including the filter that selects which subjects to train on. Edit, save, re-run. The CLI flags exist for one-off overrides, but the configured defaults are what an experiment's `config.yaml` should pin down.

```yaml
# config.yaml

dataset:
  sample_rate_hz: 1000
  num_channels: 4
  dtype: float32
  data_dir: ./data

wire_protocol:
  chunk_duration_sec: 10
  uuid_bytes: 16

runtime:
  host: 0.0.0.0
  ports: [5000, 5001]
  consumer_mode: real            # 'real' | 'mock' | 'none'
  log_level: INFO
  seed: null

filter:
  description: "Default experiment cohort"
  gender:        [female]        # OR within the list
  age:           ['>20']         # AND across constraints; can be a range
  dominant_hand: []              # empty = any
  wears_glasses: null            # null | true | false
  visit_ids:     []              # specific UUIDs (OR)
  person_ids:    []              # specific person UUIDs (OR)
  names:         []              # specific person_name values (OR)
```

To define a new experiment cohort, you have three options ordered from most to least "professional":

```bash
# 1. Most professional: copy config.yaml to a new file with the cohort baked in
cp config.yaml experiments/cohort-young-women.yaml
# (edit the filter section)
HEMISPHERIC_CONFIG=experiments/cohort-young-women.yaml python run_training.py

# 2. Edit config.yaml directly and run
python run_training.py

# 3. Override per-run via CLI flags (best for ad-hoc exploration)
python run_training.py --gender male --age "<40"
```

CLI flags override individual fields in the config's `filter` section. If you pass `--gender male`, the gender field is replaced with `['male']` while every other filter field (age, dominant_hand, etc.) still comes from `config.yaml`.

If the training team wants 5-second chunks instead of 10, change `chunk_duration_sec: 5` in the YAML. The derived constants `SAMPLES_PER_CHUNK`, `CHUNK_PAYLOAD_BYTES`, `CHUNK_WIRE_BYTES` all update on the next import. The wire size becomes 80,016 bytes; no other file needs editing.

Validation runs at config load: bad `dtype`, negative `chunk_duration_sec`, empty `ports`, unknown `consumer_mode` all raise `ValueError` at startup with a clear message, not deep inside the pipeline.

The config search order (first hit wins):

1. `$HEMISPHERIC_CONFIG` environment variable
2. `./config.yaml` in the current working directory
3. `config.yaml` next to the `hemispheric` package
4. Built-in defaults (match the shipped values)

Programmatic access:

```python
from hemispheric.config import load_config
cfg = load_config()
print(cfg.samples_per_chunk)   # 10,000 by default
print(cfg.filter.gender)       # ('female',) by default
print(cfg.runtime.ports)       # (5000, 5001) by default
```

All filter flags are repeatable.

| Flag | Semantics on repeat | Example |
|------|---------------------|---------|
| `--gender` | OR | `--gender female --gender other` |
| `--age` | AND (range) | `--age ">=18" --age "<65"` |
| `--name` | OR | `--name "Ivan KERQY"` |
| `--visit-id` | OR | `--visit-id 3e56ffb3-...` |
| `--person-id` | OR | `--person-id 3d97e9a3-...` |
| `--dominant-hand` | OR | `--dominant-hand right --dominant-hand ambidextrous` |
| `--wears-glasses` | n/a | only include subjects who wear glasses |
| `--no-glasses` | n/a | only include subjects who don't |

`--age` accepts comparators `>`, `>=`, `<`, `<=`, `==`, `!=`, `=`, or a bare number (treated as `==`). Quote the value because the shell would otherwise interpret `>` as redirection.

## Wire protocol

Every chunk on the socket is a fixed 160,016 bytes:

| offset | size | contents |
|-------:|-----:|----------|
| 0 | 16 | visit UUID, raw bytes (not the hyphenated string) |
| 16 | 160,000 | float32 samples, little-endian, time-major / channel-interleaved |

Where 160,000 = 10 sec × 1000 Hz × 4 channels × 4 bytes/float. The float32 section is C-contiguous row-major over a `(10000, 4)` array, matching the source `.npy` files. Chunks are concatenated back-to-back with no length prefix; the consumer reads exactly 160,016 bytes per chunk and parses.

## Architecture choices

**Filter language.** Repeatable flags compose naturally: numeric comparators combine with AND so `--age ">=18" --age "<65"` is a range; categorical fields combine with OR so `--gender female --gender other` is a set. New fields are one branch in `VisitFilter.matches`.

**Chunk plan as cheap pointers.** A `ChunkRef` is `(visit_uuid_bytes, npy_path, sample_start)`, around 80 bytes. We build the global list of every chunk across every matching visit, shuffle it once, and shard round-robin across ports. The 160 KB payloads are read on demand. This means we can plan tens of thousands of chunks in milliseconds.

**Round-robin sharding.** Even when the global chunk count isn't divisible by N ports, round-robin balances each shard within one. It also keeps the global random ordering intact within each shard.

**One connection per port.** Each provider port accepts exactly one consumer connection, streams its shard, then closes.

**Backpressure via `writer.drain()`.** The real consumer sleeps 0.1 s per chunk to simulate training. Without `drain`, the provider would outrun it and balloon its asyncio write buffer. With it, the producer naturally throttles to the consumer's read rate.

**Numpy mmap pool.** Each shard keeps one memory-mapped numpy array per source file open for its full duration. Re-mapping per chunk would burn syscalls when many chunks come from the same file (each bundled file holds 120 chunks).

**Sequencing via ready event.** The orchestrator awaits a ready event on the provider before launching the consumer. Without this, `connect()` can race the bind and get `ECONNREFUSED`.

## A note on the consumer's visit_db.json validation

The provided `consumer.py` validates incoming UUIDs against a `visit_db.json` it expects in its working directory. It computes the lookup key with `uuid_bytes.hex()` (32 hex characters, no hyphens), but the supplied `visit_db.json` keys are hyphenated UUIDs. The lookup will always miss and print a warning.

This is a consumer-side quirk we don't fix because the consumer is provided as-is. Two ways to make validation actually succeed: run from a directory without `visit_db.json` (the consumer logs "validation will be skipped" and proceeds normally), or pre-transform `visit_db.json` to hex-keyed form. The data path is correct in both cases; only validation logging differs.

## Tests

```bash
make test
# or:
python -m pytest tests/ -v
```

66 tests covering filter parsing, chunk planning, sharding, wire protocol round-trips, and config loading/validation. The tests don't depend on `./data/` — they fabricate small `.npy` fixtures inline as needed.

## Contributing

Issues and pull requests welcome. Before submitting:

```bash
python -m pytest tests/ -v    # all tests must pass
```

If you change behavior, add a test that demonstrates it. If you change interface, update both `README.md` and `DEPLOYMENT.md`.

## License

MIT, see [LICENSE](LICENSE).
