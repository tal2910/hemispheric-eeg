# Deployment & Installation Guide

Step-by-step instructions to download, install, and run Hemispheric EEG on your laptop or server. Pick the section for your platform.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Getting the code](#getting-the-code)
3. [Installation on Windows](#installation-on-windows)
4. [Installation on macOS and Linux](#installation-on-macos-and-linux)
5. [Add your data, then run](#add-your-data-then-run)
6. [Docker deployment](#docker-deployment)
7. [Deployment on a remote server](#deployment-on-a-remote-server)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Minimum version | How to check |
|---|---|---|
| Python | 3.10 | `python --version` |
| pip | 21.0 | `pip --version` |
| git | 2.30 (optional, for cloning) | `git --version` |
| Docker | 20.10 (optional, for containerized consumer) | `docker --version` |

If `python` doesn't resolve, try `python3`. If neither works, install Python from [python.org](https://www.python.org/downloads/) and check "Add Python to PATH" during installation.

---

## Getting the code

Three ways to obtain the project. Pick whichever is easiest for you.

### Option A — Clone with git (recommended)

```bash
git clone https://github.com/YOUR_ORG/hemispheric-eeg.git
cd hemispheric-eeg
```

### Option B — Download as ZIP

1. On the GitHub page, click the green **Code** button → **Download ZIP**.
2. Extract the ZIP to a folder you'll remember (e.g. `~/projects/` or `C:\Users\you\Documents\`).
3. Open a terminal in that folder.

### Option C — Download the tarball

If your team distributes a `hemispheric-eeg.tar.gz`:

```bash
# Linux / macOS
tar -xzf hemispheric-eeg.tar.gz
cd hemispheric-eeg

# Windows (cmd or PowerShell)
tar -xzf hemispheric-eeg.tar.gz
cd hemispheric-eeg
```

After extraction your folder should contain `run_training.py`, `consumer.py`, `config.yaml`, plus subdirectories `hemispheric/`, `data/`, and `tests/`. If any are missing, the extraction is incomplete.

**Note:** the `data/` directory ships **empty** — only a placeholder `README.md` and `.gitkeep` are inside. Production EEG datasets are gigabytes per cohort, so we don't ship them in the repo. You'll add your data files in the next section before running.

---

## Installation on Windows

```cmd
:: 1. Enter the project directory
cd path\to\hemispheric-eeg

:: 2. (Recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 3. Install dependencies
pip install numpy pyyaml

:: 4. (Optional) install dev dependencies for the test suite
pip install pytest pytest-asyncio

:: 5. Verify
python -c "import numpy, yaml; print('Installation OK')"
```

If you skip the venv, the install still works but goes into your global Python; we recommend the venv to keep things isolated.

---

## Installation on macOS and Linux

```bash
# 1. Enter the project directory
cd path/to/hemispheric-eeg

# 2. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install numpy pyyaml

# 4. (Optional) install dev dependencies for the test suite
pip install pytest pytest-asyncio

# 5. Verify
python -c "import numpy, yaml; print('Installation OK')"
```

You can verify the install by running the test suite:

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -q
```

---

## Add your data, then run

### The repo ships without data — you must add it

The `./data/` directory is intentionally empty. Production EEG datasets are too large to ship inside a deployment artifact (a single cohort can be 1–100+ GB), so the repository excludes them entirely. **Before you can run the orchestrator, you have to place your data team's `.npy` + `.json` files into `./data/`.**

Each visit is a pair of files sharing the same UUID filename stem:

```
data/
├── <visit-uuid-1>.npy           shape (timesteps, 4) float32 at 1000 Hz
├── <visit-uuid-1>.json          metadata: visit_id, person_id, person_name,
│                                age, gender, wears_glasses, date_of_visit,
│                                dominant_hand
├── <visit-uuid-2>.npy
├── <visit-uuid-2>.json
└── ...
```

See `data/README.md` inside the repo for the full schema and a synthetic-data snippet if you want to smoke-test the install without real data.

### Copy your files in

#### Linux / macOS

```bash
cp /path/to/your/visits/*.npy ./data/
cp /path/to/your/visits/*.json ./data/
```

#### Windows

```cmd
copy "C:\path\to\your\visits\*.npy" data\
copy "C:\path\to\your\visits\*.json" data\
```

#### Alternative: point at data elsewhere (no copying)

If your data lives somewhere else and you'd rather not copy it, edit `config.yaml`:

```yaml
dataset:
  data_dir: /path/to/your/visits      # absolute path, anywhere on disk
```

Then `./data/` can stay empty. The orchestrator reads from whatever `dataset.data_dir` points to.

> Note: `visit_db.json` is **not** something you need to copy or maintain — the orchestrator builds it automatically at startup from the per-visit `.json` sidecars in `./data/`. If you have a `visit_db.json` from elsewhere, ignore it; it'll be regenerated.

### Run

```bash
python run_training.py
```

That's the entire command. All parameters (filter cohort, ports, data directory, consumer mode) live in `config.yaml`. The orchestrator will:

1. Check that `./data/` contains visit files (clear error if empty).
2. Build `visit_db.json` from the per-visit sidecars.
3. Apply the filter from `config.yaml` (default: `gender=female AND age>20`).
4. Plan 10-second chunks across the configured ports.
5. Spin up TCP servers and spawn the data team's `consumer.py` as a subprocess.
6. Stream all chunks; both sides exit cleanly when done.

Expected output (the numbers depend on how many visits you added):

```
[INFO] orchestrator: loaded 5 visits from data
[INFO] hemispheric.preflight: built visit_db.json with 5 entries from data
[INFO] orchestrator: applying filter: gender in ['female'], age >20
[INFO] orchestrator: 2/5 visits matched
[INFO] orchestrator: planned 240 chunks across 2 ports (per-port: [120, 120])
[INFO] hemispheric.provider: all 2 ports listening
[INFO] orchestrator: launching consumer subprocess: python -u consumer.py 5000 5001
[INFO] Received visit_id ... shape (40000,) on port 5000
...
[INFO] orchestrator: integrity OK: 240/240 chunks streamed across 2 ports (38.4 MB total)
[INFO] orchestrator: run report written to run-reports/20260520T120000.json
```

(The example above assumes 5 visits in `./data/`; the actual line counts scale with your dataset size.)

If `./data/` is empty, you'll get a friendly error pointing you back to this section:

```
No .npy files found in 'data'.

This repository ships without bundled EEG data because production
datasets can be hundreds of GB. To run training, place your data
team's .npy and .json files into 'data'. Each visit needs
a matched pair sharing the same UUID stem ...
```

### Smoke-testing without real data

If you want to verify the install before your dataset is ready, generate one synthetic visit:

```bash
python -c "
import json, uuid, numpy as np
vid = str(uuid.uuid4())
np.save(f'data/{vid}.npy', np.random.randn(120 * 1000, 4).astype(np.float32))
json.dump({
    'visit_id': vid, 'person_id': str(uuid.uuid4()), 'person_name': 'Test Subject',
    'age': 30, 'gender': 'female', 'wears_glasses': False,
    'date_of_visit': '2026-01-01', 'dominant_hand': 'right',
}, open(f'data/{vid}.json', 'w'))
"
python run_training.py
```

This makes one 2-minute synthetic visit that satisfies the default filter so the pipeline runs end-to-end. Delete the file when done.

---

## Docker deployment

The project's `Dockerfile` containerizes **only the consumer**, matching the data team's production deployment model. The orchestrator and provider run as native Python.

### Build the consumer image

```bash
docker build -t hemispheric-consumer .
```

### Run the consumer in a container

Set `runtime.consumer_mode: none` in `config.yaml` (so the orchestrator doesn't try to spawn its own consumer), then:

```bash
# Terminal 1: provider as native Python
python run_training.py

# Terminal 2: consumer in Docker (Linux only with --network host)
docker run --rm --network host hemispheric-consumer \
    uv run python -u consumer.py 5000 5001
```

`--network host` is required because the consumer hardcodes `localhost`. On macOS or Windows Docker Desktop, run the consumer as local Python instead.

---

## Deployment on a remote server

For a private server (e.g. an internal EC2 instance, a bare metal box, a workstation accessed via SSH):

### 1. Get the code onto the server

```bash
# Option A: clone directly
ssh user@server
git clone https://github.com/YOUR_ORG/hemispheric-eeg.git
cd hemispheric-eeg

# Option B: copy from your laptop
scp -r hemispheric-eeg user@server:/path/to/destination/
ssh user@server "cd /path/to/destination/hemispheric-eeg"
```

### 2. Install Python and dependencies

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pyyaml

# CentOS / RHEL / Amazon Linux
sudo yum install -y python3 python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pyyaml
```

### 3. Verify

```bash
python -m pytest tests/ -q
# Should print "71 passed in ~1s"
```

### 4. Set up persistent operation

The filter cohort, ports, and consumer mode live in `config.yaml` on the server — edit once, then run. For one-off runs:

```bash
python run_training.py
```

For background runs that survive your SSH disconnect:

```bash
# Using nohup
nohup python run_training.py > training.log 2>&1 &

# Using tmux (recommended)
tmux new -s training
python run_training.py
# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training

# Using systemd (production-grade)
# See docs/systemd-service.md for a sample unit file
```

### 5. Firewall configuration

If your consumer runs on a different machine than the provider:

```bash
# Open ports 5000-5001 on the provider host (Ubuntu)
sudo ufw allow 5000:5001/tcp

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 5000:5001 -j ACCEPT
```

And tell the orchestrator to bind to all interfaces (not just localhost):

```bash
# Edit config.yaml to set:
# runtime:
#   host: 0.0.0.0     # already the default
```

The consumer on the other machine then connects to `<server-ip>:5000` instead of `localhost`. Note: the data team's bundled `consumer.py` hardcodes `localhost`. To connect across machines, either modify the consumer or use port forwarding.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: hemispheric` | `hemispheric/` package directory missing or empty | Re-extract / re-clone; verify the folder contains 9 `.py` files |
| `ModuleNotFoundError: yaml` | `pyyaml` not installed | `pip install pyyaml` |
| `ModuleNotFoundError: numpy` | `numpy` not installed | `pip install numpy` |
| `python: command not found` | Python not in PATH | Use `python3`, or reinstall Python with "Add to PATH" checked |
| `Address already in use` on ports 5000/5001 | Previous run didn't release ports | Wait ~30 seconds, or use `--ports 6000 6001` |
| `Permission denied` binding port | Trying to bind below port 1024 without root | Use ports above 1024 (default 5000+ is fine) |
| `No matching visits` | Filter didn't match any visit | Loosen `filter.gender` / `filter.age` in `config.yaml`, or check that the visits you copied actually match the configured cohort |
| `No .npy files found in 'data'` | Empty data folder; you skipped the "Add your data" step | Copy your `.npy` + `.json` pairs into `./data/`, or point `dataset.data_dir` in `config.yaml` at the right path |
| `Visit ID not found in visit_db.json` warnings | A visit's UUID isn't in the consumer's lookup table | Should not happen with auto-generated `visit_db.json` — if it does, delete `visit_db.json` and re-run so the orchestrator rebuilds it from your sidecars |
| `ECONNREFUSED` from consumer | Consumer started before provider bound | Should be impossible with default flow; if you see this, file an issue with the full log |
| Test failures | Wrong Python version or missing deps | Check `python --version` >= 3.10 and dependencies installed |

For any other issue, gather the following and file a GitHub issue:

```bash
python --version
pip list | grep -E "numpy|pyyaml|pytest"
python -m pytest tests/ -v 2>&1 | head -50
ls -la
ls -la hemispheric/
ls -la data/
```

---

## Next steps

Once the basic run works:

- Read [`README.md`](README.md) for the project overview.
- Read [`ARCHITECTURE.md`](ARCHITECTURE.md) for the cluster deployment design.
- Read [`config.yaml`](config.yaml) to see what's tunable.
- Run `python run_training.py --help` for the full CLI surface.
- Run `python -m pytest tests/ -v` to see what the test suite covers.
