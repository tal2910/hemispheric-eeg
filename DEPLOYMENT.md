# Deployment & Installation Guide

Step-by-step instructions to download, install, and run Hemispheric EEG on your laptop or server. Pick the section for your platform.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Getting the code](#getting-the-code)
3. [Installation on Windows](#installation-on-windows)
4. [Installation on macOS and Linux](#installation-on-macos-and-linux)
5. [First run with bundled sample data](#first-run-with-bundled-sample-data)
6. [Running with your own dataset](#running-with-your-own-dataset)
7. [Docker deployment](#docker-deployment)
8. [Deployment on a remote server](#deployment-on-a-remote-server)
9. [Troubleshooting](#troubleshooting)

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

## First run with bundled sample data

The repository ships with 5 sample EEG visits in `./data/`. You can run end-to-end immediately:

```bash
python run_training.py --gender female --age ">20"
```

This will:
1. Load the 5 bundled visits from `./data/`.
2. Filter to those matching `gender=female AND age>20` (typically 2 of the 5).
3. Plan 240 chunks (120 per visit × 2 visits).
4. Spin up two TCP servers on ports 5000 and 5001.
5. Spawn the real consumer (`consumer.py`) as a subprocess.
6. Stream all 240 chunks; both sides exit cleanly in ~12 seconds.

Expected output:

```
[INFO] orchestrator: loaded 5 visits from data
[INFO] orchestrator: applying filter: gender in ['female'], age >20
[INFO] orchestrator: 2/5 visits matched
[INFO] orchestrator: planned 240 chunks across 2 ports
[INFO] hemispheric.provider: all 2 ports listening
[INFO] orchestrator: launching consumer subprocess: python -u consumer.py 5000 5001
[INFO] Loaded visit_db.json with 99 visits
[INFO] [Thread-5000] Connected to port 5000
[INFO] [Thread-5001] Connected to port 5001
[INFO] Received visit_id ... shape (40000,) on port 5000
...
```

---

## Running with your own dataset

Replace the bundled samples with your real data:

### Linux / macOS

```bash
# Wipe the bundled samples
rm -f data/*.npy data/*.json

# Copy your visits in
cp /path/to/your/visits/*.npy data/
cp /path/to/your/visits/*.json data/

# Replace visit_db.json if needed
cp /path/to/your/visit_db.json .

# Run
python run_training.py --gender female --age ">20"
```

### Windows

```cmd
:: Wipe the bundled samples
del data\*.npy data\*.json

:: Copy your visits in
copy "C:\path\to\your\visits\*.npy" data\
copy "C:\path\to\your\visits\*.json" data\

:: Replace visit_db.json if needed
copy "C:\path\to\your\visit_db.json" .

:: Run
python run_training.py --gender female --age ">20"
```

Alternatively, leave the bundled data in place and use `--data-dir` to point elsewhere:

```bash
python run_training.py --data-dir /srv/eeg-data --gender female --age ">20"
```

---

## Docker deployment

The project's `Dockerfile` containerizes **only the consumer**, matching the data team's production deployment model. The orchestrator and provider run as native Python.

### Build the consumer image

```bash
docker build -t hemispheric-consumer .
```

### Run the consumer in a container

```bash
# Terminal 1: provider as native Python
python run_training.py --gender female --age ">20" --no-consumer

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
# Should print "65 passed in ~1s"
```

### 4. Set up persistent operation

For one-off runs:

```bash
python run_training.py --gender female --age ">20"
```

For background runs that survive your SSH disconnect:

```bash
# Using nohup
nohup python run_training.py --gender female --age ">20" > training.log 2>&1 &

# Using tmux (recommended)
tmux new -s training
python run_training.py --gender female --age ">20"
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
| `No matching visits` | Filter didn't match any visit | Loosen the filter, e.g. drop `--age ">20"` |
| `Visit ID not found in visit_db.json` warnings every chunk | Consumer's hex-vs-hyphen lookup quirk | Documented quirk, data path works fine. Silence by running: `python -c "import json; db=json.load(open('visit_db.json')); json.dump({k.replace('-',''):v for k,v in db.items()}, open('visit_db.json','w'))"` |
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
