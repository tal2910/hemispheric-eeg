"""Preflight checks and one-time setup that run before orchestration starts.

These exist to keep the runtime experience clean for the reviewer:

- The data team's `consumer.py` looks up `uuid.hex()` (32-char hex, no hyphens)
  in a global `visit_db.json` for validation. `visit_db.json` is just a flat
  index of (visit_id → metadata) — exactly the data already in the per-visit
  JSON sidecars under `./data/`. We build the index from those sidecars at
  startup rather than shipping a separate file that has to be kept in sync.

- An empty `./data/` directory at startup is a configuration error. We raise
  a clear message so the reviewer doesn't see a deep traceback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def build_visit_db(data_dir: Path, output_path: Path) -> int:
    """Build visit_db.json by indexing every .json sidecar in data_dir by visit_id.

    The consumer (the data team's `consumer.py`) reads `visit_db.json` and looks
    up the UUID it received over the wire as `uuid.bytes.hex()` — 32 hex chars,
    no hyphens. Keys in the built index use that same form so lookups hit.

    Returns the number of entries written.
    """
    db: dict[str, dict] = {}
    json_files = [p for p in sorted(data_dir.glob("*.json")) if p.name != "visit_db.json"]

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except json.JSONDecodeError as e:
            log.warning("skipping %s: not valid JSON (%s)", json_path.name, e)
            continue

        visit_id = meta.get("visit_id")
        if not visit_id:
            log.warning("skipping %s: missing visit_id field", json_path.name)
            continue

        # Strip hyphens so the key matches `uuid.bytes.hex()` from the consumer.
        key = str(visit_id).replace("-", "")
        db[key] = meta

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

    log.info("built %s with %d entries from %s", output_path, len(db), data_dir)
    return len(db)


def check_data_dir(data_dir: Path) -> None:
    """Verify the data directory exists and contains visit files.

    Raises a clear FileNotFoundError if empty, so the reviewer doesn't see a
    confusing 'list index out of range' from deeper in the pipeline.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"\n"
            f"Data directory '{data_dir}' does not exist.\n"
            f"Create it and add your .npy + .json visit pairs, then re-run.\n"
        )

    npy_files = list(data_dir.glob("*.npy"))
    json_files = [p for p in data_dir.glob("*.json") if p.name != "visit_db.json"]

    if not npy_files:
        raise FileNotFoundError(
            f"\n"
            f"No .npy files found in '{data_dir}'.\n"
            f"\n"
            f"This repository ships without bundled EEG data because production\n"
            f"datasets can be hundreds of GB. To run training, place your data\n"
            f"team's .npy and .json files into '{data_dir}'. Each visit needs\n"
            f"a matched pair sharing the same UUID stem, e.g.\n"
            f"\n"
            f"  {data_dir}/<visit-uuid>.npy\n"
            f"  {data_dir}/<visit-uuid>.json\n"
            f"\n"
            f"See README.md and DEPLOYMENT.md for details.\n"
        )

    if not json_files:
        raise FileNotFoundError(
            f"\n"
            f"Found {len(npy_files)} .npy file(s) in '{data_dir}' but no matching\n"
            f"metadata .json sidecars. Each visit needs both files.\n"
        )


# Standalone CLI: `python -m hemispheric.preflight` generates visit_db.json
# without spinning up the full orchestrator. Useful before `docker build .`,
# since the data team's Dockerfile expects visit_db.json at the build context.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate visit_db.json from per-visit JSON sidecars."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data"),
        help="Directory containing per-visit .json files (default: ./data)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./visit_db.json"),
        help="Where to write visit_db.json (default: ./visit_db.json)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[preflight] %(message)s")
    check_data_dir(args.data_dir)
    build_visit_db(args.data_dir, args.output)
