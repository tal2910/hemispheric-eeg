"""Visit metadata: load JSON sidecars and pair them with their .npy EEG files.

------------------------------------------------------------------------------
Configuration this module uses: NONE.
This module is intentionally config-free; it reads whatever JSON+NPY pairs the
caller hands it. The caller (cli.py) is what consults config.yaml for the
default data directory.
------------------------------------------------------------------------------

The data team provides one (.npy, .json) pair per visit, named after the visit
UUID, plus a global visit_db.json (used by the consumer for validation; not
needed for filtering, since the per-visit JSONs already carry every field).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class Visit:
    """One subject visit. Pairs a metadata record with its .npy EEG file."""

    visit_id: uuid.UUID
    person_id: uuid.UUID
    person_name: str
    age: int
    gender: str
    wears_glasses: bool
    date_of_visit: str          # ISO date string, kept as-is
    dominant_hand: str          # "right" | "left" | "ambidextrous"
    npy_path: Path
    metadata_path: Path
    extra: dict[str, Any]

    @property
    def visit_id_bytes(self) -> bytes:
        """16 raw bytes, for the on-wire UUID prefix the consumer expects."""
        return self.visit_id.bytes


REQUIRED_FIELDS = (
    "visit_id",
    "person_id",
    "person_name",
    "age",
    "gender",
    "wears_glasses",
    "date_of_visit",
    "dominant_hand",
)


def load_visit(metadata_path: Path) -> Visit:
    """Load one visit. The .npy is expected at the same stem with .npy extension.

    Validation performed here, in order:
      1. JSON parses
      2. All REQUIRED_FIELDS present
      3. Matching .npy file exists
      4. .npy header is readable (catches truncation or format corruption
         without paging actual data into RAM)

    Any failure raises with a descriptive message. iter_visits() catches these
    and quarantines the offending file.
    """
    with metadata_path.open("r", encoding="utf-8") as f:
        record = json.load(f)

    missing = [k for k in REQUIRED_FIELDS if k not in record]
    if missing:
        raise ValueError(f"{metadata_path}: missing required fields {missing}")

    npy_path = metadata_path.with_suffix(".npy")
    if not npy_path.exists():
        raise FileNotFoundError(
            f"{metadata_path}: matching .npy file not found at {npy_path}"
        )

    # Verify the .npy is loadable. mmap_mode='r' reads only the header (~3 KB)
    # — much cheaper than np.load(), but still catches truncation or a corrupt
    # magic number that would explode later inside the provider's hot loop.
    try:
        np.load(npy_path, mmap_mode="r")
    except (ValueError, OSError, EOFError) as e:
        raise ValueError(f"{npy_path}: .npy header unreadable ({e})") from e

    extra = {k: v for k, v in record.items() if k not in REQUIRED_FIELDS}

    return Visit(
        visit_id=uuid.UUID(record["visit_id"]),
        person_id=uuid.UUID(record["person_id"]),
        person_name=str(record["person_name"]),
        age=int(record["age"]),
        gender=str(record["gender"]),
        wears_glasses=bool(record["wears_glasses"]),
        date_of_visit=str(record["date_of_visit"]),
        dominant_hand=str(record["dominant_hand"]),
        npy_path=npy_path,
        metadata_path=metadata_path,
        extra=extra,
    )


def iter_visits(data_dir: Path) -> Iterator[Visit]:
    """Yield every loadable visit found under data_dir.

    Corrupt or incomplete visits (bad JSON, missing fields, missing or
    unreadable .npy) are quarantined: moved to <data_dir>/.quarantine/ with a
    sibling .reason.txt, and iteration continues. This prevents one bad file
    from killing the whole run.

    Convention: per-visit metadata is at <data_dir>/<uuid>.json. We skip the
    global visit_db.json (different schema; auto-generated artifact).
    """
    from .preflight import quarantine_visit  # local to break a cycle

    for metadata_path in sorted(data_dir.glob("*.json")):
        if metadata_path.name == "visit_db.json":
            continue
        try:
            yield load_visit(metadata_path)
        except (ValueError, FileNotFoundError, json.JSONDecodeError, KeyError,
                uuid.error if hasattr(uuid, "error") else ValueError) as e:
            quarantine_visit(metadata_path, str(e))


from .timing import timed

import logging
_log = logging.getLogger(__name__)


@timed
def load_all_visits(data_dir: Path) -> list[Visit]:
    """Load all loadable visits eagerly. Corrupt visits are quarantined and
    skipped; iteration continues so one bad file doesn't kill the run.

    Reports a final count of (loaded, quarantined) so the reviewer can see at
    a glance whether anything went sideways.
    """
    # Count quarantine files before and after so we can report what got moved
    # during this load. The quarantine dir may already exist from a prior run.
    quarantine_dir = data_dir / ".quarantine"
    before = (
        sum(1 for _ in quarantine_dir.glob("*.reason.txt"))
        if quarantine_dir.exists() else 0
    )

    visits = list(iter_visits(data_dir))

    after = (
        sum(1 for _ in quarantine_dir.glob("*.reason.txt"))
        if quarantine_dir.exists() else 0
    )
    newly_quarantined = after - before
    if newly_quarantined > 0:
        _log.warning(
            "%d visit(s) quarantined this run; see %s/*.reason.txt for details",
            newly_quarantined, quarantine_dir,
        )

    if not visits:
        raise FileNotFoundError(f"No visits found under {data_dir}")
    return visits
