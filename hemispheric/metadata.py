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
    """Load one visit. The .npy is expected at the same stem with .npy extension."""
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
    """Yield every visit found under data_dir.

    Convention: per-visit metadata is at <data_dir>/<uuid>.json. We skip the
    global visit_db.json (it has a different schema and lives alongside the
    per-visit files).
    """
    for metadata_path in sorted(data_dir.glob("*.json")):
        if metadata_path.name == "visit_db.json":
            continue
        yield load_visit(metadata_path)


from .timing import timed


@timed
def load_all_visits(data_dir: Path) -> list[Visit]:
    """Load all visits eagerly. Useful before filtering and chunk planning."""
    visits = list(iter_visits(data_dir))
    if not visits:
        raise FileNotFoundError(f"No visits found under {data_dir}")
    return visits
