"""Configuration loading from config.yaml.

A Config object holds the dataset properties, wire-protocol settings, and
runtime defaults that everything else in the project derives from. The values
live in config.yaml at the project root by default; override the path with the
HEMISPHERIC_CONFIG environment variable for tests or alternate deployments.

The Config dataclasses validate their fields in __post_init__: positive
integers where required, dtype must be a supported numpy type, consumer_mode
must be one of the three known modes. This catches typos and bad values at
import time instead of producing mysterious errors deeper in the pipeline.

Derived values (samples_per_chunk, chunk_payload_bytes, etc.) live on the
top-level Config as properties so they always stay consistent with the
source-of-truth fields. Change chunk_duration_sec from 10 to 5 in the YAML
and every derived value updates without code edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Mapping from dtype names we accept in config to their byte size.
_DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
}


@dataclass(frozen=True)
class DatasetConfig:
    """Properties of the source EEG data files."""

    sample_rate_hz: int = 1000
    num_channels: int = 4
    dtype: str = "float32"
    data_dir: Path = field(default_factory=lambda: Path("./data"))

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError(
                f"dataset.sample_rate_hz must be positive, got {self.sample_rate_hz}"
            )
        if self.num_channels <= 0:
            raise ValueError(
                f"dataset.num_channels must be positive, got {self.num_channels}"
            )
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(
                f"dataset.dtype must be one of {list(_DTYPE_BYTES)}, "
                f"got {self.dtype!r}"
            )


@dataclass(frozen=True)
class WireProtocolConfig:
    """On-the-wire format settings."""

    chunk_duration_sec: int = 10
    uuid_bytes: int = 16

    def __post_init__(self) -> None:
        if self.chunk_duration_sec <= 0:
            raise ValueError(
                f"wire_protocol.chunk_duration_sec must be positive, "
                f"got {self.chunk_duration_sec}"
            )
        if self.uuid_bytes != 16:
            raise ValueError(
                f"wire_protocol.uuid_bytes must be 16 (UUIDv4 is 128 bits), "
                f"got {self.uuid_bytes}"
            )


@dataclass(frozen=True)
class RuntimeConfig:
    """Orchestrator runtime defaults. CLI flags override these per run."""

    host: str = "0.0.0.0"
    ports: tuple[int, ...] = (5000, 5001)
    consumer_mode: str = "real"
    consumer_cmd: str | None = None
    log_level: str = "INFO"
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.ports:
            raise ValueError("runtime.ports must be a non-empty list")
        if self.consumer_mode not in {"real", "none"}:
            raise ValueError(
                f"runtime.consumer_mode must be 'real' or 'none', "
                f"got {self.consumer_mode!r}"
            )
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(
                f"runtime.log_level must be DEBUG|INFO|WARNING|ERROR, "
                f"got {self.log_level!r}"
            )


@dataclass(frozen=True)
class FilterConfig:
    """Subject-selection filter for a training run.

    Each field is optional; an empty filter matches every visit. Categorical
    fields combine with OR within the list; age constraints combine with AND
    (so ['>=18', '<65'] is a range). wears_glasses uses None for "either".
    """

    description: str = ""
    gender: tuple[str, ...] = ()
    age: tuple[str, ...] = ()
    dominant_hand: tuple[str, ...] = ()
    wears_glasses: bool | None = None
    visit_ids: tuple[str, ...] = ()
    person_ids: tuple[str, ...] = ()
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    """Top-level config: dataset + wire_protocol + runtime + filter, plus derived values."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    wire_protocol: WireProtocolConfig = field(default_factory=WireProtocolConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)

    @property
    def bytes_per_float(self) -> int:
        return _DTYPE_BYTES[self.dataset.dtype]

    @property
    def samples_per_chunk(self) -> int:
        return self.dataset.sample_rate_hz * self.wire_protocol.chunk_duration_sec

    @property
    def floats_per_chunk(self) -> int:
        return self.samples_per_chunk * self.dataset.num_channels

    @property
    def chunk_payload_bytes(self) -> int:
        return self.floats_per_chunk * self.bytes_per_float

    @property
    def chunk_wire_bytes(self) -> int:
        return self.chunk_payload_bytes + self.wire_protocol.uuid_bytes


def _from_dict(data: dict[str, Any]) -> Config:
    """Build a Config from a parsed YAML dict. Missing sections use defaults."""
    dataset_data = dict(data.get("dataset", {}))
    if "data_dir" in dataset_data:
        dataset_data["data_dir"] = Path(dataset_data["data_dir"])

    runtime_data = dict(data.get("runtime", {}))
    if "ports" in runtime_data:
        runtime_data["ports"] = tuple(runtime_data["ports"])

    filter_data = dict(data.get("filter", {}))
    for tuple_field in ("gender", "age", "dominant_hand",
                        "visit_ids", "person_ids", "names"):
        if tuple_field in filter_data and filter_data[tuple_field] is not None:
            filter_data[tuple_field] = tuple(filter_data[tuple_field])
        elif tuple_field in filter_data and filter_data[tuple_field] is None:
            filter_data[tuple_field] = ()

    return Config(
        dataset=DatasetConfig(**dataset_data),
        wire_protocol=WireProtocolConfig(**data.get("wire_protocol", {})),
        runtime=RuntimeConfig(**runtime_data),
        filter=FilterConfig(**filter_data),
    )


def _find_config_path() -> Path | None:
    """Find config.yaml in the standard search locations."""
    env_path = os.environ.get("HEMISPHERIC_CONFIG")
    if env_path:
        return Path(env_path)

    candidates = [
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent.parent / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path | None = None) -> Config:
    """Load configuration from a YAML file, with fallback to embedded defaults.

    Search order if `path` is None:
      1. $HEMISPHERIC_CONFIG environment variable
      2. ./config.yaml (relative to current working directory)
      3. config.yaml next to the hemispheric package (project root)
      4. Embedded defaults (matching the values in the shipped config.yaml)
    """
    if path is None:
        path = _find_config_path()

    if path is None or not path.exists():
        return Config()

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _from_dict(data)
