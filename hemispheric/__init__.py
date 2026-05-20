"""Hemispheric EEG data infrastructure for AI training.

This is the package's public surface. All tunable parameters live in
config.yaml at the project root; the constants below are loaded from there at
import time and re-exported so the rest of the package can use them via
`from hemispheric import SAMPLES_PER_CHUNK`.

------------------------------------------------------------------------------
Configuration source-of-truth: config.yaml
------------------------------------------------------------------------------
                            from config.yaml field         derived from
SAMPLE_RATE_HZ           ←  dataset.sample_rate_hz
NUM_CHANNELS             ←  dataset.num_channels
BYTES_PER_FLOAT          ←  dataset.dtype                  → 4 (float32) or 8 (float64)
CHUNK_DURATION_SEC       ←  wire_protocol.chunk_duration_sec
UUID_BYTES               ←  wire_protocol.uuid_bytes
SAMPLES_PER_CHUNK                                          sample_rate_hz × chunk_duration_sec
FLOATS_PER_CHUNK                                           samples_per_chunk × num_channels
CHUNK_PAYLOAD_BYTES                                        floats_per_chunk × bytes_per_float
CHUNK_WIRE_BYTES                                           chunk_payload_bytes + uuid_bytes
------------------------------------------------------------------------------

Wire layout (with shipped defaults):
  little-endian float32, time-major / channel-interleaved
  [c0_t0, c1_t0, c2_t0, c3_t0, c0_t1, c1_t1, ...]
  10 sec × 1000 Hz × 4 ch × 4 B = 160,000 byte payload
  16-byte UUID prefix + payload  = 160,016 byte chunk
"""

from .config import load_config

_cfg = load_config()

SAMPLE_RATE_HZ      = _cfg.dataset.sample_rate_hz
NUM_CHANNELS        = _cfg.dataset.num_channels
BYTES_PER_FLOAT     = _cfg.bytes_per_float
CHUNK_DURATION_SEC  = _cfg.wire_protocol.chunk_duration_sec
SAMPLES_PER_CHUNK   = _cfg.samples_per_chunk
FLOATS_PER_CHUNK    = _cfg.floats_per_chunk
CHUNK_PAYLOAD_BYTES = _cfg.chunk_payload_bytes
UUID_BYTES          = _cfg.wire_protocol.uuid_bytes
CHUNK_WIRE_BYTES    = _cfg.chunk_wire_bytes

__all__ = [
    "BYTES_PER_FLOAT",
    "CHUNK_DURATION_SEC",
    "CHUNK_PAYLOAD_BYTES",
    "CHUNK_WIRE_BYTES",
    "FLOATS_PER_CHUNK",
    "NUM_CHANNELS",
    "SAMPLE_RATE_HZ",
    "SAMPLES_PER_CHUNK",
    "UUID_BYTES",
]
