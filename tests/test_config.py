"""Tests for config loading and validation."""

from pathlib import Path

import pytest

from hemispheric.config import (
    Config,
    DatasetConfig,
    FilterConfig,
    RuntimeConfig,
    WireProtocolConfig,
    load_config,
)


class TestEmbeddedDefaults:
    def test_default_config_matches_shipped_yaml(self):
        cfg = Config()
        assert cfg.dataset.sample_rate_hz == 1000
        assert cfg.dataset.num_channels == 4
        assert cfg.dataset.dtype == "float32"
        assert cfg.bytes_per_float == 4
        assert cfg.wire_protocol.chunk_duration_sec == 10
        assert cfg.wire_protocol.uuid_bytes == 16
        assert cfg.runtime.host == "0.0.0.0"
        assert cfg.runtime.ports == (5000, 5001)
        assert cfg.runtime.consumer_mode == "real"
        assert cfg.runtime.log_level == "INFO"

    def test_derived_values(self):
        cfg = Config()
        assert cfg.samples_per_chunk == 10_000
        assert cfg.floats_per_chunk == 40_000
        assert cfg.chunk_payload_bytes == 160_000
        assert cfg.chunk_wire_bytes == 160_016


class TestValidation:
    def test_negative_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            DatasetConfig(sample_rate_hz=-1)

    def test_zero_channels_raises(self):
        with pytest.raises(ValueError, match="num_channels"):
            DatasetConfig(num_channels=0)

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype"):
            DatasetConfig(dtype="int16")

    def test_negative_chunk_duration_raises(self):
        with pytest.raises(ValueError, match="chunk_duration_sec"):
            WireProtocolConfig(chunk_duration_sec=-5)

    def test_wrong_uuid_size_raises(self):
        with pytest.raises(ValueError, match="uuid_bytes"):
            WireProtocolConfig(uuid_bytes=8)

    def test_empty_ports_raises(self):
        with pytest.raises(ValueError, match="ports"):
            RuntimeConfig(ports=())

    def test_unknown_consumer_mode_raises(self):
        with pytest.raises(ValueError, match="consumer_mode"):
            RuntimeConfig(consumer_mode="bananas")

    def test_unknown_log_level_raises(self):
        with pytest.raises(ValueError, match="log_level"):
            RuntimeConfig(log_level="VERBOSE")


class TestLoadFromYAML:
    def test_load_full_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "dataset:\n"
            "  sample_rate_hz: 500\n"
            "  num_channels: 8\n"
            "  dtype: float64\n"
            "  data_dir: /tmp/data\n"
            "wire_protocol:\n"
            "  chunk_duration_sec: 5\n"
            "  uuid_bytes: 16\n"
            "runtime:\n"
            "  host: 127.0.0.1\n"
            "  ports: [6000, 6001, 6002]\n"
            "  consumer_mode: none\n"
            "  log_level: DEBUG\n"
            "  seed: 42\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.dataset.sample_rate_hz == 500
        assert cfg.dataset.num_channels == 8
        assert cfg.dataset.dtype == "float64"
        assert cfg.dataset.data_dir == Path("/tmp/data")
        assert cfg.bytes_per_float == 8
        assert cfg.samples_per_chunk == 500 * 5
        assert cfg.floats_per_chunk == 500 * 5 * 8
        assert cfg.chunk_payload_bytes == 500 * 5 * 8 * 8
        assert cfg.runtime.host == "127.0.0.1"
        assert cfg.runtime.ports == (6000, 6001, 6002)
        assert cfg.runtime.consumer_mode == "none"
        assert cfg.runtime.seed == 42

    def test_partial_config_falls_back_to_defaults(self, tmp_path):
        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text(
            "dataset:\n"
            "  sample_rate_hz: 2000\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.dataset.sample_rate_hz == 2000
        assert cfg.dataset.num_channels == 4
        assert cfg.wire_protocol.chunk_duration_sec == 10
        assert cfg.runtime.ports == (5000, 5001)

    def test_missing_file_uses_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "does-not-exist.yaml")
        assert cfg == Config()

    def test_empty_file_uses_defaults(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        assert load_config(cfg_file) == Config()


class TestSearchPath:
    def test_env_var_takes_precedence(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "via_env.yaml"
        cfg_file.write_text("dataset:\n  sample_rate_hz: 999\n")
        monkeypatch.setenv("HEMISPHERIC_CONFIG", str(cfg_file))
        cfg = load_config()
        assert cfg.dataset.sample_rate_hz == 999


class TestFilterConfig:
    def test_empty_filter_defaults(self):
        f = FilterConfig()
        assert f.description == ""
        assert f.gender == ()
        assert f.age == ()
        assert f.dominant_hand == ()
        assert f.wears_glasses is None
        assert f.visit_ids == ()
        assert f.person_ids == ()
        assert f.names == ()

    def test_load_filter_from_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "filter:\n"
            "  description: 'Female subjects over 20'\n"
            "  gender: [female]\n"
            "  age: ['>20']\n"
            "  dominant_hand: [right, left]\n"
            "  wears_glasses: true\n"
            "  names: ['Alice', 'Bob']\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.filter.description == "Female subjects over 20"
        assert cfg.filter.gender == ("female",)
        assert cfg.filter.age == (">20",)
        assert cfg.filter.dominant_hand == ("right", "left")
        assert cfg.filter.wears_glasses is True
        assert cfg.filter.names == ("Alice", "Bob")

    def test_filter_with_null_fields_treated_as_empty(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "filter:\n"
            "  gender: null\n"
            "  age: null\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.filter.gender == ()
        assert cfg.filter.age == ()
