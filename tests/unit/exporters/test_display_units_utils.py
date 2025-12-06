# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.exceptions import MetricUnitError
from aiperf.common.models import MetricResult
from aiperf.exporters.display_units_utils import (
    _logger,
    convert_all_metrics_to_display_units,
    parse_unit_from_metric_name,
    to_display_unit,
)


class FakeUnit:
    def __init__(self, name: str, raise_on_convert: bool = False):
        self.value = name
        self._raise = raise_on_convert

    def __eq__(self, other):
        return isinstance(other, FakeUnit) and self.value == other.value

    def convert_to(self, target: "FakeUnit", v: float) -> float:
        if self._raise:
            raise MetricUnitError("Exception raised")
        if self.value == target.value:
            return v
        if self.value == "ns" and target.value == "ms":
            return v / NANOS_PER_MILLIS
        if self.value == "ms" and target.value == "ns":
            return v * NANOS_PER_MILLIS
        raise AssertionError(f"unsupported conversion {self.value}->{target.value}")


class FakeMetric:
    def __init__(self, base: FakeUnit, display: FakeUnit | None):
        self.unit = base
        self.display_unit = display or base
        self.display_order = 0


class FakeRegistry:
    def __init__(
        self,
        base_unit: str,
        display_unit: str | None = None,
        raise_on_convert: bool = False,
    ):
        base = FakeUnit(base_unit, raise_on_convert=raise_on_convert)
        disp = FakeUnit(display_unit) if display_unit else None
        self._metric = FakeMetric(base, disp)

    def get_class(self, _tag):
        return self._metric


class TestDisplayUnitsUtils:
    def test_noop_when_display_equals_base(self):
        reg = FakeRegistry(base_unit="ms", display_unit="ms")
        src = MetricResult(
            tag="request_latency", unit="ms", header="RL", avg=10.0, p90=12.0
        )
        out = to_display_unit(src, reg)
        # No conversion -> same object to keep it cheap
        assert out is src
        assert out.avg == 10.0
        assert out.unit == "ms"

    def test_converts_ns_to_ms_and_returns_copy(self):
        reg = FakeRegistry(base_unit="ns", display_unit="ms")
        src = MetricResult(
            tag="time_to_first_token",
            unit="ns",
            header="TTFT",
            avg=1_500_000.0,
            min=None,
            max=2_000_000.0,
            p90=1_550_000.0,
            p75=1_230_000.0,
            count=7,
        )
        out = to_display_unit(src, reg)
        assert out is not src
        assert out.unit == "ms"
        assert out.avg == pytest.approx(1.5)
        assert out.max == pytest.approx(2.0)
        assert out.p90 == pytest.approx(1.55)
        assert out.p75 == pytest.approx(1.23)
        # count isn't in STAT_KEYS and must not be converted/touched
        assert out.count == 7
        assert src.avg == 1_500_000.0  # original left untouched

    def test_preserves_none_fields(self):
        reg = FakeRegistry(base_unit="ns", display_unit="ms")
        src = MetricResult(
            tag="time_to_first_token",
            unit="ns",
            header="TTFT",
            avg=1_000_000.0,
            p95=None,
        )
        out = to_display_unit(src, reg)
        assert out.p95 is None
        assert out.avg == pytest.approx(1.0)

    def test_logs_error_on_unit_mismatch(self, monkeypatch):
        err_mock = Mock()
        monkeypatch.setattr(_logger, "error", err_mock)
        reg = FakeRegistry(base_unit="ns", display_unit="ms")
        # record claims "ms" but base is "ns"
        src = MetricResult(
            tag="time_to_first_token", unit="ms", header="TTFT", avg=1_000_000.0
        )
        to_display_unit(src, reg)
        assert err_mock.call_count == 1
        msg = err_mock.call_args[0][0]
        assert "does not match the expected unit (ns)" in msg

    def test_warns_and_continues_when_convert_raises(self, monkeypatch):
        warn_mock = Mock()
        monkeypatch.setattr(_logger, "warning", warn_mock)
        # Force convert_to to raise
        reg = FakeRegistry(base_unit="ns", display_unit="ms", raise_on_convert=True)
        src = MetricResult(
            tag="time_to_first_token", unit="ns", header="TTFT", avg=1_000_000.0
        )
        out = to_display_unit(src, reg)
        # Unit string still updated to display (ms), value left as original (since conversion failed)
        assert out.unit == "ms"
        assert out.avg == 1_000_000.0
        assert warn_mock.call_count == 1

    def test_convert_all_metrics_to_display_units(self):
        reg = FakeRegistry(base_unit="ns", display_unit="ms")
        a = MetricResult(
            tag="time_to_first_token", unit="ns", header="TTFT", avg=1_000_000.0
        )
        b = MetricResult(tag="foo", unit="ns", header="Foo", avg=2_000_000.0)
        out = convert_all_metrics_to_display_units([a, b], reg)
        assert set(out.keys()) == {"time_to_first_token", "foo"}
        assert out["time_to_first_token"].unit == "ms"
        assert out["foo"].avg == pytest.approx(2.0)


class TestParseUnitFromMetricName:
    """Tests for parse_unit_from_metric_name function."""

    @pytest.mark.parametrize(
        "metric_name,expected_unit",
        [
            # Time units
            ("request_duration_seconds", "seconds"),
            ("vllm:time_to_first_token_seconds", "seconds"),
            ("processing_time_milliseconds", "milliseconds"),
            ("dynamo_component_nats_service_processing_ms", "milliseconds"),
            ("latency_nanoseconds", "nanoseconds"),
            ("event_time_ns", "nanoseconds"),
            # Size/data units
            ("response_size_bytes", "bytes"),
            ("memory_kilobytes", "kilobytes"),
            ("disk_megabytes", "megabytes"),
            ("storage_gigabytes", "gigabytes"),
            # Count/quantity units
            ("requests_total", "count"),
            ("error_count", "count"),
            ("vllm:generation_tokens", "tokens"),
            ("vllm:prompt_tokens", "tokens"),
            ("dynamo_component_requests", "requests"),
            ("nats_client_in_messages", "messages"),
            ("kvstats_active_blocks", "blocks"),
            ("nats_client_current_connections", "connections"),
            ("frontend_disconnected_clients", "clients"),
            ("nats_service_active_services", "services"),
            ("nats_service_active_endpoints", "endpoints"),
            ("dynamo_component_errors", "errors"),
            ("cache_hits", "hits"),
            ("cache_misses", "misses"),
            ("prefix_cache_queries", "queries"),
            ("vllm:num_preemptions", "preemptions"),
            # Compound suffixes (_X_total -> X, not count)
            ("vllm:iteration_tokens_total", "tokens"),
            ("dynamo_component_nats_service_requests_total", "requests"),
            ("dynamo_component_nats_service_errors_total", "errors"),
            ("dynamo_component_nats_service_processing_ms_total", "milliseconds"),
            ("request_duration_seconds_total", "seconds"),
            ("latency_ns_total", "nanoseconds"),
            ("http_server_requests_messages_total", "messages"),
            ("network_bytes_total", "bytes"),
            ("cache_hits_total", "hits"),
            ("cache_misses_total", "misses"),
            ("scheduler_preemptions_total", "preemptions"),
            # Ratio/percentage units
            ("memory_ratio", "ratio"),
            ("cache_usage_percent", "percent"),
            ("vllm:kv_cache_usage_perc", "percent"),
            # Physical units
            ("temperature_celsius", "celsius"),
            ("voltage_volts", "volts"),
            # Special types
            ("vllm:cache_config_info", "info"),
            # Case insensitivity
            ("REQUEST_DURATION_SECONDS", "seconds"),
            ("Vllm:Kv_Cache_Usage_Perc", "percent"),
        ],
    )  # fmt: skip
    def test_parses_known_suffixes(self, metric_name: str, expected_unit: str):
        """Test that known suffixes are correctly parsed."""
        assert parse_unit_from_metric_name(metric_name) == expected_unit

    @pytest.mark.parametrize(
        "metric_name",
        [
            "dynamo_frontend_inflight_requests_gauge",  # No known suffix
            "vllm:num_requests_running",  # _running is not a unit
            "model_context_length",  # _length is not a unit
            "unknown_metric",
        ],
    )  # fmt: skip
    def test_returns_none_for_unknown_suffixes(self, metric_name: str):
        """Test that unknown suffixes return None."""
        assert parse_unit_from_metric_name(metric_name) is None

    def test_longer_suffix_takes_priority(self):
        """Test that longer suffixes match before shorter ones."""
        # _milliseconds should match before _seconds
        assert parse_unit_from_metric_name("latency_milliseconds") == "milliseconds"
        # _nanoseconds should match before _seconds
        assert parse_unit_from_metric_name("latency_nanoseconds") == "nanoseconds"
        # _tokens_total should match before _total (tokens, not count)
        assert parse_unit_from_metric_name("iteration_tokens_total") == "tokens"
        # _requests_total should match before _total (requests, not count)
        assert parse_unit_from_metric_name("http_requests_total") == "requests"
        # _errors_total should match before _total (errors, not count)
        assert parse_unit_from_metric_name("server_errors_total") == "errors"
        # _ms_total should match before _total (milliseconds, not count)
        assert parse_unit_from_metric_name("processing_ms_total") == "milliseconds"
        # _seconds_total should match before _total (seconds, not count)
        assert parse_unit_from_metric_name("duration_seconds_total") == "seconds"
