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
    infer_unit,
    parse_scale_from_description,
    parse_unit_from_description,
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

    @pytest.mark.parametrize(
        "metric_name,expected_unit",
        [
            # New suffixes: _reqs shorthand for requests
            ("sglang:num_running_reqs", "requests"),
            ("num_waiting_reqs", "requests"),
            ("queue_reqs", "requests"),
            # New suffixes: _gb_s for throughput in GB/s
            ("sglang:cache_transfer_gb_s", "GB/s"),
            ("memory_bandwidth_gb_s", "GB/s"),
        ],
    )  # fmt: skip
    def test_parses_new_suffixes(self, metric_name: str, expected_unit: str):
        """Test newly added suffixes."""
        assert parse_unit_from_metric_name(metric_name) == expected_unit


class TestParseScaleFromDescription:
    """Tests for parse_scale_from_description function."""

    @pytest.mark.parametrize(
        "description,expected",
        [
            # Ratio patterns: 0-1 range
            ("GPU cache usage as a percentage (0.0-1.0)", "ratio"),
            ("Cache hit rate (0.0 - 1.0)", "ratio"),
            ("Utilization ratio (0-1)", "ratio"),
            ("Value in range 0.0 to 1.0", "ratio"),
            ("Metric range 0-1", "ratio"),
            ("(0.0–1.0)", "ratio"),  # en-dash
            ("(0.0—1.0)", "ratio"),  # em-dash
            # Ratio patterns: "1 means/is/equals 100" style
            ("KV-cache usage. 1 means 100 percent usage.", "ratio"),
            ("Utilization where 1 means 100%", "ratio"),
            ("Cache fill level (1 = 100%)", "ratio"),
            ("1.0 = 100% full", "ratio"),
            ("Value where 1 is 100%", "ratio"),
            ("1 == 100 percent", "ratio"),
            ("1 equals 100%", "ratio"),
            # Percent patterns: 0-100 range
            ("Memory usage (0-100)", "percent"),
            ("CPU utilization (0.0-100.0)", "percent"),
            ("Value 0-100%", "percent"),
            ("Range 0 to 100", "percent"),
            # No range indicator
            ("Current number of running requests", None),
            ("Tokens processed per second", None),
            ("", None),
            (None, None),
        ],
    )  # fmt: skip
    def test_detects_scale_from_description(
        self, description: str | None, expected: str | None
    ):
        assert parse_scale_from_description(description) == expected

    def test_ratio_takes_priority_over_suffix_naming(self):
        """Test that descriptions with (0.0-1.0) return ratio even if 'percent' appears."""
        # This is the key case: metric named "_percent" but actually 0-1 range
        desc = "GPU cache usage as a percentage (0.0-1.0)"
        assert parse_scale_from_description(desc) == "ratio"


class TestParseUnitFromDescription:
    """Tests for parse_unit_from_description function."""

    @pytest.mark.parametrize(
        "description,expected",
        [
            # Explicit "in X" patterns
            ("Request latency in seconds", "seconds"),
            ("Duration measured in milliseconds", "milliseconds"),
            ("Time in ms", "milliseconds"),
            ("Interval in nanoseconds", "nanoseconds"),
            ("Delay in ns", "nanoseconds"),
            ("Response size in bytes", "bytes"),
            ("Throughput in GB/s", "GB/s"),
            ("Bandwidth in MB/s", "MB/s"),
            ("Generation rate in tokens/s", "tokens/s"),
            ("Rate in tokens/sec", "tokens/s"),
            ("Rate in tokens/second", "tokens/s"),
            ("Serving rate in requests/s", "requests/s"),
            # Parenthetical patterns
            ("Request latency (seconds)", "seconds"),
            ("Processing time (milliseconds)", "milliseconds"),
            ("Delay (ms)", "milliseconds"),
            ("Event time (nanoseconds)", "nanoseconds"),
            ("Interval (ns)", "nanoseconds"),
            ("Size (bytes)", "bytes"),
            ("Throughput (GB/s)", "GB/s"),
            ("Bandwidth (MB/s)", "MB/s"),
            ("Rate (tokens/s)", "tokens/s"),
            ("Rate (requests/s)", "requests/s"),
            # Case insensitivity
            ("Request latency IN SECONDS", "seconds"),
            ("Rate IN TOKENS/S", "tokens/s"),
            # No unit
            ("Current queue depth", None),
            ("Number of active workers", None),
            ("", None),
            (None, None),
        ],
    )  # fmt: skip
    def test_extracts_unit_from_description(
        self, description: str | None, expected: str | None
    ):
        assert parse_unit_from_description(description) == expected


class TestInferUnit:
    """Tests for the combined infer_unit function."""

    def test_scale_takes_priority_over_all(self):
        """Scale from description should override everything else."""
        # Even with _percent suffix and existing_unit="percent", (0.0-1.0) means ratio
        result = infer_unit(
            metric_name="gpu_cache_usage_percent",
            description="GPU cache usage as a percentage (0.0-1.0)",
            existing_unit="percent",
        )
        assert result == "ratio"

    def test_description_unit_over_existing_and_suffix(self):
        """Unit from description should override existing_unit and suffix."""
        result = infer_unit(
            metric_name="some_metric_total",
            description="Value in seconds",
            existing_unit="count",
        )
        assert result == "seconds"

    def test_existing_unit_over_suffix(self):
        """Existing unit should be used if no description unit."""
        result = infer_unit(
            metric_name="some_metric_total",
            description="Some generic description",
            existing_unit="bytes",
        )
        assert result == "bytes"

    def test_suffix_as_fallback(self):
        """Suffix-based inference when no other source provides unit."""
        result = infer_unit(
            metric_name="request_duration_seconds",
            description=None,
            existing_unit=None,
        )
        assert result == "seconds"

    def test_none_when_no_unit_found(self):
        """Return None when no unit can be inferred."""
        result = infer_unit(
            metric_name="some_unknown_metric",
            description="A metric with no unit info",
            existing_unit=None,
        )
        assert result is None

    @pytest.mark.parametrize(
        "metric_name,description,existing_unit,expected",
        [
            # Real-world examples from SGLang/TensorRT-LLM
            (
                "sglang:gen_throughput",
                "Generation throughput in tokens/s",
                None,
                "tokens/s",
            ),
            (
                "trtllm:e2e_request_latency_seconds",
                "End-to-end request latency",
                None,
                "seconds",
            ),
            (
                "dynamo_component_kvstats_gpu_cache_usage_percent",
                "GPU cache usage as a percentage (0.0-1.0)",
                "percent",
                "ratio",
            ),
            (
                "sglang:cache_hit_rate",
                "Cache hit rate (0.0-1.0)",
                None,
                "ratio",
            ),
            (
                "sglang:num_running_reqs",
                "Number of running requests",
                None,
                "requests",
            ),
            (
                "vllm:iteration_tokens_total",
                "Total tokens processed",
                None,
                "tokens",
            ),
        ],
    )  # fmt: skip
    def test_real_world_metrics(
        self,
        metric_name: str,
        description: str,
        existing_unit: str | None,
        expected: str,
    ):
        """Test with real-world metric examples."""
        result = infer_unit(metric_name, description, existing_unit)
        assert result == expected


# All unique metrics from concurrency50 (vLLM), concurrency51 (SGLang), concurrency52 (TensorRT-LLM)
# Format: (metric_name, description, expected_unit)
# expected_unit is what the unit SHOULD be based on metric semantics
REAL_WORLD_METRICS_FROM_EXPORTS = [
    # Dynamo metrics
    ("dynamo_component_errors", "Total number of errors in work handler processing", "errors"),
    ("dynamo_component_inflight_requests", "Number of requests currently being processed by work handler", "requests"),
    ("dynamo_component_kvstats_active_blocks", "Number of active KV cache blocks currently in use", "blocks"),
    ("dynamo_component_kvstats_gpu_cache_usage_percent", "GPU cache usage as a percentage (0.0-1.0)", "ratio"),
    ("dynamo_component_kvstats_gpu_prefix_cache_hit_rate", "GPU prefix cache hit rate as a percentage (0.0-1.0)", "ratio"),
    ("dynamo_component_kvstats_total_blocks", "Total number of KV cache blocks available", "blocks"),
    ("dynamo_component_nats_client_connection_state", "Current connection state of NATS client (0=disconnected, 1=connected, 2=reconnecting)", None),
    ("dynamo_component_nats_client_current_connections", "Current number of active connections for NATS client", "connections"),
    ("dynamo_component_nats_client_in_messages", "Total number of messages received by NATS client", "messages"),
    ("dynamo_component_nats_client_in_total_bytes", "Total number of bytes received by NATS client", "bytes"),
    ("dynamo_component_nats_client_out_messages", "Total number of messages sent by NATS client", "messages"),
    ("dynamo_component_nats_client_out_overhead_bytes", "Total number of bytes sent by NATS client", "bytes"),
    ("dynamo_component_nats_service_active_endpoints", "Number of active endpoints across all services", "endpoints"),
    ("dynamo_component_nats_service_active_services", "Number of active services in this component", "services"),
    ("dynamo_component_nats_service_errors_total", "Total number of errors across all component endpoints", "errors"),
    ("dynamo_component_nats_service_processing_ms_avg", "Average processing time across all component endpoints in milliseconds", "milliseconds"),
    ("dynamo_component_nats_service_processing_ms_total", "Total processing time across all component endpoints in milliseconds", "milliseconds"),
    ("dynamo_component_nats_service_requests_total", "Total number of requests across all component endpoints", "requests"),
    ("dynamo_component_request_bytes", "Total number of bytes received in requests by work handler", "bytes"),
    ("dynamo_component_request_duration_seconds", "Time spent processing requests by work handler", "seconds"),
    ("dynamo_component_requests", "Total number of requests processed by work handler", "requests"),
    ("dynamo_component_response_bytes", "Total number of bytes sent in responses by work handler", "bytes"),
    ("dynamo_component_uptime_seconds", "Total uptime of the DistributedRuntime in seconds", "seconds"),
    ("dynamo_frontend_disconnected_clients", "Number of disconnected clients", "clients"),
    ("dynamo_frontend_inflight_requests", "Number of inflight requests", "requests"),
    ("dynamo_frontend_input_sequence_tokens", "Input sequence length in tokens", "tokens"),
    ("dynamo_frontend_inter_token_latency_seconds", "Inter-token latency in seconds", "seconds"),
    ("dynamo_frontend_model_context_length", "Maximum context length in tokens for a worker serving the model", None),
    ("dynamo_frontend_model_kv_cache_block_size", "KV cache block size in tokens for a worker serving the model", None),
    ("dynamo_frontend_model_max_num_batched_tokens", "Maximum number of batched tokens for a worker serving the model", "tokens"),
    ("dynamo_frontend_model_max_num_seqs", "Maximum number of sequences for a worker serving the model", None),
    ("dynamo_frontend_model_migration_limit", "Maximum number of request migrations allowed for the model", None),
    ("dynamo_frontend_model_total_kv_blocks", "Total KV cache blocks available for a worker serving the model", "blocks"),
    ("dynamo_frontend_output_sequence_tokens", "Output sequence length in tokens", "tokens"),
    ("dynamo_frontend_output_tokens", "Total number of output tokens generated (updates in real-time)", "tokens"),
    ("dynamo_frontend_queued_requests", "Number of requests in HTTP processing queue", "requests"),
    ("dynamo_frontend_request_duration_seconds", "Duration of LLM requests", "seconds"),
    ("dynamo_frontend_requests", "Total number of LLM requests processed", "requests"),
    ("dynamo_frontend_time_to_first_token_seconds", "Time to first token in seconds", "seconds"),
    # SGLang metrics
    ("sglang:cache_hit_rate", "The prefix cache hit rate.", None),  # Ambiguous: _rate could be ratio or per-second
    ("sglang:engine_load_weights_time", "The time taken for the engine to load weights.", None),  # No unit specified
    ("sglang:engine_startup_time", "The time taken for the engine to start up.", None),  # No unit specified
    ("sglang:gen_throughput", "The generation throughput (token/s).", "tokens/s"),
    ("sglang:is_cuda_graph", "Whether the batch is using CUDA graph.", None),  # Boolean flag
    ("sglang:kv_transfer_alloc_ms", "The allocation waiting time of the KV transfer in ms.", "milliseconds"),
    ("sglang:kv_transfer_bootstrap_ms", "The bootstrap time of the KV transfer in ms.", "milliseconds"),
    ("sglang:kv_transfer_latency_ms", "The transfer latency of the KV cache in ms.", "milliseconds"),
    ("sglang:kv_transfer_speed_gb_s", "The transfer speed of the KV cache in GB/s.", "GB/s"),
    ("sglang:mamba_usage", "The token usage for Mamba layers.", None),  # Ambiguous: usage could be count or ratio
    ("sglang:num_decode_prealloc_queue_reqs", "The number of requests in the decode prealloc queue.", "requests"),
    ("sglang:num_decode_transfer_queue_reqs", "The number of requests in the decode transfer queue.", "requests"),
    ("sglang:num_grammar_queue_reqs", "The number of requests in the grammar waiting queue.", "requests"),
    ("sglang:num_paused_reqs", "The number of paused requests by async weight sync.", "requests"),
    ("sglang:num_prefill_inflight_queue_reqs", "The number of requests in the prefill inflight queue.", "requests"),
    ("sglang:num_prefill_prealloc_queue_reqs", "The number of requests in the prefill prealloc queue.", "requests"),
    ("sglang:num_queue_reqs", "The number of requests in the waiting queue.", "requests"),
    ("sglang:num_retracted_reqs", "The number of retracted requests.", "requests"),
    ("sglang:num_running_reqs", "The number of running requests.", "requests"),
    ("sglang:num_running_reqs_offline_batch", "The number of running low-priority offline batch requests(label is 'batch').", None),  # No _reqs suffix
    ("sglang:num_used_tokens", "The number of used tokens.", "tokens"),
    ("sglang:pending_prealloc_token_usage", "The token usage for pending preallocated tokens (not preallocated yet).", None),  # Ambiguous
    ("sglang:per_stage_req_latency_seconds", "The latency of each stage of requests.", "seconds"),
    ("sglang:queue_time_seconds", "Histogram of queueing time in seconds.", "seconds"),
    ("sglang:spec_accept_length", "The average acceptance length of speculative decoding.", None),  # Ambiguous: length of what?
    ("sglang:spec_accept_rate", "The average acceptance rate of speculative decoding (`accepted tokens / total draft tokens` in batch).", None),  # Ambiguous: rate could mean ratio or per-second
    ("sglang:swa_token_usage", "The token usage for SWA layers.", None),  # Ambiguous
    ("sglang:token_usage", "The token usage.", None),  # Ambiguous
    ("sglang:utilization", "The utilization.", None),  # Ambiguous: no range specified
    # TensorRT-LLM metrics
    ("trtllm:e2e_request_latency_seconds", "Histogram of end to end request latency in seconds.", "seconds"),
    ("trtllm:request_queue_time_seconds", "Histogram of time spent in WAITING phase for request.", "seconds"),
    ("trtllm:request_success", "Count of successfully processed requests.", None),  # No recognizable suffix
    ("trtllm:time_per_output_token_seconds", "Histogram of time per output token in seconds.", "seconds"),
    ("trtllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", "seconds"),
    # vLLM metrics
    ("vllm:cache_config_info", "Information of the LLMEngine CacheConfig", "info"),
    ("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.", "seconds"),
    ("vllm:generation_tokens", "Number of generation tokens processed.", "tokens"),
    ("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.", "seconds"),
    ("vllm:iteration_tokens_total", "Histogram of number of tokens per engine_step.", "tokens"),
    ("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", "ratio"),  # "1 means 100 percent" overrides _perc suffix
    ("vllm:num_preemptions", "Cumulative number of preemption from the engine.", "preemptions"),
    ("vllm:num_requests_running", "Number of requests in model execution batches.", None),  # No _requests suffix
    ("vllm:num_requests_waiting", "Number of requests waiting to be processed.", None),  # No _requests suffix
    ("vllm:prefix_cache_hits", "Prefix cache hits, in terms of number of cached tokens.", "hits"),
    ("vllm:prefix_cache_queries", "Prefix cache queries, in terms of number of queried tokens.", "queries"),
    ("vllm:prompt_tokens", "Number of prefill tokens processed.", "tokens"),
    ("vllm:request_decode_time_seconds", "Histogram of time spent in DECODE phase for request.", "seconds"),
    ("vllm:request_generation_tokens", "Number of generation tokens processed.", "tokens"),
    ("vllm:request_inference_time_seconds", "Histogram of time spent in RUNNING phase for request.", "seconds"),
    ("vllm:request_max_num_generation_tokens", "Histogram of maximum number of requested generation tokens.", "tokens"),
    ("vllm:request_params_max_tokens", "Histogram of the max_tokens request parameter.", "tokens"),
    ("vllm:request_params_n", "Histogram of the n request parameter.", None),
    ("vllm:request_prefill_time_seconds", "Histogram of time spent in PREFILL phase for request.", "seconds"),
    ("vllm:request_prompt_tokens", "Number of prefill tokens processed.", "tokens"),
    ("vllm:request_queue_time_seconds", "Histogram of time spent in WAITING phase for request.", "seconds"),
    ("vllm:request_success", "Count of successfully processed requests.", None),
    ("vllm:request_time_per_output_token_seconds", "Histogram of time_per_output_token_seconds per request.", "seconds"),
    ("vllm:time_per_output_token_seconds", "Histogram of time per output token in seconds.DEPRECATED: Use vllm:inter_token_latency_seconds instead.", "seconds"),
    ("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", "seconds"),
]  # fmt: skip


class TestInferUnitRealWorldMetrics:
    """Test infer_unit with all real-world metrics from vLLM, SGLang, and TensorRT-LLM exports.

    These are the EXPECTED units based on metric semantics - what the unit SHOULD be.
    The infer_unit function should be updated to match these expectations.
    """

    @pytest.mark.parametrize(
        "metric_name,description,expected",
        REAL_WORLD_METRICS_FROM_EXPORTS,
        ids=[m[0] for m in REAL_WORLD_METRICS_FROM_EXPORTS],
    )
    def test_infer_unit_real_world(
        self,
        metric_name: str,
        description: str,
        expected: str | None,
    ):
        """Test unit inference for real metrics from server exports."""
        result = infer_unit(metric_name, description, None)
        assert result == expected
