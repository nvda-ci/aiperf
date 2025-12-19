# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.models.record_models import MetricResult
from aiperf.records.live_metrics_server import (
    STAT_DISPLAY_NAMES,
    LiveMetricsServer,
    _format_info_metric,
    _format_labels,
    _sanitize_metric_name,
    format_as_prometheus,
)


class TestSanitizeMetricName:
    """Tests for metric name sanitization."""

    def test_lowercase_conversion(self):
        """Test that metric names are converted to lowercase."""
        assert _sanitize_metric_name("InterTokenLatency") == "intertokenlatency"

    def test_hyphen_replacement(self):
        """Test that hyphens are replaced with underscores."""
        assert _sanitize_metric_name("inter-token-latency") == "inter_token_latency"

    def test_space_replacement(self):
        """Test that spaces are replaced with underscores."""
        assert _sanitize_metric_name("inter token latency") == "inter_token_latency"

    def test_special_chars_replacement(self):
        """Test that special characters are replaced with underscores."""
        assert _sanitize_metric_name("metric/name.value") == "metric_name_value"

    def test_leading_digit_prefix(self):
        """Test that leading digits get an underscore prefix."""
        assert _sanitize_metric_name("99percentile") == "_99percentile"

    def test_preserves_underscores(self):
        """Test that existing underscores are preserved."""
        assert _sanitize_metric_name("inter_token_latency") == "inter_token_latency"

    def test_empty_string(self):
        """Test handling of empty string."""
        assert _sanitize_metric_name("") == ""


class TestMetricsToPrometheusText:
    """Tests for Prometheus text format conversion."""

    def test_empty_metrics_list(self):
        """Test that empty metrics list returns empty string."""
        assert format_as_prometheus([]) == ""

    def test_single_metric_with_all_stats(self):
        """Test conversion of a single metric with all stats."""
        metric = MetricResult(
            tag="custom_test_metric",
            header="Custom Test Metric",
            unit="ms",
            avg=10.5,
            p1=2.0,
            p5=3.0,
            p10=4.0,
            p25=6.0,
            p50=9.0,
            p75=12.0,
            p90=15.0,
            p95=18.0,
            p99=25.0,
            min=5.0,
            max=30.0,
            std=3.2,
            sum=1050.0,
            count=100,
        )

        result = format_as_prometheus([metric])

        for stat in STAT_DISPLAY_NAMES:
            if stat == "sum":
                # sum uses _total suffix
                assert "aiperf_custom_test_metric__total" in result
            else:
                assert f"aiperf_custom_test_metric_{stat}_" in result
            assert "# HELP aiperf_custom_test_metric_" in result
            assert "# TYPE aiperf_custom_test_metric_" in result

        # All stats should be gauges
        assert "# TYPE aiperf_custom_test_metric_avg_ gauge" in result
        assert "# TYPE aiperf_custom_test_metric__total gauge" in result

        assert "aiperf_custom_test_metric_avg_ 10.5" in result
        assert "aiperf_custom_test_metric__total 1050.0" in result
        assert "average (in ms)" in result
        assert "total (in ms)" in result

    def test_metric_with_partial_stats(self):
        """Test conversion of a metric with only some stats."""
        metric = MetricResult(
            tag="custom_throughput",
            header="Custom Throughput",
            unit="req_s",
            avg=125.5,
            sum=6275.0,
        )

        result = format_as_prometheus([metric])

        assert "aiperf_custom_throughput_avg_ 125.5" in result
        assert "aiperf_custom_throughput__total 6275.0" in result
        assert "average (in req_s)" in result

        # Check absent stats are not present
        assert "aiperf_custom_throughput_p50" not in result
        assert "aiperf_custom_throughput_p99" not in result

    def test_metric_name_sanitization(self):
        """Test that metric names are properly sanitized."""
        metric = MetricResult(
            tag="Custom-Latency-Metric",
            header="Custom Latency Metric",
            unit="ms",
            avg=10.0,
        )

        result = format_as_prometheus([metric])

        assert "aiperf_custom_latency_metric_avg_" in result
        assert "Custom-Latency-Metric" not in result

    def test_unit_in_help_text(self):
        """Test that unit appears in HELP text."""
        metric = MetricResult(
            tag="test_metric",
            header="Test Metric",
            unit="req_s",
            avg=1.0,
        )

        result = format_as_prometheus([metric])

        assert "# HELP aiperf_test_metric_avg_ Test Metric average (in req_s)" in result

    def test_multiple_metrics(self):
        """Test conversion of multiple metrics."""
        metrics = [
            MetricResult(
                tag="metric_a",
                header="Metric A",
                unit="ms",
                avg=10.0,
            ),
            MetricResult(
                tag="metric_b",
                header="Metric B",
                unit="req_s",
                avg=100.0,
            ),
        ]

        result = format_as_prometheus(metrics)

        assert "aiperf_metric_a_avg_ 10.0" in result
        assert "aiperf_metric_b_avg_ 100.0" in result
        assert "average (in ms)" in result
        assert "average (in req_s)" in result

    def test_output_ends_with_newline(self):
        """Test that output ends with a newline."""
        metric = MetricResult(
            tag="test",
            header="Test",
            unit="ms",
            avg=1.0,
        )

        result = format_as_prometheus([metric])
        assert result.endswith("\n")

    def test_registered_metric_uses_display_unit(self):
        """Test that registered metrics use display_unit from MetricRegistry."""
        # inter_token_latency is registered with display_unit=MILLISECONDS
        metric = MetricResult(
            tag="inter_token_latency",
            header="Inter Token Latency",
            unit="ns",  # Base unit (will be converted to display unit)
            avg=10_000_000.0,  # 10ms in nanoseconds
        )

        result = format_as_prometheus([metric])

        assert "aiperf_inter_token_latency_avg_ms" in result
        assert "average (in milliseconds)" in result
        assert "aiperf_inter_token_latency_avg_ms 10.0" in result


class TestLiveMetricsServer:
    """Tests for LiveMetricsServer initialization."""

    def _create_mock_user_config(
        self,
        live_metrics_port: int = 9090,
        live_metrics_host: str = "127.0.0.1",
        benchmark_id: str | None = None,
        model_names: list[str] | None = None,
        endpoint_type: str = "openai",
        streaming: bool = True,
        concurrency: int | None = 10,
        request_rate: float | None = None,
    ) -> MagicMock:
        """Create a mock UserConfig for testing."""
        user_config = MagicMock()
        user_config.live_metrics_port = live_metrics_port
        user_config.live_metrics_host = live_metrics_host
        user_config.benchmark_id = benchmark_id
        user_config.endpoint.model_names = model_names or ["test-model"]
        user_config.endpoint.type = endpoint_type
        user_config.endpoint.streaming = streaming
        user_config.loadgen.concurrency = concurrency
        user_config.loadgen.request_rate = request_rate
        user_config.model_dump_json.return_value = "{}"
        return user_config

    def test_initialization_defaults(self):
        """Test server initialization with default values from UserConfig."""
        user_config = self._create_mock_user_config(
            live_metrics_port=9090,
            live_metrics_host="127.0.0.1",
        )
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: [],
        )

        assert server._port == 9090
        assert server._host == "127.0.0.1"
        assert server._runner is None

    def test_initialization_custom_host(self):
        """Test server initialization with custom host from UserConfig."""
        user_config = self._create_mock_user_config(
            live_metrics_port=8080,
            live_metrics_host="0.0.0.0",
        )
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: [],
        )

        assert server._port == 8080
        assert server._host == "0.0.0.0"

    def test_initialization_custom_id(self):
        """Test server initialization with custom component ID."""
        user_config = self._create_mock_user_config()
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: [],
            id="custom_server_id",
        )

        assert server.id == "custom_server_id"

    def test_stores_user_config(self):
        """Test that user_config is stored for later access."""
        user_config = self._create_mock_user_config()
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: [],
        )

        assert server._user_config is user_config


class TestStatDisplayNames:
    """Tests for STAT_DISPLAY_NAMES constant."""

    def test_stats_dict_contents(self):
        """Test that STAT_DISPLAY_NAMES contains expected stats."""
        expected_keys = {
            "avg",
            "sum",
            "p1",
            "p5",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "min",
            "max",
            "std",
        }
        assert set(STAT_DISPLAY_NAMES.keys()) == expected_keys

    def test_display_names_are_human_readable(self):
        """Test that display names are human-readable."""
        assert STAT_DISPLAY_NAMES["avg"] == "average"
        assert STAT_DISPLAY_NAMES["sum"] == "total"
        assert STAT_DISPLAY_NAMES["p99"] == "99th percentile"
        assert STAT_DISPLAY_NAMES["std"] == "standard deviation"
        assert STAT_DISPLAY_NAMES["p1"] == "1st percentile"
        assert STAT_DISPLAY_NAMES["p5"] == "5th percentile"
        assert STAT_DISPLAY_NAMES["p10"] == "10th percentile"
        assert STAT_DISPLAY_NAMES["p25"] == "25th percentile"
        assert STAT_DISPLAY_NAMES["p75"] == "75th percentile"

    def test_stats_match_metric_result_fields(self):
        """Test that all stats are valid MetricResult fields."""
        metric = MetricResult(
            tag="test",
            header="Test",
            unit="ms",
        )
        for stat in STAT_DISPLAY_NAMES:
            assert hasattr(metric, stat), f"MetricResult missing field: {stat}"


class TestFormatLabels:
    """Tests for _format_labels function."""

    def test_empty_labels_returns_empty(self):
        """Test that empty labels dict returns empty string."""
        assert _format_labels({}) == ""

    def test_single_label(self):
        """Test formatting a single label."""
        result = _format_labels({"model": "llama-3"})
        assert result == '{model="llama-3"}'

    def test_multiple_labels(self):
        """Test formatting multiple labels."""
        result = _format_labels({"model": "llama-3", "streaming": "true"})
        assert 'model="llama-3"' in result
        assert 'streaming="true"' in result
        assert result.startswith("{")
        assert result.endswith("}")

    def test_escapes_quotes(self):
        """Test that quotes in values are escaped."""
        result = _format_labels({"model": 'test"model'})
        assert r'model="test\"model"' in result

    def test_escapes_backslashes(self):
        """Test that backslashes in values are escaped."""
        result = _format_labels({"path": r"C:\test"})
        assert r'path="C:\\test"' in result


class TestFormatInfoMetric:
    """Tests for _format_info_metric function."""

    def test_empty_labels_returns_empty(self):
        """Test that empty labels dict returns empty string."""
        assert _format_info_metric({}) == ""

    def test_includes_version(self):
        """Test that version is included in info metric."""
        result = _format_info_metric({"test": "value"})

        assert f'version="{version("aiperf")}"' in result

    def test_basic_labels(self):
        """Test formatting with basic labels."""
        labels = {"model": "llama-3", "endpoint_url": "http://localhost:8000"}

        result = _format_info_metric(labels)

        assert "# HELP aiperf_info AIPerf benchmark information" in result
        assert "# TYPE aiperf_info gauge" in result
        assert 'model="llama-3"' in result
        assert 'endpoint_url="http://localhost:8000"' in result
        assert "aiperf_info{" in result
        assert "} 1" in result

    def test_escapes_quotes_in_values(self):
        """Test that quotes in label values are escaped."""
        labels = {"model": 'model"with"quotes'}

        result = _format_info_metric(labels)

        assert r'model="model\"with\"quotes"' in result

    def test_escapes_backslashes_in_values(self):
        """Test that backslashes in label values are escaped."""
        labels = {"path": r"C:\path\to\model"}

        result = _format_info_metric(labels)

        assert r'path="C:\\path\\to\\model"' in result

    def test_output_ends_with_newline(self):
        """Test that output ends with a newline."""
        result = _format_info_metric({"test": "value"})
        assert result.endswith("\n")


class TestFormatAsPrometheusWithInfoLabels:
    """Tests for format_as_prometheus with info_labels parameter."""

    def test_info_metric_appears_first(self):
        """Test that info metric appears before other metrics."""
        metric = MetricResult(
            tag="test_metric",
            header="Test",
            unit="ms",
            avg=1.0,
        )
        info_labels = {"model": "test-model"}

        result = format_as_prometheus([metric], info_labels)

        # Info metric should appear before the test metric
        info_pos = result.find("aiperf_info")
        metric_pos = result.find("aiperf_test_metric")
        assert info_pos < metric_pos

    def test_no_info_metric_when_labels_none(self):
        """Test that no info metric appears when labels is None."""
        metric = MetricResult(
            tag="test_metric",
            header="Test",
            unit="ms",
            avg=1.0,
        )

        result = format_as_prometheus([metric], info_labels=None)

        assert "aiperf_info" not in result

    def test_no_info_metric_when_labels_empty(self):
        """Test that no info metric appears when labels is empty dict."""
        metric = MetricResult(
            tag="test_metric",
            header="Test",
            unit="ms",
            avg=1.0,
        )

        result = format_as_prometheus([metric], info_labels={})

        assert "aiperf_info" not in result

    def test_labels_added_to_all_metrics(self):
        """Test that key labels are added to all metrics."""
        metric = MetricResult(
            tag="test_metric",
            header="Test",
            unit="ms",
            avg=10.0,
            p99=20.0,
        )
        info_labels = {"model": "llama-3", "streaming": "true"}

        result = format_as_prometheus([metric], info_labels)

        assert (
            'aiperf_test_metric_avg_{model="llama-3",streaming="true"} 10.0' in result
        )
        assert (
            'aiperf_test_metric_p99_{model="llama-3",streaming="true"} 20.0' in result
        )

    def test_config_and_version_excluded_from_metrics(self):
        """Test that 'config' and 'version' labels are not added to metrics (only to info)."""
        metric = MetricResult(
            tag="test_metric",
            header="Test",
            unit="ms",
            avg=10.0,
        )
        info_labels = {"model": "llama-3", "config": '{"some":"json"}'}

        result = format_as_prometheus([metric], info_labels)

        # config and version should be in info metric
        assert "config=" in result
        assert "version=" in result
        # but not in the test_metric line
        metric_line = [
            line for line in result.split("\n") if line.startswith("aiperf_test_metric")
        ][0]
        assert "config=" not in metric_line
        assert "version=" not in metric_line
        assert 'model="llama-3"' in metric_line

    def test_no_labels_when_info_labels_none(self):
        """Test that metrics have no labels when info_labels is None."""
        metric = MetricResult(
            tag="test_metric",
            header="Test",
            unit="ms",
            avg=10.0,
        )

        result = format_as_prometheus([metric], info_labels=None)

        assert "aiperf_test_metric_avg_ 10.0" in result


class TestLiveMetricsServerBuildInfoLabels:
    """Tests for LiveMetricsServer._build_info_labels method."""

    def _create_mock_user_config(
        self,
        benchmark_id: str | None = None,
        model_names: list[str] | None = None,
        endpoint_type: str = "openai",
        streaming: bool = True,
        concurrency: int | None = 10,
        request_rate: float | None = None,
    ) -> MagicMock:
        """Create a mock UserConfig for testing."""
        user_config = MagicMock()
        user_config.benchmark_id = benchmark_id
        user_config.endpoint.model_names = model_names or ["test-model"]
        user_config.endpoint.type = endpoint_type
        user_config.endpoint.streaming = streaming
        user_config.loadgen.concurrency = concurrency
        user_config.loadgen.request_rate = request_rate
        user_config.model_dump_json.return_value = "{}"
        return user_config

    def test_key_labels_included(self):
        """Test that key labels are included for easy querying."""
        user_config = self._create_mock_user_config(
            benchmark_id="test-123",
            model_names=["llama-3"],
            concurrency=5,
            request_rate=100.0,
        )

        labels = LiveMetricsServer._build_info_labels(user_config)

        assert labels["benchmark_id"] == "test-123"
        assert labels["model"] == "llama-3"
        assert labels["endpoint_type"] == "openai"
        assert labels["streaming"] == "true"
        assert labels["concurrency"] == "5"
        assert labels["request_rate"] == "100.0"

    def test_config_json_included(self):
        """Test that full config is included as JSON."""
        user_config = self._create_mock_user_config()
        expected_json = '{"loadgen":{"concurrency":10}}'
        user_config.model_dump_json.return_value = expected_json

        labels = LiveMetricsServer._build_info_labels(user_config)

        assert labels["config"] == expected_json
        user_config.model_dump_json.assert_called_once_with(exclude_unset=True)

    def test_benchmark_id_excluded_when_none(self):
        """Test that benchmark_id is excluded when None."""
        user_config = self._create_mock_user_config(benchmark_id=None)

        labels = LiveMetricsServer._build_info_labels(user_config)

        assert "benchmark_id" not in labels

    def test_concurrency_excluded_when_none(self):
        """Test that concurrency is excluded when None."""
        user_config = self._create_mock_user_config(concurrency=None)

        labels = LiveMetricsServer._build_info_labels(user_config)

        assert "concurrency" not in labels

    def test_request_rate_excluded_when_none(self):
        """Test that request_rate is excluded when None."""
        user_config = self._create_mock_user_config(request_rate=None)

        labels = LiveMetricsServer._build_info_labels(user_config)

        assert "request_rate" not in labels

    def test_multiple_models_as_csv(self):
        """Test that multiple models are formatted as CSV."""
        user_config = self._create_mock_user_config(
            model_names=["model-a", "model-b", "model-c"]
        )

        labels = LiveMetricsServer._build_info_labels(user_config)

        assert labels["model"] == "model-a,model-b,model-c"

    def test_streaming_false(self):
        """Test that streaming=false is properly formatted."""
        user_config = self._create_mock_user_config(streaming=False)

        labels = LiveMetricsServer._build_info_labels(user_config)

        assert labels["streaming"] == "false"


class TestLiveMetricsServerInfoLabelsIntegration:
    """Tests for LiveMetricsServer initialization with user_config info labels."""

    def _create_mock_user_config(
        self,
        live_metrics_port: int = 9090,
        live_metrics_host: str = "127.0.0.1",
        benchmark_id: str | None = "test-benchmark",
        model_names: list[str] | None = None,
        endpoint_type: str = "openai",
        streaming: bool = True,
        concurrency: int | None = 5,
        request_rate: float | None = None,
    ) -> MagicMock:
        """Create a mock UserConfig for testing."""
        user_config = MagicMock()
        user_config.live_metrics_port = live_metrics_port
        user_config.live_metrics_host = live_metrics_host
        user_config.benchmark_id = benchmark_id
        user_config.endpoint.model_names = model_names or ["test-model"]
        user_config.endpoint.type = endpoint_type
        user_config.endpoint.streaming = streaming
        user_config.loadgen.concurrency = concurrency
        user_config.loadgen.request_rate = request_rate
        user_config.model_dump_json.return_value = "{}"
        return user_config

    def test_builds_info_labels_from_user_config(self):
        """Test that info labels are built from user_config."""
        user_config = self._create_mock_user_config()
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: [],
        )

        assert server._info_labels is not None
        assert server._info_labels["benchmark_id"] == "test-benchmark"
        assert server._info_labels["model"] == "test-model"
        assert "config" in server._info_labels


@pytest.mark.asyncio
class TestLiveMetricsServerHandlers:
    """Tests for LiveMetricsServer HTTP handlers."""

    def _create_mock_user_config(
        self,
        live_metrics_port: int = 9090,
        live_metrics_host: str = "127.0.0.1",
        benchmark_id: str | None = "test-benchmark",
        model_names: list[str] | None = None,
    ) -> MagicMock:
        """Create a mock UserConfig for testing."""
        user_config = MagicMock()
        user_config.live_metrics_port = live_metrics_port
        user_config.live_metrics_host = live_metrics_host
        user_config.benchmark_id = benchmark_id
        user_config.endpoint.model_names = model_names or ["test-model"]
        user_config.endpoint.type = "openai"
        user_config.endpoint.streaming = True
        user_config.loadgen.concurrency = 5
        user_config.loadgen.request_rate = None
        user_config.model_dump_json.return_value = '{"endpoint":{"type":"openai"}}'
        return user_config

    async def test_handle_metrics_sync_callback(self):
        """Test _handle_metrics with a synchronous callback."""
        metrics = [
            MetricResult(tag="test", header="Test", unit="ms", avg=10.0),
        ]
        user_config = self._create_mock_user_config()
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: metrics,
        )
        request = MagicMock()

        response = await server._handle_metrics(request)

        assert response.content_type == "text/plain"
        assert response.charset == "utf-8"
        body = response.body.decode("utf-8")
        assert "aiperf_test_avg_" in body

    async def test_handle_metrics_async_callback(self):
        """Test _handle_metrics with an asynchronous callback."""
        metrics = [
            MetricResult(tag="test", header="Test", unit="ms", avg=10.0),
        ]

        async def async_callback():
            return metrics

        user_config = self._create_mock_user_config()
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=async_callback,
        )
        request = MagicMock()

        response = await server._handle_metrics(request)

        assert response.content_type == "text/plain"
        body = response.body.decode("utf-8")
        assert "aiperf_test_avg_" in body

    async def test_handle_config(self):
        """Test _handle_config endpoint returns user config JSON."""
        user_config = self._create_mock_user_config()
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: [],
        )
        request = MagicMock()

        response = await server._handle_config(request)

        assert response.content_type == "application/json"
        assert response.charset == "utf-8"
        body = response.body.decode("utf-8")
        assert "openai" in body

    async def test_handle_health(self):
        """Test _handle_health endpoint returns 'ok'."""
        user_config = self._create_mock_user_config()
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: [],
        )
        request = MagicMock()

        response = await server._handle_health(request)

        assert response.text == "ok"

    async def test_handle_metrics_json_sync_callback(self):
        """Test _handle_metrics_json with a synchronous callback."""
        metrics = [
            MetricResult(
                tag="test_metric", header="Test", unit="ms", avg=10.0, p99=25.0
            ),
        ]
        user_config = self._create_mock_user_config(benchmark_id="bench-123")
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: metrics,
        )
        request = MagicMock()

        response = await server._handle_metrics_json(request)

        assert response.content_type == "application/json"
        assert response.charset == "utf-8"
        body = response.body.decode("utf-8")
        # Verify JSON structure
        data = orjson.loads(body)
        assert "aiperf_version" in data
        assert data["benchmark_id"] == "bench-123"
        assert "metrics" in data
        assert "test_metric" in data["metrics"]
        assert data["metrics"]["test_metric"]["avg"] == 10.0
        assert data["metrics"]["test_metric"]["p99"] == 25.0
        # tag should be excluded from metric dump
        assert "tag" not in data["metrics"]["test_metric"]

    async def test_handle_metrics_json_async_callback(self):
        """Test _handle_metrics_json with an asynchronous callback."""
        metrics = [
            MetricResult(tag="test_metric", header="Test", unit="ms", avg=10.0),
        ]

        async def async_callback():
            return metrics

        user_config = self._create_mock_user_config()
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=async_callback,
        )
        request = MagicMock()

        response = await server._handle_metrics_json(request)

        assert response.content_type == "application/json"
        data = orjson.loads(response.body)
        assert "metrics" in data
        assert "test_metric" in data["metrics"]

    async def test_handle_metrics_json_includes_info_labels(self):
        """Test that _handle_metrics_json includes info labels (except config/version)."""
        metrics = [
            MetricResult(tag="test", header="Test", unit="ms", avg=10.0),
        ]
        user_config = self._create_mock_user_config(
            benchmark_id="bench-id",
            model_names=["llama-3"],
        )
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: metrics,
        )
        request = MagicMock()

        response = await server._handle_metrics_json(request)

        data = orjson.loads(response.body)
        # Info labels should be present (except config which is excluded)
        assert data["model"] == "llama-3"
        assert data["endpoint_type"] == "openai"
        assert data["streaming"] is True  # coerce_value converts "true" -> True
        # config should NOT be in the response body directly (only in info_labels for prometheus)
        assert "config" not in data

    async def test_handle_metrics_json_empty_metrics(self):
        """Test _handle_metrics_json with no metrics."""
        user_config = self._create_mock_user_config(benchmark_id="bench-empty")
        server = LiveMetricsServer(
            user_config=user_config,
            metrics_callback=lambda: [],
        )
        request = MagicMock()

        response = await server._handle_metrics_json(request)

        data = orjson.loads(response.body)
        assert data["benchmark_id"] == "bench-empty"
        assert data["metrics"] == {}
