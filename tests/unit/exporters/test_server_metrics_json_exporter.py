# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsJsonExporter."""

import json

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import ProfileResults
from aiperf.common.models.export_data import (
    ServerMetricLabeledStats,
    ServerMetricsEndpointSummary,
    ServerMetricSummary,
)
from aiperf.common.models.export_stats import (
    CounterExportStats,
    GaugeExportStats,
    HistogramExportStats,
    SummaryExportStats,
)
from aiperf.common.models.metric_info_models import InfoMetricData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.exporters.server_metrics_json_exporter import ServerMetricsJsonExporter

from .conftest import create_exporter_config


@pytest.fixture
def mock_user_config(tmp_path):
    """Create a UserConfig with a temp output directory."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="/v1/chat/completions",
        ),
        output={"artifact_dir": str(tmp_path)},
    )


@pytest.fixture
def mock_profile_results():
    """Create a minimal ProfileResults for exporter config."""
    return ProfileResults(
        records=[],
        completed=100,
        start_ns=1_000_000_000_000,
        end_ns=1_300_000_000_000,
    )


@pytest.fixture
def server_metrics_results_with_summaries():
    """Create ServerMetricsResults with pre-computed endpoint_summaries.

    This mimics what records_manager produces after processing raw metrics.
    Includes all metric types and info metrics to test full export path.
    """
    # Create endpoint summaries with all metric types
    endpoint1_summary = ServerMetricsEndpointSummary(
        endpoint_url="http://localhost:8081/metrics",
        duration_seconds=300.0,
        scrape_count=60,
        avg_scrape_latency_ms=10.5,
        avg_scrape_period_ms=5084.7,
        info_metrics={
            "vllm_version_info": InfoMetricData(
                description="vLLM version information",
                labels=[{"version": "0.6.0", "build": "abc123"}],
            ),
        },
        metrics={
            "vllm:kv_cache_usage_perc": ServerMetricSummary(
                description="KV cache usage percentage",
                type="gauge",
                series=[
                    ServerMetricLabeledStats(
                        labels=None,
                        stats=GaugeExportStats(
                            min=0.4,
                            avg=0.55,
                            p50=0.54,
                            p90=0.68,
                            p95=0.72,
                            p99=0.78,
                            max=0.8,
                            std=0.1,
                        ),
                    ),
                ],
            ),
            "vllm:request_success_total": ServerMetricSummary(
                description="Total successful requests",
                type="counter",
                series=[
                    ServerMetricLabeledStats(
                        labels=None,
                        stats=CounterExportStats(
                            delta=1000.0,
                            rate_overall=3.33,
                            rate_avg=3.2,
                            rate_min=2.5,
                            rate_max=4.0,
                            rate_std=0.5,
                        ),
                    ),
                ],
            ),
            "vllm:time_to_first_token_seconds": ServerMetricSummary(
                description="Time to first token histogram",
                type="histogram",
                series=[
                    ServerMetricLabeledStats(
                        labels=None,
                        stats=HistogramExportStats(
                            count_delta=1000,
                            sum_delta=125.5,
                            avg=0.1255,
                            count_rate=3.33,
                            p50_estimate=0.05,
                            p90_estimate=0.12,
                            p95_estimate=0.18,
                            p99_estimate=0.45,
                            buckets={
                                "0.01": 50,
                                "0.1": 450,
                                "1.0": 980,
                                "+Inf": 1000,
                            },
                        ),
                    ),
                ],
            ),
            "vllm:request_latency_seconds": ServerMetricSummary(
                description="Request latency summary",
                type="summary",
                series=[
                    ServerMetricLabeledStats(
                        labels=None,
                        stats=SummaryExportStats(
                            count_delta=1000,
                            sum_delta=250.0,
                            avg=0.25,
                            count_rate=3.33,
                            quantiles={
                                "0.5": 0.2,
                                "0.9": 0.4,
                                "0.95": 0.5,
                                "0.99": 0.8,
                            },
                        ),
                    ),
                ],
            ),
        },
    )

    endpoint2_summary = ServerMetricsEndpointSummary(
        endpoint_url="http://localhost:8082/metrics",
        duration_seconds=300.0,
        scrape_count=58,
        avg_scrape_latency_ms=12.3,
        avg_scrape_period_ms=5263.2,
        info_metrics={
            "vllm_version_info": InfoMetricData(
                description="vLLM version information",
                labels=[{"version": "0.6.0", "build": "def456"}],
            ),
        },
        metrics={
            "vllm:kv_cache_usage_perc": ServerMetricSummary(
                description="KV cache usage percentage",
                type="gauge",
                series=[
                    ServerMetricLabeledStats(
                        labels=None,
                        stats=GaugeExportStats(
                            min=0.5,
                            avg=0.62,
                            p50=0.61,
                            p90=0.75,
                            p95=0.78,
                            p99=0.82,
                            max=0.85,
                            std=0.08,
                        ),
                    ),
                ],
            ),
            "vllm:request_success_total": ServerMetricSummary(
                description="Total successful requests",
                type="counter",
                series=[
                    ServerMetricLabeledStats(
                        labels=None,
                        stats=CounterExportStats(
                            delta=800.0,
                            rate_overall=2.67,
                            rate_avg=2.5,
                            rate_min=2.0,
                            rate_max=3.5,
                            rate_std=0.4,
                        ),
                    ),
                ],
            ),
        },
    )

    return ServerMetricsResults(
        server_metrics_data=None,  # Not sent over ZMQ
        endpoint_summaries={
            "localhost:8081": endpoint1_summary,
            "localhost:8082": endpoint2_summary,
        },
        start_ns=1_000_000_000_000,
        end_ns=1_300_000_000_000,
        endpoints_configured=[
            "http://localhost:8081/metrics",
            "http://localhost:8082/metrics",
        ],
        endpoints_successful=[
            "http://localhost:8081/metrics",
            "http://localhost:8082/metrics",
        ],
        error_summary=[],
    )


@pytest.fixture
def server_metrics_results_with_labeled_metrics():
    """Create ServerMetricsResults with labeled metrics to test label handling."""
    endpoint_summary = ServerMetricsEndpointSummary(
        endpoint_url="http://localhost:8081/metrics",
        duration_seconds=100.0,
        scrape_count=20,
        avg_scrape_latency_ms=8.0,
        avg_scrape_period_ms=5263.2,
        info_metrics=None,
        metrics={
            "http_requests_total": ServerMetricSummary(
                description="Total HTTP requests",
                type="counter",
                series=[
                    ServerMetricLabeledStats(
                        labels={"method": "GET", "status": "200"},
                        stats=CounterExportStats(
                            delta=500.0,
                            rate_overall=5.0,
                            rate_avg=4.8,
                            rate_min=3.0,
                            rate_max=6.0,
                            rate_std=0.8,
                        ),
                    ),
                    ServerMetricLabeledStats(
                        labels={"method": "POST", "status": "200"},
                        stats=CounterExportStats(
                            delta=300.0,
                            rate_overall=3.0,
                            rate_avg=2.9,
                            rate_min=2.0,
                            rate_max=4.0,
                            rate_std=0.5,
                        ),
                    ),
                    ServerMetricLabeledStats(
                        labels={"method": "GET", "status": "500"},
                        stats=CounterExportStats(
                            delta=5.0,
                            rate_overall=0.05,
                            rate_avg=0.04,
                            rate_min=0.0,
                            rate_max=0.1,
                            rate_std=0.02,
                        ),
                    ),
                ],
            ),
        },
    )

    return ServerMetricsResults(
        server_metrics_data=None,
        endpoint_summaries={"localhost:8081": endpoint_summary},
        start_ns=1_000_000_000_000,
        end_ns=1_100_000_000_000,
        endpoints_configured=["http://localhost:8081/metrics"],
        endpoints_successful=["http://localhost:8081/metrics"],
        error_summary=[],
    )


class TestServerMetricsJsonExporterInitialization:
    """Test exporter initialization."""

    def test_initialization_with_valid_config(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
    ):
        """Test that exporter initializes correctly with valid config."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        assert exporter is not None

    def test_initialization_disabled_without_results(
        self, mock_user_config, mock_profile_results
    ):
        """Test that exporter raises DataExporterDisabled when no results."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=None,
        )
        with pytest.raises(DataExporterDisabled):
            ServerMetricsJsonExporter(config)


class TestServerMetricsJsonExporterGetExportInfo:
    """Test get_export_info method."""

    def test_get_export_info_returns_correct_type(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
    ):
        """Test that export info contains correct type and path."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        info = exporter.get_export_info()
        assert info.export_type == "Server Metrics JSON Export"
        assert "server_metrics" in str(info.file_path)


def find_series_by_endpoint(
    metric_data: dict, endpoint: str | None = None
) -> dict | None:
    """Helper to find a series by endpoint within a metric."""
    for series in metric_data.get("series", []):
        if endpoint is None or series["endpoint"] == endpoint:
            return series
    return None


class TestServerMetricsJsonExporterGenerateContent:
    """Test JSON content generation."""

    def test_generate_content_creates_valid_json(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
    ):
        """Test that generated content is valid JSON."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        content = exporter._generate_content()
        data = json.loads(content)
        assert "summary" in data
        assert "metrics" in data

    def test_generate_content_has_normalized_endpoints(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
    ):
        """Test that endpoints are normalized (without http:// and /metrics)."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        content = exporter._generate_content()
        data = json.loads(content)

        # Check summary endpoints are normalized
        assert "localhost:8081" in data["summary"]["endpoints_configured"]
        assert "localhost:8082" in data["summary"]["endpoints_configured"]
        assert "http://" not in str(data["summary"]["endpoints_configured"])

    def test_generate_content_has_endpoint_info_in_summary(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
    ):
        """Test that endpoint metadata is in summary.endpoint_info."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        content = exporter._generate_content()
        data = json.loads(content)

        assert "endpoint_info" in data["summary"]
        endpoint_info = data["summary"]["endpoint_info"]

        # Check normalized endpoint keys
        assert "localhost:8081" in endpoint_info
        assert "localhost:8082" in endpoint_info

        # Check metadata fields
        info1 = endpoint_info["localhost:8081"]
        assert info1["endpoint_url"] == "http://localhost:8081/metrics"
        assert info1["duration_seconds"] == 300.0
        assert info1["scrape_count"] == 60
        assert info1["avg_scrape_latency_ms"] == 10.5

    def test_generate_content_has_series_from_all_endpoints(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
    ):
        """Test that series from multiple endpoints are present within each metric."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        content = exporter._generate_content()
        data = json.loads(content)

        # kv_cache_usage_perc exists in both endpoints - metric should have 2 series
        assert "vllm:kv_cache_usage_perc" in data["metrics"]
        kv_metric = data["metrics"]["vllm:kv_cache_usage_perc"]
        assert len(kv_metric["series"]) == 2  # One from each endpoint

        # Each series should have endpoint field
        endpoints_in_series = [s["endpoint"] for s in kv_metric["series"]]
        assert "localhost:8081" in endpoints_in_series
        assert "localhost:8082" in endpoints_in_series

    def test_generate_content_series_have_endpoint_fields(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
    ):
        """Test that each series has both endpoint and endpoint_url fields."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        content = exporter._generate_content()
        data = json.loads(content)

        for metric_name, metric_data in data["metrics"].items():
            for series in metric_data["series"]:
                # Normalized endpoint (no http://, no /metrics)
                assert "endpoint" in series, f"Missing endpoint in {metric_name}"
                assert not series["endpoint"].startswith("http://")
                assert not series["endpoint"].endswith("/metrics")
                # Full endpoint URL
                assert "endpoint_url" in series, (
                    f"Missing endpoint_url in {metric_name}"
                )
                assert series["endpoint_url"].startswith("http://")
                assert series["endpoint_url"].endswith("/metrics")

    def test_generate_content_includes_info_metrics_as_gauges(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
    ):
        """Test that info metrics are included as gauges with only labels (no stats)."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        content = exporter._generate_content()
        data = json.loads(content)

        # Info metrics should be in metrics dict (not a separate info_metrics key)
        assert "info_metrics" not in data
        assert "vllm_version_info" in data["metrics"]
        info_metric = data["metrics"]["vllm_version_info"]
        assert len(info_metric["series"]) == 2  # One from each endpoint

        # Each series should be a gauge with only endpoint info and labels (no stats)
        assert info_metric["type"] == "gauge"
        for series in info_metric["series"]:
            assert "endpoint" in series
            assert "endpoint_url" in series
            assert "labels" in series
            assert "version" in series["labels"]
            # Info metrics should have NO stats fields - only labels matter
            assert "observation_count" not in series
            assert "avg" not in series
            assert "min" not in series
            assert "max" not in series
            assert "std" not in series
            assert "p50" not in series

    def test_generate_content_handles_labeled_metrics(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_labeled_metrics,
    ):
        """Test that labeled metrics are handled correctly."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_labeled_metrics,
        )
        exporter = ServerMetricsJsonExporter(config)
        content = exporter._generate_content()
        data = json.loads(content)

        assert "http_requests_total" in data["metrics"]
        http_metric = data["metrics"]["http_requests_total"]
        assert len(http_metric["series"]) == 3

        # Check that labels are preserved
        for series in http_metric["series"]:
            assert "endpoint" in series
            assert "labels" in series
            assert "method" in series["labels"]
            assert "status" in series["labels"]

    def test_generate_content_includes_all_metric_types(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
    ):
        """Test that all Prometheus metric types are handled with flat series fields."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        content = exporter._generate_content()
        data = json.loads(content)

        # Gauge - flat stats fields
        assert "vllm:kv_cache_usage_perc" in data["metrics"]
        gauge_metric = data["metrics"]["vllm:kv_cache_usage_perc"]
        assert gauge_metric["type"] == "gauge"
        gauge_series = find_series_by_endpoint(gauge_metric, "localhost:8081")
        assert gauge_series is not None
        assert "avg" in gauge_series
        assert "p50" in gauge_series
        assert gauge_series["estimated_percentiles"] is False  # Exact percentiles

        # Counter - flat fields with unified naming and rate statistics
        assert "vllm:request_success_total" in data["metrics"]
        counter_metric = data["metrics"]["vllm:request_success_total"]
        assert counter_metric["type"] == "counter"
        counter_series = find_series_by_endpoint(counter_metric, "localhost:8081")
        assert counter_series is not None
        assert "delta" in counter_series
        assert "rate_per_second" in counter_series  # Overall rate (delta/duration)
        assert "rate_avg" in counter_series  # Time-weighted average rate
        assert "rate_min" in counter_series  # Minimum point-to-point rate
        assert "rate_max" in counter_series  # Maximum point-to-point rate
        assert "rate_std" in counter_series  # Standard deviation of rates

        # Histogram - flat fields with unified percentile naming
        assert "vllm:time_to_first_token_seconds" in data["metrics"]
        histogram_metric = data["metrics"]["vllm:time_to_first_token_seconds"]
        assert histogram_metric["type"] == "histogram"
        histogram_series = find_series_by_endpoint(histogram_metric, "localhost:8081")
        assert histogram_series is not None
        assert "observation_count" in histogram_series  # Standardized field
        assert "count_delta" not in histogram_series  # Removed (redundant)
        assert "buckets" in histogram_series
        assert "p99" in histogram_series  # Unified name (was p99_estimate)
        assert histogram_series["estimated_percentiles"] is True  # Estimated

        # Summary - flat fields
        assert "vllm:request_latency_seconds" in data["metrics"]
        summary_metric = data["metrics"]["vllm:request_latency_seconds"]
        assert summary_metric["type"] == "summary"
        summary_series = find_series_by_endpoint(summary_metric, "localhost:8081")
        assert summary_series is not None
        assert "quantiles" in summary_series
        assert "p99" in summary_series  # Mapped from quantiles
        assert summary_series["estimated_percentiles"] is False  # Server-computed

    def test_counter_with_zero_delta_has_minimal_output(
        self,
        mock_user_config,
        mock_profile_results,
    ):
        """Test that counters with delta=0 only include delta field."""
        # Create a fixture with a zero-delta counter
        endpoint_summary = ServerMetricsEndpointSummary(
            endpoint_url="http://localhost:8081/metrics",
            duration_seconds=100.0,
            scrape_count=20,
            avg_scrape_latency_ms=8.0,
            avg_scrape_period_ms=5000.0,
            info_metrics=None,
            metrics={
                "error_count_total": ServerMetricSummary(
                    description="Total errors",
                    type="counter",
                    series=[
                        ServerMetricLabeledStats(
                            labels=None,
                            stats=CounterExportStats(
                                delta=0,  # No errors!
                                rate_overall=None,
                                rate_avg=None,
                                rate_min=None,
                                rate_max=None,
                                rate_std=None,
                            ),
                        ),
                    ],
                ),
            },
        )
        server_metrics_results = ServerMetricsResults(
            server_metrics_data=None,
            endpoint_summaries={"localhost:8081": endpoint_summary},
            start_ns=1_000_000_000_000,
            end_ns=1_100_000_000_000,
            endpoints_configured=["http://localhost:8081/metrics"],
            endpoints_successful=["http://localhost:8081/metrics"],
            error_summary=[],
        )

        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results,
        )
        exporter = ServerMetricsJsonExporter(config)
        content = exporter._generate_content()
        data = json.loads(content)

        # Counter with delta=0 should only have delta field
        assert "error_count_total" in data["metrics"]
        counter_metric = data["metrics"]["error_count_total"]
        counter_series = counter_metric["series"][0]

        assert counter_series["delta"] == 0
        # Rate fields should NOT be present when delta=0
        assert "rate_per_second" not in counter_series
        assert "rate_avg" not in counter_series
        assert "rate_min" not in counter_series
        assert "rate_max" not in counter_series
        assert "rate_std" not in counter_series


class TestServerMetricsJsonExporterIntegration:
    """Integration tests for full export flow."""

    @pytest.mark.asyncio
    async def test_export_creates_valid_json_file(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_summaries,
        tmp_path,
    ):
        """Test that export creates a valid JSON file."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_summaries,
        )
        exporter = ServerMetricsJsonExporter(config)
        await exporter.export()

        # Read and parse the exported file
        output_file = mock_user_config.output.server_metrics_export_json_file
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "summary" in data
        assert "metrics" in data
        assert "endpoint_info" in data["summary"]
