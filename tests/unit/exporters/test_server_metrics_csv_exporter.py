# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsCsvExporter."""

import csv
import io

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import EndpointType, PrometheusMetricType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import ProfileResults
from aiperf.common.models.export_data import (
    FlatSeriesStats,
    ServerMetricsEndpointSummary,
    ServerMetricSummary,
)
from aiperf.common.models.metric_info_models import InfoMetricData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.exporters.server_metrics_csv_exporter import ServerMetricsCsvExporter

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
def server_metrics_results_with_all_types():
    """Create ServerMetricsResults with all metric types for testing.

    Includes gauge, counter, histogram, and summary metrics to test
    section separation and type-specific formatting.
    """
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
                type=PrometheusMetricType.GAUGE,
                series=[
                    FlatSeriesStats(
                        labels=None,
                        min=0.4,
                        avg=0.55,
                        p50=0.54,
                        p90=0.68,
                        p95=0.72,
                        p99=0.78,
                        max=0.8,
                        std=0.1,
                        estimated_percentiles=False,
                    ),
                ],
            ),
            "vllm:request_success_total": ServerMetricSummary(
                description="Total successful requests",
                type=PrometheusMetricType.COUNTER,
                series=[
                    FlatSeriesStats(
                        labels=None,
                        delta=1000.0,
                        rate_per_second=3.33,
                        rate_avg=3.2,
                        rate_min=2.5,
                        rate_max=4.0,
                        rate_std=0.5,
                    ),
                ],
            ),
            "vllm:time_to_first_token_seconds": ServerMetricSummary(
                description="Time to first token histogram",
                type=PrometheusMetricType.HISTOGRAM,
                series=[
                    FlatSeriesStats(
                        labels=None,
                        observation_count=1000,
                        delta=125.5,
                        avg=0.1255,
                        observations_per_second=3.33,
                        buckets={
                            "0.01": 50,
                            "0.1": 450,
                            "1.0": 980,
                            "+Inf": 1000,
                        },
                        estimated_percentiles=True,
                    ),
                ],
            ),
            "vllm:request_latency_seconds": ServerMetricSummary(
                description="Request latency summary",
                type=PrometheusMetricType.SUMMARY,
                series=[
                    FlatSeriesStats(
                        labels=None,
                        observation_count=1000,
                        delta=250.0,
                        avg=0.25,
                        observations_per_second=3.33,
                        quantiles={
                            "0.5": 0.2,
                            "0.9": 0.4,
                            "0.95": 0.5,
                            "0.99": 0.8,
                        },
                        estimated_percentiles=False,
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
        info_metrics=None,
        metrics={
            "vllm:kv_cache_usage_perc": ServerMetricSummary(
                description="KV cache usage percentage",
                type=PrometheusMetricType.GAUGE,
                series=[
                    FlatSeriesStats(
                        labels=None,
                        min=0.5,
                        avg=0.62,
                        p50=0.61,
                        p90=0.75,
                        p95=0.78,
                        p99=0.82,
                        max=0.85,
                        std=0.08,
                        estimated_percentiles=False,
                    ),
                ],
            ),
            "vllm:request_success_total": ServerMetricSummary(
                description="Total successful requests",
                type=PrometheusMetricType.COUNTER,
                series=[
                    FlatSeriesStats(
                        labels=None,
                        delta=800.0,
                        rate_per_second=2.67,
                        rate_avg=2.5,
                        rate_min=2.0,
                        rate_max=3.5,
                        rate_std=0.4,
                    ),
                ],
            ),
        },
    )

    return ServerMetricsResults(
        server_metrics_data=None,
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
                type=PrometheusMetricType.COUNTER,
                series=[
                    FlatSeriesStats(
                        labels={"method": "GET", "status": "200"},
                        delta=500.0,
                        rate_per_second=5.0,
                        rate_avg=4.8,
                        rate_min=3.0,
                        rate_max=6.0,
                        rate_std=0.8,
                    ),
                    FlatSeriesStats(
                        labels={"method": "POST", "status": "200"},
                        delta=300.0,
                        rate_per_second=3.0,
                        rate_avg=2.9,
                        rate_min=2.0,
                        rate_max=4.0,
                        rate_std=0.5,
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


def _parse_csv_content(content: str) -> list[list[str]]:
    """Parse CSV content into list of rows."""
    reader = csv.reader(io.StringIO(content))
    return list(reader)


class TestServerMetricsCsvExporterInitialization:
    """Test exporter initialization."""

    def test_initialization_with_valid_config(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that exporter initializes correctly with valid config."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
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
            ServerMetricsCsvExporter(config)


class TestServerMetricsCsvExporterGetExportInfo:
    """Test get_export_info method."""

    def test_get_export_info_returns_correct_type(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that export info contains correct type and path."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        info = exporter.get_export_info()
        assert info.export_type == "Server Metrics CSV Export"
        assert "server_metrics" in str(info.file_path)
        assert str(info.file_path).endswith(".csv")


class TestServerMetricsCsvExporterGenerateContent:
    """Test CSV content generation."""

    def test_generate_content_creates_valid_csv(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that generated content is valid CSV."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)
        assert len(rows) > 0

    def test_generate_content_has_sections_by_metric_type(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that CSV has separate sections for each metric type."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()

        # Check for column headers for each metric type (all include Type column)
        assert "Endpoint,Type,Metric,Labels,avg,min,max" in content  # gauge
        assert "Endpoint,Type,Metric,Labels,delta,rate_per_second" in content  # counter
        assert (
            "Endpoint,Type,Metric,Labels,observation_count,delta" in content
        )  # histogram/summary
        # Check that metric type values appear in the data
        assert ",gauge," in content
        assert ",counter," in content
        assert ",histogram," in content
        assert ",summary," in content

    def test_generate_content_gauge_section_has_correct_columns(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that gauge section has appropriate stat columns."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find gauge header row (starts with Endpoint and has gauge stat columns)
        gauge_header = None
        for row in rows:
            if row and row[0] == "Endpoint" and "p50" in row and "p99" in row:
                gauge_header = row
                break

        assert gauge_header is not None
        assert "Endpoint" in gauge_header
        assert "Metric" in gauge_header
        assert "Labels" in gauge_header
        assert "avg" in gauge_header
        assert "min" in gauge_header
        assert "max" in gauge_header
        assert "p50" in gauge_header
        assert "p90" in gauge_header

    def test_generate_content_counter_section_has_correct_columns(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that counter section has appropriate stat columns."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find counter header row (starts with Endpoint and has counter stat columns)
        counter_header = None
        for row in rows:
            if (
                row
                and row[0] == "Endpoint"
                and "delta" in row
                and "rate_per_second" in row
            ):
                counter_header = row
                break

        assert counter_header is not None
        assert "delta" in counter_header
        assert "rate_per_second" in counter_header
        assert "rate_avg" in counter_header

    def test_generate_content_histogram_section_has_buckets_column(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that histogram section has a buckets column with key=value pairs."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find histogram header row (has observation_count and buckets column)
        hist_header = None
        for row in rows:
            if (
                row
                and row[0] == "Endpoint"
                and "observation_count" in row
                and "buckets" in row
            ):
                hist_header = row
                break

        assert hist_header is not None
        # Buckets should be a single column at the end
        assert "buckets" in hist_header

    def test_generate_content_summary_section_has_quantiles_column(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that summary section has a quantiles column with key=value pairs."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find summary header row (has observation_count and quantiles column)
        summary_header = None
        for row in rows:
            if (
                row
                and row[0] == "Endpoint"
                and "observation_count" in row
                and "quantiles" in row
            ):
                summary_header = row
                break

        assert summary_header is not None
        # Quantiles should be a single column at the end
        assert "quantiles" in summary_header

    def test_generate_content_has_normalized_endpoints(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that endpoints are normalized (without http:// and /metrics)."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()

        assert "localhost:8081" in content
        assert "localhost:8082" in content
        assert "http://localhost:8081/metrics" not in content

    def test_generate_content_handles_labeled_metrics(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_labeled_metrics,
    ):
        """Test that labeled metrics are formatted correctly."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_labeled_metrics,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()

        # Labels should be formatted as key=value pairs
        assert "method=GET" in content
        assert "status=200" in content

    def test_generate_content_merges_metrics_from_all_endpoints(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that metrics from multiple endpoints appear in the same section."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Count rows with kv_cache_usage_perc (should be 2 - one per endpoint)
        kv_cache_rows = [r for r in rows if r and "vllm:kv_cache_usage_perc" in r]
        assert len(kv_cache_rows) == 2

    def test_generate_content_histogram_bucket_values_in_column(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that histogram bucket values are in a single column as key=value pairs."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()

        # Bucket values should be in key=value;key2=value2 format
        assert "0.01=50" in content
        assert "0.1=450" in content
        assert "1.0=980" in content
        assert "+Inf=1000" in content

    def test_generate_content_summary_quantile_values_in_column(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that summary quantile values are in a single column as key=value pairs."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()

        # Quantile values should be in key=value;key2=value2 format
        assert "0.5=0.2000" in content
        assert "0.9=0.4000" in content
        assert "0.95=0.5000" in content
        assert "0.99=0.8000" in content

    def test_generate_content_histograms_with_different_buckets_in_same_section(
        self, mock_user_config, mock_profile_results
    ):
        """Test that histograms with different bucket boundaries are in the same section."""
        endpoint_summary = ServerMetricsEndpointSummary(
            endpoint_url="http://localhost:8081/metrics",
            duration_seconds=100.0,
            scrape_count=20,
            avg_scrape_latency_ms=8.0,
            avg_scrape_period_ms=5263.2,
            info_metrics=None,
            metrics={
                "request_duration_seconds": ServerMetricSummary(
                    description="Request duration",
                    type=PrometheusMetricType.HISTOGRAM,
                    series=[
                        FlatSeriesStats(
                            labels=None,
                            observation_count=100,
                            delta=50.0,
                            avg=0.5,
                            observations_per_second=1.0,
                            buckets={
                                "0.1": 10,
                                "0.5": 50,
                                "1.0": 90,
                                "+Inf": 100,
                            },
                            estimated_percentiles=True,
                        ),
                    ],
                ),
                "queue_time_seconds": ServerMetricSummary(
                    description="Queue time",
                    type=PrometheusMetricType.HISTOGRAM,
                    series=[
                        FlatSeriesStats(
                            labels=None,
                            observation_count=200,
                            delta=10.0,
                            avg=0.05,
                            observations_per_second=2.0,
                            buckets={
                                "0.01": 50,
                                "0.05": 150,
                                "0.1": 190,
                                "+Inf": 200,
                            },
                            estimated_percentiles=True,
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
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find histogram header rows (have observation_count and buckets column)
        hist_headers = []
        for row in rows:
            if (
                row
                and row[0] == "Endpoint"
                and "observation_count" in row
                and "buckets" in row
            ):
                hist_headers.append(row)

        # All histograms should be under a single header (buckets in single column)
        assert len(hist_headers) == 1

        # Both histogram metrics should be in the data
        assert "request_duration_seconds" in content
        assert "queue_time_seconds" in content
        # Buckets from both should be present as key=value pairs
        assert "0.1=10" in content  # request_duration
        assert "0.01=50" in content  # queue_time


class TestServerMetricsCsvExporterIntegration:
    """Integration tests for full export flow."""

    @pytest.mark.asyncio
    async def test_export_creates_valid_csv_file(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
        tmp_path,
    ):
        """Test that export creates a valid CSV file."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        await exporter.export()

        # Read and parse the exported file
        output_file = mock_user_config.output.server_metrics_export_csv_file
        assert output_file.exists()

        with open(output_file) as f:
            content = f.read()

        # Verify column headers for different metric types exist
        assert "avg,min,max,std,p50,p90,p95,p99" in content  # gauge
        assert "delta,rate_per_second,rate_avg" in content  # counter
