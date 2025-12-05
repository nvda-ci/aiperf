# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsCsvExporter."""

import csv
import io

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import (
    CounterExportStats,
    GaugeExportStats,
    HistogramExportStats,
    InfoMetricData,
    ServerMetricLabeledStats,
    ServerMetricsEndpointSummary,
    ServerMetricSummary,
    SummaryExportStats,
)
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
                            count_delta=1000.0,
                            sum_delta=125.5,
                            avg=0.1255,
                            rate=3.33,
                            buckets={
                                "0.01": 50.0,
                                "0.1": 450.0,
                                "1.0": 980.0,
                                "+Inf": 1000.0,
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
                            count_delta=1000.0,
                            sum_delta=250.0,
                            avg=0.25,
                            rate=3.33,
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
        info_metrics=None,
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

        # Check for column headers for each metric type
        assert "avg,min,max,std,p50,p90,p95,p99" in content  # gauge columns
        assert (
            "delta,rate_overall,rate_avg,rate_min,rate_max,rate_std" in content
        )  # counter
        assert (
            "count_delta,sum_delta,avg,rate" in content
        )  # histogram/summary base columns

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
                and "rate_overall" in row
            ):
                counter_header = row
                break

        assert counter_header is not None
        assert "delta" in counter_header
        assert "rate_overall" in counter_header
        assert "rate_avg" in counter_header

    def test_generate_content_histogram_section_has_bucket_columns(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that histogram section has bucket values as column headers."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find histogram header row (has count_delta and bucket columns)
        hist_header = None
        for row in rows:
            if row and row[0] == "Endpoint" and "count_delta" in row and "0.01" in row:
                hist_header = row
                break

        assert hist_header is not None
        # Bucket boundaries should be in header as columns
        assert "0.01" in hist_header
        assert "0.1" in hist_header
        assert "1.0" in hist_header
        assert "+Inf" in hist_header

    def test_generate_content_summary_section_has_quantile_columns(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that summary section has quantile values as column headers."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find summary header row (has count_delta and quantile columns)
        summary_header = None
        for row in rows:
            if row and row[0] == "Endpoint" and "count_delta" in row and "0.5" in row:
                summary_header = row
                break

        assert summary_header is not None
        # Quantile keys should be in header as columns
        assert "0.5" in summary_header
        assert "0.9" in summary_header
        assert "0.95" in summary_header
        assert "0.99" in summary_header

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

    def test_generate_content_histogram_bucket_values_in_columns(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that histogram bucket values are in separate columns."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find histogram data row
        hist_data_row = None
        for row in rows:
            if row and "vllm:time_to_first_token_seconds" in row:
                hist_data_row = row
                break

        assert hist_data_row is not None
        # Bucket values should be in separate columns (50, 450, 980, 1000)
        assert "50" in hist_data_row or "50.0000" in hist_data_row
        assert "450" in hist_data_row or "450.0000" in hist_data_row
        assert "980" in hist_data_row or "980.0000" in hist_data_row
        assert "1000" in hist_data_row or "1000.0000" in hist_data_row

    def test_generate_content_summary_quantile_values_in_columns(
        self,
        mock_user_config,
        mock_profile_results,
        server_metrics_results_with_all_types,
    ):
        """Test that summary quantile values are in separate columns."""
        config = create_exporter_config(
            profile_results=mock_profile_results,
            user_config=mock_user_config,
            server_metrics_results=server_metrics_results_with_all_types,
        )
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find summary data row
        summary_data_row = None
        for row in rows:
            if row and "vllm:request_latency_seconds" in row:
                summary_data_row = row
                break

        assert summary_data_row is not None
        # Quantile values should be in separate columns (0.2, 0.4, 0.5, 0.8)
        assert "0.2000" in summary_data_row
        assert "0.4000" in summary_data_row
        assert "0.5000" in summary_data_row
        assert "0.8000" in summary_data_row

    def test_generate_content_groups_histograms_by_bucket_boundaries(
        self, mock_user_config, mock_profile_results
    ):
        """Test that histograms with different bucket boundaries get separate sections."""
        endpoint_summary = ServerMetricsEndpointSummary(
            endpoint_url="http://localhost:8081/metrics",
            duration_seconds=100.0,
            scrape_count=20,
            avg_scrape_latency_ms=8.0,
            info_metrics=None,
            metrics={
                "request_duration_seconds": ServerMetricSummary(
                    description="Request duration",
                    type="histogram",
                    series=[
                        ServerMetricLabeledStats(
                            labels=None,
                            stats=HistogramExportStats(
                                count_delta=100.0,
                                sum_delta=50.0,
                                avg=0.5,
                                rate=1.0,
                                buckets={
                                    "0.1": 10.0,
                                    "0.5": 50.0,
                                    "1.0": 90.0,
                                    "+Inf": 100.0,
                                },
                            ),
                        ),
                    ],
                ),
                "queue_time_seconds": ServerMetricSummary(
                    description="Queue time",
                    type="histogram",
                    series=[
                        ServerMetricLabeledStats(
                            labels=None,
                            stats=HistogramExportStats(
                                count_delta=200.0,
                                sum_delta=10.0,
                                avg=0.05,
                                rate=2.0,
                                buckets={
                                    "0.01": 50.0,
                                    "0.05": 150.0,
                                    "0.1": 190.0,
                                    "+Inf": 200.0,
                                },
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
        exporter = ServerMetricsCsvExporter(config)
        content = exporter._generate_content()
        rows = _parse_csv_content(content)

        # Find histogram header rows (have count_delta and bucket columns)
        hist_headers = []
        for row in rows:
            if (
                row
                and row[0] == "Endpoint"
                and "count_delta" in row
                and len(row) > 7  # has bucket columns beyond base stats
            ):
                hist_headers.append(row)

        # Should have two different histogram headers with different bucket columns
        assert len(hist_headers) == 2
        assert hist_headers[0] != hist_headers[1]


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
        assert "delta,rate_overall,rate_avg" in content  # counter
