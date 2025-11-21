# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import orjson
import pytest

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import EndpointType, PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.post_processors.server_metrics_export_results_processor import (
    ServerMetricsExportResultsProcessor,
)
from tests.unit.post_processors.conftest import aiperf_lifecycle


@pytest.fixture
def user_config_server_metrics_export(tmp_artifact_dir: Path) -> UserConfig:
    """Create UserConfig for server metrics export testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
        ),
        output=OutputConfig(
            artifact_directory=tmp_artifact_dir,
        ),
    )


@pytest.fixture
def sample_server_metrics_record_for_export() -> ServerMetricsRecord:
    """Create sample ServerMetricsRecord for export testing."""
    return ServerMetricsRecord(
        endpoint_url="http://localhost:8081/metrics",
        timestamp_ns=1_000_000_000,
        endpoint_latency_ns=5_000_000,
        metrics={
            "requests_total": MetricFamily(
                type=PrometheusMetricType.COUNTER,
                description="Total requests",
                samples=[
                    MetricSample(
                        labels={"status": "success"},
                        value=100.0,
                    )
                ],
            ),
        },
    )


class TestServerMetricsExportResultsProcessorInitialization:
    """Test ServerMetricsExportResultsProcessor initialization."""

    def test_initialization(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test processor initializes with correct file paths."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        assert (
            processor.output_file
            == user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        assert (
            processor._metadata_file
            == user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )

    def test_files_cleared_on_initialization(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        tmp_artifact_dir: Path,
    ):
        """Test that output files are cleared on initialization."""
        jsonl_file = tmp_artifact_dir / "server_metrics_export.jsonl"
        metadata_file = tmp_artifact_dir / "server_metrics_metadata.json"

        jsonl_file.write_text("old data")
        metadata_file.write_text("old metadata")

        ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        assert not jsonl_file.exists() or jsonl_file.stat().st_size == 0


class TestServerMetricsRecordProcessing:
    """Test processing ServerMetricsRecord objects."""

    @pytest.mark.asyncio
    async def test_process_single_record(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test processing single server metrics record."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        assert output_file.exists()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1

        data = orjson.loads(lines[0])
        assert data["endpoint_url"] == "http://localhost:8081/metrics"
        assert data["timestamp_ns"] == 1_000_000_000
        assert data["endpoint_latency_ns"] == 5_000_000
        assert "metrics" in data

    @pytest.mark.asyncio
    async def test_process_multiple_records(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test processing multiple server metrics records with different metrics."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            for i in range(5):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={
                        "counter": MetricFamily(
                            type=PrometheusMetricType.COUNTER,
                            description="Test counter",
                            samples=[
                                MetricSample(
                                    labels={},
                                    value=float(
                                        i
                                    ),  # Different values to avoid deduplication
                                )
                            ],
                        ),
                    },
                )
                await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 5

    @pytest.mark.asyncio
    async def test_record_converted_to_slim_format(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test that records are converted to slim format before writing."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        data = orjson.loads(output_file.read_text().strip())

        assert "metrics" in data
        assert "requests_total" in data["metrics"]


class TestMetadataExtraction:
    """Test metadata extraction and writing."""

    @pytest.mark.asyncio
    async def test_metadata_extracted_on_first_record(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test that metadata is extracted and written on first record from endpoint."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        assert metadata_file.exists()

        metadata_content = orjson.loads(metadata_file.read_bytes())
        assert "endpoints" in metadata_content
        assert "http://localhost:8081/metrics" in metadata_content["endpoints"]

    @pytest.mark.asyncio
    async def test_metadata_contains_metric_schemas(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test that metadata includes metric schemas (type, help)."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        endpoint_metadata = metadata_content["endpoints"][
            "http://localhost:8081/metrics"
        ]
        assert "metric_schemas" in endpoint_metadata
        assert "requests_total" in endpoint_metadata["metric_schemas"]

        schema = endpoint_metadata["metric_schemas"]["requests_total"]
        assert schema["type"] == "counter"
        assert schema["description"] == "Total requests"

    @pytest.mark.asyncio
    async def test_histogram_schema_includes_bucket_labels(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that histogram schemas are exported correctly."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "ttft": MetricFamily(
                    type=PrometheusMetricType.HISTOGRAM,
                    description="Time to first token",
                    samples=[
                        MetricSample(
                            labels={"model": "test"},
                            histogram=HistogramData(
                                buckets={"0.01": 5.0, "0.1": 15.0, "+Inf": 50.0},
                                sum=5.5,
                                count=50.0,
                            ),
                        )
                    ],
                )
            },
        )

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["ttft"]
        assert schema["type"] == "histogram"
        assert schema["description"] == "Time to first token"

    @pytest.mark.asyncio
    async def test_metadata_includes_unique_label_values(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metadata includes schema for metrics with multiple samples."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Total requests",
                    samples=[
                        MetricSample(
                            labels={"status": "success", "endpoint": "chat"},
                            value=100.0,
                        ),
                        MetricSample(
                            labels={"status": "error", "endpoint": "chat"},
                            value=10.0,
                        ),
                        MetricSample(
                            labels={"status": "success", "endpoint": "completions"},
                            value=50.0,
                        ),
                    ],
                )
            },
        )

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["requests_total"]
        assert schema["type"] == "counter"
        assert schema["description"] == "Total requests"

    @pytest.mark.asyncio
    async def test_unique_label_values_respects_cardinality_limit(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        monkeypatch,
    ):
        """Test that metadata handles metrics with multiple label values."""
        # Create record with 3 unique label values
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Total requests",
                    samples=[
                        MetricSample(labels={"status": "success"}, value=100.0),
                        MetricSample(labels={"status": "error"}, value=10.0),
                        MetricSample(labels={"status": "timeout"}, value=5.0),
                    ],
                )
            },
        )

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["requests_total"]
        assert schema["type"] == "counter"
        assert schema["description"] == "Total requests"

    @pytest.mark.asyncio
    async def test_metadata_updated_for_multiple_endpoints(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metadata file contains all endpoints."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            for endpoint in ["http://node1:8081/metrics", "http://node2:8081/metrics"]:
                record = ServerMetricsRecord(
                    endpoint_url=endpoint,
                    timestamp_ns=1_000_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={},
                )
                await processor.process_server_metrics_record(record)

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        assert len(metadata_content["endpoints"]) == 2
        assert "http://node1:8081/metrics" in metadata_content["endpoints"]
        assert "http://node2:8081/metrics" in metadata_content["endpoints"]

    @pytest.mark.asyncio
    async def test_metadata_only_written_once_per_endpoint(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metadata is only extracted on first record per endpoint."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            for _ in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={},
                )
                await processor.process_server_metrics_record(record)

        assert (
            "http://localhost:8081/metrics" in processor._metadata_file_model.endpoints
        )
        assert len(processor._metadata_file_model.endpoints) == 1


class TestSummarizeMethod:
    """Test summarize method behavior."""

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_list(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that summarize returns empty list (export processors don't summarize)."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            results = await processor.summarize()

        assert results == []


class TestMetadataReconciliation:
    """Test metadata reconciliation for evolving metrics."""

    @pytest.mark.asyncio
    async def test_new_metrics_appearing_later(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that new metrics appearing in later records are captured."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record with metric A
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Metric A",
                        samples=[MetricSample(labels={}, value=100.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            # Second record with metrics A and B
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Metric A",
                        samples=[MetricSample(labels={}, value=101.0)],
                    ),
                    "metric_b": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        description="Metric B",
                        samples=[MetricSample(labels={}, value=50.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

        # Verify metadata includes both metrics
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        endpoint_metadata = metadata_content["endpoints"][
            "http://localhost:8081/metrics"
        ]

        assert "metric_a" in endpoint_metadata["metric_schemas"]
        assert "metric_b" in endpoint_metadata["metric_schemas"]
        assert endpoint_metadata["metric_schemas"]["metric_a"]["type"] == "counter"
        assert endpoint_metadata["metric_schemas"]["metric_b"]["type"] == "gauge"

    @pytest.mark.asyncio
    async def test_same_count_different_metrics(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that different metrics with same count are detected."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record with metrics A, B, C
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Metric A",
                        samples=[MetricSample(labels={}, value=1.0)],
                    ),
                    "metric_b": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Metric B",
                        samples=[MetricSample(labels={}, value=2.0)],
                    ),
                    "metric_c": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Metric C",
                        samples=[MetricSample(labels={}, value=3.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            # Second record with metrics B, C, D (same count, but D is new)
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_b": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Metric B",
                        samples=[MetricSample(labels={}, value=2.0)],
                    ),
                    "metric_c": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Metric C",
                        samples=[MetricSample(labels={}, value=3.0)],
                    ),
                    "metric_d": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        description="Metric D",
                        samples=[MetricSample(labels={}, value=4.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

        # Verify metadata includes all metrics (A, B, C, D)
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        endpoint_metadata = metadata_content["endpoints"][
            "http://localhost:8081/metrics"
        ]

        assert len(endpoint_metadata["metric_schemas"]) == 4
        assert "metric_a" in endpoint_metadata["metric_schemas"]
        assert "metric_b" in endpoint_metadata["metric_schemas"]
        assert "metric_c" in endpoint_metadata["metric_schemas"]
        assert "metric_d" in endpoint_metadata["metric_schemas"]

    @pytest.mark.asyncio
    async def test_histogram_bucket_changes(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that new histogram buckets are detected and merged."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record with histogram with 3 buckets
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "request_duration": MetricFamily(
                        type=PrometheusMetricType.HISTOGRAM,
                        description="Request duration",
                        samples=[
                            MetricSample(
                                labels={},
                                histogram=HistogramData(
                                    buckets={"0.1": 10.0, "0.5": 50.0, "+Inf": 100.0},
                                    sum=25.5,
                                    count=100.0,
                                ),
                            )
                        ],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            # Second record with 5 buckets (added 0.01 and 1.0)
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "request_duration": MetricFamily(
                        type=PrometheusMetricType.HISTOGRAM,
                        description="Request duration",
                        samples=[
                            MetricSample(
                                labels={},
                                histogram=HistogramData(
                                    buckets={
                                        "0.01": 5.0,
                                        "0.1": 15.0,
                                        "0.5": 60.0,
                                        "1.0": 80.0,
                                        "+Inf": 120.0,
                                    },
                                    sum=35.5,
                                    count=120.0,
                                ),
                            )
                        ],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

        # Verify metadata includes schema
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["request_duration"]

        assert schema["type"] == "histogram"
        assert schema["description"] == "Request duration"

    @pytest.mark.asyncio
    async def test_summary_quantile_changes(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that new summary quantiles are detected and merged."""
        from aiperf.common.models.server_metrics_models import SummaryData

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record with 3 quantiles
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "response_time": MetricFamily(
                        type=PrometheusMetricType.SUMMARY,
                        description="Response time",
                        samples=[
                            MetricSample(
                                labels={},
                                summary=SummaryData(
                                    quantiles={"0.5": 0.1, "0.9": 0.5, "0.99": 1.0},
                                    sum=50.0,
                                    count=100.0,
                                ),
                            )
                        ],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            # Second record with 5 quantiles (added 0.25 and 0.95)
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "response_time": MetricFamily(
                        type=PrometheusMetricType.SUMMARY,
                        description="Response time",
                        samples=[
                            MetricSample(
                                labels={},
                                summary=SummaryData(
                                    quantiles={
                                        "0.25": 0.05,
                                        "0.5": 0.12,
                                        "0.9": 0.55,
                                        "0.95": 0.75,
                                        "0.99": 1.1,
                                    },
                                    sum=60.0,
                                    count=120.0,
                                ),
                            )
                        ],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

        # Verify metadata includes all quantiles (union)
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["response_time"]

        assert schema["type"] == "summary"
        assert schema["description"] == "Response time"

    @pytest.mark.asyncio
    async def test_no_update_for_identical_metadata(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metadata file is not rewritten when metrics don't change."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Metric A",
                        samples=[MetricSample(labels={}, value=100.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            metadata_file = user_config_server_metrics_export.output.server_metrics_metadata_json_file
            first_mtime = metadata_file.stat().st_mtime_ns

            # Wait a bit to ensure timestamp would change if file is rewritten
            await asyncio.sleep(0.01)

            # Second record with same metrics (different values)
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Metric A",
                        samples=[MetricSample(labels={}, value=105.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

            second_mtime = metadata_file.stat().st_mtime_ns

            # Metadata file should not be rewritten
            assert first_mtime == second_mtime

    @pytest.mark.asyncio
    async def test_metadata_merge_is_idempotent(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that merging the same metadata multiple times produces same result."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # Record with histogram
            record = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "duration": MetricFamily(
                        type=PrometheusMetricType.HISTOGRAM,
                        description="Duration",
                        samples=[
                            MetricSample(
                                labels={},
                                histogram=HistogramData(
                                    buckets={"0.1": 10.0, "0.5": 50.0, "+Inf": 100.0},
                                    sum=25.5,
                                    count=100.0,
                                ),
                            )
                        ],
                    ),
                },
            )

            # Process same record 3 times
            await processor.process_server_metrics_record(record)
            await processor.process_server_metrics_record(record)
            await processor.process_server_metrics_record(record)

        # Verify schema is present
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["duration"]

        assert schema["type"] == "histogram"
        assert schema["description"] == "Duration"


class TestInfoMetricsHandling:
    """Test that _info metrics are properly separated in metadata and excluded from slim records."""

    @pytest.mark.asyncio
    async def test_info_metrics_stored_separately_in_metadata(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metrics ending in _info are stored in info_metrics field."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "python_info": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Python platform information",
                    samples=[
                        MetricSample(
                            labels={"version": "3.10.0", "implementation": "CPython"},
                            value=1.0,
                        )
                    ],
                ),
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Total requests",
                    samples=[
                        MetricSample(
                            labels={"status": "success"},
                            value=100.0,
                        )
                    ],
                ),
            },
        )

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        endpoint_metadata = metadata_content["endpoints"][
            "http://localhost:8081/metrics"
        ]

        # Verify python_info is in info_metrics, not metric_schemas
        assert "python_info" in endpoint_metadata["info_metrics"]
        assert "python_info" not in endpoint_metadata["metric_schemas"]

        # Verify regular metrics are still in metric_schemas
        assert "requests_total" in endpoint_metadata["metric_schemas"]
        assert "requests_total" not in endpoint_metadata["info_metrics"]

        # Verify the info_metric contains description + labels (no values or type)
        info_data = endpoint_metadata["info_metrics"]["python_info"]
        assert "type" not in info_data
        assert info_data["description"] == "Python platform information"

        # Verify labels are stored as list of dicts (values omitted)
        assert "labels" in info_data
        assert len(info_data["labels"]) == 1
        labels = info_data["labels"][0]
        assert labels == {"version": "3.10.0", "implementation": "CPython"}
        # Verify no value field
        assert "value" not in info_data

    @pytest.mark.asyncio
    async def test_info_metrics_excluded_from_slim_records(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metrics ending in _info are excluded from slim JSONL records."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "python_info": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Python platform information",
                    samples=[
                        MetricSample(
                            labels={"version": "3.10.0"},
                            value=1.0,
                        )
                    ],
                ),
                "process_info": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Process information",
                    samples=[
                        MetricSample(
                            labels={"pid": "1234"},
                            value=1.0,
                        )
                    ],
                ),
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Total requests",
                    samples=[
                        MetricSample(
                            labels={"status": "success"},
                            value=100.0,
                        )
                    ],
                ),
            },
        )

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        jsonl_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = jsonl_file.read_text().strip().split("\n")

        # Should have 1 line
        assert len(lines) == 1

        slim_record = orjson.loads(lines[0])

        # Verify _info metrics are NOT in the slim record
        assert "python_info" not in slim_record["metrics"]
        assert "process_info" not in slim_record["metrics"]

        # Verify regular metrics ARE in the slim record
        assert "requests_total" in slim_record["metrics"]

    @pytest.mark.asyncio
    async def test_mixed_info_and_regular_metrics(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test handling of multiple _info metrics alongside regular metrics."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "python_info": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Python info",
                    samples=[MetricSample(labels={}, value=1.0)],
                ),
                "server_info": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Server info",
                    samples=[MetricSample(labels={}, value=1.0)],
                ),
                "cpu_usage": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="CPU usage",
                    samples=[MetricSample(labels={}, value=42.0)],
                ),
                "memory_usage": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Memory usage",
                    samples=[MetricSample(labels={}, value=1024.0)],
                ),
            },
        )

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        # Check metadata file
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        endpoint_metadata = metadata_content["endpoints"][
            "http://localhost:8081/metrics"
        ]

        # Verify correct classification
        assert len(endpoint_metadata["info_metrics"]) == 2
        assert "python_info" in endpoint_metadata["info_metrics"]
        assert "server_info" in endpoint_metadata["info_metrics"]

        assert len(endpoint_metadata["metric_schemas"]) == 2
        assert "cpu_usage" in endpoint_metadata["metric_schemas"]
        assert "memory_usage" in endpoint_metadata["metric_schemas"]

        # Verify info metrics have labels (no values)
        python_info = endpoint_metadata["info_metrics"]["python_info"]
        assert "labels" in python_info
        assert len(python_info["labels"]) == 1
        assert isinstance(python_info["labels"][0], dict)
        assert "value" not in python_info

        server_info = endpoint_metadata["info_metrics"]["server_info"]
        assert "labels" in server_info
        assert len(server_info["labels"]) == 1
        assert isinstance(server_info["labels"][0], dict)
        assert "value" not in server_info

        # Check JSONL file
        jsonl_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        slim_record = orjson.loads(jsonl_file.read_text().strip())

        # Only regular metrics in slim record
        assert len(slim_record["metrics"]) == 2
        assert "cpu_usage" in slim_record["metrics"]
        assert "memory_usage" in slim_record["metrics"]
        assert "python_info" not in slim_record["metrics"]
        assert "server_info" not in slim_record["metrics"]
