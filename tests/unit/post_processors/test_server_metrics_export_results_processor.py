# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
                help="Total requests",
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
        """Test processing multiple server metrics records."""
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
                    metrics={},
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
        assert schema["help"] == "Total requests"

    @pytest.mark.asyncio
    async def test_histogram_schema_includes_bucket_labels(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that histogram schemas include bucket labels."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "ttft": MetricFamily(
                    type=PrometheusMetricType.HISTOGRAM,
                    help="Time to first token",
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
        assert "bucket_labels" in schema
        assert schema["bucket_labels"] == ["0.01", "0.1", "+Inf"]

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

        assert "http://localhost:8081/metrics" in processor._seen_endpoints
        assert len(processor._seen_endpoints) == 1


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
