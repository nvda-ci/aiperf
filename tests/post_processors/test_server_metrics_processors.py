# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for ServerMetrics processors."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.config import UserConfig
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.post_processors.server_metrics_export_results_processor import (
    ServerMetricsExportResultsProcessor,
)
from aiperf.post_processors.server_metrics_results_processor import (
    ServerMetricsResultsProcessor,
)


@pytest.fixture
def user_config(tmp_path):
    """Create mock user config."""
    config = MagicMock(spec=UserConfig)
    config.output = MagicMock()
    # Provide a default temporary path
    config.output.profile_export_server_metrics_jsonl_file = (
        tmp_path / "default_export.jsonl"
    )
    return config


@pytest.fixture
def sample_record():
    """Create sample server metrics record."""
    sample = MetricSample(labels={"method": "GET"}, value=100.0)
    family = MetricFamily(type="counter", help="Test metric", samples=[sample])

    return ServerMetricsRecord(
        timestamp_ns=1000000000,
        endpoint_url="http://localhost:8081/metrics",
        metrics={"test_metric": family},
    )


@pytest.mark.asyncio
async def test_results_processor_initialization(user_config):
    """Test ServerMetricsResultsProcessor initialization."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    hierarchy = processor.get_server_metrics_hierarchy()
    assert hierarchy is not None
    assert len(hierarchy.endpoints) == 0


@pytest.mark.asyncio
async def test_results_processor_single_record(user_config, sample_record):
    """Test processing a single record."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    await processor.process_server_metrics_record(sample_record)

    hierarchy = processor.get_server_metrics_hierarchy()
    assert "http://localhost:8081/metrics" in hierarchy.endpoints

    endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
    assert len(endpoint_data.time_series.snapshots) == 1


@pytest.mark.asyncio
async def test_results_processor_multiple_records_same_endpoint(
    user_config, sample_record
):
    """Test processing multiple records from same endpoint."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    # Process multiple records
    for i in range(5):
        record = ServerMetricsRecord(
            timestamp_ns=1000000000 + i * 1000000000,
            endpoint_url="http://localhost:8081/metrics",
            metrics=sample_record.metrics,
        )
        await processor.process_server_metrics_record(record)

    hierarchy = processor.get_server_metrics_hierarchy()
    endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
    assert len(endpoint_data.time_series.snapshots) == 5


@pytest.mark.asyncio
async def test_results_processor_multiple_endpoints(user_config, sample_record):
    """Test processing records from multiple endpoints."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    # Create records for different endpoints
    endpoints = [
        "http://localhost:8081/metrics",
        "http://localhost:9090/metrics",
        "http://192.168.1.1:8080/metrics",
    ]

    for endpoint in endpoints:
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url=endpoint,
            metrics=sample_record.metrics,
        )
        await processor.process_server_metrics_record(record)

    hierarchy = processor.get_server_metrics_hierarchy()
    assert len(hierarchy.endpoints) == 3

    for endpoint in endpoints:
        assert endpoint in hierarchy.endpoints


@pytest.mark.asyncio
async def test_results_processor_summarize(user_config, sample_record):
    """Test summarize method returns empty list."""
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    await processor.process_server_metrics_record(sample_record)

    results = await processor.summarize()
    # Currently returns empty list - placeholder for future metric extraction
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_export_processor_initialization(user_config):
    """Test ServerMetricsExportResultsProcessor initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        # Mock output path in processor
        processor = ServerMetricsExportResultsProcessor(user_config=user_config)

        # Override output file for testing
        processor.output_file = output_file

        assert processor.output_file == output_file


@pytest.mark.asyncio
async def test_export_processor_writes_record(user_config, sample_record):
    """Test export processor writes records to JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        processor = ServerMetricsExportResultsProcessor(user_config=user_config)
        processor.output_file = output_file

        # Initialize processor
        await processor.initialize()

        # Process a record
        await processor.process_server_metrics_record(sample_record)

        # Stop processor (which flushes)
        await processor.stop()

        # Verify file exists and contains data
        assert output_file.exists()
        content = output_file.read_text()
        assert len(content) > 0

        # Parse JSON and verify structure
        record_data = orjson.loads(content)
        assert record_data["timestamp_ns"] == sample_record.timestamp_ns
        assert record_data["endpoint_url"] == sample_record.endpoint_url
        assert "metrics" in record_data


@pytest.mark.asyncio
async def test_export_processor_multiple_records(user_config, sample_record):
    """Test export processor handles multiple records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        processor = ServerMetricsExportResultsProcessor(user_config=user_config)
        processor.output_file = output_file

        # Initialize processor
        await processor.initialize()

        # Process multiple records
        for i in range(10):
            record = ServerMetricsRecord(
                timestamp_ns=1000000000 + i * 1000000000,
                endpoint_url="http://localhost:8081/metrics",
                metrics=sample_record.metrics,
            )
            await processor.process_server_metrics_record(record)

        # Stop processor (which flushes)
        await processor.stop()

        # Verify file has multiple lines
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 10

        # Verify each line is valid JSON
        for line in lines:
            data = orjson.loads(line)
            assert "timestamp_ns" in data
            assert "endpoint_url" in data


@pytest.mark.asyncio
async def test_export_processor_different_endpoints(user_config, sample_record):
    """Test export processor handles records from different endpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        processor = ServerMetricsExportResultsProcessor(user_config=user_config)
        processor.output_file = output_file

        # Initialize processor
        await processor.initialize()

        # Create records for different endpoints
        endpoints = ["http://localhost:8081/metrics", "http://localhost:9090/metrics"]

        for endpoint in endpoints:
            record = ServerMetricsRecord(
                timestamp_ns=1000000000,
                endpoint_url=endpoint,
                metrics=sample_record.metrics,
            )
            await processor.process_server_metrics_record(record)

        await processor.stop()

        # Verify all endpoints present
        lines = output_file.read_text().strip().split("\n")
        endpoint_urls = [orjson.loads(line)["endpoint_url"] for line in lines]
        assert set(endpoint_urls) == set(endpoints)


@pytest.mark.asyncio
async def test_export_processor_preserves_snapshot_structure(user_config):
    """Test that export processor preserves full snapshot structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        processor = ServerMetricsExportResultsProcessor(user_config=user_config)
        processor.output_file = output_file

        # Initialize processor
        await processor.initialize()

        # Create complex snapshot with multiple metrics
        samples1 = [
            MetricSample(labels={"method": "GET", "status": "200"}, value=100.0),
            MetricSample(labels={"method": "POST", "status": "200"}, value=50.0),
        ]
        samples2 = [MetricSample(labels={"type": "heap"}, value=1024.0)]

        metrics = {
            "http_requests": MetricFamily(
                type="counter", help="Requests", samples=samples1
            ),
            "memory_bytes": MetricFamily(type="gauge", help="Memory", samples=samples2),
        }

        record = ServerMetricsRecord(
            timestamp_ns=1500000000,
            endpoint_url="http://localhost:8081/metrics",
            metrics=metrics,
        )

        await processor.process_server_metrics_record(record)
        await processor.stop()

        # Parse and verify structure
        data = orjson.loads(output_file.read_text())
        assert "metrics" in data
        assert "http_requests" in data["metrics"]
        assert "memory_bytes" in data["metrics"]

        # Verify http_requests metric
        http_metric = data["metrics"]["http_requests"]
        assert http_metric["type"] == "counter"
        assert len(http_metric["samples"]) == 2


@pytest.mark.asyncio
async def test_export_processor_summarize_returns_empty(user_config):
    """Test export processor summarize returns empty list."""
    processor = ServerMetricsExportResultsProcessor(user_config=user_config)

    results = await processor.summarize()
    assert results == []


@pytest.mark.asyncio
async def test_export_processor_handles_write_error(user_config, sample_record):
    """Test export processor handles write errors gracefully."""
    # Use invalid path to trigger error
    processor = ServerMetricsExportResultsProcessor(user_config=user_config)
    processor.output_file = Path("/invalid/path/test.jsonl")

    # Should not raise, just log error
    await processor.process_server_metrics_record(sample_record)


@pytest.mark.asyncio
async def test_results_processor_hierarchy_isolation(user_config):
    """Test that different processor instances have isolated hierarchies."""
    processor1 = ServerMetricsResultsProcessor(user_config=user_config)
    processor2 = ServerMetricsResultsProcessor(user_config=user_config)

    sample = MetricSample(labels={}, value=100.0)
    family = MetricFamily(type="counter", help="Test", samples=[sample])

    record = ServerMetricsRecord(
        timestamp_ns=1000000000,
        endpoint_url="http://localhost:8081/metrics",
        metrics={"test": family},
    )

    # Process in first processor only
    await processor1.process_server_metrics_record(record)

    # Verify isolation
    hierarchy1 = processor1.get_server_metrics_hierarchy()
    hierarchy2 = processor2.get_server_metrics_hierarchy()

    assert len(hierarchy1.endpoints) == 1
    assert len(hierarchy2.endpoints) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
