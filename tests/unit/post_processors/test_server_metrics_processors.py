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


@pytest.fixture
def user_config(tmp_path):
    """Create mock user config."""
    config = MagicMock(spec=UserConfig)
    config.output = MagicMock()
    # Provide a default temporary path
    config.output.server_metrics_export_jsonl_file = tmp_path / "default_export.jsonl"
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

        # Parse JSON and verify slim structure (no endpoint_url, type, help)
        record_data = orjson.loads(content)
        assert record_data["timestamp_ns"] == sample_record.timestamp_ns
        assert "endpoint_url" not in record_data  # Slim format excludes endpoint_url
        assert "metrics" in record_data
        # Verify flat structure: metrics map directly to sample lists
        assert isinstance(record_data["metrics"], dict)
        for _metric_name, samples in record_data["metrics"].items():
            assert isinstance(samples, list)  # Direct list, no 'samples' key nesting


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

        # Verify each line is valid JSON with slim structure
        for line in lines:
            data = orjson.loads(line)
            assert "timestamp_ns" in data
            assert "endpoint_url" not in data  # Slim format excludes endpoint_url
            assert "metrics" in data


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

        # Verify both records written (slim format doesn't include endpoint_url)
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == len(endpoints)
        # Each line should have slim structure
        for line in lines:
            data = orjson.loads(line)
            assert "timestamp_ns" in data
            assert "metrics" in data
            assert "endpoint_url" not in data  # Slim format excludes endpoint_url


@pytest.mark.asyncio
async def test_export_processor_preserves_snapshot_structure(user_config):
    """Test that export processor writes slim snapshot structure (flat, no type/help)."""
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

        # Parse and verify slim structure (flat, no type/help)
        data = orjson.loads(output_file.read_text())
        assert "metrics" in data
        assert "http_requests" in data["metrics"]
        assert "memory_bytes" in data["metrics"]

        # Verify http_requests metric in slim format (direct list, no type/help/samples keys)
        http_samples = data["metrics"]["http_requests"]
        assert isinstance(
            http_samples, list
        )  # Direct list, not nested in 'samples' key
        assert len(http_samples) == 2
        # Verify samples have expected structure
        assert http_samples[0]["labels"]["method"] == "GET"
        assert http_samples[0]["value"] == 100.0


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
