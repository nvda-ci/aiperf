# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for full server metrics pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import pytest

from aiperf.common.config import UserConfig
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.post_processors.server_metrics_export_results_processor import (
    ServerMetricsExportResultsProcessor,
)
from aiperf.post_processors.server_metrics_results_processor import (
    ServerMetricsResultsProcessor,
)
from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)

SAMPLE_METRICS = """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200"} 1547.0
http_requests_total{method="POST",status="200"} 892.0

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",le="0.1"} 100
http_request_duration_seconds_bucket{method="GET",le="0.5"} 450
http_request_duration_seconds_bucket{method="GET",le="+Inf"} 500
http_request_duration_seconds_sum{method="GET"} 125.5
http_request_duration_seconds_count{method="GET"} 500

# HELP memory_usage_bytes Current memory usage
# TYPE memory_usage_bytes gauge
memory_usage_bytes{type="heap"} 1073741824
"""


@pytest.mark.asyncio
async def test_collector_to_processor_pipeline():
    """Test full pipeline from collector to results processor."""
    records_collected = []

    async def record_callback(records, collector_id):
        records_collected.extend(records)

    # Create collector
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
        record_callback=record_callback,
    )

    await collector.initialize()

    # Mock fetch to return sample metrics
    with patch.object(collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS):
        await collector._collect_and_process_metrics()

    await collector.stop()

    # Verify records were collected
    assert len(records_collected) == 1
    assert isinstance(records_collected[0], ServerMetricsRecord)

    # Now process through results processor
    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    for record in records_collected:
        await processor.process_server_metrics_record(record)

    # Verify hierarchy is populated
    hierarchy = processor.get_server_metrics_hierarchy()
    assert "http://localhost:8081/metrics" in hierarchy.endpoints

    endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
    assert len(endpoint_data.time_series.snapshots) == 1

    # Verify metrics are present
    _, metrics_dict = endpoint_data.time_series.snapshots[0]
    assert "http_requests" in metrics_dict
    assert "http_request_duration_seconds" in metrics_dict
    assert "memory_usage_bytes" in metrics_dict


@pytest.mark.asyncio
async def test_collector_to_export_processor_pipeline():
    """Test full pipeline from collector to export processor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        records_collected = []

        async def record_callback(records, collector_id):
            records_collected.extend(records)

        # Create collector
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8081/metrics",
            collection_interval=1.0,
            record_callback=record_callback,
        )

        await collector.initialize()

        # Mock fetch
        with patch.object(
            collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS
        ):
            await collector._collect_and_process_metrics()

        await collector.stop()

        # Process through export processor
        user_config = MagicMock(spec=UserConfig)
        user_config.output = MagicMock()
        user_config.output.profile_export_server_metrics_jsonl_file = output_file
        export_processor = ServerMetricsExportResultsProcessor(user_config=user_config)

        # Initialize processor
        await export_processor.initialize()

        for record in records_collected:
            await export_processor.process_server_metrics_record(record)

        await export_processor.stop()

        # Verify JSONL file
        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1

        # Parse and verify structure
        data = orjson.loads(lines[0])
        assert "timestamp_ns" in data
        assert "endpoint_url" in data
        assert "metrics" in data


@pytest.mark.asyncio
async def test_multiple_collectors_to_single_processor():
    """Test multiple collectors feeding into one results processor."""
    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    endpoints = [
        "http://localhost:8081/metrics",
        "http://localhost:9090/metrics",
        "http://192.168.1.1:8080/metrics",
    ]

    # Create multiple collectors
    for endpoint in endpoints:
        collector = ServerMetricsDataCollector(
            endpoint_url=endpoint,
            collection_interval=1.0,
        )

        await collector.initialize()

        # Mock fetch
        with patch.object(
            collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS
        ):
            records = collector._parse_metrics_to_records(SAMPLE_METRICS)

            # Process records
            for record in records:
                await processor.process_server_metrics_record(record)

        await collector.stop()

    # Verify all endpoints are in hierarchy
    hierarchy = processor.get_server_metrics_hierarchy()
    assert len(hierarchy.endpoints) == 3

    for endpoint in endpoints:
        assert endpoint in hierarchy.endpoints


@pytest.mark.asyncio
async def test_collector_to_both_processors():
    """Test collector feeding to both results and export processors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_export.jsonl"

        records_collected = []

        async def record_callback(records, collector_id):
            records_collected.extend(records)

        # Create collector
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8081/metrics",
            collection_interval=1.0,
            record_callback=record_callback,
        )

        await collector.initialize()

        # Collect multiple times
        for _ in range(3):
            with patch.object(
                collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS
            ):
                await collector._collect_and_process_metrics()

        await collector.stop()

        # Create both processors
        user_config = MagicMock(spec=UserConfig)
        user_config.output = MagicMock()
        user_config.output.profile_export_server_metrics_jsonl_file = output_file

        results_processor = ServerMetricsResultsProcessor(user_config=user_config)
        export_processor = ServerMetricsExportResultsProcessor(user_config=user_config)

        # Initialize export processor
        await export_processor.initialize()

        # Process through both
        for record in records_collected:
            await results_processor.process_server_metrics_record(record)
            await export_processor.process_server_metrics_record(record)

        await export_processor.stop()

        # Verify results processor
        hierarchy = results_processor.get_server_metrics_hierarchy()
        endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
        assert len(endpoint_data.time_series.snapshots) == 3

        # Verify export processor
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 3


@pytest.mark.asyncio
async def test_error_handling_in_pipeline():
    """Test error handling throughout the pipeline."""
    error_received = []

    async def error_callback(error, collector_id):
        error_received.append(error)

    # Create collector with error callback
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
        error_callback=error_callback,
    )

    await collector.initialize()

    # Mock fetch to raise error
    with patch.object(
        collector, "_fetch_metrics_text", side_effect=RuntimeError("Connection failed")
    ):
        # Should call error callback
        await collector._collect_metrics_task()

    await collector.stop()

    # Verify error was captured
    assert len(error_received) == 1


@pytest.mark.asyncio
async def test_empty_metrics_handling():
    """Test pipeline handles empty metrics gracefully."""
    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Return empty metrics
    with patch.object(collector, "_fetch_metrics_text", return_value=""):
        records = collector._parse_metrics_to_records("")

        for record in records:
            await processor.process_server_metrics_record(record)

    await collector.stop()

    # Should not crash, hierarchy should be empty
    hierarchy = processor.get_server_metrics_hierarchy()
    assert len(hierarchy.endpoints) == 0


@pytest.mark.asyncio
async def test_concurrent_collection_cycles():
    """Test multiple collection cycles in sequence."""
    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Simulate multiple collection cycles
    for _ in range(5):
        with patch.object(
            collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS
        ):
            records = collector._parse_metrics_to_records(SAMPLE_METRICS)

            for record in records:
                await processor.process_server_metrics_record(record)

    await collector.stop()

    # Verify all cycles were recorded
    hierarchy = processor.get_server_metrics_hierarchy()
    endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
    assert len(endpoint_data.time_series.snapshots) == 5


@pytest.mark.asyncio
async def test_metrics_variety_preservation():
    """Test that all metric types are preserved through pipeline."""
    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    with patch.object(collector, "_fetch_metrics_text", return_value=SAMPLE_METRICS):
        records = collector._parse_metrics_to_records(SAMPLE_METRICS)

        for record in records:
            await processor.process_server_metrics_record(record)

    await collector.stop()

    # Get the snapshot and verify metric types
    hierarchy = processor.get_server_metrics_hierarchy()
    endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
    _, metrics_dict = endpoint_data.time_series.snapshots[0]

    # Verify counter
    assert "http_requests" in metrics_dict
    assert metrics_dict["http_requests"].type == "counter"

    # Verify histogram
    assert "http_request_duration_seconds" in metrics_dict
    assert metrics_dict["http_request_duration_seconds"].type == "histogram"
    histogram_sample = metrics_dict["http_request_duration_seconds"].samples[0]
    assert histogram_sample.histogram is not None
    assert histogram_sample.histogram.count == 500.0

    # Verify gauge
    assert "memory_usage_bytes" in metrics_dict
    assert metrics_dict["memory_usage_bytes"].type == "gauge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
