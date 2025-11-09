# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsResultsProcessor aggregation functionality."""

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.post_processors.server_metrics_results_processor import (
    ServerMetricsResultsProcessor,
)


@pytest.fixture
def user_config():
    """Create minimal user config."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
        )
    )


@pytest.fixture
def processor(user_config):
    """Create ServerMetricsResultsProcessor instance."""
    return ServerMetricsResultsProcessor(user_config=user_config)


@pytest.fixture
def sample_counter_records():
    """Create sample counter metric records for testing aggregation."""
    records = []
    for i in range(10):
        metrics = {
            "http_requests_total": MetricFamily(
                type="counter",
                help="Total HTTP requests",
                samples=[
                    MetricSample(
                        labels={"method": "GET", "status": "200"},
                        value=100.0 + i * 10,
                    ),
                    MetricSample(
                        labels={"method": "POST", "status": "200"},
                        value=50.0 + i * 5,
                    ),
                ],
            ),
        }
        record = ServerMetricsRecord(
            timestamp_ns=1000000000 + i * 100000000,
            endpoint_url="http://localhost:8081/metrics",
            metrics=metrics,
        )
        records.append(record)
    return records


@pytest.fixture
def sample_gauge_records():
    """Create sample gauge metric records for testing aggregation."""
    records = []
    for i in range(10):
        metrics = {
            "memory_usage_bytes": MetricFamily(
                type="gauge",
                help="Memory usage in bytes",
                samples=[
                    MetricSample(
                        labels={"type": "heap"},
                        value=1024.0 * (100 + i * 10),
                    ),
                ],
            ),
            "cpu_usage_percent": MetricFamily(
                type="gauge",
                help="CPU usage percentage",
                samples=[
                    MetricSample(
                        labels={},
                        value=50.0 + i * 2,
                    ),
                ],
            ),
        }
        record = ServerMetricsRecord(
            timestamp_ns=1000000000 + i * 100000000,
            endpoint_url="http://localhost:8081/metrics",
            metrics=metrics,
        )
        records.append(record)
    return records


@pytest.mark.asyncio
async def test_processor_initialization(processor):
    """Test processor initializes correctly."""
    assert processor is not None
    assert processor._server_metrics_hierarchy is not None
    assert processor._discovered_metrics is not None


@pytest.mark.asyncio
async def test_process_server_metrics_record(processor, sample_counter_records):
    """Test processing individual server metrics records."""
    for record in sample_counter_records:
        await processor.process_server_metrics_record(record)

    hierarchy = processor.get_server_metrics_hierarchy()
    assert "http://localhost:8081/metrics" in hierarchy.endpoints

    endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
    assert len(endpoint_data.time_series.snapshots) == 10


@pytest.mark.asyncio
async def test_metric_discovery(processor, sample_counter_records):
    """Test auto-discovery of metrics during processing."""
    for record in sample_counter_records:
        await processor.process_server_metrics_record(record)

    # Should have discovered the counter metric with two label combinations
    assert len(processor._discovered_metrics) >= 2

    # Check that specific metrics were discovered
    metric_names = {item[0] for item in processor._discovered_metrics}
    assert "http_requests_total" in metric_names


@pytest.mark.asyncio
async def test_summarize_counter_metrics(processor, sample_counter_records):
    """Test aggregation of counter metrics."""
    for record in sample_counter_records:
        await processor.process_server_metrics_record(record)

    results = await processor.summarize()

    assert len(results) > 0

    # Find the http_requests_total metric with GET method
    get_metrics = [
        r for r in results if "http_requests_total" in r.tag and "method_GET" in r.tag
    ]
    assert len(get_metrics) == 1

    metric = get_metrics[0]
    # For counters, we compute the delta between first and last value
    # First GET value: 100.0, Last GET value: 190.0, Delta: 90.0
    assert metric.min == 90.0
    assert metric.max == 90.0
    assert metric.avg == 90.0
    assert metric.count == 1  # Counter deltas are single values


@pytest.mark.asyncio
async def test_summarize_gauge_metrics(processor, sample_gauge_records):
    """Test aggregation of gauge metrics."""
    for record in sample_gauge_records:
        await processor.process_server_metrics_record(record)

    results = await processor.summarize()

    assert len(results) > 0

    # Find CPU usage metric
    cpu_metrics = [r for r in results if "cpu_usage_percent" in r.tag]
    assert len(cpu_metrics) == 1

    metric = cpu_metrics[0]
    assert metric.min == 50.0
    assert metric.max == 68.0
    assert metric.count == 10


@pytest.mark.asyncio
async def test_summarize_with_multiple_endpoints(processor, sample_counter_records):
    """Test aggregation with multiple endpoints."""
    # Add records for first endpoint
    for record in sample_counter_records:
        await processor.process_server_metrics_record(record)

    # Add records for second endpoint
    for _i, record in enumerate(sample_counter_records):
        modified_record = ServerMetricsRecord(
            timestamp_ns=record.timestamp_ns,
            endpoint_url="http://localhost:9090/metrics",
            metrics=record.metrics,
        )
        await processor.process_server_metrics_record(modified_record)

    results = await processor.summarize()

    # Should have metrics for both endpoints
    endpoint_tags = {r.tag.split(".")[1] for r in results if "server_metrics." in r.tag}
    assert len(endpoint_tags) == 2


@pytest.mark.asyncio
async def test_unit_inference(processor):
    """Test unit inference from metric names."""
    assert processor._infer_unit_from_metric_name("request_duration_seconds") == "s"
    assert processor._infer_unit_from_metric_name("latency_ms") == "ms"
    assert processor._infer_unit_from_metric_name("queue_time_us") == "us"
    assert processor._infer_unit_from_metric_name("memory_bytes") == "bytes"
    assert processor._infer_unit_from_metric_name("cpu_percent") == "%"
    assert processor._infer_unit_from_metric_name("requests_per_s") == "/s"
    assert processor._infer_unit_from_metric_name("request_count") == "count"
    assert processor._infer_unit_from_metric_name("unknown_metric") == ""


@pytest.mark.asyncio
async def test_summarize_with_no_data(processor):
    """Test summarize with no data returns empty list."""
    results = await processor.summarize()
    assert results == []


@pytest.mark.asyncio
async def test_percentile_calculations(processor, sample_gauge_records):
    """Test that percentile calculations are correct."""
    for record in sample_gauge_records:
        await processor.process_server_metrics_record(record)

    results = await processor.summarize()

    cpu_metrics = [r for r in results if "cpu_usage_percent" in r.tag]
    assert len(cpu_metrics) == 1

    metric = cpu_metrics[0]
    # Verify percentiles are within expected range
    assert metric.p50 >= metric.min
    assert metric.p50 <= metric.max
    assert metric.p95 >= metric.p50
    assert metric.p95 <= metric.max
    assert metric.p99 >= metric.p95
    assert metric.p99 <= metric.max


@pytest.mark.asyncio
async def test_metric_with_empty_labels(processor):
    """Test aggregation of metrics with no labels."""
    metrics = {
        "simple_counter": MetricFamily(
            type="counter",
            help="Simple counter",
            samples=[MetricSample(labels={}, value=42.0)],
        ),
    }

    for i in range(5):
        record = ServerMetricsRecord(
            timestamp_ns=1000000000 + i * 100000000,
            endpoint_url="http://localhost:8081/metrics",
            metrics=metrics,
        )
        await processor.process_server_metrics_record(record)

    results = await processor.summarize()

    # Should find the metric with empty labels
    simple_metrics = [r for r in results if "simple_counter" in r.tag]
    assert len(simple_metrics) == 1


@pytest.mark.asyncio
async def test_hierarchical_tag_format(processor, sample_counter_records):
    """Test that tags follow hierarchical format."""
    for record in sample_counter_records:
        await processor.process_server_metrics_record(record)

    results = await processor.summarize()

    # All tags should start with server_metrics prefix
    for result in results:
        assert result.tag.startswith("server_metrics.")

        # Tag should contain endpoint display name
        assert "localhost:8081" in result.tag or "8081" in result.tag


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
