# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsDataCollector."""

import pytest

from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)

SAMPLE_PROMETHEUS_METRICS = """# HELP http_requests_total Total HTTP requests
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

# HELP process_cpu_seconds_total Total CPU time
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 45.67
"""


@pytest.mark.asyncio
async def test_parse_simple_counter():
    """Test parsing simple counter metrics."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(SAMPLE_PROMETHEUS_METRICS)

    assert len(records) == 1
    record = records[0]
    assert record.endpoint_url == "http://localhost:8081/metrics"
    # Note: prometheus_client parser strips _total suffix from counters
    assert "http_requests" in record.metrics
    assert "process_cpu_seconds" in record.metrics

    # Verify counter metric
    http_requests = record.metrics["http_requests"]
    assert http_requests.type == "counter"
    assert len(http_requests.samples) == 2

    # Verify labels and values
    labels_to_values = {
        tuple(sorted(s.labels.items())): s.value for s in http_requests.samples
    }
    assert labels_to_values[(("method", "GET"), ("status", "200"))] == 1547.0
    assert labels_to_values[(("method", "POST"), ("status", "200"))] == 892.0


@pytest.mark.asyncio
async def test_parse_histogram():
    """Test parsing histogram metrics."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(SAMPLE_PROMETHEUS_METRICS)

    assert len(records) == 1
    record = records[0]
    assert "http_request_duration_seconds" in record.metrics

    # Verify histogram metric
    duration_histogram = record.metrics["http_request_duration_seconds"]
    assert duration_histogram.type == "histogram"
    assert len(duration_histogram.samples) == 1

    # Verify histogram structure
    sample = duration_histogram.samples[0]
    assert sample.labels["method"] == "GET"
    assert sample.histogram is not None
    assert sample.histogram.sum == 125.5
    assert sample.histogram.count == 500.0
    assert sample.histogram.buckets["0.1"] == 100
    assert sample.histogram.buckets["0.5"] == 450
    assert sample.histogram.buckets["+Inf"] == 500


@pytest.mark.asyncio
async def test_empty_metrics():
    """Test parsing empty metrics."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records("")
    assert len(records) == 0

    records = collector._parse_metrics_to_records("   \n  \n  ")
    assert len(records) == 0


@pytest.mark.asyncio
async def test_deduplication():
    """Test that duplicate labels are de-duplicated (last wins)."""
    metrics_with_duplicates = """# HELP test_metric Test metric
# TYPE test_metric counter
test_metric{label="a"} 100
test_metric{label="a"} 200
"""

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(metrics_with_duplicates)
    assert len(records) == 1

    test_metric = records[0].metrics["test_metric"]
    assert len(test_metric.samples) == 1
    assert test_metric.samples[0].value == 200.0  # Last value wins


@pytest.mark.asyncio
async def test_created_metrics_filtered():
    """Test that _created metrics are filtered out and not stored separately."""
    metrics_with_created = """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET"} 1547.0

# HELP http_requests_total_created Total HTTP requests
# TYPE http_requests_total_created gauge
http_requests_total_created{method="GET"} 1.7624927793601315e+09

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",le="0.1"} 100
http_request_duration_seconds_bucket{method="GET",le="+Inf"} 500
http_request_duration_seconds_sum{method="GET"} 125.5
http_request_duration_seconds_count{method="GET"} 500

# HELP http_request_duration_seconds_created HTTP request duration
# TYPE http_request_duration_seconds_created gauge
http_request_duration_seconds_created{method="GET"} 1.7624927793601475e+09

# HELP active_requests Current active requests
# TYPE active_requests gauge
active_requests 42.0
"""

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(metrics_with_created)
    assert len(records) == 1

    record = records[0]
    # Verify _created metrics are not in the parsed metrics
    assert "http_requests_total_created" not in record.metrics
    assert "http_request_duration_seconds_created" not in record.metrics

    # Verify the actual metrics are still there
    assert (
        "http_requests" in record.metrics
    )  # _total suffix stripped by prometheus_client
    assert "http_request_duration_seconds" in record.metrics
    assert "active_requests" in record.metrics

    # Verify the metrics have correct structure
    http_requests = record.metrics["http_requests"]
    assert http_requests.type == "counter"
    assert len(http_requests.samples) == 1
    assert http_requests.samples[0].value == 1547.0

    duration_histogram = record.metrics["http_request_duration_seconds"]
    assert duration_histogram.type == "histogram"
    assert len(duration_histogram.samples) == 1
    assert duration_histogram.samples[0].histogram is not None

    active = record.metrics["active_requests"]
    assert active.type == "gauge"
    assert len(active.samples) == 1
    assert active.samples[0].value == 42.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
