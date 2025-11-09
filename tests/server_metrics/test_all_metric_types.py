# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for all Prometheus metric types."""

import pytest

from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)

# Counter metrics
COUNTER_METRICS = """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200"} 1547.0
http_requests_total{method="POST",status="200"} 892.0
http_requests_total{method="GET",status="404"} 23.0
http_requests_total{method="DELETE",status="200"} 15.0

# HELP process_cpu_seconds_total Total CPU time in seconds
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 45.67
"""

# Gauge metrics
GAUGE_METRICS = """# HELP memory_usage_bytes Current memory usage in bytes
# TYPE memory_usage_bytes gauge
memory_usage_bytes{type="heap"} 1073741824
memory_usage_bytes{type="stack"} 8388608

# HELP temperature_celsius Current temperature
# TYPE temperature_celsius gauge
temperature_celsius{sensor="cpu"} 65.5
temperature_celsius{sensor="gpu"} 72.3

# HELP active_connections Current number of active connections
# TYPE active_connections gauge
active_connections 42
"""

# Histogram metrics
HISTOGRAM_METRICS = """# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",le="0.005"} 10
http_request_duration_seconds_bucket{method="GET",le="0.01"} 25
http_request_duration_seconds_bucket{method="GET",le="0.025"} 50
http_request_duration_seconds_bucket{method="GET",le="0.05"} 80
http_request_duration_seconds_bucket{method="GET",le="0.1"} 100
http_request_duration_seconds_bucket{method="GET",le="0.25"} 120
http_request_duration_seconds_bucket{method="GET",le="0.5"} 450
http_request_duration_seconds_bucket{method="GET",le="1.0"} 490
http_request_duration_seconds_bucket{method="GET",le="+Inf"} 500
http_request_duration_seconds_sum{method="GET"} 125.5
http_request_duration_seconds_count{method="GET"} 500

http_request_duration_seconds_bucket{method="POST",le="0.005"} 5
http_request_duration_seconds_bucket{method="POST",le="0.01"} 10
http_request_duration_seconds_bucket{method="POST",le="0.025"} 20
http_request_duration_seconds_bucket{method="POST",le="0.05"} 30
http_request_duration_seconds_bucket{method="POST",le="0.1"} 35
http_request_duration_seconds_bucket{method="POST",le="0.25"} 38
http_request_duration_seconds_bucket{method="POST",le="0.5"} 40
http_request_duration_seconds_bucket{method="POST",le="1.0"} 42
http_request_duration_seconds_bucket{method="POST",le="+Inf"} 45
http_request_duration_seconds_sum{method="POST"} 8.7
http_request_duration_seconds_count{method="POST"} 45
"""

# Summary metrics
SUMMARY_METRICS = """# HELP rpc_duration_seconds RPC duration summary
# TYPE rpc_duration_seconds summary
rpc_duration_seconds{service="auth",quantile="0.5"} 0.1
rpc_duration_seconds{service="auth",quantile="0.9"} 0.5
rpc_duration_seconds{service="auth",quantile="0.99"} 1.0
rpc_duration_seconds_sum{service="auth"} 100.0
rpc_duration_seconds_count{service="auth"} 1000

rpc_duration_seconds{service="database",quantile="0.5"} 0.05
rpc_duration_seconds{service="database",quantile="0.9"} 0.2
rpc_duration_seconds{service="database",quantile="0.99"} 0.5
rpc_duration_seconds_sum{service="database"} 50.0
rpc_duration_seconds_count{service="database"} 2000
"""

# Untyped metrics
UNTYPED_METRICS = """# HELP custom_metric Some custom metric without type
custom_metric{label="a"} 100
custom_metric{label="b"} 200
"""

# Mixed metrics
MIXED_METRICS = (
    COUNTER_METRICS
    + "\n"
    + GAUGE_METRICS
    + "\n"
    + HISTOGRAM_METRICS
    + "\n"
    + SUMMARY_METRICS
    + "\n"
    + UNTYPED_METRICS
)


@pytest.mark.asyncio
async def test_counter_metrics():
    """Test parsing counter metrics with multiple labels."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(COUNTER_METRICS)

    assert len(records) == 1
    record = records[0]

    # Check http_requests counter
    http_requests = record.metrics["http_requests"]
    assert http_requests.type == "counter"
    assert len(http_requests.samples) == 4

    # Verify all samples
    samples_dict = {
        (s.labels.get("method"), s.labels.get("status")): s.value
        for s in http_requests.samples
    }
    assert samples_dict[("GET", "200")] == 1547.0
    assert samples_dict[("POST", "200")] == 892.0
    assert samples_dict[("GET", "404")] == 23.0
    assert samples_dict[("DELETE", "200")] == 15.0

    # Check process_cpu_seconds counter (no labels)
    process_cpu = record.metrics["process_cpu_seconds"]
    assert process_cpu.type == "counter"
    assert len(process_cpu.samples) == 1
    assert process_cpu.samples[0].value == 45.67
    assert process_cpu.samples[0].labels == {}


@pytest.mark.asyncio
async def test_gauge_metrics():
    """Test parsing gauge metrics."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(GAUGE_METRICS)

    assert len(records) == 1
    record = records[0]

    # Check memory_usage_bytes gauge
    memory = record.metrics["memory_usage_bytes"]
    assert memory.type == "gauge"
    assert len(memory.samples) == 2

    samples_dict = {s.labels.get("type"): s.value for s in memory.samples}
    assert samples_dict["heap"] == 1073741824
    assert samples_dict["stack"] == 8388608

    # Check temperature gauge
    temp = record.metrics["temperature_celsius"]
    assert temp.type == "gauge"
    assert len(temp.samples) == 2

    temp_dict = {s.labels.get("sensor"): s.value for s in temp.samples}
    assert temp_dict["cpu"] == 65.5
    assert temp_dict["gpu"] == 72.3

    # Check active_connections gauge (no labels)
    connections = record.metrics["active_connections"]
    assert connections.type == "gauge"
    assert len(connections.samples) == 1
    assert connections.samples[0].value == 42


@pytest.mark.asyncio
async def test_histogram_metrics():
    """Test parsing histogram metrics with multiple label combinations."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(HISTOGRAM_METRICS)

    assert len(records) == 1
    record = records[0]

    duration = record.metrics["http_request_duration_seconds"]
    assert duration.type == "histogram"
    assert len(duration.samples) == 2  # GET and POST

    # Find GET histogram
    get_histogram = next(s for s in duration.samples if s.labels.get("method") == "GET")
    assert get_histogram.histogram is not None
    assert get_histogram.histogram.count == 500.0
    assert get_histogram.histogram.sum == 125.5
    assert len(get_histogram.histogram.buckets) == 9

    # Verify all buckets for GET
    expected_buckets = {
        "0.005": 10,
        "0.01": 25,
        "0.025": 50,
        "0.05": 80,
        "0.1": 100,
        "0.25": 120,
        "0.5": 450,
        "1.0": 490,
        "+Inf": 500,
    }
    for le, expected_count in expected_buckets.items():
        assert get_histogram.histogram.buckets[le] == expected_count

    # Find POST histogram
    post_histogram = next(
        s for s in duration.samples if s.labels.get("method") == "POST"
    )
    assert post_histogram.histogram is not None
    assert post_histogram.histogram.count == 45.0
    assert post_histogram.histogram.sum == 8.7
    assert len(post_histogram.histogram.buckets) == 9


@pytest.mark.asyncio
async def test_summary_metrics():
    """Test parsing summary metrics with quantiles."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(SUMMARY_METRICS)

    assert len(records) == 1
    record = records[0]

    rpc_duration = record.metrics["rpc_duration_seconds"]
    assert rpc_duration.type == "summary"
    assert len(rpc_duration.samples) == 2  # auth and database

    # Find auth summary
    auth_summary = next(
        s for s in rpc_duration.samples if s.labels.get("service") == "auth"
    )
    assert auth_summary.summary is not None
    assert auth_summary.summary.count == 1000.0
    assert auth_summary.summary.sum == 100.0
    assert len(auth_summary.summary.quantiles) == 3

    # Verify quantiles for auth
    assert auth_summary.summary.quantiles["0.5"] == 0.1
    assert auth_summary.summary.quantiles["0.9"] == 0.5
    assert auth_summary.summary.quantiles["0.99"] == 1.0

    # Find database summary
    db_summary = next(
        s for s in rpc_duration.samples if s.labels.get("service") == "database"
    )
    assert db_summary.summary is not None
    assert db_summary.summary.count == 2000.0
    assert db_summary.summary.sum == 50.0

    # Verify quantiles for database
    assert db_summary.summary.quantiles["0.5"] == 0.05
    assert db_summary.summary.quantiles["0.9"] == 0.2
    assert db_summary.summary.quantiles["0.99"] == 0.5


@pytest.mark.asyncio
async def test_untyped_metrics():
    """Test parsing untyped metrics."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(UNTYPED_METRICS)

    assert len(records) == 1
    record = records[0]

    custom = record.metrics["custom_metric"]
    assert custom.type == "unknown"  # Prometheus parser uses "unknown" not "untyped"
    assert len(custom.samples) == 2

    samples_dict = {s.labels.get("label"): s.value for s in custom.samples}
    assert samples_dict["a"] == 100
    assert samples_dict["b"] == 200


@pytest.mark.asyncio
async def test_mixed_metrics():
    """Test parsing multiple metric types in one response."""
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(MIXED_METRICS)

    assert len(records) == 1
    record = records[0]

    # Verify we have all metric types
    assert "http_requests" in record.metrics
    assert "memory_usage_bytes" in record.metrics
    assert "http_request_duration_seconds" in record.metrics
    assert "rpc_duration_seconds" in record.metrics
    assert "custom_metric" in record.metrics

    # Verify counts
    assert len(record.metrics) >= 8  # At least 8 different metrics

    # Verify types
    assert record.metrics["http_requests"].type == "counter"
    assert record.metrics["memory_usage_bytes"].type == "gauge"
    assert record.metrics["http_request_duration_seconds"].type == "histogram"
    assert record.metrics["rpc_duration_seconds"].type == "summary"
    assert (
        record.metrics["custom_metric"].type == "unknown"
    )  # Prometheus parser uses "unknown"


@pytest.mark.asyncio
async def test_histogram_with_missing_data():
    """Test histogram with missing sum or count."""
    incomplete_histogram = """# HELP test_histogram Test
# TYPE test_histogram histogram
test_histogram_bucket{le="0.1"} 10
test_histogram_bucket{le="+Inf"} 20
"""

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(incomplete_histogram)

    assert len(records) == 1
    record = records[0]

    # Should not create histogram sample without sum and count
    test_histogram = record.metrics["test_histogram"]
    assert test_histogram.type == "histogram"
    assert len(test_histogram.samples) == 0  # No complete histogram


@pytest.mark.asyncio
async def test_summary_with_missing_data():
    """Test summary with missing sum or count."""
    incomplete_summary = """# HELP test_summary Test
# TYPE test_summary summary
test_summary{quantile="0.5"} 0.1
test_summary{quantile="0.9"} 0.5
"""

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(incomplete_summary)

    assert len(records) == 1
    record = records[0]

    # Should not create summary sample without sum and count
    test_summary = record.metrics["test_summary"]
    assert test_summary.type == "summary"
    assert len(test_summary.samples) == 0  # No complete summary


@pytest.mark.asyncio
async def test_metrics_with_special_characters_in_labels():
    """Test metrics with special characters in label values."""
    special_labels = """# HELP http_requests Total requests
# TYPE http_requests counter
http_requests{path="/api/v1/users",method="GET"} 100
http_requests{path="/api/v1/users/123",method="GET"} 50
http_requests{path="/api/v2/items?filter=active",method="GET"} 25
"""

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(special_labels)

    assert len(records) == 1
    record = records[0]

    http_requests = record.metrics["http_requests"]
    assert len(http_requests.samples) == 3

    # Verify label values preserved correctly
    paths = {s.labels.get("path") for s in http_requests.samples}
    assert "/api/v1/users" in paths
    assert "/api/v1/users/123" in paths
    assert "/api/v2/items?filter=active" in paths


@pytest.mark.asyncio
async def test_metrics_with_scientific_notation():
    """Test metrics with scientific notation values."""
    scientific = """# HELP large_values Large values
# TYPE large_values counter
large_values{type="a"} 1.23e10
large_values{type="b"} 4.56e-5
large_values{type="c"} 7.89e+3
"""

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(scientific)

    assert len(records) == 1
    record = records[0]

    large_values = record.metrics["large_values"]
    assert len(large_values.samples) == 3

    values_dict = {s.labels.get("type"): s.value for s in large_values.samples}
    assert values_dict["a"] == 1.23e10
    assert values_dict["b"] == 4.56e-5
    assert values_dict["c"] == 7.89e3


@pytest.mark.asyncio
async def test_metrics_with_negative_values():
    """Test metrics with negative values (valid for gauges)."""
    negative = """# HELP temperature Temperature
# TYPE temperature gauge
temperature{location="outside"} -5.5
temperature{location="inside"} 20.0
"""

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )

    records = collector._parse_metrics_to_records(negative)

    assert len(records) == 1
    record = records[0]

    temperature = record.metrics["temperature"]
    assert len(temperature.samples) == 2

    values_dict = {s.labels.get("location"): s.value for s in temperature.samples}
    assert values_dict["outside"] == -5.5
    assert values_dict["inside"] == 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
