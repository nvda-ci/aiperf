# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for server_metrics tests."""

import pytest

from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    SummaryData,
)


@pytest.fixture
def sample_counter_metrics():
    """Sample Prometheus counter metrics."""
    return """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200"} 1547.0
http_requests_total{method="POST",status="200"} 892.0
"""


@pytest.fixture
def sample_gauge_metrics():
    """Sample Prometheus gauge metrics."""
    return """# HELP memory_usage_bytes Current memory usage
# TYPE memory_usage_bytes gauge
memory_usage_bytes{type="heap"} 1073741824
"""


@pytest.fixture
def sample_histogram_metrics():
    """Sample Prometheus histogram metrics."""
    return """# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",le="0.1"} 100
http_request_duration_seconds_bucket{method="GET",le="0.5"} 450
http_request_duration_seconds_bucket{method="GET",le="+Inf"} 500
http_request_duration_seconds_sum{method="GET"} 125.5
http_request_duration_seconds_count{method="GET"} 500
"""


@pytest.fixture
def sample_summary_metrics():
    """Sample Prometheus summary metrics."""
    return """# HELP rpc_duration_seconds RPC duration summary
# TYPE rpc_duration_seconds summary
rpc_duration_seconds{service="auth",quantile="0.5"} 0.1
rpc_duration_seconds{service="auth",quantile="0.9"} 0.5
rpc_duration_seconds{service="auth",quantile="0.99"} 1.0
rpc_duration_seconds_sum{service="auth"} 100.0
rpc_duration_seconds_count{service="auth"} 1000
"""


@pytest.fixture
def simple_metric_sample():
    """Create a simple metric sample."""
    return MetricSample(labels={"method": "GET"}, value=100.0)


@pytest.fixture
def histogram_metric_sample():
    """Create a histogram metric sample."""
    histogram = HistogramData(
        buckets={"0.1": 10.0, "0.5": 50.0, "+Inf": 100.0},
        sum=25.5,
        count=100.0,
    )
    return MetricSample(labels={"method": "GET"}, histogram=histogram)


@pytest.fixture
def summary_metric_sample():
    """Create a summary metric sample."""
    summary = SummaryData(
        quantiles={"0.5": 0.1, "0.9": 0.5, "0.99": 1.0},
        sum=100.0,
        count=1000.0,
    )
    return MetricSample(labels={"service": "auth"}, summary=summary)


@pytest.fixture
def sample_metric_family(simple_metric_sample):
    """Create a sample metric family."""
    return MetricFamily(
        type="counter",
        help="Test metric",
        samples=[simple_metric_sample],
    )


@pytest.fixture
def sample_server_metrics_record(sample_metric_family):
    """Create a sample server metrics record."""
    return ServerMetricsRecord(
        timestamp_ns=1000000000,
        endpoint_url="http://localhost:8081/metrics",
        metrics={"test_metric": sample_metric_family},
    )


@pytest.fixture
def multiple_endpoints():
    """Return a list of test endpoints."""
    return [
        "http://localhost:8081/metrics",
        "http://localhost:9090/metrics",
        "http://192.168.1.1:8080/metrics",
    ]


@pytest.fixture
def server_metrics_collector():
    """Create a ServerMetricsDataCollector instance for testing."""
    from aiperf.server_metrics.server_metrics_data_collector import (
        ServerMetricsDataCollector,
    )

    return ServerMetricsDataCollector(
        endpoint_url="http://localhost:8081/metrics",
        collection_interval=1.0,
    )


@pytest.fixture
def sample_prometheus_text():
    """Sample Prometheus metrics in text format with multiple types."""
    return """# HELP http_requests_total Total HTTP requests
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
