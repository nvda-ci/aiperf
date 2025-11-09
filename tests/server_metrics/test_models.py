# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic tests for ServerMetrics system."""

import pytest

from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    ServerMetricsHierarchy,
    ServerMetricsRecord,
    SummaryData,
)


def test_histogram_data_creation():
    """Test HistogramData model creation."""
    histogram = HistogramData(
        buckets={"0.1": 10.0, "0.5": 50.0, "+Inf": 100.0},
        sum=25.5,
        count=100.0,
    )
    assert histogram.buckets["0.1"] == 10.0
    assert histogram.sum == 25.5
    assert histogram.count == 100.0


def test_summary_data_creation():
    """Test SummaryData model creation."""
    summary = SummaryData(
        quantiles={"0.5": 0.1, "0.9": 0.5, "0.99": 1.0},
        sum=100.0,
        count=1000.0,
    )
    assert summary.quantiles["0.5"] == 0.1
    assert summary.sum == 100.0
    assert summary.count == 1000.0


def test_metric_sample_counter():
    """Test MetricSample with counter type."""
    sample = MetricSample(
        labels={"method": "GET", "status": "200"},
        value=1547.0,
    )
    assert sample.labels["method"] == "GET"
    assert sample.value == 1547.0
    assert sample.histogram is None
    assert sample.summary is None


def test_metric_sample_histogram():
    """Test MetricSample with histogram type."""
    histogram = HistogramData(
        buckets={"0.1": 10.0, "0.5": 50.0, "+Inf": 100.0},
        sum=25.5,
        count=100.0,
    )
    sample = MetricSample(
        labels={"method": "GET"},
        histogram=histogram,
    )
    assert sample.labels["method"] == "GET"
    assert sample.value is None
    assert sample.histogram is not None
    assert sample.histogram.sum == 25.5


def test_metric_family_creation():
    """Test MetricFamily model creation."""
    samples = [
        MetricSample(labels={"method": "GET"}, value=100.0),
        MetricSample(labels={"method": "POST"}, value=50.0),
    ]
    family = MetricFamily(
        type="counter",
        help="Total HTTP requests",
        samples=samples,
    )
    assert family.type == "counter"
    assert family.help == "Total HTTP requests"
    assert len(family.samples) == 2


def test_server_metrics_record_creation():
    """Test ServerMetricsRecord model creation."""
    samples = [
        MetricSample(labels={}, value=100.0),
    ]
    family = MetricFamily(
        type="counter",
        help="Total HTTP requests",
        samples=samples,
    )
    record = ServerMetricsRecord(
        timestamp_ns=1699564800123456789,
        endpoint_url="http://localhost:8081/metrics",
        metrics={"http_requests_total": family},
    )
    assert record.timestamp_ns == 1699564800123456789
    assert record.endpoint_url == "http://localhost:8081/metrics"
    assert "http_requests_total" in record.metrics


def test_server_metrics_hierarchy():
    """Test ServerMetricsHierarchy accumulation."""
    hierarchy = ServerMetricsHierarchy()

    # Create first record
    samples = [MetricSample(labels={}, value=100.0)]
    family = MetricFamily(type="counter", help="Test", samples=samples)
    record1 = ServerMetricsRecord(
        timestamp_ns=1000000000,
        endpoint_url="http://localhost:8081/metrics",
        metrics={"test_metric": family},
    )

    # Create second record
    samples = [MetricSample(labels={}, value=200.0)]
    family = MetricFamily(type="counter", help="Test", samples=samples)
    record2 = ServerMetricsRecord(
        timestamp_ns=2000000000,
        endpoint_url="http://localhost:8081/metrics",
        metrics={"test_metric": family},
    )

    # Add records to hierarchy
    hierarchy.add_record(record1)
    hierarchy.add_record(record2)

    # Verify structure
    assert "http://localhost:8081/metrics" in hierarchy.endpoints
    endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
    assert len(endpoint_data.time_series.snapshots) == 2
    assert endpoint_data.metadata.endpoint_url == "http://localhost:8081/metrics"


def test_multiple_endpoints():
    """Test ServerMetricsHierarchy with multiple endpoints."""
    hierarchy = ServerMetricsHierarchy()

    # Add records from two different endpoints
    for i, endpoint in enumerate(
        ["http://localhost:8081/metrics", "http://localhost:9090/metrics"]
    ):
        samples = [MetricSample(labels={}, value=float(i * 100))]
        family = MetricFamily(type="counter", help="Test", samples=samples)
        record = ServerMetricsRecord(
            timestamp_ns=i * 1000000000,
            endpoint_url=endpoint,
            metrics={"test_metric": family},
        )
        hierarchy.add_record(record)

    # Verify both endpoints are present
    assert len(hierarchy.endpoints) == 2
    assert "http://localhost:8081/metrics" in hierarchy.endpoints
    assert "http://localhost:9090/metrics" in hierarchy.endpoints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
