# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic tests for ServerMetrics system."""

import pytest

from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    SummaryData,
)


class TestHistogramData:
    """Tests for HistogramData model."""

    def test_creation(self, histogram_metric_sample):
        """Test HistogramData model creation."""
        histogram = histogram_metric_sample.histogram
        assert histogram.buckets["0.1"] == 10.0
        assert histogram.sum == 25.5
        assert histogram.count == 100.0

    @pytest.mark.parametrize(
        "buckets,expected_sum,expected_count",
        [
            ({"0.1": 5.0, "+Inf": 10.0}, 5.5, 10.0),
            ({"1.0": 100.0, "5.0": 200.0, "+Inf": 250.0}, 500.0, 250.0),
            ({"0.01": 1.0, "0.1": 10.0, "1.0": 100.0, "+Inf": 150.0}, 75.0, 150.0),
        ],
    )
    def test_various_bucket_configurations(self, buckets, expected_sum, expected_count):
        """Test HistogramData with various bucket configurations."""
        histogram = HistogramData(
            buckets=buckets,
            sum=expected_sum,
            count=expected_count,
        )
        assert histogram.sum == expected_sum
        assert histogram.count == expected_count
        assert len(histogram.buckets) == len(buckets)


class TestSummaryData:
    """Tests for SummaryData model."""

    def test_creation(self, summary_metric_sample):
        """Test SummaryData model creation."""
        summary = summary_metric_sample.summary
        assert summary.quantiles["0.5"] == 0.1
        assert summary.sum == 100.0
        assert summary.count == 1000.0

    @pytest.mark.parametrize(
        "quantiles,expected_sum,expected_count",
        [
            ({"0.5": 0.05, "0.99": 0.5}, 50.0, 100.0),
            ({"0.25": 0.1, "0.5": 0.2, "0.75": 0.3, "0.99": 1.0}, 250.0, 500.0),
        ],
    )
    def test_various_quantile_configurations(
        self, quantiles, expected_sum, expected_count
    ):
        """Test SummaryData with various quantile configurations."""
        summary = SummaryData(
            quantiles=quantiles,
            sum=expected_sum,
            count=expected_count,
        )
        assert summary.sum == expected_sum
        assert summary.count == expected_count
        assert len(summary.quantiles) == len(quantiles)


class TestMetricSample:
    """Tests for MetricSample model."""

    def test_counter_sample(self):
        """Test MetricSample with counter type (value only)."""
        sample = MetricSample(
            labels={"method": "GET", "status": "200"},
            value=1547.0,
        )
        assert sample.labels["method"] == "GET"
        assert sample.value == 1547.0
        assert sample.histogram is None
        assert sample.summary is None

    def test_histogram_sample(self, histogram_metric_sample):
        """Test MetricSample with histogram type."""
        assert histogram_metric_sample.labels["method"] == "GET"
        assert histogram_metric_sample.value is None
        assert histogram_metric_sample.histogram is not None
        assert histogram_metric_sample.histogram.sum == 25.5

    def test_summary_sample(self, summary_metric_sample):
        """Test MetricSample with summary type."""
        assert summary_metric_sample.labels["service"] == "auth"
        assert summary_metric_sample.value is None
        assert summary_metric_sample.summary is not None
        assert summary_metric_sample.summary.count == 1000.0

    @pytest.mark.parametrize(
        "labels,value",
        [
            ({"method": "GET"}, 100.0),
            ({"method": "POST"}, 50.0),
            ({"method": "PUT", "status": "201"}, 25.0),
            (None, 42.0),  # No labels case
        ],
    )
    def test_various_label_configurations(self, labels, value):
        """Test MetricSample with various label configurations."""
        sample = MetricSample(labels=labels, value=value)
        assert sample.value == value
        assert sample.labels == labels


class TestMetricFamily:
    """Tests for MetricFamily model."""

    def test_creation(self, sample_metric_family):
        """Test MetricFamily model creation."""
        assert sample_metric_family.type == "counter"
        assert sample_metric_family.help == "Test metric"
        assert len(sample_metric_family.samples) == 1

    @pytest.mark.parametrize(
        "metric_type,help_text,num_samples",
        [
            ("counter", "Total HTTP requests", 2),
            ("gauge", "Current memory usage", 1),
            ("histogram", "Request duration", 3),
            ("summary", "RPC latency", 4),
        ],
    )
    def test_various_types(self, metric_type, help_text, num_samples):
        """Test MetricFamily with various metric types."""
        samples = [
            MetricSample(labels={"id": str(i)}, value=float(i * 10))
            for i in range(num_samples)
        ]
        family = MetricFamily(
            type=metric_type,
            help=help_text,
            samples=samples,
        )
        assert family.type == metric_type
        assert family.help == help_text
        assert len(family.samples) == num_samples


class TestServerMetricsRecord:
    """Tests for ServerMetricsRecord model."""

    def test_creation(self, sample_server_metrics_record):
        """Test ServerMetricsRecord model creation."""
        assert sample_server_metrics_record.timestamp_ns == 1000000000
        assert (
            sample_server_metrics_record.endpoint_url == "http://localhost:8081/metrics"
        )
        assert "test_metric" in sample_server_metrics_record.metrics

    @pytest.mark.parametrize(
        "endpoint_url",
        [
            "http://localhost:8081/metrics",
            "http://localhost:9090/metrics",
            "http://192.168.1.1:8080/metrics",
            "http://triton-server:8002/metrics",
        ],
    )
    def test_various_endpoints(self, endpoint_url, simple_metric_sample):
        """Test ServerMetricsRecord with various endpoint URLs."""
        family = MetricFamily(
            type="counter",
            help="Test metric",
            samples=[simple_metric_sample],
        )
        record = ServerMetricsRecord(
            timestamp_ns=1699564800123456789,
            endpoint_url=endpoint_url,
            metrics={"http_requests_total": family},
        )
        assert record.endpoint_url == endpoint_url
        assert "http_requests_total" in record.metrics

    def test_multiple_metrics(self, simple_metric_sample):
        """Test ServerMetricsRecord with multiple metric families."""
        metrics = {
            f"metric_{i}": MetricFamily(
                type="counter",
                help=f"Metric {i}",
                samples=[simple_metric_sample],
            )
            for i in range(5)
        }
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8081/metrics",
            metrics=metrics,
        )
        assert len(record.metrics) == 5
        for i in range(5):
            assert f"metric_{i}" in record.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
