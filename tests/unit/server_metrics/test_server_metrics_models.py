# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    MetricSchema,
    ServerMetricsEndpointData,
    ServerMetricsHierarchy,
    ServerMetricsMetadata,
    ServerMetricsRecord,
    ServerMetricsResults,
    SlimMetricSample,
    SummaryData,
)


class TestMetricSampleConversion:
    """Test MetricSample to SlimMetricSample conversion."""

    def test_counter_to_slim(self):
        """Test converting simple counter metric to slim format."""
        sample = MetricSample(
            labels={"model": "test-model", "status": "success"},
            value=100.0,
        )
        slim = sample.to_slim()

        assert slim.labels == {"model": "test-model", "status": "success"}
        assert slim.value == 100.0
        assert slim.histogram is None
        assert slim.summary is None

    def test_gauge_to_slim(self):
        """Test converting gauge metric to slim format."""
        sample = MetricSample(labels=None, value=0.75)
        slim = sample.to_slim()

        assert slim.labels is None
        assert slim.value == 0.75
        assert slim.histogram is None

    def test_histogram_to_slim(self):
        """Test converting histogram metric to slim dict format."""
        sample = MetricSample(
            labels={"model": "test"},
            histogram=HistogramData(
                buckets={"0.01": 5.0, "0.1": 15.0, "1.0": 50.0, "+Inf": 100.0},
                sum=125.5,
                count=100.0,
            ),
        )
        slim = sample.to_slim()

        assert slim.labels == {"model": "test"}
        assert slim.value is None
        assert slim.histogram == {"0.01": 5.0, "0.1": 15.0, "1.0": 50.0, "+Inf": 100.0}
        assert slim.sum == 125.5
        assert slim.count == 100.0

    def test_summary_to_slim(self):
        """Test converting summary metric to slim dict format."""
        sample = MetricSample(
            labels={"endpoint": "/v1/chat"},
            summary=SummaryData(
                quantiles={"0.5": 0.1, "0.9": 0.5, "0.99": 1.0},
                sum=50.0,
                count=100.0,
            ),
        )
        slim = sample.to_slim()

        assert slim.labels == {"endpoint": "/v1/chat"}
        assert slim.summary == {"0.5": 0.1, "0.9": 0.5, "0.99": 1.0}
        assert slim.sum == 50.0
        assert slim.count == 100.0


class TestSlimMetricSampleValidation:
    """Test SlimMetricSample mutual exclusivity validation."""

    def test_value_only_is_valid(self):
        """Test that value-only sample is valid."""
        sample = SlimMetricSample(value=42.0)
        assert sample.value == 42.0

    def test_histogram_only_is_valid(self):
        """Test that histogram-only sample is valid."""
        sample = SlimMetricSample(
            histogram={"0.1": 10, "1.0": 50},
            sum=100.0,
            count=50,
        )
        assert sample.histogram == {"0.1": 10, "1.0": 50}

    def test_summary_only_is_valid(self):
        """Test that summary-only sample is valid."""
        sample = SlimMetricSample(
            summary={"0.5": 0.1, "0.99": 1.0},
            sum=50.0,
            count=100,
        )
        assert sample.summary == {"0.5": 0.1, "0.99": 1.0}

    def test_none_fields_is_valid(self):
        """Test that sample with no value/histogram/summary is valid (labels only)."""
        sample = SlimMetricSample(labels={"key": "value"})
        assert sample.labels == {"key": "value"}
        assert sample.value is None
        assert sample.histogram is None
        assert sample.summary is None

    def test_value_and_histogram_raises(self):
        """Test that setting both value and histogram raises ValidationError."""
        import pytest

        with pytest.raises(ValueError, match="Only one of"):
            SlimMetricSample(value=42.0, histogram={"0.1": 10})

    def test_value_and_summary_raises(self):
        """Test that setting both value and summary raises ValidationError."""
        import pytest

        with pytest.raises(ValueError, match="Only one of"):
            SlimMetricSample(value=42.0, summary={"0.5": 0.1})

    def test_histogram_and_summary_raises(self):
        """Test that setting both histogram and summary raises ValidationError."""
        import pytest

        with pytest.raises(ValueError, match="Only one of"):
            SlimMetricSample(histogram={"0.1": 10}, summary={"0.5": 0.1})

    def test_all_three_raises(self):
        """Test that setting all three raises ValidationError."""
        import pytest

        with pytest.raises(ValueError, match="Only one of"):
            SlimMetricSample(value=42.0, histogram={"0.1": 10}, summary={"0.5": 0.1})


class TestServerMetricsRecordConversion:
    """Test ServerMetricsRecord to slim format conversion."""

    def test_full_record_to_slim(
        self,
        sample_counter_metric: MetricFamily,
        sample_histogram_metric: MetricFamily,
    ):
        """Test converting complete record with multiple metric types."""
        record = ServerMetricsRecord(
            endpoint_url="http://node1:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "requests_total": sample_counter_metric,
                "ttft": sample_histogram_metric,
            },
        )

        slim = record.to_slim()

        assert slim.endpoint_url == "http://node1:8081/metrics"
        assert slim.timestamp_ns == 1_000_000_000
        assert slim.endpoint_latency_ns == 5_000_000
        assert len(slim.metrics) == 2
        assert "requests_total" in slim.metrics
        assert "ttft" in slim.metrics

        assert isinstance(slim.metrics["requests_total"][0], SlimMetricSample)
        assert slim.metrics["requests_total"][0].value == 150.0

        assert isinstance(slim.metrics["ttft"][0], SlimMetricSample)
        assert slim.metrics["ttft"][0].histogram is not None

    def test_slim_record_preserves_endpoint_url(self, sample_server_metrics_record):
        """Test that endpoint_url is preserved in slim format."""
        slim = sample_server_metrics_record.to_slim()
        assert slim.endpoint_url == sample_server_metrics_record.endpoint_url


class TestMetricSchema:
    """Test MetricSchema model for metadata."""

    def test_counter_schema(self):
        """Test schema for counter metric."""
        schema = MetricSchema(
            type=PrometheusMetricType.COUNTER,
            description="Total number of requests",
        )

        assert schema.type == PrometheusMetricType.COUNTER
        assert schema.description == "Total number of requests"

    def test_histogram_schema_with_buckets(self):
        """Test schema for histogram."""
        schema = MetricSchema(
            type=PrometheusMetricType.HISTOGRAM,
            description="Request duration histogram",
        )

        assert schema.type == PrometheusMetricType.HISTOGRAM
        assert schema.description == "Request duration histogram"

    def test_summary_schema_with_quantiles(self):
        """Test schema for summary."""
        schema = MetricSchema(
            type=PrometheusMetricType.SUMMARY,
            description="Request latency quantiles",
        )

        assert schema.type == PrometheusMetricType.SUMMARY
        assert schema.description == "Request latency quantiles"


class TestHistogramAndSummaryData:
    """Test HistogramData and SummaryData models."""

    def test_histogram_data_structure(self):
        """Test HistogramData contains buckets, sum, and count."""
        hist = HistogramData(
            buckets={"0.01": 10.0, "0.1": 25.0, "+Inf": 50.0},
            sum=5.5,
            count=50.0,
        )

        assert len(hist.buckets) == 3
        assert hist.buckets["0.01"] == 10.0
        assert hist.sum == 5.5
        assert hist.count == 50.0

    def test_summary_data_structure(self):
        """Test SummaryData contains quantiles, sum, and count."""
        summary = SummaryData(
            quantiles={"0.5": 0.1, "0.9": 0.5, "0.99": 1.0},
            sum=50.0,
            count=100.0,
        )

        assert len(summary.quantiles) == 3
        assert summary.quantiles["0.5"] == 0.1
        assert summary.sum == 50.0
        assert summary.count == 100.0

    def test_histogram_optional_fields(self):
        """Test histogram with optional fields as None."""
        hist = HistogramData(
            buckets={"0.01": 10.0},
            sum=None,
            count=None,
        )

        assert hist.buckets == {"0.01": 10.0}
        assert hist.sum is None
        assert hist.count is None


class TestServerMetricsEndpointData:
    """Test ServerMetricsEndpointData model with grouped approach."""

    def test_add_record_grouped(self, sample_server_metrics_record):
        """Test adding ServerMetricsRecord stores data in NumPy columnar storage."""
        metadata = ServerMetricsMetadata(
            endpoint_url="http://localhost:8081/metrics",
            metric_schemas={},
            info_metrics={},
        )
        endpoint_data = ServerMetricsEndpointData(
            endpoint_url="http://localhost:8081/metrics",
            metadata=metadata,
        )

        endpoint_data.add_record(sample_server_metrics_record)

        # Verify data stored in columnar storage
        assert len(endpoint_data.time_series) == 1
        # Should have extracted gauge metrics (new format uses | separator)
        assert (
            "vllm:gpu_cache_usage_perc|model_name=meta-llama/Llama-3.1-8B-Instruct"
            in endpoint_data.time_series.gauges
        )
        # Should have extracted counter metrics
        assert (
            "vllm:request_success_total|model_name=meta-llama/Llama-3.1-8B-Instruct"
            in endpoint_data.time_series.counters
        )

    def test_add_multiple_records(self, sample_gauge_metric, sample_counter_metric):
        """Test adding multiple records accumulates data in columnar storage."""
        metadata = ServerMetricsMetadata(
            endpoint_url="http://localhost:8081/metrics",
            metric_schemas={},
            info_metrics={},
        )
        endpoint_data = ServerMetricsEndpointData(
            endpoint_url="http://localhost:8081/metrics",
            metadata=metadata,
        )

        for i in range(3):
            record = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "gauge_metric": sample_gauge_metric,
                    "counter_metric": sample_counter_metric,
                },
            )
            endpoint_data.add_record(record)

        # Verify 3 snapshots stored in columnar storage
        assert len(endpoint_data.time_series) == 3

    def test_get_available_gauge_metrics(self, sample_gauge_metric):
        """Test getting available gauge metric names."""
        metadata = ServerMetricsMetadata(
            endpoint_url="http://localhost:8081/metrics",
            metric_schemas={},
            info_metrics={},
        )
        endpoint_data = ServerMetricsEndpointData(
            endpoint_url="http://localhost:8081/metrics",
            metadata=metadata,
        )

        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={"gauge_metric": sample_gauge_metric},
        )
        endpoint_data.add_record(record)

        available = endpoint_data.get_available_gauge_metrics()
        assert len(available) == 1
        assert any("gauge_metric" in m for m in available)

    def test_get_available_counter_metrics(self, sample_counter_metric):
        """Test getting available counter metric names."""
        metadata = ServerMetricsMetadata(
            endpoint_url="http://localhost:8081/metrics",
            metric_schemas={},
            info_metrics={},
        )
        endpoint_data = ServerMetricsEndpointData(
            endpoint_url="http://localhost:8081/metrics",
            metadata=metadata,
        )

        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={"counter_metric": sample_counter_metric},
        )
        endpoint_data.add_record(record)

        available = endpoint_data.get_available_counter_metrics()
        assert len(available) == 1
        assert any("counter_metric" in m for m in available)


class TestServerMetricsHierarchy:
    """Test ServerMetricsHierarchy storage model."""

    def test_add_record_creates_endpoint(self, sample_server_metrics_record):
        """Test adding a record creates endpoint entry."""
        hierarchy = ServerMetricsHierarchy()

        hierarchy.add_record(sample_server_metrics_record)

        assert "http://localhost:8081/metrics" in hierarchy.endpoints
        endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
        assert endpoint_data.endpoint_url == "http://localhost:8081/metrics"

    def test_add_record_multiple_endpoints(self, sample_gauge_metric):
        """Test adding records from multiple endpoints."""
        hierarchy = ServerMetricsHierarchy()

        for i, endpoint in enumerate(
            ["http://node1:8081/metrics", "http://node2:8081/metrics"]
        ):
            record = ServerMetricsRecord(
                endpoint_url=endpoint,
                timestamp_ns=1_000_000_000 + i,
                endpoint_latency_ns=5_000_000,
                metrics={"gauge_metric": sample_gauge_metric},
            )
            hierarchy.add_record(record)

        assert len(hierarchy.endpoints) == 2
        assert "http://node1:8081/metrics" in hierarchy.endpoints
        assert "http://node2:8081/metrics" in hierarchy.endpoints

    def test_add_record_updates_existing_endpoint(self, sample_gauge_metric):
        """Test adding multiple records to same endpoint accumulates data in columnar storage."""
        hierarchy = ServerMetricsHierarchy()

        for i in range(3):
            record = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={"gauge_metric": sample_gauge_metric},
            )
            hierarchy.add_record(record)

        assert len(hierarchy.endpoints) == 1
        endpoint_data = hierarchy.endpoints["http://localhost:8081/metrics"]
        # Verify 3 snapshots stored in columnar storage
        assert len(endpoint_data.time_series) == 3


class TestServerMetricsResults:
    """Test ServerMetricsResults model."""

    def test_results_creation(self):
        """Test creating ServerMetricsResults with all fields."""
        hierarchy = ServerMetricsHierarchy()
        results = ServerMetricsResults(
            server_metrics_data=hierarchy,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            endpoints_configured=["http://node1:8081/metrics"],
            endpoints_successful=["http://node1:8081/metrics"],
            error_summary=[],
        )

        assert results.start_ns == 1_000_000_000
        assert results.end_ns == 2_000_000_000
        assert results.endpoints_configured == ["http://node1:8081/metrics"]
        assert results.endpoints_successful == ["http://node1:8081/metrics"]
        assert len(results.error_summary) == 0

    def test_results_with_errors(self):
        """Test creating ServerMetricsResults with error summary."""
        from aiperf.common.models import ErrorDetails, ErrorDetailsCount

        hierarchy = ServerMetricsHierarchy()
        error = ErrorDetails(message="Connection failed")
        results = ServerMetricsResults(
            server_metrics_data=hierarchy,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            error_summary=[ErrorDetailsCount(error_details=error, count=5)],
        )

        assert len(results.error_summary) == 1
        assert results.error_summary[0].count == 5
