# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for server metrics slim format and reconstruction.

These tests verify the functional behavior of slim format conversion,
schema extraction, and full record reconstruction without testing
implementation details.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.config import UserConfig
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    MetricSchema,
    ServerMetricsMetadata,
    ServerMetricsRecord,
    SummaryData,
)
from aiperf.post_processors.server_metrics_export_results_processor import (
    ServerMetricsExportResultsProcessor,
)


@pytest.fixture
def user_config(tmp_path):
    """Create mock user config."""
    config = MagicMock(spec=UserConfig)
    config.output = MagicMock()
    config.output.server_metrics_export_jsonl_file = tmp_path / "export.jsonl"
    return config


class TestHistogramSlimFormat:
    """Test histogram metrics are correctly converted to slim format and preserved."""

    @pytest.mark.asyncio
    async def test_histogram_with_multiple_buckets_preserves_counts(self, user_config):
        """Verify histogram bucket counts are preserved in slim format."""
        # Create histogram with multiple buckets
        histogram = HistogramData(
            buckets={
                "0.1": 10.0,
                "0.5": 50.0,
                "1.0": 100.0,
                "5.0": 200.0,
                "+Inf": 250.0,
            },
            sum=450.5,
            count=250.0,
        )
        sample = MetricSample(labels={"path": "/api"}, value=None, histogram=histogram)
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM,
            help="Request duration",
            samples=[sample],
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"http_request_duration_seconds": family},
        )

        # Convert to slim and export
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.jsonl"
            processor = ServerMetricsExportResultsProcessor(user_config=user_config)
            processor.output_file = output_file

            await processor.initialize()
            await processor.process_server_metrics_record(record)
            await processor.stop()

            # Verify slim format has counts array (not bucket dict)
            data = orjson.loads(output_file.read_text())
            metric_samples = data["metrics"]["http_request_duration_seconds"]
            assert len(metric_samples) == 1

            # Verify histogram is in slim format (direct array, not nested in dict)
            sample = metric_samples[0]
            assert "histogram" in sample
            assert isinstance(sample["histogram"], list)
            assert len(sample["histogram"]) == 5
            # Counts should be in sorted order by bucket label
            assert sample["histogram"] == [10.0, 50.0, 100.0, 200.0, 250.0]
            assert sample["sum"] == 450.5
            assert sample["count"] == 250.0
            # Verify buckets dict is not present (slim format)
            assert "buckets" not in sample

    @pytest.mark.asyncio
    async def test_histogram_with_single_bucket(self, user_config):
        """Verify single-bucket histograms work correctly."""
        histogram = HistogramData(buckets={"+Inf": 100.0}, sum=50.0, count=100.0)
        sample = MetricSample(labels=None, histogram=histogram)
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM, help="Test", samples=[sample]
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"test_metric": family},
        )

        slim = record.to_slim()
        assert len(slim.metrics["test_metric"][0].histogram) == 1
        assert slim.metrics["test_metric"][0].histogram[0] == 100.0

    @pytest.mark.asyncio
    async def test_histogram_order_preserved_across_conversion(self, user_config):
        """Verify bucket order is preserved when converting to/from slim format."""
        # Create histogram with unsorted bucket labels
        histogram = HistogramData(
            buckets={
                "5.0": 200.0,
                "0.1": 10.0,
                "+Inf": 250.0,
                "1.0": 100.0,
                "0.5": 50.0,
            },
            sum=450.5,
            count=250.0,
        )
        sample = MetricSample(labels=None, histogram=histogram)
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM, help="Test", samples=[sample]
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"test_metric": family},
        )

        # Convert to slim - should sort buckets
        slim = record.to_slim()
        counts = slim.metrics["test_metric"][0].histogram

        # Verify counts are in sorted order by bucket label (numeric sort)
        assert counts == [10.0, 50.0, 100.0, 200.0, 250.0]


class TestSummarySlimFormat:
    """Test summary metrics are correctly converted to slim format and preserved."""

    @pytest.mark.asyncio
    async def test_summary_with_multiple_quantiles_preserves_values(self, user_config):
        """Verify summary quantile values are preserved in slim format."""
        summary = SummaryData(
            quantiles={"0.5": 0.05, "0.9": 0.09, "0.95": 0.095, "0.99": 0.099},
            sum=10.5,
            count=100.0,
        )
        sample = MetricSample(labels={"handler": "/api"}, value=None, summary=summary)
        family = MetricFamily(
            type=PrometheusMetricType.SUMMARY,
            help="Request duration summary",
            samples=[sample],
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"http_request_duration_summary": family},
        )

        # Convert to slim and export
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.jsonl"
            processor = ServerMetricsExportResultsProcessor(user_config=user_config)
            processor.output_file = output_file

            await processor.initialize()
            await processor.process_server_metrics_record(record)
            await processor.stop()

            # Verify slim format has values array (not quantiles dict)
            data = orjson.loads(output_file.read_text())
            metric_samples = data["metrics"]["http_request_duration_summary"]
            assert len(metric_samples) == 1

            # Verify summary is in slim format (direct array, not nested in dict)
            sample = metric_samples[0]
            assert "summary" in sample
            assert isinstance(sample["summary"], list)
            assert len(sample["summary"]) == 4
            # Values should be in sorted order by quantile label
            assert sample["summary"] == [0.05, 0.09, 0.095, 0.099]
            assert sample["sum"] == 10.5
            assert sample["count"] == 100.0
            # Verify quantiles dict is not present (slim format)
            assert "quantiles" not in sample

    @pytest.mark.asyncio
    async def test_summary_order_preserved(self, user_config):
        """Verify quantile order is preserved when converting to slim format."""
        # Create summary with unsorted quantiles
        summary = SummaryData(
            quantiles={"0.99": 0.099, "0.5": 0.05, "0.95": 0.095, "0.9": 0.09},
            sum=10.5,
            count=100.0,
        )
        sample = MetricSample(labels=None, summary=summary)
        family = MetricFamily(
            type=PrometheusMetricType.SUMMARY, help="Test", samples=[sample]
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"test_metric": family},
        )

        # Convert to slim - should sort quantiles
        slim = record.to_slim()
        values = slim.metrics["test_metric"][0].summary

        # Verify values are in sorted order by quantile label
        assert values == [0.05, 0.09, 0.095, 0.099]


class TestSchemaExtraction:
    """Test metric schemas are correctly extracted with bucket/quantile labels."""

    def test_schema_includes_histogram_bucket_labels(self):
        """Verify schema extraction includes histogram bucket labels."""
        histogram = HistogramData(
            buckets={"0.1": 10.0, "0.5": 50.0, "1.0": 100.0, "+Inf": 250.0},
            sum=450.5,
            count=250.0,
        )
        sample = MetricSample(labels=None, histogram=histogram)
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM,
            help="Request duration",
            samples=[sample],
        )

        # Manually extract schema (simulating what ServerMetricsManager does)
        schema = MetricSchema(
            type=family.type,
            help=family.help,
            bucket_labels=sorted(
                family.samples[0].histogram.buckets.keys(), key=lambda x: float(x)
            ),
        )

        # Verify schema has bucket labels in correct order
        assert schema.bucket_labels == ["0.1", "0.5", "1.0", "+Inf"]
        assert schema.type == PrometheusMetricType.HISTOGRAM
        assert schema.help == "Request duration"

    def test_schema_includes_summary_quantile_labels(self):
        """Verify schema extraction includes summary quantile labels."""
        summary = SummaryData(
            quantiles={"0.5": 0.05, "0.9": 0.09, "0.99": 0.099}, sum=10.5, count=100.0
        )
        sample = MetricSample(labels=None, summary=summary)
        family = MetricFamily(
            type=PrometheusMetricType.SUMMARY, help="Request summary", samples=[sample]
        )

        # Manually extract schema
        schema = MetricSchema(
            type=family.type,
            help=family.help,
            quantile_labels=sorted(
                family.samples[0].summary.quantiles.keys(), key=lambda x: float(x)
            ),
        )

        # Verify schema has quantile labels in correct order
        assert schema.quantile_labels == ["0.5", "0.9", "0.99"]
        assert schema.type == PrometheusMetricType.SUMMARY
        assert schema.help == "Request summary"

    def test_schema_for_counter_has_no_bucket_labels(self):
        """Verify counter/gauge schemas don't have bucket/quantile labels."""
        sample = MetricSample(labels=None, value=100.0)
        family = MetricFamily(
            type=PrometheusMetricType.COUNTER, help="Total requests", samples=[sample]
        )

        schema = MetricSchema(
            type=family.type,
            help=family.help,
            bucket_labels=None,
            quantile_labels=None,
        )

        assert schema.bucket_labels is None
        assert schema.quantile_labels is None
        assert schema.type == PrometheusMetricType.COUNTER


class TestMixedMetricTypes:
    """Test records with multiple metric types (counter, gauge, histogram, summary)."""

    @pytest.mark.asyncio
    async def test_record_with_all_metric_types_exports_correctly(self, user_config):
        """Verify a record with counters, gauges, histograms, and summaries exports correctly."""
        # Counter
        counter_sample = MetricSample(labels={"method": "GET"}, value=1000.0)
        counter_family = MetricFamily(
            type=PrometheusMetricType.COUNTER,
            help="Total requests",
            samples=[counter_sample],
        )

        # Gauge
        gauge_sample = MetricSample(labels=None, value=50.0)
        gauge_family = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            help="Active connections",
            samples=[gauge_sample],
        )

        # Histogram
        histogram = HistogramData(
            buckets={"0.1": 100.0, "0.5": 500.0, "+Inf": 1000.0},
            sum=250.0,
            count=1000.0,
        )
        histogram_sample = MetricSample(labels=None, histogram=histogram)
        histogram_family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM,
            help="Request duration",
            samples=[histogram_sample],
        )

        # Summary
        summary = SummaryData(
            quantiles={"0.5": 0.05, "0.9": 0.09}, sum=10.0, count=100.0
        )
        summary_sample = MetricSample(labels=None, summary=summary)
        summary_family = MetricFamily(
            type=PrometheusMetricType.SUMMARY,
            help="Response size",
            samples=[summary_sample],
        )

        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={
                "http_requests_total": counter_family,
                "active_connections": gauge_family,
                "http_duration_seconds": histogram_family,
                "response_size_bytes": summary_family,
            },
        )

        # Export slim format
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.jsonl"
            processor = ServerMetricsExportResultsProcessor(user_config=user_config)
            processor.output_file = output_file

            await processor.initialize()
            await processor.process_server_metrics_record(record)
            await processor.stop()

            # Verify all metrics are present in slim format
            data = orjson.loads(output_file.read_text())
            metrics = data["metrics"]

            assert "http_requests_total" in metrics
            assert "active_connections" in metrics
            assert "http_duration_seconds" in metrics
            assert "response_size_bytes" in metrics

            # Counter has value
            assert metrics["http_requests_total"][0]["value"] == 1000.0

            # Gauge has value
            assert metrics["active_connections"][0]["value"] == 50.0

            # Histogram is direct array
            assert isinstance(metrics["http_duration_seconds"][0]["histogram"], list)
            assert len(metrics["http_duration_seconds"][0]["histogram"]) == 3

            # Summary is direct array
            assert isinstance(metrics["response_size_bytes"][0]["summary"], list)
            assert len(metrics["response_size_bytes"][0]["summary"]) == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_metrics_dict(self, user_config):
        """Verify record with no metrics converts to slim format."""
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={},
        )

        slim = record.to_slim()
        assert slim.timestamp_ns == record.timestamp_ns
        assert len(slim.metrics) == 0

    def test_metric_with_no_samples(self, user_config):
        """Verify metric family with no samples converts to slim format."""
        family = MetricFamily(
            type=PrometheusMetricType.COUNTER, help="Test", samples=[]
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"test_metric": family},
        )

        slim = record.to_slim()
        assert len(slim.metrics["test_metric"]) == 0

    def test_histogram_with_zero_counts(self, user_config):
        """Verify histogram with all zero counts works correctly."""
        histogram = HistogramData(
            buckets={"0.1": 0.0, "0.5": 0.0, "1.0": 0.0, "+Inf": 0.0},
            sum=0.0,
            count=0.0,
        )
        sample = MetricSample(labels=None, histogram=histogram)
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM, help="Test", samples=[sample]
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"test_metric": family},
        )

        slim = record.to_slim()
        counts = slim.metrics["test_metric"][0].histogram
        assert all(c == 0.0 for c in counts)
        assert slim.metrics["test_metric"][0].sum == 0.0
        assert slim.metrics["test_metric"][0].count == 0.0

    def test_sample_with_many_labels(self, user_config):
        """Verify samples with many labels are preserved."""
        sample = MetricSample(
            labels={
                "method": "GET",
                "path": "/api/v1/users",
                "status": "200",
                "host": "server1",
                "region": "us-west",
                "env": "prod",
            },
            value=100.0,
        )
        family = MetricFamily(
            type=PrometheusMetricType.COUNTER, help="Test", samples=[sample]
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"test_metric": family},
        )

        slim = record.to_slim()
        slim_labels = slim.metrics["test_metric"][0].labels
        assert len(slim_labels) == 6
        assert slim_labels["method"] == "GET"
        assert slim_labels["path"] == "/api/v1/users"
        assert slim_labels["status"] == "200"

    def test_metric_with_multiple_samples(self, user_config):
        """Verify metric with multiple samples (different label sets) works correctly."""
        samples = [
            MetricSample(labels={"method": "GET"}, value=1000.0),
            MetricSample(labels={"method": "POST"}, value=500.0),
            MetricSample(labels={"method": "DELETE"}, value=50.0),
        ]
        family = MetricFamily(
            type=PrometheusMetricType.COUNTER,
            help="Requests by method",
            samples=samples,
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"http_requests_total": family},
        )

        slim = record.to_slim()
        slim_samples = slim.metrics["http_requests_total"]
        assert len(slim_samples) == 3
        # Verify each sample preserved
        methods = [s.labels["method"] for s in slim_samples]
        assert set(methods) == {"GET", "POST", "DELETE"}

    def test_histogram_with_large_bucket_counts(self, user_config):
        """Verify histograms with large bucket counts work correctly."""
        # Create histogram with 20 buckets
        buckets = {f"{i * 0.1:.1f}": float(i * 100) for i in range(1, 21)}
        buckets["+Inf"] = 2000.0

        histogram = HistogramData(buckets=buckets, sum=10000.0, count=2000.0)
        sample = MetricSample(labels=None, histogram=histogram)
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM, help="Test", samples=[sample]
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"test_metric": family},
        )

        slim = record.to_slim()
        counts = slim.metrics["test_metric"][0].histogram
        assert len(counts) == 21  # 20 buckets + +Inf
        # Verify counts are in order
        assert counts[0] == 100.0
        assert counts[19] == 2000.0
        assert counts[20] == 2000.0  # +Inf


class TestFullRecordReconstruction:
    """Test that full records can be reconstructed from slim format + schema."""

    def test_reconstruct_histogram_from_slim_and_schema(self):
        """Verify histogram can be reconstructed with correct bucket labels."""
        # Original record with histogram
        histogram = HistogramData(
            buckets={"0.1": 10.0, "0.5": 50.0, "1.0": 100.0, "+Inf": 150.0},
            sum=75.0,
            count=150.0,
        )
        sample = MetricSample(labels={"path": "/api"}, histogram=histogram)
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM,
            help="Request duration",
            samples=[sample],
        )
        original_record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"http_duration": family},
        )

        # Convert to slim
        slim_record = original_record.to_slim()

        # Create schema (simulating what would be in metadata)
        schema = MetricSchema(
            type=PrometheusMetricType.HISTOGRAM,
            help="Request duration",
            bucket_labels=["0.1", "0.5", "1.0", "+Inf"],
        )

        # Simulate reconstruction (what RecordsManager does)
        slim_sample = slim_record.metrics["http_duration"][0]
        reconstructed_buckets = dict(
            zip(schema.bucket_labels, slim_sample.histogram, strict=True)
        )
        reconstructed_histogram = HistogramData(
            buckets=reconstructed_buckets,
            sum=slim_sample.sum,
            count=slim_sample.count,
        )

        # Verify reconstructed histogram matches original
        assert reconstructed_histogram.buckets == histogram.buckets
        assert reconstructed_histogram.sum == histogram.sum
        assert reconstructed_histogram.count == histogram.count

    def test_reconstruct_summary_from_slim_and_schema(self):
        """Verify summary can be reconstructed with correct quantile labels."""
        # Original record with summary
        summary = SummaryData(
            quantiles={"0.5": 0.05, "0.9": 0.09, "0.99": 0.099},
            sum=10.0,
            count=100.0,
        )
        sample = MetricSample(labels={"handler": "/api"}, summary=summary)
        family = MetricFamily(
            type=PrometheusMetricType.SUMMARY,
            help="Response size",
            samples=[sample],
        )
        original_record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"response_size": family},
        )

        # Convert to slim
        slim_record = original_record.to_slim()

        # Create schema
        schema = MetricSchema(
            type=PrometheusMetricType.SUMMARY,
            help="Response size",
            quantile_labels=["0.5", "0.9", "0.99"],
        )

        # Simulate reconstruction
        slim_sample = slim_record.metrics["response_size"][0]
        reconstructed_quantiles = dict(
            zip(schema.quantile_labels, slim_sample.summary, strict=True)
        )
        reconstructed_summary = SummaryData(
            quantiles=reconstructed_quantiles,
            sum=slim_sample.sum,
            count=slim_sample.count,
        )

        # Verify reconstructed summary matches original
        assert reconstructed_summary.quantiles == summary.quantiles
        assert reconstructed_summary.sum == summary.sum
        assert reconstructed_summary.count == summary.count

    def test_reconstruct_preserves_all_metric_data(self):
        """Verify complete round-trip: original -> slim -> reconstructed preserves all data."""
        # Create complex record with multiple metric types
        counter_sample = MetricSample(labels={"method": "GET"}, value=1000.0)
        histogram = HistogramData(
            buckets={"0.1": 100.0, "1.0": 500.0, "+Inf": 1000.0},
            sum=250.0,
            count=1000.0,
        )
        histogram_sample = MetricSample(labels={"path": "/api"}, histogram=histogram)

        original_record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    help="Total requests",
                    samples=[counter_sample],
                ),
                "request_duration": MetricFamily(
                    type=PrometheusMetricType.HISTOGRAM,
                    help="Request duration",
                    samples=[histogram_sample],
                ),
            },
        )

        # Convert to slim
        slim_record = original_record.to_slim()

        # Create schemas
        schemas = {
            "requests_total": MetricSchema(
                type=PrometheusMetricType.COUNTER,
                help="Total requests",
                bucket_labels=None,
                quantile_labels=None,
            ),
            "request_duration": MetricSchema(
                type=PrometheusMetricType.HISTOGRAM,
                help="Request duration",
                bucket_labels=["0.1", "1.0", "+Inf"],
                quantile_labels=None,
            ),
        }

        # Verify timestamp preserved
        assert slim_record.timestamp_ns == original_record.timestamp_ns

        # Verify counter sample preserved
        slim_counter = slim_record.metrics["requests_total"][0]
        assert slim_counter.labels == counter_sample.labels
        assert slim_counter.value == counter_sample.value

        # Verify histogram can be reconstructed
        slim_histogram = slim_record.metrics["request_duration"][0]
        reconstructed_buckets = dict(
            zip(
                schemas["request_duration"].bucket_labels,
                slim_histogram.histogram,
                strict=True,
            )
        )
        assert reconstructed_buckets == histogram.buckets

    def test_schema_bucket_count_mismatch_raises_error(self):
        """Verify mismatched bucket counts between schema and data raises error."""
        # Slim data with 3 counts
        histogram = HistogramData(
            buckets={"0.1": 10.0, "1.0": 50.0, "+Inf": 100.0}, sum=50.0, count=100.0
        )
        sample = MetricSample(labels=None, histogram=histogram)
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM, help="Test", samples=[sample]
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"test": family},
        )

        slim = record.to_slim()
        slim_histogram = slim.metrics["test"][0].histogram

        # Schema with wrong number of bucket labels (4 instead of 3)
        wrong_schema_labels = ["0.1", "0.5", "1.0", "+Inf"]

        # Attempting to zip with strict=True should raise ValueError
        with pytest.raises(ValueError):
            dict(zip(wrong_schema_labels, slim_histogram, strict=True))


class TestSchemaMetadata:
    """Test schema extraction and metadata handling."""

    def test_metadata_includes_all_schemas_for_multi_metric_record(self):
        """Verify metadata extraction creates schemas for all metrics in a record."""
        counter = MetricSample(labels=None, value=100.0)
        histogram = HistogramData(
            buckets={"0.1": 10.0, "+Inf": 50.0}, sum=25.0, count=50.0
        )
        hist_sample = MetricSample(labels=None, histogram=histogram)

        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={
                "metric1": MetricFamily(
                    type=PrometheusMetricType.COUNTER, help="Counter", samples=[counter]
                ),
                "metric2": MetricFamily(
                    type=PrometheusMetricType.HISTOGRAM,
                    help="Histogram",
                    samples=[hist_sample],
                ),
            },
        )

        # Manually create metadata (simulating ServerMetricsManager)
        schemas = {}
        for metric_name, family in record.metrics.items():
            bucket_labels = None
            if family.samples and family.samples[0].histogram:
                bucket_labels = sorted(
                    family.samples[0].histogram.buckets.keys(), key=lambda x: float(x)
                )
            schemas[metric_name] = MetricSchema(
                type=family.type, help=family.help, bucket_labels=bucket_labels
            )

        metadata = ServerMetricsMetadata(
            endpoint_url=record.endpoint_url,
            endpoint_display="localhost:8080",
            kubernetes_pod_info=None,
            metric_schemas=schemas,
        )

        # Verify metadata has schemas for both metrics
        assert len(metadata.metric_schemas) == 2
        assert "metric1" in metadata.metric_schemas
        assert "metric2" in metadata.metric_schemas

        # Verify counter schema has no bucket labels
        assert metadata.metric_schemas["metric1"].bucket_labels is None

        # Verify histogram schema has bucket labels
        assert metadata.metric_schemas["metric2"].bucket_labels == ["0.1", "+Inf"]

    def test_different_endpoints_have_separate_schemas(self):
        """Verify different endpoints can have different schemas for same metric name."""
        # Endpoint 1 with histogram having 3 buckets
        histogram1 = HistogramData(
            buckets={"0.1": 10.0, "1.0": 50.0, "+Inf": 100.0}, sum=50.0, count=100.0
        )
        MetricSample(labels=None, histogram=histogram1)

        # Endpoint 2 with histogram having 5 buckets (different configuration)
        histogram2 = HistogramData(
            buckets={
                "0.1": 5.0,
                "0.5": 25.0,
                "1.0": 50.0,
                "5.0": 90.0,
                "+Inf": 100.0,
            },
            sum=60.0,
            count=100.0,
        )
        MetricSample(labels=None, histogram=histogram2)

        # Create schemas for each endpoint
        schema1_buckets = sorted(histogram1.buckets.keys(), key=lambda x: float(x))
        schema2_buckets = sorted(histogram2.buckets.keys(), key=lambda x: float(x))

        # Verify schemas are different
        assert len(schema1_buckets) == 3
        assert len(schema2_buckets) == 5
        assert schema1_buckets != schema2_buckets


class TestDataIntegrity:
    """Test data integrity through the complete flow."""

    def test_no_data_loss_in_slim_conversion(self):
        """Verify no data is lost when converting to slim format."""
        # Create record with all possible data
        histogram = HistogramData(
            buckets={"0.1": 10.0, "0.5": 50.0, "1.0": 100.0, "+Inf": 150.0},
            sum=75.5,
            count=150.0,
        )
        summary = SummaryData(
            quantiles={"0.5": 0.055, "0.9": 0.091, "0.99": 0.0991},
            sum=10.123,
            count=100.0,
        )

        original = ServerMetricsRecord(
            timestamp_ns=1234567890,
            endpoint_url="http://test:8080/metrics",
            kubernetes_pod_info=None,
            metrics={
                "counter": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    help="Test counter",
                    samples=[MetricSample(labels={"tag": "value"}, value=999.0)],
                ),
                "gauge": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    help="Test gauge",
                    samples=[MetricSample(labels={"tag": "value"}, value=50.5)],
                ),
                "histogram": MetricFamily(
                    type=PrometheusMetricType.HISTOGRAM,
                    help="Test histogram",
                    samples=[
                        MetricSample(labels={"tag": "value"}, histogram=histogram)
                    ],
                ),
                "summary": MetricFamily(
                    type=PrometheusMetricType.SUMMARY,
                    help="Test summary",
                    samples=[MetricSample(labels={"tag": "value"}, summary=summary)],
                ),
            },
        )

        # Convert to slim
        slim = original.to_slim()

        # Verify timestamp preserved
        assert slim.timestamp_ns == original.timestamp_ns

        # Verify counter value preserved
        assert slim.metrics["counter"][0].value == 999.0

        # Verify gauge value preserved
        assert slim.metrics["gauge"][0].value == 50.5

        # Verify histogram counts preserved (in order)
        hist_counts = slim.metrics["histogram"][0].histogram
        assert hist_counts == [10.0, 50.0, 100.0, 150.0]
        assert slim.metrics["histogram"][0].sum == 75.5
        assert slim.metrics["histogram"][0].count == 150.0

        # Verify summary values preserved (in order)
        summ_values = slim.metrics["summary"][0].summary
        assert summ_values == [0.055, 0.091, 0.0991]
        assert slim.metrics["summary"][0].sum == 10.123
        assert slim.metrics["summary"][0].count == 100.0

        # Verify labels preserved
        for metric_name in slim.metrics:
            assert slim.metrics[metric_name][0].labels == {"tag": "value"}

    @pytest.mark.asyncio
    async def test_exported_slim_format_is_valid_json(self, user_config):
        """Verify exported slim records are valid JSON."""
        histogram = HistogramData(
            buckets={"0.1": 10.0, "+Inf": 100.0}, sum=50.0, count=100.0
        )
        sample = MetricSample(labels=None, histogram=histogram)
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM, help="Test", samples=[sample]
        )
        record = ServerMetricsRecord(
            timestamp_ns=1000000000,
            endpoint_url="http://localhost:8080/metrics",
            metrics={"test_metric": family},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.jsonl"
            processor = ServerMetricsExportResultsProcessor(user_config=user_config)
            processor.output_file = output_file

            await processor.initialize()
            await processor.process_server_metrics_record(record)
            await processor.stop()

            # Verify file is valid JSON
            content = output_file.read_text()
            data = orjson.loads(content)  # Will raise if invalid JSON

            # Verify basic structure
            assert "timestamp_ns" in data
            assert "metrics" in data
            assert isinstance(data["metrics"], dict)
