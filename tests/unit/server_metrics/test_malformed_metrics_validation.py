# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validation of malformed histogram and summary metrics."""

import pytest

from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)


class TestMalformedHistogramValidation:
    """Test that malformed histograms are properly rejected."""

    @pytest.mark.asyncio
    async def test_histogram_with_no_buckets_skipped(self):
        """Test that histograms with sum/count but no buckets are skipped."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        # Malformed histogram: has sum and count but no bucket samples
        malformed_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_sum 100.0
http_request_duration_seconds_count 50
"""
        records = collector._parse_metrics_to_records(malformed_prometheus, 1000)

        # Should return empty list because histogram is incomplete (empty snapshots are suppressed)
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_histogram_with_only_buckets_skipped(self):
        """Test that histograms with only buckets (no sum/count) are skipped."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        # Malformed histogram: has buckets but no sum/count
        malformed_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 10
http_request_duration_seconds_bucket{le="1.0"} 25
http_request_duration_seconds_bucket{le="+Inf"} 50
"""
        records = collector._parse_metrics_to_records(malformed_prometheus, 1000)

        # Should return empty list because histogram is incomplete (empty snapshots are suppressed)
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_histogram_with_only_sum_skipped(self):
        """Test that histograms with only sum (no count/buckets) are skipped."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        # Malformed histogram: has only sum
        malformed_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_sum 100.0
"""
        records = collector._parse_metrics_to_records(malformed_prometheus, 1000)

        # Should return empty list because histogram is incomplete (empty snapshots are suppressed)
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_valid_histogram_accepted(self):
        """Test that valid histograms with all required fields are accepted."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        # Valid histogram: has buckets, sum, and count
        valid_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 10
http_request_duration_seconds_bucket{le="1.0"} 25
http_request_duration_seconds_bucket{le="+Inf"} 50
http_request_duration_seconds_sum 100.0
http_request_duration_seconds_count 50
"""
        records = collector._parse_metrics_to_records(valid_prometheus, 1000)

        # Should successfully parse the complete histogram
        assert len(records) == 1
        record = records[0]
        assert "http_request_duration_seconds" in record.metrics
        metric_family = record.metrics["http_request_duration_seconds"]
        assert len(metric_family.samples) == 1
        assert metric_family.samples[0].histogram is not None
        assert len(metric_family.samples[0].histogram.buckets) == 3
        assert metric_family.samples[0].histogram.sum == 100.0
        assert metric_family.samples[0].histogram.count == 50


class TestMalformedSummaryValidation:
    """Test that malformed summaries are properly rejected."""

    @pytest.mark.asyncio
    async def test_summary_with_no_quantiles_skipped(self):
        """Test that summaries with sum/count but no quantiles are skipped."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        # Malformed summary: has sum and count but no quantile samples
        malformed_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds summary
http_request_duration_seconds_sum 100.0
http_request_duration_seconds_count 50
"""
        records = collector._parse_metrics_to_records(malformed_prometheus, 1000)

        # Should return empty list because summary is incomplete (empty snapshots are suppressed)
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_summary_with_only_quantiles_skipped(self):
        """Test that summaries with only quantiles (no sum/count) are skipped."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        # Malformed summary: has quantiles but no sum/count
        malformed_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds summary
http_request_duration_seconds{quantile="0.5"} 0.1
http_request_duration_seconds{quantile="0.9"} 0.5
http_request_duration_seconds{quantile="0.99"} 1.0
"""
        records = collector._parse_metrics_to_records(malformed_prometheus, 1000)

        # Should return empty list because summary is incomplete (empty snapshots are suppressed)
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_summary_with_only_sum_skipped(self):
        """Test that summaries with only sum (no count/quantiles) are skipped."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        # Malformed summary: has only sum
        malformed_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds summary
http_request_duration_seconds_sum 100.0
"""
        records = collector._parse_metrics_to_records(malformed_prometheus, 1000)

        # Should return empty list because summary is incomplete (empty snapshots are suppressed)
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_valid_summary_accepted(self):
        """Test that valid summaries with all required fields are accepted."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        # Valid summary: has quantiles, sum, and count
        valid_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds summary
http_request_duration_seconds{quantile="0.5"} 0.1
http_request_duration_seconds{quantile="0.9"} 0.5
http_request_duration_seconds{quantile="0.99"} 1.0
http_request_duration_seconds_sum 100.0
http_request_duration_seconds_count 50
"""
        records = collector._parse_metrics_to_records(valid_prometheus, 1000)

        # Should successfully parse the complete summary
        assert len(records) == 1
        record = records[0]
        assert "http_request_duration_seconds" in record.metrics
        metric_family = record.metrics["http_request_duration_seconds"]
        assert len(metric_family.samples) == 1
        assert metric_family.samples[0].summary is not None
        assert len(metric_family.samples[0].summary.quantiles) == 3
        assert metric_family.samples[0].summary.sum == 100.0
        assert metric_family.samples[0].summary.count == 50


class TestMixedValidAndInvalidMetrics:
    """Test behavior when some metrics are valid and others are malformed."""

    @pytest.mark.asyncio
    async def test_valid_metrics_preserved_when_invalid_skipped(self):
        """Test that valid metrics are still processed when invalid ones are skipped."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        # Mix of valid counter, invalid histogram, and valid gauge
        mixed_prometheus = """# HELP requests_total Total requests
# TYPE requests_total counter
requests_total 1000

# HELP http_duration_seconds Request duration (malformed - no buckets)
# TYPE http_duration_seconds histogram
http_duration_seconds_sum 100.0
http_duration_seconds_count 50

# HELP active_connections Active connections
# TYPE active_connections gauge
active_connections 42
"""
        records = collector._parse_metrics_to_records(mixed_prometheus, 1000)

        # Should have valid counter and gauge, but not the malformed histogram
        assert len(records) == 1
        record = records[0]
        # Note: prometheus_client strips _total suffix from counter names
        assert "requests" in record.metrics
        assert "active_connections" in record.metrics
        assert "http_duration_seconds" not in record.metrics
