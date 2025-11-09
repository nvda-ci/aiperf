# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for histogram utility functions."""

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.metric_utils import (
    compute_histogram_delta,
    compute_metric_statistics_from_histogram,
)


class TestComputeHistogramDelta:
    """Tests for compute_histogram_delta function."""

    def test_simple_delta(self):
        """Test simple histogram delta calculation."""
        start_histogram = {
            "0.1": 10,
            "0.5": 45,
            "+Inf": 100,
        }
        end_histogram = {
            "0.1": 30,
            "0.5": 95,
            "+Inf": 200,
        }

        delta = compute_histogram_delta(start_histogram, end_histogram)

        assert delta["0.1"] == 20
        assert delta["0.5"] == 50
        assert delta["+Inf"] == 100

    def test_no_change_delta(self):
        """Test delta when histograms are identical."""
        histogram = {
            "0.5": 50,
            "1.0": 100,
            "+Inf": 100,
        }

        delta = compute_histogram_delta(histogram, histogram)

        assert delta["0.5"] == 0
        assert delta["1.0"] == 0
        assert delta["+Inf"] == 0

    def test_mismatched_buckets_error(self):
        """Test error when bucket boundaries don't match."""
        start_histogram = {
            "0.1": 10,
            "0.5": 45,
            "+Inf": 100,
        }
        end_histogram = {
            "0.2": 30,  # Different boundary
            "0.5": 95,
            "+Inf": 200,
        }

        with pytest.raises(ValueError, match="boundaries don't match"):
            compute_histogram_delta(start_histogram, end_histogram)


class TestComputeMetricStatisticsFromHistogram:
    """Tests for compute_metric_statistics_from_histogram function."""

    def test_basic_histogram_statistics(self):
        """Test computing statistics from histogram."""
        buckets = {
            "0.1": 10,
            "0.5": 45,
            "1.0": 100,
            "+Inf": 100,
        }
        sum_value = 35.0  # Total sum of all observations
        count = 100

        result = compute_metric_statistics_from_histogram(
            buckets=buckets,
            sum_value=sum_value,
            count=count,
            tag="test_metric",
            header="Test Metric",
            unit="ms",
        )

        assert result.tag == "test_metric"
        assert result.header == "Test Metric"
        assert result.unit == "ms"
        assert result.count == 100
        assert result.avg == 0.35  # 35.0 / 100
        assert result.min == 0.0
        assert result.max == 0.0
        # Percentiles are not computed for histograms
        assert result.p50 == 0.0
        assert result.p95 == 0.0
        assert result.p99 == 0.0

    def test_zero_count_error(self):
        """Test error when count is zero."""
        buckets = {
            "0.5": 0,
            "+Inf": 0,
        }

        with pytest.raises(NoMetricValue, match="zero observations"):
            compute_metric_statistics_from_histogram(
                buckets=buckets,
                sum_value=0,
                count=0,
                tag="test",
                header="Test",
                unit="ms",
            )

    def test_invalid_buckets_error(self):
        """Test that invalid buckets don't cause errors (no validation)."""
        # We no longer validate bucket structure since we don't compute percentiles
        # Missing +Inf bucket is fine
        buckets = {
            "0.5": 50,
        }

        result = compute_metric_statistics_from_histogram(
            buckets=buckets,
            sum_value=25.0,
            count=50,
            tag="test",
            header="Test",
            unit="ms",
        )

        assert result.count == 50
        assert result.avg == 0.5  # 25.0 / 50

    def test_with_metric_name_in_error(self):
        """Test that metric name appears in error messages."""
        buckets = {
            "0.5": 0,
            "+Inf": 0,
        }

        with pytest.raises(
            NoMetricValue, match="Histogram metric 'request_latency' has zero"
        ):
            compute_metric_statistics_from_histogram(
                buckets=buckets,
                sum_value=0,
                count=0,
                tag="test",
                header="Test",
                unit="ms",
                metric_name="request_latency",
            )
