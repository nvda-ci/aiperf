# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Prometheus formatter."""

import pytest

from aiperf.api.prometheus_formatter import (
    format_as_prometheus,
    format_labels,
    sanitize_metric_name,
)
from aiperf.common.models import MetricResult


class TestSanitizeMetricName:
    """Test metric name sanitization."""

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("TestMetric", "testmetric"),
            ("test-metric", "test_metric"),
            ("test.metric", "test_metric"),
            ("test/metric", "test_metric"),
            ("123metric", "_123metric"),
            ("valid_metric_123", "valid_metric_123"),
        ],
    )  # fmt: skip
    def test_sanitize(self, input_name: str, expected: str) -> None:
        """Test metric name sanitization handles various inputs."""
        assert sanitize_metric_name(input_name) == expected


class TestFormatLabels:
    """Test label formatting."""

    def test_empty_labels(self) -> None:
        """Test formatting empty labels returns empty string."""
        assert format_labels({}) == ""

    def test_single_label(self) -> None:
        """Test formatting single label."""
        assert format_labels({"key": "value"}) == '{key="value"}'

    def test_multiple_labels(self) -> None:
        """Test formatting multiple labels."""
        result = format_labels({"key1": "value1", "key2": "value2"})
        assert 'key1="value1"' in result
        assert 'key2="value2"' in result

    @pytest.mark.parametrize(
        "value,expected",
        [
            ('single"value', '{key="single\\"value"}'),
            ("double\"value", '{key="double\\"value"}'),
        ],
    )  # fmt: skip
    def test_escape_special_chars(self, value: str, expected: str) -> None:
        """Test that special characters in values are escaped."""
        assert format_labels({"key": value}) == expected


class TestFormatAsPrometheus:
    """Test Prometheus format generation."""

    def test_empty_metrics(self) -> None:
        """Test formatting empty metrics list returns empty string."""
        assert format_as_prometheus([]) == ""

    def test_single_metric(self) -> None:
        """Test formatting single metric."""
        metric = MetricResult(
            tag="test_metric",
            header="Test Metric",
            unit="ms",
            avg=100.0,
            min=50.0,
            max=150.0,
        )
        result = format_as_prometheus([metric])

        assert "aiperf_test_metric_avg_" in result
        assert "# TYPE" in result
        assert "# HELP" in result
        assert "100.0" in result

    def test_info_labels(self) -> None:
        """Test that info labels are included."""
        metric = MetricResult(
            tag="test_metric", header="Test Metric", unit="ms", avg=100.0
        )
        result = format_as_prometheus(
            [metric],
            info_labels={"model": "gpt-4", "endpoint_type": "openai"},
        )

        assert "aiperf_info" in result
        assert 'model="gpt-4"' in result
        assert 'endpoint_type="openai"' in result

    def test_percentiles(self) -> None:
        """Test that percentiles are included."""
        metric = MetricResult(
            tag="latency",
            header="Latency",
            unit="ms",
            p50=100.0,
            p95=200.0,
            p99=300.0,
        )
        result = format_as_prometheus([metric])

        assert "p50" in result
        assert "p95" in result
        assert "p99" in result
