# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the metrics utilities."""

import orjson
import pytest

from aiperf.api.metrics_utils import build_info_labels, format_metrics_json
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.models import MetricResult


class TestBuildInfoLabels:
    """Test info label building from UserConfig."""

    def test_basic_labels(self) -> None:
        """Test basic label extraction from config."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["gpt-4"]))
        labels = build_info_labels(config)

        assert labels["model"] == "gpt-4"
        assert labels["endpoint_type"] == "chat"
        assert labels["streaming"] == "false"
        assert "config" in labels

    def test_multiple_models(self) -> None:
        """Test multiple model names are comma-separated."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["gpt-4", "gpt-3.5-turbo"])
        )
        labels = build_info_labels(config)
        assert labels["model"] == "gpt-4,gpt-3.5-turbo"

    def test_benchmark_id_included(self) -> None:
        """Test benchmark_id is included when set."""
        config = UserConfig(
            benchmark_id="test-bench-123",
            endpoint=EndpointConfig(model_names=["model"]),
        )
        labels = build_info_labels(config)
        assert labels["benchmark_id"] == "test-bench-123"

    def test_benchmark_id_auto_generated(self) -> None:
        """Test benchmark_id is auto-generated when not explicitly set."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["model"]))
        labels = build_info_labels(config)
        assert "benchmark_id" in labels

    @pytest.mark.parametrize(
        "loadgen_kwargs,label_key,expected",
        [
            ({"concurrency": 10}, "concurrency", "10"),
            ({"request_rate": 5.0}, "request_rate", "5.0"),
        ],
    )  # fmt: skip
    def test_loadgen_labels(
        self, loadgen_kwargs: dict, label_key: str, expected: str
    ) -> None:
        """Test loadgen parameters are included in labels."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["model"]),
            loadgen=LoadGeneratorConfig(**loadgen_kwargs),
        )
        labels = build_info_labels(config)
        assert labels[label_key] == expected

    def test_streaming_false(self) -> None:
        """Test streaming label when disabled."""
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["model"], streaming=False)
        )
        labels = build_info_labels(config)
        assert labels["streaming"] == "false"


class TestFormatMetricsJson:
    """Test JSON metrics formatting."""

    def test_empty_metrics(self) -> None:
        """Test formatting empty metrics list."""
        result = format_metrics_json([])
        data = orjson.loads(result)

        assert "aiperf_version" in data
        assert data["metrics"] == {}

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
        result = format_metrics_json([metric])
        data = orjson.loads(result)

        assert "test_metric" in data["metrics"]
        assert data["metrics"]["test_metric"]["avg"] == 100.0
        assert data["metrics"]["test_metric"]["min"] == 50.0
        assert data["metrics"]["test_metric"]["max"] == 150.0
        assert "tag" not in data["metrics"]["test_metric"]

    def test_benchmark_id_included(self) -> None:
        """Test benchmark_id is included when provided."""
        result = format_metrics_json([], benchmark_id="bench-123")
        data = orjson.loads(result)
        assert data["benchmark_id"] == "bench-123"

    def test_benchmark_id_excluded_when_none(self) -> None:
        """Test benchmark_id is not included when None."""
        result = format_metrics_json([])
        data = orjson.loads(result)
        assert "benchmark_id" not in data

    def test_info_labels_included(self) -> None:
        """Test info labels are merged into response."""
        result = format_metrics_json(
            [], info_labels={"model": "gpt-4", "endpoint_type": "openai"}
        )
        data = orjson.loads(result)

        assert data["model"] == "gpt-4"
        assert data["endpoint_type"] == "openai"

    @pytest.mark.parametrize("excluded_key", ["config", "version"])  # fmt: skip
    def test_excluded_labels(self, excluded_key: str) -> None:
        """Test that certain labels are excluded from output."""
        result = format_metrics_json(
            [], info_labels={"model": "gpt-4", excluded_key: "some_value"}
        )
        data = orjson.loads(result)

        assert excluded_key not in data or data.get(excluded_key) != "some_value"
        assert data["model"] == "gpt-4"

    def test_multiple_metrics(self) -> None:
        """Test formatting multiple metrics."""
        metrics = [
            MetricResult(tag="latency", header="Latency", unit="ms", avg=100.0),
            MetricResult(tag="throughput", header="Throughput", unit="req/s", avg=50.0),
        ]
        result = format_metrics_json(metrics)
        data = orjson.loads(result)

        assert data["metrics"]["latency"]["avg"] == 100.0
        assert data["metrics"]["throughput"]["avg"] == 50.0
