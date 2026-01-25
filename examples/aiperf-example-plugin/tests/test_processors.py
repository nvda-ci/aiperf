# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for result processors demonstrating 2025 pytest best practices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiperf_example_plugin.processors import (
    ExampleMetricsProcessor,
    ExampleMetricsProcessorConfig,
    ExampleResultsAggregator,
    ProcessingResult,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for test files.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture

    Returns:
        Path to temporary directory
    """
    return tmp_path


@pytest.fixture
def sample_results() -> list[dict[str, Any]]:
    """Create sample results for testing.

    Returns:
        List of sample result records with varying data
    """
    return [
        {
            "id": 1,
            "timestamp": 1000.0,
            "latency_ms": 50.5,
            "status": "success",
            "error": None,
        },
        {
            "id": 2,
            "timestamp": 1001.0,
            "latency_ms": 45.2,
            "status": "success",
            "error": None,
        },
        {
            "id": 3,
            "timestamp": 1002.0,
            "latency_ms": 100.1,
            "status": "success",
            "error": None,
        },
        {
            "id": 4,
            "timestamp": 1003.0,
            "latency_ms": None,
            "status": "error",
            "error": "Timeout",
        },
    ]


class TestProcessingResult:
    """Tests for ProcessingResult Pydantic model."""

    def test_success_result_creation(self) -> None:
        """Test creating a successful ProcessingResult."""
        result = ProcessingResult(
            success=True,
            record_count=100,
            error_count=0,
            metrics={"custom_metric": 42},
            output_path="/tmp/output.txt",
        )

        assert result.success
        assert result.record_count == 100
        assert result.error_count == 0
        assert result.metrics == {"custom_metric": 42}
        assert result.output_path == "/tmp/output.txt"

    def test_default_values(self) -> None:
        """Test ProcessingResult default values."""
        result = ProcessingResult(success=False)

        assert not result.success
        assert result.record_count == 0
        assert result.error_count == 0
        assert result.metrics is None
        assert result.output_path is None

    def test_frozen_model_immutable(self) -> None:
        """Test that ProcessingResult is immutable."""
        result = ProcessingResult(success=True)

        with pytest.raises(Exception):  # Pydantic ValidationError
            result.success = False  # type: ignore


class TestExampleMetricsProcessorConfig:
    """Tests for ExampleMetricsProcessorConfig Pydantic model."""

    def test_default_config(self) -> None:
        """Test config with default values."""
        config = ExampleMetricsProcessorConfig()

        assert config.output_file == Path("/tmp/aiperf_metrics.txt")
        assert config.include_percentiles is True
        assert config.create_dirs is True
        assert config.percentiles == [50, 75, 90, 95, 99]

    def test_custom_config(self, temp_dir: Path) -> None:
        """Test config with custom values."""
        output_file = temp_dir / "custom.txt"
        config = ExampleMetricsProcessorConfig(
            output_file=output_file,
            include_percentiles=False,
            create_dirs=False,
            percentiles=[50, 90, 99],
        )

        assert config.output_file == output_file
        assert config.include_percentiles is False
        assert config.create_dirs is False
        assert config.percentiles == [50, 90, 99]

    def test_percentile_validation(self) -> None:
        """Test that invalid percentiles raise validation error."""
        with pytest.raises(ValueError):
            ExampleMetricsProcessorConfig(percentiles=[50, 150])  # 150 is invalid

    def test_percentile_deduplication(self) -> None:
        """Test that duplicate percentiles are removed."""
        config = ExampleMetricsProcessorConfig(percentiles=[50, 90, 50, 90])

        assert config.percentiles == [50, 90]


class TestExampleMetricsProcessor:
    """Tests for ExampleMetricsProcessor."""

    def test_init_creates_output_directory(self, temp_dir: Path) -> None:
        """Test that output directory is created on init."""
        output_file = temp_dir / "metrics" / "output.txt"
        config = ExampleMetricsProcessorConfig(output_file=output_file)
        ExampleMetricsProcessor(config)

        assert output_file.parent.exists()

    def test_from_config_factory_method(self, temp_dir: Path) -> None:
        """Test factory method creates processor correctly."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(
            output_file=output_file,
            include_percentiles=False,
            percentiles=[50, 95],
        )

        assert processor.config.output_file == output_file
        assert processor.config.include_percentiles is False
        assert processor.config.percentiles == [50, 95]

    @pytest.mark.asyncio
    async def test_process_empty_results(self, temp_dir: Path) -> None:
        """Test processing empty results."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(output_file=output_file)

        result = await processor.process([])

        assert not result.success
        assert result.record_count == 0

    @pytest.mark.asyncio
    async def test_process_results_calculates_metrics(
        self, temp_dir: Path, sample_results: list[dict[str, Any]]
    ) -> None:
        """Test that processing calculates metrics."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(output_file=output_file)

        result = await processor.process(sample_results)

        assert result.success
        assert result.record_count == 4
        assert result.output_path == str(output_file)
        assert result.metrics is not None

    @pytest.mark.asyncio
    async def test_process_writes_output_file(
        self, temp_dir: Path, sample_results: list[dict[str, Any]]
    ) -> None:
        """Test that metrics are written to output file."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(output_file=output_file)

        await processor.process(sample_results)

        assert output_file.exists()
        content = output_file.read_text()
        assert "AIPerf Example Plugin Metrics" in content
        assert "request_counts" in content

    @pytest.mark.asyncio
    async def test_calculate_request_stats(
        self, temp_dir: Path, sample_results: list[dict[str, Any]]
    ) -> None:
        """Test request statistics calculation."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(output_file=output_file)

        result = await processor.process(sample_results)

        assert result.metrics is not None
        assert result.metrics["request_counts"]["total"] == 4
        assert result.metrics["request_counts"]["successful"] == 3
        assert result.metrics["request_counts"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_calculate_latency_percentiles(
        self, temp_dir: Path, sample_results: list[dict[str, Any]]
    ) -> None:
        """Test latency percentile calculation."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(
            output_file=output_file, include_percentiles=True
        )

        result = await processor.process(sample_results)

        assert result.metrics is not None
        assert "latency_percentiles" in result.metrics

        percentiles = result.metrics["latency_percentiles"]
        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        assert "mean" in percentiles
        assert "stdev" in percentiles

        # Verify percentile values are reasonable
        assert percentiles["min"] <= percentiles["p50"] <= percentiles["max"]
        assert percentiles["p95"] >= percentiles["p50"]

    @pytest.mark.asyncio
    async def test_error_rate_calculation(
        self, temp_dir: Path, sample_results: list[dict[str, Any]]
    ) -> None:
        """Test error rate calculation."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(output_file=output_file)

        result = await processor.process(sample_results)

        assert result.metrics is not None
        assert result.metrics["error_count"] == 1
        assert result.metrics["error_rate"] == 0.25  # 1 out of 4

    @pytest.mark.asyncio
    async def test_no_latency_data(self, temp_dir: Path) -> None:
        """Test processing results without latency data."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(output_file=output_file)

        results = [
            {"id": 1, "timestamp": 1000.0, "status": "success"},
            {"id": 2, "timestamp": 1001.0, "status": "success"},
        ]

        result = await processor.process(results)

        assert result.success
        assert result.record_count == 2
        # Latency percentiles should not be calculated without data
        assert "latency_percentiles" not in result.metrics

    @pytest.mark.asyncio
    async def test_percentile_calculation_edge_cases(self, temp_dir: Path) -> None:
        """Test percentile calculation with edge cases."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(output_file=output_file)

        # Single value
        results = [{"id": 1, "timestamp": 1000.0, "latency_ms": 50.0, "status": "success"}]

        result = await processor.process(results)
        assert result.success
        assert result.metrics["latency_percentiles"]["mean"] == 50.0

    @pytest.mark.asyncio
    async def test_custom_percentiles(self, temp_dir: Path) -> None:
        """Test processing with custom percentile list."""
        output_file = temp_dir / "metrics.txt"
        processor = ExampleMetricsProcessor.from_config(
            output_file=output_file, percentiles=[10, 50, 90]
        )

        results = [{"latency_ms": float(i), "status": "success"} for i in range(1, 101)]

        result = await processor.process(results)

        assert result.success
        percentiles = result.metrics["latency_percentiles"]
        assert "p10" in percentiles
        assert "p50" in percentiles
        assert "p90" in percentiles
        # p75, p95, p99 should not be present
        assert "p75" not in percentiles


class TestExampleResultsAggregator:
    """Tests for ExampleResultsAggregator."""

    @pytest.mark.asyncio
    async def test_aggregate_single_result_set(self, sample_results: list[dict[str, Any]]) -> None:
        """Test aggregating a single result set."""
        aggregator = ExampleResultsAggregator()

        summary = await aggregator.aggregate([sample_results])

        assert summary["total_records"] == 4
        assert summary["total_sets"] == 1
        assert summary["records_per_set"] == 4.0

    @pytest.mark.asyncio
    async def test_aggregate_multiple_result_sets(
        self, sample_results: list[dict[str, Any]]
    ) -> None:
        """Test aggregating multiple result sets."""
        aggregator = ExampleResultsAggregator()

        summary = await aggregator.aggregate([sample_results, sample_results])

        assert summary["total_records"] == 8
        assert summary["total_sets"] == 2
        assert summary["records_per_set"] == 4.0

    @pytest.mark.asyncio
    async def test_calculate_success_rate(self, sample_results: list[dict[str, Any]]) -> None:
        """Test success rate calculation."""
        aggregator = ExampleResultsAggregator()

        summary = await aggregator.aggregate([sample_results])

        # 3 successful out of 4
        assert summary["success_rate"] == 0.75

    @pytest.mark.asyncio
    async def test_calculate_throughput(self, sample_results: list[dict[str, Any]]) -> None:
        """Test throughput calculation."""
        aggregator = ExampleResultsAggregator()

        summary = await aggregator.aggregate([sample_results])

        # Results span from 1000.0 to 1003.0 (3 seconds)
        # 4 results / 3 seconds = 1.33 RPS
        assert "throughput_rps" in summary
        assert summary["throughput_rps"] > 0

    @pytest.mark.asyncio
    async def test_aggregate_empty_results(self) -> None:
        """Test aggregating empty results."""
        aggregator = ExampleResultsAggregator()

        summary = await aggregator.aggregate([])

        assert "error" in summary

    @pytest.mark.asyncio
    async def test_generate_report(self, sample_results: list[dict[str, Any]]) -> None:
        """Test report generation."""
        aggregator = ExampleResultsAggregator()

        summary = await aggregator.aggregate([sample_results])
        report = await aggregator.generate_report(summary)

        assert "AIPerf Aggregated Results Report" in report
        assert "Total Records: 4" in report
        assert "Result Sets: 1" in report
        assert "Success Rate:" in report

    @pytest.mark.asyncio
    async def test_throughput_calculation_no_timestamps(self) -> None:
        """Test throughput returns None when no timestamps available."""
        aggregator = ExampleResultsAggregator()

        results = [
            {"id": 1, "status": "success"},
            {"id": 2, "status": "success"},
        ]

        summary = await aggregator.aggregate([results])

        assert "throughput_rps" not in summary

    @pytest.mark.asyncio
    async def test_success_rate_all_successful(self) -> None:
        """Test success rate when all results are successful."""
        aggregator = ExampleResultsAggregator()

        results = [
            {"id": 1, "status": "success"},
            {"id": 2, "status": "success"},
            {"id": 3, "status": "success"},
        ]

        summary = await aggregator.aggregate([results])

        assert summary["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_success_rate_all_failed(self) -> None:
        """Test success rate when all results failed."""
        aggregator = ExampleResultsAggregator()

        results = [
            {"id": 1, "error": "Failed"},
            {"id": 2, "error": "Failed"},
            {"id": 3, "error": "Failed"},
        ]

        summary = await aggregator.aggregate([results])

        assert summary["success_rate"] == 0.0
