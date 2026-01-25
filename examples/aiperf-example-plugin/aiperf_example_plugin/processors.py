# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example post-processors demonstrating 2025 Python best practices.

This module demonstrates:
- Pydantic models for data structures with validation
- Modern type hints with type aliases
- Async I/O with pathlib
- Comprehensive docstrings with examples
- Immutable frozen models for results

These processors can be chained together to build complex
analysis pipelines.

Example:
    >>> processor = ExampleMetricsProcessor.from_config(
    ...     output_file="/tmp/metrics.txt",
    ...     include_percentiles=True
    ... )
    >>> results = await processor.process(raw_results)
    >>> print(f"Processed {results.record_count} records")

    >>> aggregator = ExampleResultsAggregator()
    >>> summary = await aggregator.aggregate([results_phase1, results_phase2])
    >>> print(f"Throughput: {summary['throughput_rps']:.2f} req/s")
"""

from __future__ import annotations

import contextlib
import statistics
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, Field, field_validator

# Type aliases for better readability
ResultRecord = dict[str, Any]
ResultSet = list[ResultRecord]
Percentiles = dict[str, float]
RequestStats = dict[str, int | float]

__all__ = [
    "ProcessingResult",
    "ExampleMetricsProcessorConfig",
    "ExampleMetricsProcessor",
    "ExampleResultsAggregator",
]


class ProcessingResult(BaseModel):
    """Result of processing operations.

    This is an immutable frozen model to ensure result integrity.

    Attributes:
        success: Whether processing succeeded
        record_count: Number of records processed
        error_count: Number of errors encountered
        metrics: Dictionary of calculated metrics
        output_path: Path where results were written (if applicable)
    """

    success: bool = Field(description="Whether processing completed successfully")
    record_count: int = Field(default=0, description="Total number of records processed")
    error_count: int = Field(default=0, description="Number of errors encountered")
    metrics: dict[str, Any] | None = Field(default=None, description="Calculated metrics")
    output_path: str | None = Field(default=None, description="Output file path")

    model_config = {"frozen": True}


class ExampleMetricsProcessorConfig(BaseModel):
    """Configuration for ExampleMetricsProcessor.

    Attributes:
        output_file: Path where processed metrics will be written
        include_percentiles: Whether to calculate latency percentiles
        create_dirs: Whether to create parent directories
        percentiles: List of percentiles to calculate (0-100)
    """

    output_file: Path = Field(
        default=Path("/tmp/aiperf_metrics.txt"),
        description="Path to write processed metrics",
    )
    include_percentiles: bool = Field(
        default=True,
        description="Calculate latency percentiles",
    )
    create_dirs: bool = Field(
        default=True,
        description="Create parent directories if they don't exist",
    )
    percentiles: list[float] = Field(
        default=[50, 75, 90, 95, 99],
        description="Percentiles to calculate (0-100)",
    )

    @field_validator("output_file", mode="before")
    @classmethod
    def validate_output_file(cls, v: str | Path) -> Path:
        """Ensure output_file is a Path object.

        Args:
            v: Output file path as string or Path

        Returns:
            Path object for output file
        """
        return Path(v) if isinstance(v, str) else v

    @field_validator("percentiles")
    @classmethod
    def validate_percentiles(cls, v: list[float]) -> list[float]:
        """Validate percentile values are in range 0-100.

        Args:
            v: List of percentile values

        Returns:
            Validated percentile list

        Raises:
            ValueError: If any percentile is out of range
        """
        if not all(0 <= p <= 100 for p in v):
            raise ValueError("Percentiles must be between 0 and 100")
        return sorted(set(v))  # Remove duplicates and sort

    model_config = {"frozen": False}


class ExampleMetricsProcessor:
    """Example post-processor that calculates custom metrics.

    This processor demonstrates:
    - Pydantic configuration with validation
    - Comprehensive metric calculation
    - Type-safe result handling
    - Proper error handling

    Metrics calculated:
    - Request count statistics (total, successful, failed)
    - Latency percentiles (configurable)
    - Error rates
    - Success rates

    Example:
        >>> config = ExampleMetricsProcessorConfig(
        ...     output_file="/tmp/metrics.json",
        ...     include_percentiles=True,
        ...     percentiles=[50, 90, 95, 99]
        ... )
        >>> processor = ExampleMetricsProcessor(config)
        >>> result = await processor.process(results)
        >>> if result.success:
        ...     print(f"Processed {result.record_count} records")
        ...     print(f"Metrics: {result.metrics}")

    Thread Safety:
        This processor is not thread-safe. Create separate instances
        for concurrent processing.
    """

    def __init__(self, config: ExampleMetricsProcessorConfig) -> None:
        """Initialize metrics processor.

        Args:
            config: Processor configuration
        """
        self.config = config

        # Create parent directory if requested
        if self.config.create_dirs:
            self.config.output_file.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        output_file: str | Path = "/tmp/aiperf_metrics.txt",
        include_percentiles: bool = True,
        create_dirs: bool = True,
        percentiles: list[float] | None = None,
    ) -> ExampleMetricsProcessor:
        """Factory method to create processor from parameters.

        Args:
            output_file: Path to write processed metrics
            include_percentiles: Calculate latency percentiles
            create_dirs: Create parent directories
            percentiles: List of percentiles to calculate

        Returns:
            Configured ExampleMetricsProcessor instance

        Example:
            >>> processor = ExampleMetricsProcessor.from_config(
            ...     output_file="/tmp/metrics.txt",
            ...     percentiles=[50, 90, 95, 99]
            ... )
        """
        config = ExampleMetricsProcessorConfig(
            output_file=output_file,
            include_percentiles=include_percentiles,
            create_dirs=create_dirs,
            percentiles=percentiles or [50, 75, 90, 95, 99],
        )
        return cls(config)

    async def process(self, results: ResultSet) -> ProcessingResult:
        """Process results and calculate metrics.

        Args:
            results: List of result records to process

        Returns:
            ProcessingResult with calculated metrics

        Example:
            >>> results = [
            ...     {"latency_ms": 50.5, "status": "success"},
            ...     {"latency_ms": 45.2, "status": "success"},
            ... ]
            >>> result = await processor.process(results)
            >>> print(result.metrics["request_counts"]["total"])
        """
        if not results:
            return ProcessingResult(
                success=False,
                record_count=0,
                error_count=0,
                metrics=None,
            )

        try:
            metrics = await self._calculate_metrics(results)
            await self._write_metrics(metrics)

            return ProcessingResult(
                success=True,
                record_count=len(results),
                error_count=0,
                metrics=metrics,
                output_path=str(self.config.output_file),
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                record_count=0,
                error_count=1,
                metrics={"error": str(e)},
            )

    async def _calculate_metrics(self, results: ResultSet) -> dict[str, Any]:
        """Calculate metrics from results.

        Args:
            results: List of results

        Returns:
            Dictionary of calculated metrics
        """
        metrics: dict[str, Any] = {
            "total_records": len(results),
            "request_counts": self._calculate_request_stats(results),
        }

        # Calculate latency percentiles if data available
        if self.config.include_percentiles:
            latencies = self._extract_latencies(results)
            if latencies:
                metrics["latency_percentiles"] = self._calculate_percentiles(latencies)

        # Calculate error statistics
        error_count = sum(1 for r in results if r.get("error") or r.get("status") == "error")
        if error_count > 0:
            metrics["error_count"] = error_count
            metrics["error_rate"] = error_count / len(results)

        return metrics

    def _calculate_request_stats(self, results: ResultSet) -> RequestStats:
        """Calculate request count statistics.

        Args:
            results: List of results

        Returns:
            Dictionary with request statistics
        """
        successful = sum(1 for r in results if not r.get("error"))
        failed = sum(1 for r in results if r.get("error"))

        return {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results) if results else 0.0,
        }

    def _extract_latencies(self, results: ResultSet) -> list[float]:
        """Extract latency values from results.

        Supports multiple field names for compatibility.

        Args:
            results: List of results

        Returns:
            List of latency values (in milliseconds)
        """
        latencies: list[float] = []

        for result in results:
            # Support multiple latency field names
            for latency_field in ["latency_ms", "latency", "response_time"]:
                if latency_field in result:
                    with contextlib.suppress(TypeError, ValueError):
                        latencies.append(float(result[latency_field]))
                    break

        return latencies

    def _calculate_percentiles(self, values: list[float]) -> Percentiles:
        """Calculate percentiles for values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary of percentile values with min/max/mean/stdev
        """
        if not values:
            return {}

        sorted_values = sorted(values)
        percentile_results: Percentiles = {}

        # Calculate requested percentiles
        for p in self.config.percentiles:
            percentile_results[f"p{int(p)}"] = self._percentile(sorted_values, p)

        # Add summary statistics
        percentile_results.update(
            {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )

        return percentile_results

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        """Calculate percentile value using linear interpolation.

        Args:
            sorted_values: Sorted list of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not sorted_values:
            return 0.0

        index = (percentile / 100.0) * (len(sorted_values) - 1)

        if index % 1 == 0:
            return sorted_values[int(index)]

        # Linear interpolation between values
        lower_idx = int(index)
        upper_idx = lower_idx + 1
        fraction = index - lower_idx

        if upper_idx >= len(sorted_values):
            return sorted_values[lower_idx]

        lower_val = sorted_values[lower_idx]
        upper_val = sorted_values[upper_idx]

        return lower_val + (upper_val - lower_val) * fraction

    async def _write_metrics(self, metrics: dict[str, Any]) -> None:
        """Write metrics to output file.

        Uses orjson for fast, correct JSON serialization.

        Args:
            metrics: Metrics to write
        """
        try:
            json_bytes = orjson.dumps(metrics, option=orjson.OPT_INDENT_2)

            # Write header and metrics
            output = "=== AIPerf Example Plugin Metrics ===\n\n" + json_bytes.decode("utf-8") + "\n"

            self.config.output_file.write_text(output, encoding="utf-8")
        except OSError as e:
            print(f"Warning: Failed to write metrics file {self.config.output_file}: {e}")


class ExampleResultsAggregator:
    """Example aggregator for combining multiple result sets.

    This aggregator demonstrates:
    - Aggregation patterns for multi-phase results
    - Throughput calculation
    - Success rate analysis
    - Report generation

    Example:
        >>> aggregator = ExampleResultsAggregator()
        >>> summary = await aggregator.aggregate([
        ...     results_phase1,
        ...     results_phase2,
        ... ])
        >>> print(f"Throughput: {summary['throughput_rps']:.2f} req/s")
        >>> print(f"Success rate: {summary['success_rate']:.2%}")

        >>> report = await aggregator.generate_report(summary)
        >>> print(report)
    """

    async def aggregate(self, result_sets: list[ResultSet]) -> dict[str, Any]:
        """Aggregate multiple result sets.

        Args:
            result_sets: List of result lists to aggregate

        Returns:
            Aggregated statistics dictionary

        Example:
            >>> summary = await aggregator.aggregate([
            ...     phase1_results,
            ...     phase2_results,
            ... ])
        """
        if not result_sets:
            return {"error": "No results to aggregate"}

        all_results: ResultSet = []
        for results in result_sets:
            all_results.extend(results)

        aggregated: dict[str, Any] = {
            "total_records": len(all_results),
            "total_sets": len(result_sets),
            "records_per_set": len(all_results) / len(result_sets) if result_sets else 0.0,
            "success_rate": self._calculate_success_rate(all_results),
        }

        # Calculate throughput if timing data available
        throughput = self._calculate_throughput(all_results)
        if throughput is not None:
            aggregated["throughput_rps"] = throughput

        return aggregated

    def _calculate_success_rate(self, results: ResultSet) -> float:
        """Calculate overall success rate.

        Args:
            results: All results to analyze

        Returns:
            Success rate (0.0-1.0)
        """
        if not results:
            return 0.0

        successful = sum(1 for r in results if not r.get("error"))
        return successful / len(results)

    def _calculate_throughput(self, results: ResultSet) -> float | None:
        """Calculate throughput in requests per second.

        Args:
            results: Results with timing data

        Returns:
            Throughput in RPS, or None if timing data unavailable
        """
        timestamps: list[float] = []

        for result in results:
            if "timestamp" in result:
                with contextlib.suppress(TypeError, ValueError):
                    timestamps.append(float(result["timestamp"]))

        if not timestamps or len(timestamps) < 2:
            return None

        start_time = min(timestamps)
        end_time = max(timestamps)
        duration_sec = end_time - start_time

        if duration_sec <= 0:
            return None

        return len(results) / duration_sec

    async def generate_report(self, aggregated_stats: dict[str, Any]) -> str:
        """Generate human-readable report from aggregated stats.

        Args:
            aggregated_stats: Aggregated statistics dictionary

        Returns:
            Formatted report string

        Example:
            >>> summary = await aggregator.aggregate([results])
            >>> report = await aggregator.generate_report(summary)
            >>> print(report)
        """
        lines = [
            "=== AIPerf Aggregated Results Report ===",
            "",
            f"Total Records: {aggregated_stats.get('total_records', 0):,}",
            f"Result Sets: {aggregated_stats.get('total_sets', 0)}",
        ]

        success_rate = aggregated_stats.get("success_rate", 0.0)
        lines.append(f"Success Rate: {success_rate:.2%}")

        throughput = aggregated_stats.get("throughput_rps")
        if throughput:
            lines.append(f"Throughput: {throughput:.2f} req/s")

        return "\n".join(lines) + "\n"
