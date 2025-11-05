# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic base classes for hierarchical metrics storage.

This module provides type-parameterized base classes that eliminate duplication between
GPU telemetry and server metrics implementations. Both systems follow identical patterns:
    1. MetricSnapshot - all metrics at one timestamp
    2. MetricTimeSeries - chronological list of snapshots
    3. ResourceMetricsData - metadata + time series for one resource
    4. MetricsHierarchy - endpoint -> resource_id -> data

The generic base classes capture this common structure while allowing concrete
implementations to customize:
    - Record types (TelemetryRecord, ServerMetricRecord)
    - Metadata types (GpuMetadata, ServerMetadata)
    - Endpoint field names (dcgm_url, server_url)
    - Resource ID extraction logic (gpu_uuid, server_id)
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from pydantic import Field, field_serializer, field_validator

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.record_models import MetricResult

# Type variables for generic base classes
MetadataT = TypeVar("MetadataT", bound=AIPerfBaseModel)
RecordT = TypeVar("RecordT", bound=AIPerfBaseModel)


class HistogramBucket(AIPerfBaseModel):
    """Single bucket in a Prometheus histogram.

    Prometheus histograms use cumulative buckets where each bucket's count
    includes all observations less than or equal to the bucket's upper bound.
    """

    le: float = Field(
        description="Upper bound (less than or equal) for this bucket. +Inf is represented as float('inf')"
    )
    count: float = Field(
        description="Cumulative count of observations up to and including this bucket"
    )

    @field_serializer("le", when_used="json")
    def serialize_le(self, value: float) -> float | str:
        """Serialize le field, converting inf to string to avoid JSON null."""
        import math

        if math.isinf(value):
            return "+Inf" if value > 0 else "-Inf"
        return value

    @field_validator("le", mode="before")
    @classmethod
    def validate_le(cls, value: float | str | None) -> float:
        """Validate le field, converting string representations of inf to float."""

        if value is None:
            raise ValueError(
                "le cannot be None - infinity must be represented as float('inf') or '+Inf' string"
            )
        if isinstance(value, str):
            if value in ("+Inf", "Infinity", "inf"):
                return float("inf")
            if value in ("-Inf", "-Infinity", "-inf"):
                return float("-inf")
            return float(value)
        if isinstance(value, int | float):
            return float(value)
        raise ValueError(f"Invalid le value: {value!r}")


class HistogramSnapshot(AIPerfBaseModel):
    """Complete histogram data at a single point in time.

    Captures the full distribution of observations including all bucket counts,
    sum, and total count. This allows accurate percentile calculation from
    bucket boundaries using linear interpolation.
    """

    buckets: list[HistogramBucket] = Field(
        description="Ordered list of histogram buckets (sorted by le)"
    )
    sum: float = Field(description="Sum of all observed values")
    count: int = Field(description="Total number of observations")
    created_at_ns: int | None = Field(
        default=None,
        description="Unix timestamp in nanoseconds when this histogram was first created (from Prometheus _created metric)",
    )


@dataclass(slots=True)
class HistogramComponents:
    """Mutable accumulator for histogram components during parsing.

    This replaces dict[str, Any] with a typed structure for accumulating
    histogram data from Prometheus metrics before creating HistogramSnapshot.

    Used during metric parsing to collect:
    - Individual bucket observations from _bucket metrics
    - Sum from _sum metric
    - Count from _count metric
    - Creation timestamp from _created metric

    After parsing completes, convert to HistogramSnapshot for immutable storage.
    """

    buckets: list[tuple[float, float]]
    sum: float = 0.0
    count: int = 0
    created_at_ns: int | None = None

    def to_snapshot(self) -> HistogramSnapshot:
        """Convert to immutable HistogramSnapshot.

        Sorts buckets by le value and creates HistogramBucket objects.
        Filters out any buckets with invalid (None) le values as a safety measure.

        Returns:
            HistogramSnapshot with sorted buckets ready for percentile computation
        """
        # Filter out buckets with None le values (defensive programming)
        valid_buckets = [(le, count) for le, count in self.buckets if le is not None]
        sorted_buckets = sorted(valid_buckets, key=lambda x: x[0])
        return HistogramSnapshot(
            buckets=[
                HistogramBucket(le=le, count=count) for le, count in sorted_buckets
            ],
            sum=self.sum,
            count=self.count,
            created_at_ns=self.created_at_ns,
        )


@dataclass(slots=True, frozen=True)
class HistogramCollection:
    """Collection of histograms keyed by metric name.

    This is a high-performance, immutable collection that replaces dict[str, HistogramSnapshot].
    Uses dataclass with slots for memory efficiency and frozen=True for immutability.

    Provides dict-like interface for compatibility with existing code:
    - collection[name] - access by name
    - name in collection - membership test
    - collection.keys() - iterate over names
    - collection.items() - iterate over (name, histogram) pairs
    - collection.values() - iterate over histograms
    - len(collection) - number of histograms
    """

    _data: dict[str, HistogramSnapshot]

    def __getitem__(self, name: str) -> HistogramSnapshot:
        """Access histogram by name."""
        return self._data[name]

    def __contains__(self, name: str) -> bool:
        """Check if histogram name exists in collection."""
        return name in self._data

    def keys(self) -> Iterator[str]:
        """Return iterator over histogram names."""
        return iter(self._data.keys())

    def values(self) -> Iterator[HistogramSnapshot]:
        """Return iterator over histogram snapshots."""
        return iter(self._data.values())

    def items(self) -> Iterator[tuple[str, HistogramSnapshot]]:
        """Return iterator over (name, histogram) pairs."""
        return iter(self._data.items())

    def get(
        self, name: str, default: HistogramSnapshot | None = None
    ) -> HistogramSnapshot | None:
        """Get histogram by name with optional default."""
        return self._data.get(name, default)

    def __len__(self) -> int:
        """Return number of histograms in collection."""
        return len(self._data)

    def __bool__(self) -> bool:
        """Return True if collection has any histograms."""
        return bool(self._data)

    @classmethod
    def from_dict(cls, data: dict[str, HistogramSnapshot]) -> "HistogramCollection":
        """Create collection from dictionary."""
        return cls(_data=data)

    @classmethod
    def empty(cls) -> "HistogramCollection":
        """Create empty collection."""
        return cls(_data={})


def compute_histogram_percentile(
    buckets: list[HistogramBucket], percentile: float, total_count: int
) -> float:
    """Compute percentile from histogram buckets using linear interpolation.

    Args:
        buckets: Ordered list of cumulative histogram buckets (sorted by le)
        percentile: Percentile to compute (0-100, e.g., 50 for median, 99 for p99)
        total_count: Total number of observations in the histogram

    Returns:
        Estimated value at the given percentile

    Note:
        Uses linear interpolation within buckets. If all observations fall into
        the first bucket, returns the bucket's upper bound. For percentiles beyond
        the data, returns the last finite bucket boundary or 0 if none exists.
    """
    if not buckets or total_count == 0:
        return 0.0

    target_count = total_count * (percentile / 100.0)

    # Find the bucket containing the target percentile
    prev_bucket = None
    for bucket in buckets:
        if bucket.count >= target_count:
            # Target falls in this bucket - interpolate between prev and current
            if prev_bucket is None:
                # All observations in first bucket, return its upper bound
                return bucket.le if bucket.le != float("inf") else 0.0

            # Linear interpolation between bucket boundaries
            count_in_bucket = bucket.count - prev_bucket.count
            if count_in_bucket == 0:
                return prev_bucket.le

            # How far into this bucket is our target?
            position_in_bucket = (target_count - prev_bucket.count) / count_in_bucket

            # Interpolate between previous bucket's upper bound and current bucket's upper bound
            lower_bound = prev_bucket.le
            upper_bound = bucket.le if bucket.le != float("inf") else lower_bound * 2

            return lower_bound + position_in_bucket * (upper_bound - lower_bound)

        prev_bucket = bucket

    # Target percentile is beyond all data - return last finite bucket boundary
    for bucket in reversed(buckets):
        if bucket.le != float("inf"):
            return bucket.le

    return 0.0


def compute_histogram_statistics(histogram: HistogramSnapshot) -> dict[str, float]:
    """Compute statistical summary from a histogram snapshot.

    Args:
        histogram: Histogram snapshot with buckets, sum, and count

    Returns:
        Dictionary with keys: min, max, avg, p1, p5, p10, p25, p50, p75, p90, p95, p99

    Note:
        - avg is computed from sum/count
        - min is estimated as 0 or first bucket with non-zero count
        - max is estimated from highest bucket with count
        - percentiles are computed via linear interpolation
    """
    if histogram.count == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
            "p1": 0.0,
            "p5": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    avg = histogram.sum / histogram.count

    # Estimate min from first bucket with observations
    min_val = 0.0
    for i, bucket in enumerate(histogram.buckets):
        prev_count = histogram.buckets[i - 1].count if i > 0 else 0
        if bucket.count > prev_count:
            min_val = histogram.buckets[i - 1].le if i > 0 else 0.0
            break

    # Estimate max from highest bucket boundary with observations
    max_val = 0.0
    for bucket in reversed(histogram.buckets):
        if bucket.count > 0 and bucket.le != float("inf"):
            max_val = bucket.le
            break

    # Compute percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = {
        f"p{p}": compute_histogram_percentile(histogram.buckets, p, histogram.count)
        for p in percentiles
    }

    return {
        "min": min_val,
        "max": max_val,
        "avg": avg,
        **percentile_values,
    }


class BaseMetricSnapshot(AIPerfBaseModel):
    """All metrics for a single resource at one point in time.

    Groups all metric values collected during a single collection cycle,
    eliminating timestamp duplication across individual metrics.

    Supports both scalar metrics (gauges, counters) and histogram distributions.

    Generic base for GpuTelemetrySnapshot and ServerMetricSnapshot.
    """

    timestamp_ns: int = Field(description="Collection timestamp for all metrics")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="All scalar metric values at this timestamp"
    )
    histograms: dict[str, HistogramSnapshot] = Field(
        default_factory=dict,
        description="All histogram distributions at this timestamp",
    )


class BaseMetricTimeSeries(AIPerfBaseModel):
    """Time series data for all metrics on a single resource.

    Uses grouped snapshots instead of individual metric time series to eliminate
    timestamp duplication and improve storage efficiency.

    Generic base for GpuMetricTimeSeries and ServerMetricTimeSeries.
    """

    snapshots: list[BaseMetricSnapshot] = Field(
        default_factory=list, description="Chronological snapshots of all metrics"
    )

    def append_snapshot(
        self,
        metrics: dict[str, float],
        timestamp_ns: int,
        histograms: dict[str, HistogramSnapshot] | None = None,
    ) -> None:
        """Add new snapshot with all metrics at once.

        Args:
            metrics: Dictionary of metric_name -> value for this timestamp
            timestamp_ns: Timestamp when measurements were taken
            histograms: Optional dictionary of histogram_name -> HistogramSnapshot
        """
        snapshot = BaseMetricSnapshot(
            timestamp_ns=timestamp_ns,
            metrics={k: v for k, v in metrics.items() if v is not None},
            histograms=histograms or {},
        )
        self.snapshots.append(snapshot)

    def get_metric_values(self, metric_name: str) -> list[tuple[float, int]]:
        """Extract time series data for a specific metric.

        Args:
            metric_name: Name of the metric to extract

        Returns:
            List of (value, timestamp_ns) tuples for the specified metric
        """
        return [
            (snapshot.metrics[metric_name], snapshot.timestamp_ns)
            for snapshot in self.snapshots
            if metric_name in snapshot.metrics
        ]

    def to_metric_result(
        self, metric_name: str, tag: str, header: str, unit: str
    ) -> MetricResult:
        """Convert metric time series to MetricResult with statistical summary.

        Args:
            metric_name: Name of the metric to analyze
            tag: Unique identifier for this metric (used by dashboard, exports, API)
            header: Human-readable name for display
            unit: Unit of measurement (e.g., "W" for Watts, "%" for percentage)

        Returns:
            MetricResult with min/max/avg/percentiles computed from time series

        Raises:
            NoMetricValue: If no data points are available for the specified metric
        """
        data_points = self.get_metric_values(metric_name)

        if not data_points:
            raise NoMetricValue(f"No metric data available for metric '{metric_name}'")

        values = np.array([point[0] for point in data_points])
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
            values, [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )

        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=np.min(values),
            max=np.max(values),
            avg=float(np.mean(values)),
            std=float(np.std(values)),
            count=len(values),
            current=float(data_points[-1][0]),
            p1=p1,
            p5=p5,
            p10=p10,
            p25=p25,
            p50=p50,
            p75=p75,
            p90=p90,
            p95=p95,
            p99=p99,
        )

    def get_histogram_values(
        self, histogram_name: str
    ) -> list[tuple[HistogramSnapshot, int]]:
        """Extract time series data for a specific histogram.

        Args:
            histogram_name: Name of the histogram to extract

        Returns:
            List of (HistogramSnapshot, timestamp_ns) tuples for the specified histogram
        """
        return [
            (snapshot.histograms[histogram_name], snapshot.timestamp_ns)
            for snapshot in self.snapshots
            if histogram_name in snapshot.histograms
        ]

    def histogram_to_metric_result(
        self, histogram_name: str, tag: str, header: str, unit: str
    ) -> MetricResult:
        """Convert histogram time series to MetricResult with statistical summary.

        For histograms, statistics are computed from the most recent snapshot's buckets,
        as histogram buckets already represent the distribution of all observations.
        Uses linear interpolation between bucket boundaries for accurate percentiles.

        Args:
            histogram_name: Name of the histogram to analyze
            tag: Unique identifier for this metric (used by dashboard, exports, API)
            header: Human-readable name for display
            unit: Unit of measurement (e.g., "s" for seconds, "bytes" for bytes)

        Returns:
            MetricResult with min/max/avg/percentiles computed from histogram buckets

        Raises:
            NoMetricValue: If no histogram data is available for the specified metric
        """
        histogram_data = self.get_histogram_values(histogram_name)

        if not histogram_data:
            raise NoMetricValue(
                f"No histogram data available for metric '{histogram_name}'"
            )

        # Use the most recent histogram snapshot for statistics
        latest_histogram, _ = histogram_data[-1]

        # Compute statistics from histogram buckets
        stats = compute_histogram_statistics(latest_histogram)

        # For std and current, we use approximations since histograms don't provide these directly
        # std is not available from histogram buckets, so we use 0.0
        # current is approximated as the average
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=stats["min"],
            max=stats["max"],
            avg=stats["avg"],
            std=0.0,  # Not available from histogram buckets
            count=latest_histogram.count,
            current=stats["avg"],  # Approximate as average
            p1=stats["p1"],
            p5=stats["p5"],
            p10=stats["p10"],
            p25=stats["p25"],
            p50=stats["p50"],
            p75=stats["p75"],
            p90=stats["p90"],
            p95=stats["p95"],
            p99=stats["p99"],
        )


class BaseResourceMetricsData(AIPerfBaseModel, Generic[MetadataT]):
    """Complete metrics data for one resource: metadata + grouped metric time series.

    This combines static resource information (GPU, server, etc.) with dynamic
    time-series data, providing the complete picture for one resource's metrics
    using efficient grouped snapshots.

    Generic base for GpuTelemetryData and ServerMetricsData.

    Type Parameters:
        MetadataT: The metadata model type (e.g., GpuMetadata, ServerMetadata)
    """

    metadata: MetadataT = Field(description="Static resource information")
    time_series: BaseMetricTimeSeries = Field(
        default_factory=BaseMetricTimeSeries,
        description="Grouped time series for all metrics",
    )

    def get_metric_result(
        self, metric_name: str, tag: str, header: str, unit: str
    ) -> MetricResult:
        """Get MetricResult for a specific metric.

        Args:
            metric_name: Name of the metric to analyze
            tag: Unique identifier for this metric
            header: Human-readable name for display
            unit: Unit of measurement

        Returns:
            MetricResult with statistical summary for the specified metric
        """
        return self.time_series.to_metric_result(metric_name, tag, header, unit)
