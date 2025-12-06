# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, model_validator

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetailsCount
from aiperf.common.models.export_models import (
    CounterExportStats,
    GaugeExportStats,
    HistogramExportStats,
    InfoMetricData,
    ServerMetricsEndpointSummary,
    SummaryExportStats,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HistogramData(AIPerfBaseModel):
    """Structured histogram data with buckets, sum, and count."""

    buckets: dict[str, float] = Field(
        description='Bucket upper bounds (le="less than or equal") to counts. Keys are strings (e.g., "0.01", "0.1", "1.0")'
    )
    sum: float | None = Field(default=None, description="Sum of all observed values")
    count: float | None = Field(
        default=None, description="Total number of observations"
    )


class SummaryData(AIPerfBaseModel):
    """Structured summary data with quantiles, sum, and count."""

    quantiles: dict[str, float] = Field(
        description="Quantile to value {quantile: value}"
    )
    sum: float | None = Field(default=None, description="Sum of all observed values")
    count: float | None = Field(
        default=None, description="Total number of observations"
    )


class SlimMetricSample(AIPerfBaseModel):
    """Slim metric sample with minimal data using dictionary-based format.

    Optimized for JSONL export. Uses dictionary format for
    histogram/summary data for clarity:
    - Type and help text are in schema
    - Histogram bucket labels (le values) map to their counts
    - Summary quantile labels map to their values
    - sum/count are optional fields at sample level (used for histogram/summary)

    Format examples:
    - Counter/Gauge: {"value": 42.0} or {"labels": {...}, "value": 42.0}
    - Histogram: {"histogram": {"0.01": 10, "0.1": 25, "1.0": 50, ...}, "sum": 100.0, "count": 50}
    - Summary: {"summary": {"0.5": 0.1, "0.9": 0.5, "0.99": 1.0, ...}, "sum": 100.0, "count": 50}
    """

    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels (excluding histogram/summary special labels). None if no labels.",
    )
    value: float | None = Field(
        default=None, description="Simple metric value (counter/gauge)"
    )
    histogram: dict[str, float] | None = Field(
        default=None,
        description='Histogram bucket upper bounds (le="less than or equal") to counts. Keys are strings like "0.01", "0.1", "1.0"',
    )
    summary: dict[str, float] | None = Field(
        default=None,
        description="Summary quantile to value mapping",
    )
    sum: float | None = Field(
        default=None,
        description="Sum of all observed values (for histogram/summary)",
    )
    count: float | None = Field(
        default=None,
        description="Total number of observations (for histogram/summary)",
    )


class MetricSample(AIPerfBaseModel):
    """Single metric sample with labels and value."""

    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels (excluding histogram/summary special labels). None if no labels.",
    )
    value: float | None = Field(
        default=None, description="Simple metric value (counter/gauge)"
    )
    histogram: HistogramData | None = Field(
        default=None, description="Histogram data if metric is histogram type"
    )
    summary: SummaryData | None = Field(
        default=None, description="Summary data if metric is summary type"
    )

    def to_slim(self) -> SlimMetricSample:
        """Convert to slim metric sample format.

        For histograms and summaries, converts to dictionary format where
        bucket/quantile labels map to their counts/values.

        Returns:
            SlimMetricSample with dictionary-based histogram/summary data
        """
        if self.histogram:
            return SlimMetricSample(
                labels=self.labels,
                value=self.value,
                histogram=self.histogram.buckets,
                sum=self.histogram.sum,
                count=self.histogram.count,
            )

        if self.summary:
            return SlimMetricSample(
                labels=self.labels,
                value=self.value,
                summary=self.summary.quantiles,
                sum=self.summary.sum,
                count=self.summary.count,
            )

        return SlimMetricSample(
            labels=self.labels,
            value=self.value,
        )


class MetricFamily(AIPerfBaseModel):
    """Group of related metrics with same name and type."""

    type: PrometheusMetricType = Field(description="Metric type as enum")
    description: str = Field(description="Metric description from HELP text")
    samples: list[MetricSample] = Field(
        description="Metric samples grouped by base labels"
    )


class MetricSchema(AIPerfBaseModel):
    """Schema information for a metric (type and help text).

    Provides documentation for each metric collected from Prometheus endpoints.
    Sent once per metric in ServerMetricsMetadata to avoid repeating in every record.
    """

    type: PrometheusMetricType = Field(description="Metric type as enum")
    description: str = Field(description="Metric description from HELP text")


class ServerMetricsSlimRecord(AIPerfBaseModel):
    """Slim server metrics record containing only time-varying data.

    This record excludes static metadata (endpoint_url, metric types, help text)
    to reduce JSONL file size. The metadata and schemas are stored separately in the
    ServerMetricsMetadataFile.
    """

    endpoint_url: str = Field(
        description="Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')"
    )
    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when metrics were collected (time_ns)"
    )
    endpoint_latency_ns: int = Field(
        description="Nanoseconds it took to collect the metrics from the endpoint"
    )
    metrics: dict[str, list[SlimMetricSample]] = Field(
        description="Metrics grouped by family name, mapping directly to slim sample list"
    )


class ServerMetricsMetadata(AIPerfBaseModel):
    """Metadata for a server metrics endpoint that doesn't change over time.

    Includes metric schemas (type and help text) to avoid sending them in every record.
    Info metrics (ending in _info) are stored with their complete label keys and values,
    since they represent static information that doesn't change over time.
    """

    endpoint_url: str = Field(description="Prometheus metrics endpoint URL")
    info_metrics: dict[str, InfoMetricData] = Field(
        default_factory=dict,
        description="Info metrics (ending in _info) with complete label keys and values as reported by the Prometheus endpoint",
    )
    metric_schemas: dict[str, MetricSchema] = Field(
        default_factory=dict,
        description="Metric schemas (name, type, and help) as reported by the Prometheus endpoint",
    )


class ServerMetricsMetadataFile(AIPerfBaseModel):
    """Container for all server metrics endpoint metadata.

    This model represents the complete server_metrics_metadata.json file structure,
    mapping endpoint URLs to their metadata.
    """

    endpoints: dict[str, ServerMetricsMetadata] = Field(
        default_factory=dict,
        description="Dict mapping endpoint_url to ServerMetricsMetadata",
    )


class ServerMetricsRecord(AIPerfBaseModel):
    """Single server metrics data point from Prometheus endpoint.

    This record contains all metrics scraped from one Prometheus endpoint at one point in time.
    Used for hierarchical storage: endpoint_url -> time series data.
    """

    endpoint_url: str = Field(
        description="Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')"
    )
    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when metrics were collected (time_ns)"
    )
    endpoint_latency_ns: int = Field(
        description="Nanoseconds it took to collect the metrics from the endpoint"
    )
    metrics: dict[str, MetricFamily] = Field(
        description="Metrics grouped by family name"
    )

    def to_slim(self) -> ServerMetricsSlimRecord:
        """Convert to slim record using array-based format for histograms/summaries.

        Creates flat structure where metrics map directly to slim sample lists.
        For histograms and summaries, uses array format with bucket counts/quantile values
        and sum/count at the sample level.

        Excludes metrics ending in _info as they are stored separately in metadata.

        Returns:
            ServerMetricsSlimRecord with only timestamp and slim samples (flat structure)
        """
        slim_metrics = {
            name: [sample.to_slim() for sample in family.samples]
            for name, family in self.metrics.items()
            if not name.endswith("_info")
        }

        return ServerMetricsSlimRecord(
            timestamp_ns=self.timestamp_ns,
            endpoint_latency_ns=self.endpoint_latency_ns,
            endpoint_url=self.endpoint_url,
            metrics=slim_metrics,
        )

    def extract_metadata(self) -> ServerMetricsMetadata:
        """Extract metadata from this record.

        Extracts metric schemas (type and description) and separates _info metrics
        with their complete label data.

        Returns:
            ServerMetricsMetadata with schemas and info metrics
        """
        metric_schemas: dict[str, MetricSchema] = {}
        info_metrics: dict[str, InfoMetricData] = {}

        for metric_name, metric_family in self.metrics.items():
            if metric_name.endswith("_info"):
                labels_list = [
                    sample.labels if sample.labels else {}
                    for sample in metric_family.samples
                ]
                info_metrics[metric_name] = InfoMetricData(
                    description=metric_family.description,
                    labels=labels_list,
                )
            else:
                metric_schemas[metric_name] = MetricSchema(
                    type=metric_family.type,
                    description=metric_family.description,
                )

        return ServerMetricsMetadata(
            endpoint_url=self.endpoint_url,
            metric_schemas=metric_schemas,
            info_metrics=info_metrics,
        )


# ============================================================================
# Hierarchy Models for Aggregation
# ============================================================================


class HistogramSnapshot(AIPerfBaseModel):
    """Snapshot of histogram data at a point in time."""

    buckets: dict[str, float] = Field(
        description='Bucket upper bounds (le="less than or equal") to cumulative counts. Keys are strings like "0.01", "0.1", "1.0"'
    )
    sum: float | None = Field(default=None, description="Sum of all observed values")
    count: float | None = Field(
        default=None, description="Total number of observations"
    )


class SummarySnapshot(AIPerfBaseModel):
    """Snapshot of summary data at a point in time."""

    quantiles: dict[str, float] = Field(description="Quantile to value mapping")
    sum: float | None = Field(default=None, description="Sum of all observed values")
    count: float | None = Field(
        default=None, description="Total number of observations"
    )


class TimeRangeFilter(AIPerfBaseModel):
    """Filter for selecting metrics within a specific time range.

    Used to exclude warmup periods and end buffer times from aggregation.
    """

    start_ns: int | None = Field(
        default=None,
        description="Start of valid time range (exclusive of warmup). None means from beginning.",
    )
    end_ns: int | None = Field(
        default=None,
        description="End of valid time range (exclusive of flush buffer). None means to end.",
    )

    @model_validator(mode="after")
    def validate_range(self) -> TimeRangeFilter:
        """Validate that start_ns < end_ns if both are specified."""
        if (
            self.start_ns is not None
            and self.end_ns is not None
            and self.start_ns >= self.end_ns
        ):
            raise ValueError(
                f"start_ns ({self.start_ns}) must be less than end_ns ({self.end_ns})"
            )
        return self

    def includes(self, timestamp_ns: int) -> bool:
        """Check if a timestamp falls within this time range."""
        return not (
            (self.start_ns is not None and timestamp_ns < self.start_ns)
            or (self.end_ns is not None and timestamp_ns > self.end_ns)
        )


# ============================================================================
# Optimized Per-Metric Storage with NumPy Arrays
# ============================================================================
#
# Design principles:
# 1. Each metric type has storage optimized for its semantics
# 2. All types support time-based filtering (warmup exclusion, end buffer)
# 3. No global timestamp alignment - each metric is self-contained
# 4. NumPy arrays for memory efficiency and vectorized operations
#
# Storage by type:
# - Gauge: (timestamp, value) pairs → distribution stats over time
# - Counter: (timestamp, cumulative) pairs → rate distribution analysis
# - Histogram: (ts, sum, count) arrays + first/last buckets → avg value + rates
# - Summary: (ts, sum, count) arrays + quantile history → avg value + quantile trends
# ============================================================================

_INITIAL_CAPACITY = 256


class ScalarTimeSeries:
    """NumPy-backed (timestamp, value) storage for gauges and counters.

    Supports:
    - Time range filtering
    - Reference point lookup for delta calculations
    - Vectorized statistics computation
    """

    __slots__ = ("_timestamps", "_values", "_size")

    def __init__(self) -> None:
        self._timestamps: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.int64)
        self._values: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.float64)
        self._size: int = 0

    def append(self, timestamp_ns: int, value: float) -> None:
        if self._size >= len(self._values):
            new_cap = len(self._values) * 2
            new_ts = np.empty(new_cap, dtype=np.int64)
            new_val = np.empty(new_cap, dtype=np.float64)
            new_ts[: self._size] = self._timestamps[: self._size]
            new_val[: self._size] = self._values[: self._size]
            self._timestamps, self._values = new_ts, new_val
        self._timestamps[self._size] = timestamp_ns
        self._values[self._size] = value
        self._size += 1

    @property
    def timestamps(self) -> NDArray[np.int64]:
        return self._timestamps[: self._size]

    @property
    def values(self) -> NDArray[np.float64]:
        return self._values[: self._size]

    def __len__(self) -> int:
        return self._size

    def get_time_mask(self, time_filter: TimeRangeFilter | None) -> NDArray[np.bool_]:
        """Get boolean mask for points within time range."""
        if time_filter is None:
            return np.ones(self._size, dtype=bool)
        mask = np.ones(self._size, dtype=bool)
        ts = self.timestamps
        if time_filter.start_ns is not None:
            mask &= ts >= time_filter.start_ns
        if time_filter.end_ns is not None:
            mask &= ts <= time_filter.end_ns
        return mask

    def get_reference_idx(self, time_filter: TimeRangeFilter | None) -> int | None:
        """Get index of last point BEFORE time filter start (for delta calculation)."""
        if time_filter is None or time_filter.start_ns is None:
            return None
        candidates = np.where(self.timestamps < time_filter.start_ns)[0]
        return int(candidates[-1]) if len(candidates) > 0 else None


class HistogramTimeSeries:
    """Storage for histogram metrics optimized for rate analysis.

    Stores:
    - (timestamp, sum, count) as NumPy arrays for rate time-series
    - First + last bucket snapshots only (intermediates unnecessary for final stats)

    Enables:
    - Observation rate (count/sec) - e.g., requests/second
    - Value rate (sum/sec) - e.g., total latency/second
    - Average value (sum/count) - e.g., avg latency
    """

    __slots__ = (
        "_timestamps",
        "_sums",
        "_counts",
        "_size",
        "_bucket_snapshots",
    )

    def __init__(self) -> None:
        self._timestamps: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.int64)
        self._sums: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.float64)
        self._counts: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.float64)
        self._size: int = 0
        self._bucket_snapshots: list[dict[str, float]] = []

    def append(self, timestamp_ns: int, histogram: HistogramSnapshot) -> None:
        if self._size >= len(self._timestamps):
            new_cap = len(self._timestamps) * 2
            new_ts = np.empty(new_cap, dtype=np.int64)
            new_sums = np.empty(new_cap, dtype=np.float64)
            new_counts = np.empty(new_cap, dtype=np.float64)
            new_ts[: self._size] = self._timestamps[: self._size]
            new_sums[: self._size] = self._sums[: self._size]
            new_counts[: self._size] = self._counts[: self._size]
            self._timestamps, self._sums, self._counts = new_ts, new_sums, new_counts

        self._timestamps[self._size] = timestamp_ns
        self._sums[self._size] = histogram.sum or 0.0
        self._counts[self._size] = histogram.count or 0.0
        self._bucket_snapshots.append(dict(histogram.buckets))
        self._size += 1

    @property
    def timestamps(self) -> NDArray[np.int64]:
        return self._timestamps[: self._size]

    @property
    def sums(self) -> NDArray[np.float64]:
        return self._sums[: self._size]

    @property
    def counts(self) -> NDArray[np.float64]:
        return self._counts[: self._size]

    def __len__(self) -> int:
        return self._size

    def get_indices_for_filter(
        self, time_filter: TimeRangeFilter | None
    ) -> tuple[int | None, int]:
        """Get (reference_idx, final_idx) for time filter."""
        ts = self.timestamps
        ref_idx = None
        final_idx = self._size - 1

        if time_filter is not None:
            if time_filter.start_ns is not None:
                candidates = np.where(ts < time_filter.start_ns)[0]
                ref_idx = int(candidates[-1]) if len(candidates) > 0 else None
            if time_filter.end_ns is not None:
                candidates = np.where(ts <= time_filter.end_ns)[0]
                final_idx = (
                    int(candidates[-1]) if len(candidates) > 0 else self._size - 1
                )

        return ref_idx, final_idx

    def get_observation_rates(
        self, time_filter: TimeRangeFilter | None = None
    ) -> NDArray[np.float64]:
        """Get point-to-point observation rates (count deltas / time deltas).

        Zero-duration intervals are filtered out. Returns empty array if no valid rates.
        """
        ref_idx, final_idx = self.get_indices_for_filter(time_filter)
        start_idx = ref_idx if ref_idx is not None else 0

        ts = self.timestamps[start_idx : final_idx + 1]
        counts = self.counts[start_idx : final_idx + 1]

        if len(ts) < 2:
            return np.array([], dtype=np.float64)

        count_deltas = np.diff(counts)
        time_deltas_ns = np.diff(ts)

        # Filter out zero-duration intervals
        valid_mask = time_deltas_ns > 0
        if not np.any(valid_mask):
            return np.array([], dtype=np.float64)

        time_deltas_s = time_deltas_ns[valid_mask] / NANOS_PER_SECOND
        return count_deltas[valid_mask] / time_deltas_s


class SummaryTimeSeries:
    """Storage for summary metrics with quantile trend analysis.

    Stores:
    - (timestamp, sum, count) as NumPy arrays for rate analysis
    - All quantile snapshots for trend analysis (quantiles are NOT cumulative)

    Enables:
    - Observation rate (count/sec)
    - Average value (sum/count)
    - Quantile trends: How did p99 change over time?
    - Min/max/avg of each quantile
    """

    __slots__ = ("_timestamps", "_sums", "_counts", "_size", "_quantile_snapshots")

    def __init__(self) -> None:
        self._timestamps: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.int64)
        self._sums: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.float64)
        self._counts: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.float64)
        self._size: int = 0
        self._quantile_snapshots: list[dict[str, float]] = []

    def append(self, timestamp_ns: int, summary: SummarySnapshot) -> None:
        if self._size >= len(self._timestamps):
            new_cap = len(self._timestamps) * 2
            new_ts = np.empty(new_cap, dtype=np.int64)
            new_sums = np.empty(new_cap, dtype=np.float64)
            new_counts = np.empty(new_cap, dtype=np.float64)
            new_ts[: self._size] = self._timestamps[: self._size]
            new_sums[: self._size] = self._sums[: self._size]
            new_counts[: self._size] = self._counts[: self._size]
            self._timestamps, self._sums, self._counts = new_ts, new_sums, new_counts

        self._timestamps[self._size] = timestamp_ns
        self._sums[self._size] = summary.sum or 0.0
        self._counts[self._size] = summary.count or 0.0
        self._quantile_snapshots.append(dict(summary.quantiles))
        self._size += 1

    @property
    def timestamps(self) -> NDArray[np.int64]:
        return self._timestamps[: self._size]

    @property
    def sums(self) -> NDArray[np.float64]:
        return self._sums[: self._size]

    @property
    def counts(self) -> NDArray[np.float64]:
        return self._counts[: self._size]

    def __len__(self) -> int:
        return self._size

    def get_indices_for_filter(
        self, time_filter: TimeRangeFilter | None
    ) -> tuple[int | None, int]:
        """Get (reference_idx, final_idx) for time filter."""
        ts = self.timestamps
        ref_idx = None
        final_idx = self._size - 1

        if time_filter is not None:
            if time_filter.start_ns is not None:
                candidates = np.where(ts < time_filter.start_ns)[0]
                ref_idx = int(candidates[-1]) if len(candidates) > 0 else None
            if time_filter.end_ns is not None:
                candidates = np.where(ts <= time_filter.end_ns)[0]
                final_idx = (
                    int(candidates[-1]) if len(candidates) > 0 else self._size - 1
                )

        return ref_idx, final_idx

    def get_quantile_stats(
        self, quantile_key: str, time_filter: TimeRangeFilter | None = None
    ) -> dict[str, float] | None:
        """Get min/max/avg/last of a specific quantile over time."""
        ref_idx, final_idx = self.get_indices_for_filter(time_filter)
        start_idx = (ref_idx + 1) if ref_idx is not None else 0

        values = []
        for i in range(start_idx, final_idx + 1):
            if quantile_key in self._quantile_snapshots[i]:
                values.append(self._quantile_snapshots[i][quantile_key])

        if not values:
            return None

        arr = np.array(values)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "avg": float(np.mean(arr)),
            "last": float(arr[-1]),
        }


class ServerMetricsTimeSeries:
    """Optimized per-metric storage for server metrics.

    Design:
    - Each metric is self-contained with its own timestamps
    - No global alignment, no NaN padding for sparse data
    - NumPy arrays for memory efficiency and vectorized operations
    - Time filtering supported via index lookups

    Per-type optimization:
    - Gauges: Full (ts, value) history for distribution stats
    - Counters: Full (ts, value) history for rate distribution
    - Histograms: (ts, sum, count) history + first/last buckets only
    - Summaries: (ts, sum, count) history + full quantile history for trends
    """

    __slots__ = (
        "gauges",
        "counters",
        "histograms",
        "summaries",
        "first_timestamp_ns",
        "last_timestamp_ns",
        "_snapshot_count",
        "_scrape_latencies_ns",
    )

    def __init__(self) -> None:
        self.gauges: dict[str, ScalarTimeSeries] = {}
        self.counters: dict[str, ScalarTimeSeries] = {}
        self.histograms: dict[str, HistogramTimeSeries] = {}
        self.summaries: dict[str, SummaryTimeSeries] = {}
        self.first_timestamp_ns: int = 0
        self.last_timestamp_ns: int = 0
        self._snapshot_count: int = 0
        self._scrape_latencies_ns: list[int] = []

    def append_snapshot(
        self,
        timestamp_ns: int,
        gauge_metrics: dict[str, float] | None = None,
        counter_metrics: dict[str, float] | None = None,
        histogram_metrics: dict[str, HistogramSnapshot] | None = None,
        summary_metrics: dict[str, SummarySnapshot] | None = None,
        scrape_latency_ns: int | None = None,
    ) -> None:
        """Append all metrics from a single collection cycle."""
        if self._snapshot_count == 0:
            self.first_timestamp_ns = timestamp_ns
        self.last_timestamp_ns = timestamp_ns
        self._snapshot_count += 1
        if scrape_latency_ns is not None:
            self._scrape_latencies_ns.append(scrape_latency_ns)

        for key, value in (gauge_metrics or {}).items():
            self.gauges.setdefault(key, ScalarTimeSeries()).append(timestamp_ns, value)

        for key, value in (counter_metrics or {}).items():
            self.counters.setdefault(key, ScalarTimeSeries()).append(
                timestamp_ns, value
            )

        for key, histogram in (histogram_metrics or {}).items():
            self.histograms.setdefault(key, HistogramTimeSeries()).append(
                timestamp_ns, histogram
            )

        for key, summary in (summary_metrics or {}).items():
            self.summaries.setdefault(key, SummaryTimeSeries()).append(
                timestamp_ns, summary
            )

    def __len__(self) -> int:
        return self._snapshot_count

    def iter_export_stats(
        self, time_filter: TimeRangeFilter | None = None
    ) -> Iterator[
        tuple[
            str,
            PrometheusMetricType,
            GaugeExportStats
            | CounterExportStats
            | HistogramExportStats
            | SummaryExportStats,
        ]
    ]:
        """Iterate over all metrics, yielding (key, type, export_stats) tuples."""
        metric_configs = [
            (self.gauges, PrometheusMetricType.GAUGE, GaugeExportStats),
            (self.counters, PrometheusMetricType.COUNTER, CounterExportStats),
            (self.histograms, PrometheusMetricType.HISTOGRAM, HistogramExportStats),
            (self.summaries, PrometheusMetricType.SUMMARY, SummaryExportStats),
        ]

        for storage, metric_type, stats_cls in metric_configs:
            for key, ts in storage.items():
                if len(ts) > 0:
                    yield key, metric_type, stats_cls.from_time_series(ts, time_filter)


class ServerMetricsEndpointData(AIPerfBaseModel):
    """Complete server metrics data for one endpoint: metadata + time series storage.

    Uses NumPy-backed time series storage for efficient memory usage and fast aggregation.
    Provides both ExportStats API (optimized JSON aggregation) and MetricResult API (CSV export).
    """

    endpoint_url: str = Field(description="Source Prometheus metrics endpoint URL")
    metadata: ServerMetricsMetadata = Field(
        description="Static endpoint metadata (schemas, info metrics)"
    )
    time_series: ServerMetricsTimeSeries = Field(
        default_factory=ServerMetricsTimeSeries,
        description="NumPy-backed time series storage",
        exclude=True,  # Don't serialize (NumPy arrays not JSON-serializable)
    )

    model_config = {"arbitrary_types_allowed": True}

    def add_record(self, record: ServerMetricsRecord) -> None:
        """Add server metrics record to time series storage.

        Extracts all metric types (gauge, counter, histogram, summary) from the record.
        Metrics are keyed by "{metric_name}{label_suffix}" where label_suffix starts
        with "|" (e.g., "http_requests|method=GET,status=200").
        """
        gauge_metrics: dict[str, float] = {}
        counter_metrics: dict[str, float] = {}
        histogram_metrics: dict[str, HistogramSnapshot] = {}
        summary_metrics: dict[str, SummarySnapshot] = {}

        for metric_name, metric_family in record.metrics.items():
            if metric_name.endswith("_info"):
                continue

            metric_type = metric_family.type

            for sample in metric_family.samples:
                label_suffix = self._get_label_suffix(sample.labels)
                key = f"{metric_name}{label_suffix}"

                match metric_type:
                    case PrometheusMetricType.GAUGE:
                        if sample.value is not None:
                            gauge_metrics[key] = sample.value
                    case PrometheusMetricType.COUNTER:
                        if sample.value is not None:
                            counter_metrics[key] = sample.value
                    case PrometheusMetricType.HISTOGRAM:
                        if sample.histogram is not None:
                            histogram_metrics[key] = HistogramSnapshot(
                                buckets=sample.histogram.buckets,
                                sum=sample.histogram.sum,
                                count=sample.histogram.count,
                            )
                    case PrometheusMetricType.SUMMARY:
                        if sample.summary is not None:
                            summary_metrics[key] = SummarySnapshot(
                                quantiles=sample.summary.quantiles,
                                sum=sample.summary.sum,
                                count=sample.summary.count,
                            )
                    case _:
                        warnings.warn(
                            f"Unsupported metric type: {metric_type}",
                            stacklevel=2,
                        )
                        continue

        # Only add if we have any metrics
        if gauge_metrics or counter_metrics or histogram_metrics or summary_metrics:
            self.time_series.append_snapshot(
                timestamp_ns=record.timestamp_ns,
                gauge_metrics=gauge_metrics,
                counter_metrics=counter_metrics,
                histogram_metrics=histogram_metrics,
                summary_metrics=summary_metrics,
                scrape_latency_ns=record.endpoint_latency_ns,
            )

    def _get_label_suffix(self, labels: dict[str, str] | None) -> str:
        """Create a suffix string from labels for unique metric identification.

        Uses "|" as separator between metric name and labels to avoid ambiguity
        with underscores in metric names. Labels are sorted and comma-separated
        as key=value pairs. Empty string if no labels.

        Examples:
            No labels: "" (empty string)
            With labels: "|method=GET,status=200"
            Multiple labels: "|model_name=llama-8b,status=success"
        """
        if not labels:
            return ""
        sorted_items = sorted(labels.items())
        return "|" + ",".join(f"{k}={v}" for k, v in sorted_items)

    # ========================================================================
    # Export Stats Methods - Use optimized columnar storage
    # ========================================================================

    def get_gauge_export_stats(
        self, metric_key: str, time_filter: TimeRangeFilter | None = None
    ) -> GaugeExportStats:
        """Get GaugeExportStats for a specific gauge metric."""
        return self.time_series.to_gauge_export_stats(metric_key, time_filter)

    def get_counter_export_stats(
        self, metric_key: str, time_filter: TimeRangeFilter | None = None
    ) -> CounterExportStats:
        """Get CounterExportStats for a specific counter metric."""
        return self.time_series.to_counter_export_stats(metric_key, time_filter)

    def get_histogram_export_stats(
        self, metric_key: str, time_filter: TimeRangeFilter | None = None
    ) -> HistogramExportStats:
        """Get HistogramExportStats for a specific histogram metric."""
        return self.time_series.to_histogram_export_stats(metric_key, time_filter)

    def get_summary_export_stats(
        self, metric_key: str, time_filter: TimeRangeFilter | None = None
    ) -> SummaryExportStats:
        """Get SummaryExportStats for a specific summary metric."""
        return self.time_series.to_summary_export_stats(metric_key, time_filter)

    # ========================================================================
    # Available Metrics
    # ========================================================================

    def get_available_gauge_metrics(self) -> set[str]:
        """Get the set of available gauge metric names (including label suffixes)."""
        return set(self.time_series.gauges.keys())

    def get_available_counter_metrics(self) -> set[str]:
        """Get the set of available counter metric names (including label suffixes)."""
        return set(self.time_series.counters.keys())

    def get_available_histogram_metrics(self) -> set[str]:
        """Get the set of available histogram metric names (including label suffixes)."""
        return set(self.time_series.histograms.keys())

    def get_available_summary_metrics(self) -> set[str]:
        """Get the set of available summary metric names (including label suffixes)."""
        return set(self.time_series.summaries.keys())


class ServerMetricsHierarchy(AIPerfBaseModel):
    """Hierarchical storage: endpoint_url -> complete server metrics data.

    Structure:
    {
        "http://localhost:8081/metrics": ServerMetricsEndpointData(metadata + time series),
        "http://localhost:8082/metrics": ServerMetricsEndpointData(metadata + time series)
    }
    """

    endpoints: dict[str, ServerMetricsEndpointData] = Field(
        default_factory=dict,
        description="Dict mapping endpoint_url to server metrics data",
    )

    def add_record(self, record: ServerMetricsRecord) -> None:
        """Add server metrics record to hierarchical storage.

        Automatically creates new endpoints as needed.
        """
        url = record.endpoint_url

        if url not in self.endpoints:
            metadata = record.extract_metadata()
            self.endpoints[url] = ServerMetricsEndpointData(
                endpoint_url=url, metadata=metadata
            )
        else:
            existing = self.endpoints[url]
            new_metadata = record.extract_metadata()
            existing.metadata.metric_schemas.update(new_metadata.metric_schemas)
            existing.metadata.info_metrics.update(new_metadata.info_metrics)

        self.endpoints[url].add_record(record)


class ServerMetricsResults(AIPerfBaseModel):
    """Results from server metrics collection during a profile run.

    The hierarchy (server_metrics_data) contains raw NumPy arrays and is NOT serializable
    over ZMQ. Pre-computed summaries (endpoint_summaries) are computed in the subprocess
    and sent as JSON-serializable Pydantic models.
    """

    server_metrics_data: ServerMetricsHierarchy | None = Field(
        default=None,
        description="Hierarchical server metrics data (NOT sent over ZMQ - local use only)",
    )
    endpoint_summaries: dict[str, ServerMetricsEndpointSummary] | None = Field(
        default=None,
        description="Pre-computed endpoint summaries ready for export (sent over ZMQ)",
    )
    start_ns: int = Field(
        description="Start time of server metrics collection in nanoseconds"
    )
    end_ns: int = Field(
        description="End time of server metrics collection in nanoseconds"
    )
    endpoints_configured: list[str] = Field(
        default_factory=list,
        description="List of server metrics endpoint URLs in configured scope for display",
    )
    endpoints_successful: list[str] = Field(
        default_factory=list,
        description="List of server metrics endpoint URLs that successfully provided data",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )
    # Time filter for aggregation (excludes warmup and end buffer)
    aggregation_time_filter: TimeRangeFilter | None = Field(
        default=None,
        description="Time filter for aggregation, excluding warmup and end buffer periods",
    )


class ProcessServerMetricsResult(AIPerfBaseModel):
    """Result of server metrics processing - mirrors ProcessTelemetryResult pattern."""

    results: ServerMetricsResults = Field(
        description="The processed server metrics results"
    )
    errors: list = Field(
        default_factory=list,
        description="Any errors that occurred while processing server metrics data",
    )
