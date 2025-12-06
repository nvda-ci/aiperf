# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy-backed time series storage classes for server metrics.

This module provides efficient storage for time series data collected from
Prometheus endpoints. Each metric type has storage optimized for its semantics:

- ScalarTimeSeries: (timestamp, value) pairs for gauges and counters
- HistogramTimeSeries: (timestamp, sum, count) + bucket snapshots
- SummaryTimeSeries: (timestamp, sum, count) + quantile snapshots

Design principles:
1. Each metric type has storage optimized for its semantics
2. All types support time-based filtering (warmup exclusion, end buffer)
3. No global timestamp alignment - each metric is self-contained
4. NumPy arrays for memory efficiency and vectorized operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, model_validator

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.base_models import AIPerfBaseModel

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Snapshot Models (Point-in-time metric data)
# =============================================================================


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


# =============================================================================
# Time Series Storage Classes
# =============================================================================

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
    - All bucket snapshots for percentile computation

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
