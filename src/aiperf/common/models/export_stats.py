# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export statistics computation for server metrics.

This module provides functions to compute statistics from time series data
directly into FlatSeriesStats, the canonical model for server metric statistics.

Functions:
- compute_gauge_stats(): Point-in-time sampled values
- compute_counter_stats(): Monotonically increasing totals with rate analysis
- compute_histogram_stats(): Value distributions with percentile estimation
- compute_summary_stats(): Server-computed quantiles

Legacy classes (GaugeExportStats, etc.) are kept for backward compatibility
but new code should use the compute_*_stats() functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.export_data import FlatSeriesStats
from aiperf.common.models.histogram_analysis import (
    accumulate_bucket_statistics,
)
from aiperf.common.models.histogram_percentiles import compute_estimated_percentiles

if TYPE_CHECKING:
    from aiperf.common.models.timeseries_storage import (
        HistogramTimeSeries,
        ScalarTimeSeries,
        SummaryTimeSeries,
        TimeRangeFilter,
    )


# =============================================================================
# Factory Functions - Compute stats directly into FlatSeriesStats
# =============================================================================


def compute_gauge_stats(
    ts: ScalarTimeSeries,
    time_filter: TimeRangeFilter | None = None,
    labels: dict[str, str] | None = None,
) -> FlatSeriesStats | None:
    """Compute gauge statistics from a ScalarTimeSeries.

    Gauges represent instantaneous values (e.g., current queue depth, cache usage %).
    Statistics are computed over all samples in the aggregation window.

    For constant gauges (std == 0), returns simplified stats with only avg and
    observation_count=1.

    Args:
        ts: The scalar time series to compute stats from
        time_filter: Optional time range filter
        labels: Optional labels for the time series

    Returns:
        FlatSeriesStats with gauge statistics, or None if no data in range
    """
    mask = ts.get_time_mask(time_filter)
    values = ts.values[mask]

    # Return None if time filter excludes all data
    if len(values) == 0:
        return None

    # Use sample std (ddof=1) for unbiased estimate; 0 for single sample
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # Simplified format for constant gauges
    if std == 0:
        return FlatSeriesStats(
            labels=labels,
            observation_count=1,
            avg=float(np.mean(values)),
        )

    pcts = np.percentile(values, [50, 90, 95, 99])

    return FlatSeriesStats(
        labels=labels,
        avg=float(np.mean(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        std=std,
        p50=float(pcts[0]),
        p90=float(pcts[1]),
        p95=float(pcts[2]),
        p99=float(pcts[3]),
        estimated_percentiles=False,
    )


def compute_counter_stats(
    ts: ScalarTimeSeries,
    time_filter: TimeRangeFilter | None = None,
    labels: dict[str, str] | None = None,
) -> FlatSeriesStats | None:
    """Compute counter statistics from a ScalarTimeSeries.

    Counters represent cumulative totals (e.g., total requests, total bytes).
    We report the delta and rate statistics over the aggregation window.

    Change-point detection: Rates are computed between points where the counter value
    actually changed, not between every sample. This avoids misleading statistics when
    sampling faster than the server updates.

    Args:
        ts: The scalar time series to compute stats from
        time_filter: Optional time range filter
        labels: Optional labels for the time series

    Returns:
        FlatSeriesStats with counter statistics, or None if no data in range
    """
    ref_idx = ts.get_reference_idx(time_filter)
    mask = ts.get_time_mask(time_filter)

    filtered_ts = ts.timestamps[mask]
    filtered_vals = ts.values[mask]

    # Return None if time filter excludes all data
    if len(filtered_vals) == 0:
        return None

    # Reference for delta calculation
    ref_value = (
        float(ts.values[ref_idx]) if ref_idx is not None else float(filtered_vals[0])
    )
    ref_ts = ts.timestamps[ref_idx] if ref_idx is not None else filtered_ts[0]

    # Total delta and duration
    total_delta = float(filtered_vals[-1]) - ref_value
    duration_ns = filtered_ts[-1] - ref_ts

    # Rate calculation - None if duration is zero
    if duration_ns <= 0:
        return FlatSeriesStats(
            labels=labels,
            delta=total_delta,
        )

    duration_s = duration_ns / NANOS_PER_SECOND

    # Build full series including reference point
    if ref_idx is not None:
        all_ts = np.concatenate([[ref_ts], filtered_ts])
        all_vals = np.concatenate([[ref_value], filtered_vals])
    else:
        all_ts, all_vals = filtered_ts, filtered_vals

    # Find change points (indices where value differs from previous)
    if len(all_vals) > 1:
        value_changed = np.diff(all_vals) != 0
        change_indices = np.concatenate([[0], np.where(value_changed)[0] + 1])

        if len(change_indices) > 1:
            change_ts = all_ts[change_indices]
            change_vals = all_vals[change_indices]

            deltas = np.diff(change_vals)
            time_deltas_ns = np.diff(change_ts)

            valid_mask = time_deltas_ns > 0
            if np.any(valid_mask):
                time_deltas_s = time_deltas_ns[valid_mask] / NANOS_PER_SECOND
                valid_deltas = deltas[valid_mask]
                rates = valid_deltas / time_deltas_s

                rate_avg = float(np.sum(valid_deltas) / np.sum(time_deltas_s))
                rate_std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0

                return FlatSeriesStats(
                    labels=labels,
                    delta=total_delta,
                    rate_per_second=total_delta / duration_s,
                    rate_avg=rate_avg,
                    rate_min=float(np.min(rates)),
                    rate_max=float(np.max(rates)),
                    rate_std=rate_std,
                )

    # Not enough change points for rate statistics
    return FlatSeriesStats(
        labels=labels,
        delta=total_delta,
        rate_per_second=total_delta / duration_s,
    )


def compute_histogram_stats(
    ts: HistogramTimeSeries,
    time_filter: TimeRangeFilter | None = None,
    labels: dict[str, str] | None = None,
) -> FlatSeriesStats:
    """Compute histogram statistics from a HistogramTimeSeries.

    Histograms track distributions (e.g., request latencies). We report:
    - Observation count and rate
    - Sum delta and rate
    - Average value per observation
    - Estimated percentiles using polynomial histogram algorithm
    - Raw bucket data for downstream analysis

    For histograms with no observations (observation_count == 0), returns
    simplified stats with only observation_count=0.

    Args:
        ts: The histogram time series to compute stats from
        time_filter: Optional time range filter
        labels: Optional labels for the time series

    Returns:
        FlatSeriesStats with histogram statistics
    """
    ref_idx, final_idx = ts.get_indices_for_filter(time_filter)

    # Reference values
    if ref_idx is not None:
        ref_sum = float(ts.sums[ref_idx])
        ref_count = float(ts.counts[ref_idx])
        ref_ts = ts.timestamps[ref_idx]
    else:
        ref_sum = float(ts.sums[0])
        ref_count = float(ts.counts[0])
        ref_ts = ts.timestamps[0]

    # Final values
    final_sum = float(ts.sums[final_idx])
    final_count = float(ts.counts[final_idx])
    final_ts = ts.timestamps[final_idx]
    final_buckets = (
        ts._bucket_snapshots[final_idx] if final_idx < len(ts._bucket_snapshots) else {}
    )

    # Compute deltas
    sum_delta = final_sum - ref_sum
    observation_count = int(final_count - ref_count)
    duration_ns = final_ts - ref_ts

    # Simplified format for empty histograms
    if observation_count == 0:
        return FlatSeriesStats(
            labels=labels,
            observation_count=0,
        )

    avg_value = sum_delta / observation_count
    duration_s = duration_ns / NANOS_PER_SECOND if duration_ns > 0 else 0
    observations_per_second = observation_count / duration_s if duration_s > 0 else None
    rate_per_second = sum_delta / duration_s if duration_s > 0 else None

    # Bucket delta calculation
    ref_bucket_idx = ref_idx if ref_idx is not None else 0
    ref_buckets = (
        ts._bucket_snapshots[ref_bucket_idx]
        if ref_bucket_idx < len(ts._bucket_snapshots)
        else {}
    )
    bucket_deltas: dict[str, int] | None = {}
    for le, final_val in final_buckets.items():
        ref_val = ref_buckets.get(le, 0.0)
        delta = final_val - ref_val
        if delta < 0:
            bucket_deltas = None
            break
        bucket_deltas[le] = int(delta)

    # Compute estimated percentiles
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None

    if bucket_deltas:
        start_idx = ref_idx if ref_idx is not None else 0
        bucket_stats = accumulate_bucket_statistics(
            ts.timestamps,
            ts.sums,
            ts.counts,
            ts._bucket_snapshots,
            start_idx=start_idx,
        )

        estimated = compute_estimated_percentiles(
            bucket_deltas=bucket_deltas,
            bucket_stats=bucket_stats,
            total_sum=sum_delta,
            total_count=observation_count,
        )

        if estimated:
            p50 = estimated.p50_estimate
            p90 = estimated.p90_estimate
            p95 = estimated.p95_estimate
            p99 = estimated.p99_estimate

    return FlatSeriesStats(
        labels=labels,
        observation_count=observation_count,
        avg=avg_value,
        delta=sum_delta,
        rate_per_second=rate_per_second,
        observations_per_second=observations_per_second,
        p50=p50,
        p90=p90,
        p95=p95,
        p99=p99,
        estimated_percentiles=True,
        buckets=bucket_deltas,
    )


def compute_summary_stats(
    ts: SummaryTimeSeries,
    time_filter: TimeRangeFilter | None = None,
    labels: dict[str, str] | None = None,
) -> FlatSeriesStats:
    """Compute summary statistics from a SummaryTimeSeries.

    Summaries provide pre-computed quantiles from the server. We report:
    - Observation count and rate
    - Sum delta and rate
    - Average value per observation
    - Server-computed quantiles (cumulative over server lifetime)

    For summaries with no observations (observation_count == 0), returns
    simplified stats with only observation_count=0.

    Args:
        ts: The summary time series to compute stats from
        time_filter: Optional time range filter
        labels: Optional labels for the time series

    Returns:
        FlatSeriesStats with summary statistics
    """
    ref_idx, final_idx = ts.get_indices_for_filter(time_filter)

    # Reference values
    if ref_idx is not None:
        ref_sum = float(ts.sums[ref_idx])
        ref_count = float(ts.counts[ref_idx])
        ref_ts = ts.timestamps[ref_idx]
    else:
        ref_sum = float(ts.sums[0])
        ref_count = float(ts.counts[0])
        ref_ts = ts.timestamps[0]

    final_sum = float(ts.sums[final_idx])
    final_count = float(ts.counts[final_idx])
    final_ts = ts.timestamps[final_idx]
    final_quantiles = ts._quantile_snapshots[final_idx]

    sum_delta = final_sum - ref_sum
    observation_count = int(final_count - ref_count)
    duration_ns = final_ts - ref_ts

    # Simplified format for empty summaries
    if observation_count == 0:
        return FlatSeriesStats(
            labels=labels,
            observation_count=0,
        )

    avg_value = sum_delta / observation_count
    duration_s = duration_ns / NANOS_PER_SECOND if duration_ns > 0 else 0
    observations_per_second = observation_count / duration_s if duration_s > 0 else None
    rate_per_second = sum_delta / duration_s if duration_s > 0 else None

    # Map quantiles to percentile fields
    quantiles = dict(final_quantiles)

    return FlatSeriesStats(
        labels=labels,
        observation_count=observation_count,
        avg=avg_value,
        delta=sum_delta,
        rate_per_second=rate_per_second,
        observations_per_second=observations_per_second,
        p50=quantiles.get("0.5"),
        p90=quantiles.get("0.9"),
        p95=quantiles.get("0.95"),
        p99=quantiles.get("0.99"),
        estimated_percentiles=False,
        quantiles=quantiles,
    )
