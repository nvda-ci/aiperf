# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export statistics models for server metrics.

This module provides type-specific statistics models for Prometheus metric types:
- GaugeExportStats: Point-in-time sampled values
- CounterExportStats: Monotonically increasing totals with rate analysis
- HistogramExportStats: Value distributions with percentile estimation
- SummaryExportStats: Server-computed quantiles

Each model includes a from_time_series() classmethod to create statistics
from the corresponding time series storage class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from pydantic import Field

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.base_models import AIPerfBaseModel
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
# Gauge Export Stats
# =============================================================================


class GaugeExportStats(AIPerfBaseModel):
    """Export statistics for gauge metrics - point-in-time sampled values.

    Gauges represent instantaneous values (e.g., current queue depth, cache usage %).
    Statistics are computed over all samples in the aggregation window.
    """

    min: float = Field(description="Minimum observed value")
    avg: float = Field(description="Average value across all samples")
    p50: float = Field(description="50th percentile (median)")
    p90: float = Field(description="90th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")
    max: float = Field(description="Maximum observed value")
    std: float = Field(description="Standard deviation of values")

    @classmethod
    def from_time_series(
        cls, ts: ScalarTimeSeries, time_filter: TimeRangeFilter | None = None
    ) -> GaugeExportStats:
        """Create GaugeExportStats from a ScalarTimeSeries."""
        mask = ts.get_time_mask(time_filter)
        values = ts.values[mask]

        pcts = np.percentile(values, [50, 90, 95, 99])

        # Use sample std (ddof=1) for unbiased estimate; 0 for single sample
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        return cls(
            avg=float(np.mean(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            std=std,
            p50=float(pcts[0]),
            p90=float(pcts[1]),
            p95=float(pcts[2]),
            p99=float(pcts[3]),
        )


# =============================================================================
# Counter Export Stats
# =============================================================================


class CounterExportStats(AIPerfBaseModel):
    """Export statistics for counter metrics - monotonically increasing totals.

    Counters represent cumulative totals (e.g., total requests, total bytes).
    We report the delta and rate statistics over the aggregation window.

    Note on rate metrics:
    - rate_overall: Overall throughput (delta/duration) - always available if duration > 0
    - rate_avg/min/max/std: Statistics computed between *change points* only

    Change-point detection: Rates are computed between points where the counter value
    actually changed, not between every sample. This avoids misleading statistics when
    sampling faster than the server updates (e.g., sampling at 10Hz when server updates
    at 1Hz would otherwise show many 0/s rates followed by a spike).

    Rate fields are None when:
    - Duration is zero (insufficient time)
    - No value changes occurred (nothing to compute rates from)
    """

    # Delta statistics
    delta: float = Field(description="Change over the aggregation period")
    # Overall rate (best measure of throughput)
    rate_overall: float | None = Field(
        default=None, description="Overall rate per second (delta/duration)"
    )
    # Instantaneous rate statistics (from rates between change points)
    rate_avg: float | None = Field(
        default=None, description="Time-weighted average rate between change points"
    )
    rate_min: float | None = Field(
        default=None, description="Minimum point-to-point rate per second"
    )
    rate_max: float | None = Field(
        default=None, description="Maximum point-to-point rate per second"
    )
    rate_std: float | None = Field(
        default=None, description="Standard deviation of point-to-point rates"
    )

    @classmethod
    def from_time_series(
        cls, ts: ScalarTimeSeries, time_filter: TimeRangeFilter | None = None
    ) -> CounterExportStats:
        """Create CounterExportStats from a ScalarTimeSeries."""
        ref_idx = ts.get_reference_idx(time_filter)
        mask = ts.get_time_mask(time_filter)

        filtered_ts = ts.timestamps[mask]
        filtered_vals = ts.values[mask]

        # Reference for delta calculation
        ref_value = (
            float(ts.values[ref_idx])
            if ref_idx is not None
            else float(filtered_vals[0])
        )
        ref_ts = ts.timestamps[ref_idx] if ref_idx is not None else filtered_ts[0]

        # Total delta and duration
        total_delta = float(filtered_vals[-1]) - ref_value
        duration_ns = filtered_ts[-1] - ref_ts

        # Rate calculation - None if duration is zero
        if duration_ns <= 0:
            return cls(
                delta=total_delta,
                rate_overall=None,
                rate_avg=None,
                rate_min=None,
                rate_max=None,
                rate_std=None,
            )

        duration_s = duration_ns / NANOS_PER_SECOND

        # Build full series including reference point
        if ref_idx is not None:
            all_ts = np.concatenate([[ref_ts], filtered_ts])
            all_vals = np.concatenate([[ref_value], filtered_vals])
        else:
            all_ts, all_vals = filtered_ts, filtered_vals

        # Find change points (indices where value differs from previous)
        # This avoids the "0/s 0/s 0/s 1000/s" problem when sampling faster than server updates
        if len(all_vals) > 1:
            # Always include first point, then points where value changed
            value_changed = np.diff(all_vals) != 0
            change_indices = np.concatenate([[0], np.where(value_changed)[0] + 1])

            if len(change_indices) > 1:
                # Extract timestamps and values at change points
                change_ts = all_ts[change_indices]
                change_vals = all_vals[change_indices]

                # Compute rates between consecutive change points
                deltas = np.diff(change_vals)
                time_deltas_ns = np.diff(change_ts)

                # Filter out any zero-duration intervals (shouldn't happen, but safety)
                valid_mask = time_deltas_ns > 0
                if np.any(valid_mask):
                    time_deltas_s = time_deltas_ns[valid_mask] / NANOS_PER_SECOND
                    valid_deltas = deltas[valid_mask]
                    rates = valid_deltas / time_deltas_s

                    # Time-weighted average: sum(deltas) / sum(durations)
                    # This weights each rate by how long that rate was observed
                    rate_avg = float(np.sum(valid_deltas) / np.sum(time_deltas_s))

                    # Use sample std (ddof=1) for unbiased estimate; 0 for single rate
                    rate_std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0

                    return cls(
                        delta=total_delta,
                        rate_overall=total_delta / duration_s,
                        rate_avg=rate_avg,
                        rate_min=float(np.min(rates)),
                        rate_max=float(np.max(rates)),
                        rate_std=rate_std,
                    )

        # Not enough change points for rate statistics
        return cls(
            delta=total_delta,
            rate_overall=total_delta / duration_s,
            rate_avg=None,
            rate_min=None,
            rate_max=None,
            rate_std=None,
        )


# =============================================================================
# Histogram Export Stats
# =============================================================================


class HistogramExportStats(AIPerfBaseModel):
    """Export statistics for histogram metrics - value distribution + rates.

    Histograms track distributions (e.g., request latencies). We report:
    - Delta stats: count_delta, sum_delta, avg over the aggregation period
    - Rate: observations per second
    - Estimated percentiles: Polynomial histogram algorithm with +Inf bucket estimation
    - Raw bucket data for downstream analysis
    """

    # Delta statistics over the aggregation period
    count_delta: int = Field(
        description="Change in observation count over the aggregation period"
    )
    count_rate: float | None = Field(
        default=None,
        description="Observations per second (count_delta/duration)",
    )
    sum_delta: float = Field(
        description="Change in sum of observed values over the aggregation period"
    )
    sum_rate: float | None = Field(
        default=None,
        description="Sum per second (sum_delta/duration). Useful for token/byte histograms.",
    )
    avg: float = Field(
        description="Average value per observation (sum_delta/count_delta)"
    )
    # Estimated percentiles using polynomial histogram algorithm
    p50_estimate: float | None = Field(
        default=None, description="Estimated 50th percentile (median)"
    )
    p90_estimate: float | None = Field(
        default=None, description="Estimated 90th percentile"
    )
    p95_estimate: float | None = Field(
        default=None, description="Estimated 95th percentile"
    )
    p99_estimate: float | None = Field(
        default=None, description="Estimated 99th percentile"
    )
    # Raw bucket data for custom analysis (None if counter reset detected)
    buckets: dict[str, int] | None = Field(
        default=None,
        description='Bucket upper bounds (le="less than or equal") to delta counts. None if counter reset detected during collection.',
    )

    @classmethod
    def from_time_series(
        cls, ts: HistogramTimeSeries, time_filter: TimeRangeFilter | None = None
    ) -> HistogramExportStats:
        """Create HistogramExportStats from a HistogramTimeSeries."""
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
            ts._bucket_snapshots[final_idx]
            if final_idx < len(ts._bucket_snapshots)
            else {}
        )

        # Compute deltas
        sum_delta = final_sum - ref_sum
        count_delta = final_count - ref_count
        duration_ns = final_ts - ref_ts

        avg_value = sum_delta / count_delta if count_delta > 0 else 0.0
        duration_s = duration_ns / NANOS_PER_SECOND if duration_ns > 0 else 0
        count_rate = count_delta / duration_s if duration_s > 0 else None
        sum_rate = sum_delta / duration_s if duration_s > 0 else None

        # Bucket delta calculation
        # If any delta is negative (counter reset), return None for buckets
        # since the data is invalid/incomplete
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
                # Counter reset detected - data is invalid
                bucket_deltas = None
                break
            bucket_deltas[le] = int(delta)

        # Compute estimated percentiles using polynomial histogram algorithm
        p50_estimate: float | None = None
        p90_estimate: float | None = None
        p95_estimate: float | None = None
        p99_estimate: float | None = None

        if bucket_deltas:
            # Learn per-bucket means from single-bucket scrape intervals
            start_idx = ref_idx if ref_idx is not None else 0
            bucket_stats = accumulate_bucket_statistics(
                ts.timestamps,
                ts.sums,
                ts.counts,
                ts._bucket_snapshots,
                start_idx=start_idx,
            )

            # Compute estimated percentiles including +Inf bucket estimation
            estimated = compute_estimated_percentiles(
                bucket_deltas=bucket_deltas,
                bucket_stats=bucket_stats,
                total_sum=sum_delta,
                total_count=int(count_delta),
            )

            if estimated:
                p50_estimate = estimated.p50_estimate
                p90_estimate = estimated.p90_estimate
                p95_estimate = estimated.p95_estimate
                p99_estimate = estimated.p99_estimate

        return cls(
            count_delta=int(count_delta),
            sum_delta=sum_delta,
            avg=avg_value,
            count_rate=count_rate,
            sum_rate=sum_rate,
            p50_estimate=p50_estimate,
            p90_estimate=p90_estimate,
            p95_estimate=p95_estimate,
            p99_estimate=p99_estimate,
            buckets=bucket_deltas,
        )


# =============================================================================
# Summary Export Stats
# =============================================================================


class SummaryExportStats(AIPerfBaseModel):
    """Export statistics for summary metrics - server-computed quantiles.

    Summaries provide pre-computed quantiles from the server. We report:
    - Delta stats: count_delta, sum_delta, avg over the aggregation period
    - Quantiles: Final values from server (exact, not estimated)
    - Rate: observations per second
    """

    # Delta statistics over the aggregation period
    count_delta: int = Field(
        description="Change in observation count over the aggregation period"
    )
    sum_delta: float = Field(
        description="Change in sum of observed values over the aggregation period"
    )
    avg: float = Field(
        description="Average value per observation (sum_delta/count_delta)"
    )
    # Server-computed quantiles - NOTE: These are cumulative values over the server's lifetime,
    # not period-specific. Prometheus summaries cannot provide quantiles for a specific time window.
    quantiles: dict[str, float] = Field(
        default_factory=dict,
        description="Server-computed quantiles (cumulative over server lifetime, not period-specific). Keys are quantile strings (e.g., '0.5', '0.9', '0.99')",
    )
    # Rates - None if duration is zero
    count_rate: float | None = Field(
        default=None,
        description="Observations per second (count_delta/duration)",
    )
    sum_rate: float | None = Field(
        default=None,
        description="Sum per second (sum_delta/duration). Useful for token/byte summaries.",
    )

    @classmethod
    def from_time_series(
        cls, ts: SummaryTimeSeries, time_filter: TimeRangeFilter | None = None
    ) -> SummaryExportStats:
        """Create SummaryExportStats from a SummaryTimeSeries."""
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
        count_delta = final_count - ref_count
        duration_ns = final_ts - ref_ts

        avg_value = sum_delta / count_delta if count_delta > 0 else 0.0
        duration_s = duration_ns / NANOS_PER_SECOND if duration_ns > 0 else 0
        count_rate = count_delta / duration_s if duration_s > 0 else None
        sum_rate = sum_delta / duration_s if duration_s > 0 else None

        return cls(
            count_delta=int(count_delta),
            sum_delta=sum_delta,
            avg=avg_value,
            quantiles=dict(final_quantiles),
            count_rate=count_rate,
            sum_rate=sum_rate,
        )


# =============================================================================
# Type Alias
# =============================================================================

# Union type for any server metric stats (no discriminator needed - type is at family level)
ServerMetricStats: TypeAlias = (
    GaugeExportStats | CounterExportStats | HistogramExportStats | SummaryExportStats
)
