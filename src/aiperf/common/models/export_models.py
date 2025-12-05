# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from pydantic import ConfigDict, Field, SerializeAsAny

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import ErrorDetailsCount
from aiperf.common.models.base_models import AIPerfBaseModel

if TYPE_CHECKING:
    from aiperf.common.models.server_metrics_models import (
        HistogramTimeSeries,
        ScalarTimeSeries,
        SummaryTimeSeries,
        TimeRangeFilter,
    )

# ============================================================================
# Prometheus Terminology Glossary
# ============================================================================
#
# - **Metric Family**: A group of related metrics with the same name and type.
#   Example: All "http_request_duration_seconds" metrics form one family.
#
# - **Time Series**: A unique combination of metric name + labels.
#   Example: http_request_duration_seconds{method="GET", status="200"}
#   is a different time series from {method="POST", status="200"}.
#
# - **le** (less than or equal): Histogram bucket upper bound notation.
#   A bucket with le="0.1" contains all observations <= 0.1 seconds.
#
# - **Quantiles**: Pre-computed percentiles from the server (p50, p90, p95, p99).
#   These are exact values calculated by Prometheus, not estimated from buckets.
#
# ============================================================================


class InfoMetricData(AIPerfBaseModel):
    """Complete data for an info metric including label data.

    Info metrics (ending in _info) contain static system information that doesn't
    change over time. We store only the labels (not values) since the labels contain
    the actual information and values are typically just 1.0.
    """

    description: str = Field(description="Metric description from HELP text")
    labels: list[dict[str, str]] = Field(
        description="List of label keys and values as reported by the Prometheus endpoint"
    )


class JsonMetricResult(AIPerfBaseModel):
    """The result values of a single metric for JSON export.

    NOTE:
    This model has been designed to mimic the structure of the GenAI-Perf JSON output
    as closely as possible. Be careful not to add or remove fields that are not present in the
    GenAI-Perf JSON output.
    """

    unit: str = Field(description="The unit of the metric, e.g. 'ms' or 'requests/sec'")
    avg: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: int | float | None = None
    max: int | float | None = None
    std: float | None = None


# ============================================================================
# Server Metrics Export Stats - Type-specific models for semantic correctness
# ============================================================================


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


class HistogramExportStats(AIPerfBaseModel):
    """Export statistics for histogram metrics - value distribution + rates.

    Histograms track distributions (e.g., request latencies). We report:
    - Delta stats: count_delta, sum_delta, avg over the aggregation period
    - Rate: observations per second
    - Raw bucket data for downstream analysis
    """

    # Delta statistics over the aggregation period
    count_delta: float = Field(
        description="Change in observation count over the aggregation period"
    )
    sum_delta: float = Field(
        description="Change in sum of observed values over the aggregation period"
    )
    avg: float = Field(
        description="Average value per observation (sum_delta/count_delta)"
    )
    # Rate - None if duration is zero
    rate: float | None = Field(
        default=None,
        description="Observations per second (count_delta/duration)",
    )
    # Raw bucket data for custom analysis (None if counter reset detected)
    buckets: dict[str, float] | None = Field(
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
        rate = (
            count_delta / (duration_ns / NANOS_PER_SECOND) if duration_ns > 0 else None
        )

        # Bucket delta calculation
        # If any delta is negative (counter reset), return None for buckets
        # since the data is invalid/incomplete
        ref_bucket_idx = ref_idx if ref_idx is not None else 0
        ref_buckets = (
            ts._bucket_snapshots[ref_bucket_idx]
            if ref_bucket_idx < len(ts._bucket_snapshots)
            else {}
        )
        bucket_deltas: dict[str, float] | None = {}
        for le, final_val in final_buckets.items():
            ref_val = ref_buckets.get(le, 0.0)
            delta = final_val - ref_val
            if delta < 0:
                # Counter reset detected - data is invalid
                bucket_deltas = None
                break
            bucket_deltas[le] = delta

        return cls(
            count_delta=count_delta,
            sum_delta=sum_delta,
            avg=avg_value,
            rate=rate,
            buckets=bucket_deltas,
        )


class SummaryExportStats(AIPerfBaseModel):
    """Export statistics for summary metrics - server-computed quantiles.

    Summaries provide pre-computed quantiles from the server. We report:
    - Delta stats: count_delta, sum_delta, avg over the aggregation period
    - Quantiles: Final values from server (exact, not estimated)
    - Rate: observations per second
    """

    # Delta statistics over the aggregation period
    count_delta: float = Field(
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
    # Rate - None if duration is zero
    rate: float | None = Field(
        default=None,
        description="Observations per second (count_delta/duration)",
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
        rate = (
            count_delta / (duration_ns / NANOS_PER_SECOND) if duration_ns > 0 else None
        )

        return cls(
            count_delta=count_delta,
            sum_delta=sum_delta,
            avg=avg_value,
            quantiles=dict(final_quantiles),
            rate=rate,
        )


# Union type for any server metric stats (no discriminator needed - type is at family level)
ServerMetricStats: TypeAlias = (
    GaugeExportStats | CounterExportStats | HistogramExportStats | SummaryExportStats
)


class TelemetrySummary(AIPerfBaseModel):
    """Summary information for telemetry collection."""

    endpoints_configured: list[str]
    endpoints_successful: list[str]
    start_time: datetime
    end_time: datetime


class GpuSummary(AIPerfBaseModel):
    """Summary of GPU telemetry data."""

    gpu_index: int
    gpu_name: str
    gpu_uuid: str
    hostname: str | None
    metrics: dict[str, JsonMetricResult]  # metric_key -> {stat_key -> value}


class EndpointData(AIPerfBaseModel):
    """Data for a single endpoint."""

    gpus: dict[str, GpuSummary]


class TelemetryExportData(AIPerfBaseModel):
    """Telemetry data structure for JSON export."""

    summary: TelemetrySummary
    endpoints: dict[str, EndpointData]


class ServerMetricsEndpointInfo(AIPerfBaseModel):
    """Metadata about a single endpoint's collection statistics."""

    endpoint_url: str = Field(description="Full endpoint URL")
    duration_seconds: float = Field(
        description="Total duration of metrics collection for this endpoint"
    )
    scrape_count: int = Field(
        description="Number of successful scrapes from this endpoint"
    )
    avg_scrape_latency_ms: float = Field(
        description="Average time to scrape metrics from this endpoint in milliseconds"
    )


class ServerMetricsSummary(AIPerfBaseModel):
    """Summary information for server metrics collection."""

    endpoints_configured: list[str] = Field(
        description="List of configured endpoint identifiers (normalized)"
    )
    endpoints_successful: list[str] = Field(
        description="List of successful endpoint identifiers (normalized)"
    )
    start_time: datetime
    end_time: datetime
    endpoint_info: dict[str, ServerMetricsEndpointInfo] | None = Field(
        default=None,
        description="Per-endpoint collection metadata keyed by normalized endpoint identifier",
    )


class ServerMetricLabeledStats(AIPerfBaseModel):
    """Aggregated statistics for a single time series (unique label combination).

    In Prometheus, each unique label combination is a separate time series.
    This model represents statistics for one such combination.
    """

    endpoint: str | None = Field(
        default=None,
        description="Endpoint URL this series came from (used in merged export format)",
    )
    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels for this series. None if the metric has no labels.",
    )
    stats: SerializeAsAny[ServerMetricStats] = Field(
        description="Type-specific aggregated statistics (gauge, counter, histogram, or summary)",
    )


class ServerMetricSummary(AIPerfBaseModel):
    """Summary of a server metric with type, description, and per-label statistics.

    Combines metadata (metric type and description) with aggregated statistics.
    Each item in 'series' represents statistics for a unique label combination.
    """

    description: str = Field(description="Metric description from HELP text")
    type: str = Field(description="Metric type (gauge, counter, histogram, summary)")
    series: list[ServerMetricLabeledStats] = Field(
        default_factory=list,
        description="Statistics for each unique label combination",
    )


class ServerMetricsEndpointSummary(AIPerfBaseModel):
    """Summary of server metrics data for a single endpoint.

    Unified structure combining metadata and type-specific aggregated statistics:
    - Each metric uses stats matching its semantic type (gauge, counter, histogram, summary)
    - Mirrors JSONL structure with labels as proper objects
    - Includes metric description from metadata
    """

    endpoint_url: str
    # Collection metadata
    duration_seconds: float = Field(
        description="Total duration of metrics collection for this endpoint"
    )
    scrape_count: int = Field(
        description="Number of successful scrapes from this endpoint"
    )
    avg_scrape_latency_ms: float = Field(
        description="Average time to scrape metrics from this endpoint in milliseconds"
    )
    # Metric data
    info_metrics: dict[str, InfoMetricData] | None = Field(
        default=None,
        description="Static info metrics (ending in _info) with their label data",
    )
    metrics: dict[str, ServerMetricSummary] = Field(
        default_factory=dict,
        description="All metrics keyed by metric name, with description and type-specific statistics",
    )


class ServerMetricsExportData(AIPerfBaseModel):
    """Server metrics data structure for JSON export."""

    summary: ServerMetricsSummary
    endpoints: dict[str, ServerMetricsEndpointSummary]


class ServerMetricsMergedExportData(AIPerfBaseModel):
    """Server metrics data structure with all endpoints merged into a single metrics dict.

    This format merges series from all endpoints into each metric, with each series
    item containing an 'endpoint' field to identify its source.
    """

    summary: ServerMetricsSummary
    info_metrics: dict[str, InfoMetricData] | None = Field(
        default=None,
        description="Static info metrics merged from all endpoints",
    )
    metrics: dict[str, ServerMetricSummary] = Field(
        default_factory=dict,
        description="All metrics merged across endpoints, with endpoint field in each series item",
    )


class TimesliceData(AIPerfBaseModel):
    """Data for a single timeslice.

    Contains metrics for one time slice with dynamic metric fields
    added via Pydantic's extra="allow" setting.
    """

    model_config = ConfigDict(extra="allow")

    timeslice_index: int


class TimesliceCollectionExportData(AIPerfBaseModel):
    """Export data for all timeslices in a single file.

    Contains an array of timeslice data objects with metadata.
    """

    timeslices: list[TimesliceData]
    input_config: UserConfig | None = None


class JsonExportData(AIPerfBaseModel):
    """Summary data to be exported to a JSON file.

    NOTE:
    This model has been designed to mimic the structure of the GenAI-Perf JSON output
    as closely as possible. Be careful when modifying this model to not break the
    compatibility with the GenAI-Perf JSON output.
    """

    # NOTE: The extra="allow" setting is needed to allow additional metrics not defined in this class
    #       to be added to the export data. It is also already set in the AIPerfBaseModel,
    #       but we are setting it here to guard against base model changes.
    model_config = ConfigDict(extra="allow")

    request_throughput: JsonMetricResult | None = None
    request_latency: JsonMetricResult | None = None
    request_count: JsonMetricResult | None = None
    time_to_first_token: JsonMetricResult | None = None
    time_to_second_token: JsonMetricResult | None = None
    inter_token_latency: JsonMetricResult | None = None
    output_token_throughput: JsonMetricResult | None = None
    output_token_throughput_per_user: JsonMetricResult | None = None
    output_sequence_length: JsonMetricResult | None = None
    input_sequence_length: JsonMetricResult | None = None
    goodput: JsonMetricResult | None = None
    good_request_count: JsonMetricResult | None = None
    output_token_count: JsonMetricResult | None = None
    reasoning_token_count: JsonMetricResult | None = None
    min_request_timestamp: JsonMetricResult | None = None
    max_response_timestamp: JsonMetricResult | None = None
    inter_chunk_latency: JsonMetricResult | None = None
    total_output_tokens: JsonMetricResult | None = None
    total_reasoning_tokens: JsonMetricResult | None = None
    benchmark_duration: JsonMetricResult | None = None
    total_isl: JsonMetricResult | None = None
    total_osl: JsonMetricResult | None = None
    error_request_count: JsonMetricResult | None = None
    error_isl: JsonMetricResult | None = None
    total_error_isl: JsonMetricResult | None = None
    telemetry_data: TelemetryExportData | None = None
    server_metrics_data: ServerMetricsExportData | None = None
    input_config: UserConfig | None = None
    was_cancelled: bool | None = None
    error_summary: list[ErrorDetailsCount] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
