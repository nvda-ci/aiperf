# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from pydantic import ConfigDict, Field, SerializeAsAny

from aiperf.common.config import UserConfig
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

    # Collection metadata
    sample_count: int = Field(description="Number of samples collected")
    duration_seconds: float = Field(
        default=0.0, description="Aggregation period duration"
    )
    # Value statistics
    avg: float = Field(description="Average value across all samples")
    min: float = Field(description="Minimum observed value")
    max: float = Field(description="Maximum observed value")
    std: float = Field(description="Standard deviation of values")
    # Key percentiles (industry standard)
    p50: float = Field(description="50th percentile (median)")
    p90: float = Field(description="90th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")
    # Time-weighted average (more accurate for sparse/irregular samples)
    time_weighted_avg: float = Field(
        description="Time-weighted average accounting for duration between samples. More accurate than simple avg for metrics that change infrequently (e.g., KV cache usage, queue depth)"
    )

    @classmethod
    def from_time_series(
        cls, ts: ScalarTimeSeries, time_filter: TimeRangeFilter | None = None
    ) -> GaugeExportStats:
        """Create GaugeExportStats from a ScalarTimeSeries."""
        mask = ts.get_time_mask(time_filter)
        values = ts.values[mask]
        timestamps = ts.timestamps[mask]

        pcts = np.percentile(values, [50, 90, 95, 99])
        duration_s = (
            (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0.0
        )

        # Time-weighted average: weight each value by duration until next sample
        if len(values) > 1:
            durations = np.diff(timestamps).astype(np.float64)
            # Last value has no "next" sample, so we exclude it from weighting
            # (alternative: extend to end of window, but this is simpler and standard)
            total_weighted = np.sum(values[:-1] * durations)
            total_duration = np.sum(durations)
            time_weighted_avg = (
                float(total_weighted / total_duration)
                if total_duration > 0
                else float(np.mean(values))
            )
        else:
            time_weighted_avg = float(values[0]) if len(values) > 0 else 0.0

        return cls(
            sample_count=len(values),
            duration_seconds=duration_s,
            avg=float(np.mean(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            std=float(np.std(values)),
            p50=float(pcts[0]),
            p90=float(pcts[1]),
            p95=float(pcts[2]),
            p99=float(pcts[3]),
            time_weighted_avg=time_weighted_avg,
        )


class CounterExportStats(AIPerfBaseModel):
    """Export statistics for counter metrics - monotonically increasing totals.

    Counters represent cumulative totals (e.g., total requests, total bytes).
    We report the delta and rate statistics over the aggregation window.
    """

    # Delta statistics
    delta: float = Field(description="Change over the aggregation period")
    # Rate statistics (delta per second)
    rate_avg: float = Field(description="Average rate per second (delta/duration)")
    rate_min: float = Field(description="Minimum instantaneous rate per second")
    rate_max: float = Field(description="Maximum instantaneous rate per second")
    rate_std: float = Field(description="Standard deviation of instantaneous rates")

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
        duration_s = max((filtered_ts[-1] - ref_ts) / 1e9, 1e-9)

        # Point-to-point rates (include ref→first transition)
        if ref_idx is not None:
            all_ts = np.concatenate([[ref_ts], filtered_ts])
            all_vals = np.concatenate([[ref_value], filtered_vals])
        else:
            all_ts, all_vals = filtered_ts, filtered_vals

        if len(all_vals) > 1:
            deltas = np.diff(all_vals)
            time_deltas_s = np.maximum(np.diff(all_ts) / 1e9, 1e-9)
            rates = deltas / time_deltas_s
        else:
            rates = np.array([total_delta / duration_s])

        return cls(
            delta=total_delta,
            rate_avg=total_delta / duration_s,
            rate_min=float(np.min(rates)),
            rate_max=float(np.max(rates)),
            rate_std=float(np.std(rates)),
        )


class HistogramExportStats(AIPerfBaseModel):
    """Export statistics for histogram metrics - value distribution + observation rates.

    Histograms track distributions (e.g., request latencies). We report:
    - Value stats: avg observed value (sum/count)
    - Rate stats: observation_rate (count/sec) with variability over time
    - Estimated percentiles from bucket distribution using histogram_quantile
    - Raw bucket data for downstream analysis

    Note: Estimated percentiles use linear interpolation between bucket boundaries
    and may have 10-30% error depending on bucket granularity. These are standard
    Prometheus histogram_quantile estimates used for SLO monitoring.
    """

    # Observation statistics
    observation_count: int = Field(
        description="Total number of observations recorded (e.g., total requests measured)"
    )
    sum: float = Field(
        description="Sum of all observed values (e.g., total latency in seconds across all requests)"
    )
    avg: float = Field(
        description="Average value per observation (sum/observation_count). E.g., for latency histograms, this is average latency per request"
    )
    # Observation rate
    observation_rate: float = Field(
        default=0.0,
        description="Average observation rate (observations per second)",
    )
    # Raw bucket data for custom analysis
    buckets: dict[str, float] = Field(
        description='Bucket upper bounds (le="less than or equal") to delta counts. Keys are strings like "0.01", "0.1", "1.0"'
    )
    # Enhanced metrics
    estimated_percentiles: dict[str, float] = Field(
        description="Estimated percentiles from bucket distribution using histogram_quantile (p50, p90, p95, p99). Uses linear interpolation with ±10-30% accuracy depending on bucket granularity"
    )

    @staticmethod
    def _process_buckets(buckets: dict[str, float]) -> list[tuple[float, float]]:
        """Process and sort histogram buckets for percentile computation."""
        if not buckets:
            return []

        numeric: list[tuple[float, float]] = []
        inf_count = None

        for le_str, count in buckets.items():
            if le_str == "+Inf":
                inf_count = count
            else:
                numeric.append((float(le_str), count))

        numeric.sort()

        if inf_count is not None:
            upper_bound = numeric[-1][0] * 2 if numeric else 1e10
            numeric.append((upper_bound, inf_count))

        return numeric

    @staticmethod
    def _estimate_percentiles(
        buckets: dict[str, float], total_count: float
    ) -> dict[str, float]:
        """Estimate p50, p90, p95, p99 from histogram buckets."""
        if total_count == 0 or not buckets:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

        processed = HistogramExportStats._process_buckets(buckets)

        result = {}
        for percentile, key in [
            (0.50, "p50"),
            (0.90, "p90"),
            (0.95, "p95"),
            (0.99, "p99"),
        ]:
            target_rank = percentile * total_count
            prev_le, prev_count = 0.0, 0.0

            for le, count in processed:
                if count >= target_rank:
                    if count == prev_count:
                        result[key] = le
                    else:
                        fraction = (target_rank - prev_count) / (count - prev_count)
                        result[key] = prev_le + fraction * (le - prev_le)
                    break
                prev_le, prev_count = le, count
            else:
                result[key] = processed[-1][0] if processed else 0.0

        return result

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
        total_sum = final_sum - ref_sum
        total_count = int(final_count - ref_count)
        duration_s = max((final_ts - ref_ts) / 1e9, 1e-9)

        avg_value = total_sum / total_count if total_count > 0 else 0.0
        observation_rate = total_count / duration_s

        # Bucket delta calculation
        if ref_idx is not None:
            ref_buckets = (
                ts._bucket_snapshots[ref_idx]
                if ref_idx < len(ts._bucket_snapshots)
                else {}
            )
            bucket_deltas = {
                le: max(0.0, final_val - ref_buckets.get(le, 0.0))
                for le, final_val in final_buckets.items()
            }
        else:
            bucket_deltas = dict(final_buckets)

        estimated_percentiles = cls._estimate_percentiles(bucket_deltas, total_count)

        return cls(
            observation_count=total_count,
            sum=total_sum,
            avg=avg_value,
            observation_rate=observation_rate,
            buckets=bucket_deltas,
            estimated_percentiles=estimated_percentiles,
        )


class SummaryExportStats(AIPerfBaseModel):
    """Export statistics for summary metrics - server-computed quantiles.

    Summaries provide pre-computed quantiles from the server. We report:
    - Value stats: avg observed value (sum/count)
    - Quantiles: Final values from server (exact, not estimated)
    - Rate stats: observation_rate (count/sec)
    """

    # Observation statistics
    observation_count: int = Field(
        description="Total number of observations recorded (e.g., total requests measured)"
    )
    sum: float = Field(
        description="Sum of all observed values (e.g., total latency in seconds across all requests)"
    )
    avg: float = Field(
        description="Average value per observation (sum/observation_count). E.g., for latency summaries, this is average latency per request"
    )
    # Server-computed quantiles (exact values from server, keys are quantile strings like "0.5", "0.99")
    quantiles: dict[str, float] = Field(
        default_factory=dict,
        description="Server-computed quantiles (exact values). Keys are quantile strings (e.g., '0.5', '0.9', '0.99')",
    )
    # Observation rate
    observation_rate: float = Field(
        default=0.0,
        description="Average observation rate (observations per second)",
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

        total_sum = final_sum - ref_sum
        total_count = int(final_count - ref_count)
        duration_s = max((final_ts - ref_ts) / 1e9, 1e-9)

        avg_value = total_sum / total_count if total_count > 0 else 0.0
        observation_rate = total_count / duration_s

        return cls(
            observation_count=total_count,
            sum=total_sum,
            avg=avg_value,
            quantiles=dict(final_quantiles),
            observation_rate=observation_rate,
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


class ServerMetricsSummary(AIPerfBaseModel):
    """Summary information for server metrics collection."""

    endpoints_configured: list[str]
    endpoints_successful: list[str]
    start_time: datetime
    end_time: datetime


class ServerMetricLabeledStats(AIPerfBaseModel):
    """Aggregated statistics for a single time series (unique label combination).

    In Prometheus, each unique label combination is a separate time series.
    This model represents statistics for one such combination.
    """

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
