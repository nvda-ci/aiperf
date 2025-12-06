# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export data models for JSON output.

This module provides data structures for exporting benchmark results to JSON:

- JsonMetricResult: Single metric result with statistics
- TelemetryExportData: GPU telemetry data for export
- ServerMetricsExportData: Server metrics data for export (nested format)
- ServerMetricsFlatExportData: Server metrics data for export (flat format)
- JsonExportData: Complete benchmark results for JSON export

These models are designed to be compatible with the GenAI-Perf JSON output format.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field

from aiperf.common.config import UserConfig
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetailsCount
from aiperf.common.models.metric_info_models import InfoMetricData

# =============================================================================
# JSON Metric Result
# =============================================================================


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


# =============================================================================
# Telemetry Export Data
# =============================================================================


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


# =============================================================================
# Server Metrics Export Data
# =============================================================================


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
        description="Average time to complete each scrape in milliseconds"
    )
    avg_scrape_period_ms: float = Field(
        description="Average time between consecutive scrapes in milliseconds"
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


class ServerMetricSummary(AIPerfBaseModel):
    """Summary of a server metric with type, description, and per-label statistics.

    Combines metadata (metric type and description) with aggregated statistics.
    Each item in 'series' represents statistics for a unique label combination
    using FlatSeriesStats (the canonical model for server metric statistics).
    """

    description: str = Field(description="Metric description from HELP text")
    type: PrometheusMetricType = Field(description="Metric type")
    series: list[FlatSeriesStats] = Field(
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
    avg_scrape_period_ms: float = Field(
        description="Average time between consecutive scrapes in milliseconds"
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

    Info metrics (ending in _info) are included as gauges with value=1.0 since they
    represent static configuration/version information that doesn't change.
    """

    summary: ServerMetricsSummary
    metrics: dict[str, ServerMetricSummary] = Field(
        default_factory=dict,
        description="All metrics merged across endpoints, with endpoint field in each series item",
    )


# =============================================================================
# Server Metrics Hybrid Export Data (keyed metrics + flat stats)
# =============================================================================


class FlatSeriesStats(AIPerfBaseModel):
    """Flat statistics for a single time series within a metric.

    This is the canonical model for server metric statistics. Used both internally
    during computation and for final JSON export. Contains endpoint info, labels,
    and flattened stats fields.

    Endpoint fields are optional during computation and filled in when building
    the final export structure.
    """

    # Endpoint fields (optional during computation, filled in for export)
    endpoint: str | None = Field(
        default=None,
        description="Normalized endpoint identifier (e.g., 'localhost:8081')",
    )
    endpoint_url: str | None = Field(
        default=None,
        description="Full endpoint URL (e.g., 'http://localhost:8081/metrics')",
    )

    # Labels
    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels. None if the metric has no labels.",
    )

    # Common statistics
    observation_count: int | None = Field(
        default=None,
        description="Number of observations/samples. Computed differently per type.",
    )
    # Histogram/Summary shared fields
    observations_per_second: float | None = Field(
        default=None, description="Observations per second (histogram/summary)"
    )

    avg: float | None = Field(default=None, description="Average value")

    # Gauge-specific fields
    min: float | None = Field(default=None, description="Minimum value (gauge only)")
    max: float | None = Field(default=None, description="Maximum value (gauge only)")
    std: float | None = Field(
        default=None, description="Standard deviation (gauge only)"
    )

    # Unified percentile fields (no more p99_estimate vs p99)
    p50: float | None = Field(default=None, description="50th percentile")
    p90: float | None = Field(default=None, description="90th percentile")
    p95: float | None = Field(default=None, description="95th percentile")
    p99: float | None = Field(default=None, description="99th percentile")
    estimated_percentiles: bool | None = Field(
        default=None,
        description="True if percentiles are estimated (histogram), False if exact (gauge/summary)",
    )

    # Counter and Histogram/Summary shared fields
    delta: float | None = Field(
        default=None,
        description="Value change over collection period. "
        "Counter: change in counter value. Histogram/Summary: change in sum.",
    )
    rate_per_second: float | None = Field(
        default=None,
        description="Delta per second. Counter: counter rate. Histogram/Summary: sum rate.",
    )

    # Counter-specific rate statistics
    rate_avg: float | None = Field(
        default=None,
        description="Time-weighted average rate between change points (counter)",
    )
    rate_min: float | None = Field(
        default=None, description="Minimum point-to-point rate per second (counter)"
    )
    rate_max: float | None = Field(
        default=None, description="Maximum point-to-point rate per second (counter)"
    )
    rate_std: float | None = Field(
        default=None, description="Standard deviation of point-to-point rates (counter)"
    )

    # Histogram-specific
    buckets: dict[str, int] | None = Field(
        default=None,
        description="Histogram bucket upper bounds to delta counts (e.g., {'0.1': 2000, '+Inf': 5000})",
    )

    # Summary-specific
    quantiles: dict[str, float] | None = Field(
        default=None,
        description="Server-computed quantiles from Prometheus summary (e.g., {'0.5': 0.1, '0.99': 0.5})",
    )


class HybridMetricData(AIPerfBaseModel):
    """Metric data with type, description, unit, and flat series stats.

    Used in hybrid export format where metrics are keyed by name for O(1) lookup,
    but stats within each series are flattened for easy access.
    """

    type: PrometheusMetricType = Field(description="Metric type")
    description: str = Field(description="Metric description from HELP text")
    unit: str | None = Field(
        default=None,
        description="Unit inferred from metric name suffix (_seconds, _bytes, etc.)",
    )
    series: list[FlatSeriesStats] = Field(
        default_factory=list,
        description="Statistics for each unique endpoint + label combination",
    )


class ServerMetricsHybridExportData(AIPerfBaseModel):
    """Server metrics in hybrid format: keyed metrics with flat stats.

    Provides O(1) metric lookup by name while keeping stats flat within each series.
    Best of both worlds: easy to find specific metrics AND easy to access their stats.

    Example access:
        data["metrics"]["vllm:kv_cache_usage_perc"]["series"][0]["p99"]
    """

    summary: ServerMetricsSummary
    metrics: dict[str, HybridMetricData] = Field(
        default_factory=dict,
        description="Metrics keyed by name, each with flat series stats",
    )


# =============================================================================
# Timeslice Export Data
# =============================================================================


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


# =============================================================================
# Main JSON Export Data
# =============================================================================


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
