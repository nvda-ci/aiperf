# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export data models for JSON output.

This module provides data structures for exporting benchmark results to JSON:

- JsonMetricResult: Single metric result with statistics
- TelemetryExportData: GPU telemetry data for export
- ServerMetricsExportData: Server metrics data for export
- JsonExportData: Complete benchmark results for JSON export

These models are designed to be compatible with the GenAI-Perf JSON output format.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field, SerializeAsAny

from aiperf.common.config import UserConfig
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetailsCount
from aiperf.common.models.export_stats import ServerMetricStats
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
