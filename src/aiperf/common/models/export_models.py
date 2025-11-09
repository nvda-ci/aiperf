# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field

from aiperf.common.enums.prometheus_enums import PrometheusMetricType
from aiperf.common.models import ErrorDetailsCount
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.server_metrics_models import HistogramData, SummaryData

if TYPE_CHECKING:
    from aiperf.common.config import UserConfig


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


class ServerMetricsSummary(AIPerfBaseModel):
    """Summary information for server metrics collection."""

    endpoints_configured: list[str]
    endpoints_successful: list[str]
    start_time: datetime
    end_time: datetime


class AggregatedMetricSample(AIPerfBaseModel):
    """Aggregated data for a single metric sample with specific labels.

    For histograms: Contains bucket deltas over the time period
    For summaries: Contains quantiles, sum, count deltas over the time period
    For counters: Contains delta value
    For gauges: Contains statistics (avg, min, max, percentiles)
    """

    labels: dict[str, str] = Field(
        default_factory=dict, description="Label key-value pairs for this sample"
    )
    value: float | None = Field(
        default=None, description="Delta value for counters, or single value for gauges"
    )
    histogram: HistogramData | None = Field(
        default=None,
        description="Histogram bucket deltas over time period",
    )
    summary: SummaryData | None = Field(
        default=None,
        description="Summary quantiles and deltas over time period",
    )
    statistics: JsonMetricResult | None = Field(
        default=None,
        description="Aggregated statistics for gauges (avg, min, max, percentiles)",
    )


class AggregatedMetricFamily(AIPerfBaseModel):
    """Aggregated metric family with data for each label combination.

    Similar structure to raw MetricFamily but with aggregated data over time period:
    - Histograms: bucket deltas (delta between first and last snapshot)
    - Counters: delta value (increase over time period)
    - Gauges: computed statistics (avg, min, max, percentiles)
    """

    type: PrometheusMetricType = Field(
        description="Metric type as defined in the Prometheus exposition format"
    )
    help: str | None = Field(default=None, description="Help text for this metric")
    unit: str | None = Field(default=None, description="Unit of measurement")
    samples: list[AggregatedMetricSample] = Field(
        description="List of samples with different label combinations and aggregated data over time period"
    )


class ServerMetricsEndpointData(AIPerfBaseModel):
    """Server metrics data for a single endpoint.

    Contains metrics organized by metric name with their statistics,
    preserving label structure similar to raw snapshot data.
    """

    endpoint_url: str = Field(
        description="Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')"
    )
    metrics: dict[str, AggregatedMetricFamily] = Field(
        description="Metrics organized by metric name with their statistics, "
        "preserving label structure similar to raw snapshot data. Structure: metric_name -> AggregatedMetricFamily"
    )


class ServerMetricsExportData(AIPerfBaseModel):
    """Server metrics data structure for JSON export.

    Contains server metrics data organized by endpoint with their statistics,
    preserving label structure similar to raw snapshot data.
    """

    summary: ServerMetricsSummary
    endpoints: dict[str, ServerMetricsEndpointData]


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


def _rebuild_models_with_forward_refs() -> None:
    """Rebuild models that have forward references to UserConfig.

    This must be called after UserConfig is fully defined to resolve forward references.
    """
    try:
        from aiperf.common.config import UserConfig  # noqa: F401

        JsonExportData.model_rebuild()
        TimesliceCollectionExportData.model_rebuild()
    except ImportError:
        # UserConfig not available yet, models will be rebuilt when needed
        pass
