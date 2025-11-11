# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic summary result models for results processors.

This module provides a type-safe, extensible system for processor summary results
using AutoRoutedModel with discriminated unions. Each processor type returns its
own specialized result class, enabling automatic routing and type-safe handling
without isinstance checks.

Architecture:
- ProcessorSummaryResult: Base class with processor_type discriminator
- Subclasses: Specific result types for each processor category
- AutoRouted: Automatic deserialization to correct subclass

Example:
    # Processor returns typed result
    result = await processor.summarize()

    # Type-safe handling with match
    match result:
        case MetricSummaryResult():
            return result.results
        case TimesliceSummaryResult():
            return result.timeslice_results
"""

from pathlib import Path
from typing import ClassVar

from pydantic import Field

from aiperf.common.enums import ResultsProcessorType
from aiperf.common.models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetailsCount
from aiperf.common.models.metric_result_models import MetricResult
from aiperf.common.models.server_metrics_models import ServerMetricsHierarchy
from aiperf.common.models.telemetry_models import TelemetryHierarchy


class ProcessorSummaryResult(AIPerfBaseModel):
    """Base class for processor summary results with automatic routing by processor_type.

    Uses AutoRoutedModel for type-safe deserialization:
    - Routes by processor_type discriminator to specific subclass
    - Each processor returns its own typed result
    - Enables type-safe handling without isinstance checks

    The discriminator field enables automatic routing when deserializing from JSON
    or dict, ensuring the correct subclass is instantiated based on processor_type.
    """

    discriminator_field: ClassVar[str] = "processor_type"

    processor_type: ResultsProcessorType = Field(
        ...,
        description="Type of processor that generated this summary result. Used as the discriminator field for automatic routing.",
    )


class MetricSummaryResult(ProcessorSummaryResult):
    """Summary result containing aggregated metrics from a results processor.

    This result type is used by processors that aggregate metrics across all records
    and produce a final list of computed metrics.

    Used by:
    - MetricResultsProcessor (METRIC_RESULTS)

    Attributes:
        processor_type: Type of processor (METRIC_RESULTS)
        results: List of computed metric results with aggregated statistics
    """

    processor_type: ResultsProcessorType = ResultsProcessorType.METRIC_RESULTS

    results: list[MetricResult] = Field(
        ..., description="List of computed metric results with aggregated statistics"
    )


class TimesliceSummaryResult(ProcessorSummaryResult):
    """Summary result containing metrics grouped by time slices.

    This result type is used by processors that divide the benchmark run into
    user-configurable time slices and compute metrics for each slice independently.

    Attributes:
        processor_type: Type of processor (TIMESLICE)
        timeslice_results: Metric results grouped by timeslice index (0-based sequential)
    """

    processor_type: ResultsProcessorType = ResultsProcessorType.TIMESLICE

    timeslice_results: dict[int, list[MetricResult]] = Field(
        ...,
        description="Metric results grouped by timeslice index (0-based sequential)",
    )


class TelemetrySummaryResult(ProcessorSummaryResult):
    """Summary result containing GPU telemetry hierarchy.

    This result type is used by processors that aggregate GPU telemetry data
    from DCGM endpoints. The telemetry hierarchy provides the native, structured
    format for all GPU data organized by endpoint and GPU UUID.

    MetricResult summaries can be generated on-demand via the hierarchy's
    get_metric_result() methods when needed for display or export.

    Attributes:
        processor_type: Type of processor (TELEMETRY_RESULTS)
        telemetry_data: Complete telemetry hierarchy with all GPU data
        endpoints_tested: List of DCGM endpoints tested
        endpoints_successful: List of DCGM endpoints that succeeded
        error_summary: Errors encountered during collection
    """

    processor_type: ResultsProcessorType = ResultsProcessorType.TELEMETRY_RESULTS
    telemetry_data: TelemetryHierarchy = Field(
        ...,
        description="Complete telemetry hierarchy with all GPU data organized by endpoint and GPU",
    )
    endpoints_tested: list[str] = Field(
        ..., description="List of DCGM endpoints that were tested"
    )
    endpoints_successful: list[str] = Field(
        ..., description="List of DCGM endpoints that successfully provided telemetry"
    )
    error_summary: list[ErrorDetailsCount] = Field(
        ..., description="Summary of errors encountered during telemetry collection"
    )


class ServerMetricsSummaryResult(ProcessorSummaryResult):
    """Summary result containing server metrics hierarchy.

    This result type is used by processors that aggregate server metrics data
    from Prometheus endpoints. The server metrics hierarchy provides the native,
    structured format for all Prometheus data organized by endpoint.

    MetricResult summaries can be generated on-demand via the hierarchy's
    get_metric_result() methods when needed for display or export.

    Attributes:
        processor_type: Type of processor (SERVER_METRICS_RESULTS)
        server_metrics_data: Complete server metrics hierarchy with all endpoint data
        endpoints_tested: List of Prometheus endpoints tested
        endpoints_successful: List of Prometheus endpoints that succeeded
        error_summary: Errors encountered during collection
    """

    processor_type: ResultsProcessorType = ResultsProcessorType.SERVER_METRICS_RESULTS
    server_metrics_data: ServerMetricsHierarchy = Field(
        ..., description="Complete server metrics hierarchy with all endpoint data"
    )
    endpoints_tested: list[str] = Field(
        ..., description="List of Prometheus endpoints that were tested"
    )
    endpoints_successful: list[str] = Field(
        ...,
        description="List of Prometheus endpoints that successfully provided server metrics",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        ...,
        description="Summary of errors encountered during server metrics collection",
    )


class BaseFileExportSummaryResult(ProcessorSummaryResult):
    """Summary result for any file-exporting processor."""

    file_path: Path = Field(
        ...,
        description="Path to exported file",
    )
    record_count: int = Field(
        default=0, description="Number of records exported to file"
    )


class RecordExportSummaryResult(BaseFileExportSummaryResult):
    """Summary result for record export processor."""

    processor_type: ResultsProcessorType = ResultsProcessorType.RECORD_EXPORT


class TelemetryExportSummaryResult(BaseFileExportSummaryResult):
    """Summary result for telemetry export processor."""

    processor_type: ResultsProcessorType = ResultsProcessorType.TELEMETRY_EXPORT


class ServerMetricsExportSummaryResult(BaseFileExportSummaryResult):
    """Summary result for server metrics export processor."""

    processor_type: ResultsProcessorType = ResultsProcessorType.SERVER_METRICS_EXPORT
