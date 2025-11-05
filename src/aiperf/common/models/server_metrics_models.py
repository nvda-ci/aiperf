# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ConfigDict, Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.metrics_hierarchy_base import (
    BaseMetricSnapshot,
    BaseMetricTimeSeries,
    BaseResourceMetricsData,
    HistogramSnapshot,
)


class ServerMetrics(AIPerfBaseModel):
    """Dynamic server metrics collected at a single point in time from Prometheus /metrics endpoint.

    This model uses a flexible schema that accepts any metric fields dynamically.
    All metrics are stored as float values and can be accessed as attributes.

    This design allows the system to:
    - Support any Prometheus metrics without code changes
    - Work with different server types (Dynamo, vLLM, TensorRT-LLM, etc.)
    - Automatically handle new metrics as they're added to servers
    - Scale to support custom metrics from user applications

    Common Prometheus metrics automatically supported:
    - Request metrics: requests_total, requests_in_flight, request_duration_seconds
    - Resource metrics: cpu_usage_percent, memory_usage_bytes, process_resident_memory_bytes
    - Dynamo metrics: component_*, frontend_*, kvstats_*, model_*
    - Custom metrics: Any metric name from your server's /metrics endpoint

    Example usage:
        # Create with any fields
        metrics = ServerMetrics(
            requests_total=1000.0,
            cpu_usage_percent=45.2,
            custom_metric=123.4
        )

        # Access metrics as attributes
        print(metrics.requests_total)  # 1000.0
        print(metrics.custom_metric)   # 123.4

        # Convert to dict for processing
        data = metrics.model_dump()  # Includes all fields
        filtered = {k: v for k, v in data.items() if v is not None}

    Note:
        Prometheus metric names are used directly as field names (with "_total" suffix
        removed). Display names and units are inferred dynamically from field names using
        format_metric_display_name() and infer_metric_unit() helper functions.
    """

    model_config = ConfigDict(extra="allow")  # Allow any additional fields dynamically


class ServerMetricRecord(AIPerfBaseModel):
    """Single server metric data point from monitoring.

    This record contains all metrics for one server at one point in time,
    along with metadata to identify the source metrics endpoint and specific server.
    Used for hierarchical storage: server_url -> server_id -> time series data.
    """

    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when metrics were collected (time_ns)"
    )

    server_url: str = Field(
        description="Source server metrics endpoint URL (e.g., 'http://frontend:8080/metrics')"
    )

    server_id: str = Field(
        description="Unique server identifier (e.g., 'frontend-0', 'worker-1')"
    )
    server_type: str | None = Field(
        default=None, description="Server type (e.g., 'frontend', 'worker', 'backend')"
    )
    hostname: str | None = Field(
        default=None, description="Hostname where server is running"
    )
    instance: str | None = Field(
        default=None, description="Instance identifier (e.g., 'localhost:8080')"
    )

    metrics_data: ServerMetrics = Field(
        description="Server metrics snapshot collected at this timestamp"
    )
    histograms_data: dict[str, HistogramSnapshot] = Field(
        default_factory=dict,
        description="Histogram distributions collected at this timestamp",
    )


class ServerMetadata(AIPerfBaseModel):
    """Static metadata for a server that doesn't change over time.

    This is stored once per server and referenced by all metric data points
    to avoid duplicating metadata in every time-series entry.
    """

    server_id: str = Field(description="Unique server identifier - primary key")
    server_type: str | None = Field(
        default=None, description="Server type classification"
    )
    hostname: str | None = Field(default=None, description="Host machine name")
    instance: str | None = Field(default=None, description="Instance identifier")


class ServerMetricSnapshot(BaseMetricSnapshot):
    """All metrics for a single server at one point in time.

    Groups all metric values collected during a single collection cycle,
    eliminating timestamp duplication across individual metrics.
    """

    pass  # Inherits all functionality from BaseMetricSnapshot


class ServerMetricTimeSeries(BaseMetricTimeSeries):
    """Time series data for all metrics on a single server.

    Uses grouped snapshots instead of individual metric time series to eliminate
    timestamp duplication and improve storage efficiency.
    """

    pass  # Inherits all functionality from BaseMetricTimeSeries


class ServerMetricsData(BaseResourceMetricsData[ServerMetadata]):
    """Complete metrics data for one server: metadata + grouped metric time series.

    This combines static server information with dynamic time-series data,
    providing the complete picture for one server's metrics using efficient grouped snapshots.
    """

    time_series: ServerMetricTimeSeries = Field(
        default_factory=ServerMetricTimeSeries,
        description="Grouped time series for all metrics",
    )

    def add_record(self, record: ServerMetricRecord) -> None:
        """Add metric record as a grouped snapshot.

        Args:
            record: New metric data point from server collector

        Note: Groups all metric values and histograms from the record into a single snapshot
        """
        metric_mapping = record.metrics_data.model_dump()
        valid_metrics = {k: v for k, v in metric_mapping.items() if v is not None}
        if valid_metrics or record.histograms_data:
            self.time_series.append_snapshot(
                valid_metrics, record.timestamp_ns, record.histograms_data
            )


class ServerMetricsHierarchy(AIPerfBaseModel):
    """Hierarchical storage: server_url -> server_id -> complete server metrics data.

    This provides hierarchical structure while maintaining efficient
    access patterns for both real-time display and final aggregation.

    Structure:
    {
        "http://frontend:8080/metrics": {
            "frontend-0": ServerMetricsData(metadata + time series),
            "frontend-1": ServerMetricsData(metadata + time series)
        },
        "http://worker:8081/metrics": {
            "worker-0": ServerMetricsData(metadata + time series)
        }
    }
    """

    server_endpoints: dict[str, dict[str, ServerMetricsData]] = Field(
        default_factory=dict,
        description="Nested dict: server_url -> server_id -> metrics data",
    )

    def add_record(self, record: ServerMetricRecord) -> None:
        """Add metric record to hierarchical storage.

        Args:
            record: New metric data from server monitoring

        Note: Automatically creates hierarchy levels as needed:
        - New server endpoints get empty server dict
        - New servers get initialized with metadata and empty metrics
        """

        if record.server_url not in self.server_endpoints:
            self.server_endpoints[record.server_url] = {}

        server_data = self.server_endpoints[record.server_url]

        if record.server_id not in server_data:
            metadata = ServerMetadata(
                server_id=record.server_id,
                server_type=record.server_type,
                hostname=record.hostname,
                instance=record.instance,
            )
            server_data[record.server_id] = ServerMetricsData(metadata=metadata)

        server_data[record.server_id].add_record(record)


class ServerMetricsResults(AIPerfBaseModel):
    """Results from server metrics collection during a profile run.

    This class contains all server metrics data and metadata collected during
    a benchmarking session, separate from inference performance results.
    """

    metrics_data: ServerMetricsHierarchy = Field(
        description="Hierarchical metrics data organized by server endpoint and server ID"
    )
    start_ns: int = Field(description="Start time of metrics collection in nanoseconds")
    end_ns: int = Field(description="End time of metrics collection in nanoseconds")
    endpoints_configured: list[str] = Field(
        default_factory=list,
        description="List of server endpoint URLs configured for monitoring",
    )
    endpoints_successful: list[str] = Field(
        default_factory=list,
        description="List of server endpoint URLs that successfully provided metrics data",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )


class ProcessServerMetricsResult(AIPerfBaseModel):
    """Result of server metrics processing - mirrors ProcessTelemetryResult pattern.

    This provides a parallel structure to ProcessTelemetryResult for the server metrics pipeline,
    maintaining complete separation while following the same architectural patterns.
    """

    results: ServerMetricsResults = Field(
        description="The processed server metrics results"
    )
    errors: list[ErrorDetails] = Field(
        default_factory=list,
        description="Any errors that occurred while processing server metrics data",
    )
