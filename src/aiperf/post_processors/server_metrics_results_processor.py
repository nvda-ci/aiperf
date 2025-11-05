# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.enums.metric_enums import MetricUnitT
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.server_metrics_models import (
    ServerMetadata,
    ServerMetricRecord,
    ServerMetricsData,
    ServerMetricsHierarchy,
)
from aiperf.common.protocols import ServerMetricsResultsProcessorProtocol
from aiperf.post_processors.base_hierarchical_results_processor import (
    BaseHierarchicalResultsProcessor,
)
from aiperf.server_metrics.constants import (
    format_metric_display_name,
    infer_metric_unit,
)


@implements_protocol(ServerMetricsResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.SERVER_METRICS_RESULTS)
class ServerMetricsResultsProcessor(
    BaseHierarchicalResultsProcessor[ServerMetricsHierarchy, ServerMetricRecord]
):
    """Process individual ServerMetricRecord objects into hierarchical storage.

    Uses dynamic field discovery - metrics are discovered from the actual server
    response data without requiring pre-defined configuration. Display names and
    units are inferred automatically from field names.
    """

    HIERARCHY_CLASS = ServerMetricsHierarchy
    METRICS_CONFIG = None  # None enables dynamic field discovery
    ENDPOINTS_DICT_FIELD = "server_endpoints"

    def get_server_metrics_hierarchy(self) -> ServerMetricsHierarchy:
        """Get the accumulated server metrics hierarchy."""
        return self.get_hierarchy()

    def _discover_metrics_config(
        self, resource_data: ServerMetricsData
    ) -> list[tuple[str, str, MetricUnitT]]:
        """Discover metrics configuration dynamically from server metrics data.

        Examines the actual metrics present in the time series data and generates
        display names and units automatically from field names.

        Args:
            resource_data: ServerMetricsData containing time series for this server

        Returns:
            List of (display_name, field_name, unit_enum) tuples for all metrics
        """
        metrics_config = []

        # Get all metric field names from the time series data
        if resource_data.time_series.snapshots:
            # Get field names from the first snapshot (all snapshots have same structure)
            first_snapshot = resource_data.time_series.snapshots[0]
            field_names = first_snapshot.metrics.keys()

            # Generate config for each field
            for field_name in sorted(field_names):
                display_name = format_metric_display_name(field_name)
                unit_enum = infer_metric_unit(field_name)
                metrics_config.append((display_name, field_name, unit_enum))

        return metrics_config

    async def process_server_metric_record(self, record: ServerMetricRecord) -> None:
        """Process individual server metric record into hierarchical storage.

        Args:
            record: ServerMetricRecord containing server metrics and hierarchical metadata
        """
        await self._process_record_internal(record)

    # Abstract method implementations for server metrics

    def _create_tag(
        self, endpoint_url: str, resource_id: str, metric_name: str, metadata: Any
    ) -> str:
        """Create a unique tag for server metric.

        Args:
            endpoint_url: Server metrics endpoint URL
            resource_id: Server ID
            metric_name: Metric field name
            metadata: ServerMetadata instance

        Returns:
            Tag string like "requests_total_server_frontend_8080_frontend-0"
        """
        server_tag = endpoint_url.replace(":", "_").replace("/", "_").replace(".", "_")
        return f"{metric_name}_server_{server_tag}_{resource_id}"

    def _create_header(
        self, metric_display: str, endpoint_display: str, metadata: Any
    ) -> str:
        """Create a human-readable header for server metric.

        Args:
            metric_display: Human-readable metric name
            endpoint_display: Formatted server endpoint for display
            metadata: ServerMetadata instance

        Returns:
            Header string like "Requests Total | frontend:8080 | frontend | host1"
        """
        server_metadata: ServerMetadata = metadata
        server_type = server_metadata.server_type or "server"
        hostname = server_metadata.hostname or "unknown"
        return f"{metric_display} | {endpoint_display} | {server_type} | {hostname}"

    def _format_no_metric_debug(
        self, metric_name: str, resource_id: str, endpoint_url: str
    ) -> str:
        """Format debug message for missing server metric data."""
        return f"No data available for metric '{metric_name}' on server {resource_id} from {endpoint_url}"

    def _format_error_message(
        self, metric_name: str, resource_id: str, endpoint_url: str, error: Exception
    ) -> str:
        """Format error message for server metric processing failures."""
        return f"Unexpected error generating metric result for '{metric_name}' on server {resource_id} from {endpoint_url}: {error}"
