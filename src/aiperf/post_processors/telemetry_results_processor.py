# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.enums.metric_enums import MetricUnitT
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.telemetry_models import (
    GpuMetadata,
    GpuTelemetryData,
    TelemetryHierarchy,
    TelemetryRecord,
)
from aiperf.common.protocols import (
    TelemetryResultsProcessorProtocol,
)
from aiperf.gpu_telemetry.constants import (
    format_metric_display_name,
    infer_metric_unit,
)
from aiperf.post_processors.base_hierarchical_results_processor import (
    BaseHierarchicalResultsProcessor,
)


@implements_protocol(TelemetryResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.TELEMETRY_RESULTS)
class TelemetryResultsProcessor(
    BaseHierarchicalResultsProcessor[TelemetryHierarchy, TelemetryRecord]
):
    """Process individual TelemetryRecord objects into hierarchical storage.

    Uses dynamic field discovery - metrics are discovered from the actual GPU
    telemetry data without requiring pre-defined configuration. Display names and
    units are inferred automatically from field names.
    """

    HIERARCHY_CLASS = TelemetryHierarchy
    METRICS_CONFIG = None  # None enables dynamic field discovery
    ENDPOINTS_DICT_FIELD = "dcgm_endpoints"

    def get_telemetry_hierarchy(self) -> TelemetryHierarchy:
        """Get the accumulated telemetry hierarchy."""
        return self.get_hierarchy()

    def _discover_metrics_config(
        self, resource_data: GpuTelemetryData
    ) -> list[tuple[str, str, MetricUnitT]]:
        """Discover metrics configuration dynamically from GPU telemetry data.

        Examines the actual metrics present in the time series data and generates
        display names and units automatically from field names.

        Args:
            resource_data: GpuTelemetryData containing time series for this GPU

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

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        """Process individual telemetry record into hierarchical storage.

        Args:
            record: TelemetryRecord containing GPU metrics and hierarchical metadata
        """
        await self._process_record_internal(record)

    # Abstract method implementations for GPU telemetry

    def _create_tag(
        self, endpoint_url: str, resource_id: str, metric_name: str, metadata: Any
    ) -> str:
        """Create a unique tag for GPU telemetry metric.

        Args:
            endpoint_url: DCGM endpoint URL
            resource_id: GPU UUID
            metric_name: Metric field name
            metadata: GpuMetadata instance

        Returns:
            Tag string like "gpu_power_usage_dcgm_node1_gpu0_ef6ef310"
        """
        gpu_metadata: GpuMetadata = metadata
        dcgm_tag = endpoint_url.replace(":", "_").replace("/", "_").replace(".", "_")
        return f"{metric_name}_dcgm_{dcgm_tag}_gpu{gpu_metadata.gpu_index}_{resource_id[:12]}"

    def _create_header(
        self, metric_display: str, endpoint_display: str, metadata: Any
    ) -> str:
        """Create a human-readable header for GPU telemetry metric.

        Args:
            metric_display: Human-readable metric name
            endpoint_display: Formatted DCGM endpoint for display
            metadata: GpuMetadata instance

        Returns:
            Header string like "GPU Power Usage | node1:9401 | GPU 0 | RTX 6000"
        """
        gpu_metadata: GpuMetadata = metadata
        return f"{metric_display} | {endpoint_display} | GPU {gpu_metadata.gpu_index} | {gpu_metadata.model_name}"

    def _format_no_metric_debug(
        self, metric_name: str, resource_id: str, endpoint_url: str
    ) -> str:
        """Format debug message for missing GPU telemetry metric data."""
        return f"No data available for metric '{metric_name}' on GPU {resource_id[:12]} from {endpoint_url}"

    def _format_error_message(
        self, metric_name: str, resource_id: str, endpoint_url: str, error: Exception
    ) -> str:
        """Format error message for GPU telemetry metric processing failures."""
        return f"Unexpected error generating metric result for '{metric_name}' on GPU {resource_id[:12]} from {endpoint_url}: {error}"
