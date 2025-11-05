# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for hierarchical metrics results processors.

This module provides a generic base class that eliminates duplication between
TelemetryResultsProcessor and ServerMetricsResultsProcessor. Both processors follow
identical patterns:
    1. Maintain hierarchical storage (endpoint -> resource_id -> metrics data)
    2. Process individual records by adding them to hierarchy
    3. Summarize by iterating hierarchy and generating MetricResult list

The generic base class captures this common structure while allowing concrete
implementations to customize:
    - Hierarchy type (TelemetryHierarchy, ServerMetricsHierarchy)
    - Record type (TelemetryRecord, ServerMetricRecord)
    - Metrics configuration (static config or dynamic field discovery)
    - Tag/header formatting logic
"""

from abc import abstractmethod
from typing import Any, Generic

from aiperf.common.config import UserConfig
from aiperf.common.enums.metric_enums import MetricUnitT
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult
from aiperf.common.types import HierarchyT, RecordT
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


class BaseHierarchicalResultsProcessor(
    BaseMetricsProcessor, Generic[HierarchyT, RecordT]
):
    """Base class for processors that build hierarchical metrics storage.

    This class provides common functionality for:
    - Maintaining hierarchical storage structure
    - Processing individual records
    - Generating MetricResult summaries with statistics

    Subclasses must implement abstract methods to customize:
    - Tag/header formatting for display

    ClassVars to override:
        HIERARCHY_CLASS: The hierarchy model class to instantiate
        METRICS_CONFIG: List of (display_name, field_name, unit) tuples
        ENDPOINTS_DICT_FIELD: Field name in hierarchy to access endpoints dict

    Type Parameters:
        HierarchyT: The hierarchy model type (e.g., TelemetryHierarchy, ServerMetricsHierarchy)
        RecordT: The record model type (e.g., TelemetryRecord, ServerMetricRecord)
    """

    # Subclasses must override these ClassVars
    HIERARCHY_CLASS: type[HierarchyT]
    METRICS_CONFIG: (
        list[tuple[str, str, MetricUnitT]] | None
    )  # None enables dynamic discovery
    ENDPOINTS_DICT_FIELD: str

    def __init__(
        self,
        user_config: UserConfig,
        **kwargs: Any,
    ):
        """Initialize the hierarchical results processor.

        Args:
            user_config: User configuration
            **kwargs: Additional keyword arguments passed to parent
        """
        super().__init__(user_config=user_config, **kwargs)
        self._hierarchy = self.HIERARCHY_CLASS()

    def get_hierarchy(self) -> HierarchyT:
        """Get the accumulated metrics hierarchy.

        Returns:
            The hierarchy instance containing all processed metrics data
        """
        return self._hierarchy

    async def _process_record_internal(self, record: RecordT) -> None:
        """Internal method to process individual record into hierarchical storage.

        This is the core processing logic that's identical across all hierarchical
        metrics processors. Subclasses should delegate their protocol-specific
        methods (e.g., process_telemetry_record, process_server_metric_record) to this.

        Args:
            record: Record containing metrics and hierarchical metadata
        """
        self._hierarchy.add_record(record)

    async def summarize(self) -> list[MetricResult]:
        """Generate MetricResult list for real-time display and final export.

        This method is called by RecordsManager for:
        1. Final results generation when profiling completes
        2. Real-time dashboard updates (when enabled)

        The implementation follows a template method pattern:
        - Iterates through hierarchy structure (common across all types)
        - Delegates endpoint/resource specific logic to abstract methods
        - Handles errors consistently

        Supports two modes:
        - Static mode (METRICS_CONFIG set): Uses pre-defined metric configurations
        - Dynamic mode (METRICS_CONFIG is None): Discovers fields from actual data

        Returns:
            List of MetricResult objects, one per resource per metric type.
            Tags follow hierarchical naming pattern for dashboard filtering.
        """
        results = []

        # Get endpoints dict from hierarchy using class attribute field name
        endpoints_dict = getattr(self._hierarchy, self.ENDPOINTS_DICT_FIELD)

        for endpoint_url, resource_data_dict in endpoints_dict.items():
            endpoint_display = normalize_endpoint_display(endpoint_url)

            for resource_id, resource_data in resource_data_dict.items():
                # Extract metadata - subclass-specific
                metadata = resource_data.metadata

                # Choose between static config or dynamic discovery
                if self.METRICS_CONFIG is not None:
                    # Static mode: Use pre-defined METRICS_CONFIG
                    metrics_config = self.METRICS_CONFIG
                else:
                    # Dynamic mode: Discover fields from resource data
                    metrics_config = self._discover_metrics_config(resource_data)

                # Process each metric configuration
                for (
                    metric_display,
                    metric_name,
                    unit_enum,
                ) in metrics_config:
                    try:
                        # Create tag - subclass-specific formatting
                        tag = self._create_tag(
                            endpoint_url, resource_id, metric_name, metadata
                        )

                        # Create header - subclass-specific formatting
                        header = self._create_header(
                            metric_display, endpoint_display, metadata
                        )

                        # Get unit string
                        unit = unit_enum.value

                        # Get metric result - common across all types
                        result = resource_data.get_metric_result(
                            metric_name, tag, header, unit
                        )
                        results.append(result)

                    except NoMetricValue:
                        self.debug(
                            self._format_no_metric_debug(
                                metric_name, resource_id, endpoint_url
                            )
                        )
                        continue

                    except Exception as e:
                        self.exception(
                            self._format_error_message(
                                metric_name, resource_id, endpoint_url, e
                            )
                        )
                        continue

        return results

    def _discover_metrics_config(
        self, resource_data: Any
    ) -> list[tuple[str, str, MetricUnitT]]:
        """Discover metrics configuration dynamically from resource data.

        This method is called when METRICS_CONFIG is None to enable dynamic field discovery.
        Subclasses can override this method to customize the discovery logic.

        Args:
            resource_data: Resource data object (e.g., ServerMetricsData, GpuTelemetryData)

        Returns:
            List of (display_name, field_name, unit_enum) tuples for metrics in this resource
        """
        # Default implementation returns empty list (subclasses should override)
        return []

    @abstractmethod
    def _create_tag(
        self, endpoint_url: str, resource_id: str, metric_name: str, metadata: Any
    ) -> str:
        """Create a unique tag for the metric result.

        Tags are used for:
        - Dashboard filtering (hierarchical navigation)
        - Export file naming
        - API result identification

        Args:
            endpoint_url: Source endpoint URL
            resource_id: Resource identifier (gpu_uuid, server_id, etc.)
            metric_name: Metric field name
            metadata: Resource metadata (GpuMetadata, ServerMetadata, etc.)

        Returns:
            Unique tag string (e.g., "gpu_power_usage_dcgm_node1_gpu0_ef6ef310")
        """
        pass

    @abstractmethod
    def _create_header(
        self, metric_display: str, endpoint_display: str, metadata: Any
    ) -> str:
        """Create a human-readable header for the metric result.

        Headers are shown in:
        - Console output tables
        - Dashboard displays
        - Export summaries

        Args:
            metric_display: Human-readable metric name
            endpoint_display: Formatted endpoint URL for display
            metadata: Resource metadata (GpuMetadata, ServerMetadata, etc.)

        Returns:
            Header string (e.g., "GPU Power Usage | node1:9401 | GPU 0 | RTX 6000")
        """
        pass

    @abstractmethod
    def _format_no_metric_debug(
        self, metric_name: str, resource_id: str, endpoint_url: str
    ) -> str:
        """Format debug message for missing metric data.

        Args:
            metric_name: Metric field name
            resource_id: Resource identifier
            endpoint_url: Source endpoint URL

        Returns:
            Debug message string
        """
        pass

    @abstractmethod
    def _format_error_message(
        self, metric_name: str, resource_id: str, endpoint_url: str, error: Exception
    ) -> str:
        """Format error message for metric processing failures.

        Args:
            metric_name: Metric field name
            resource_id: Resource identifier
            endpoint_url: Source endpoint URL
            error: Exception that occurred

        Returns:
            Error message string
        """
        pass
