# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models import MetricResult
from aiperf.common.models.server_metrics_models import (
    ServerMetricsHierarchy,
    ServerMetricsRecord,
    TimeRangeFilter,
)
from aiperf.common.protocols import ServerMetricsResultsProcessorProtocol
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ServerMetricsResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.SERVER_METRICS_RESULTS)
class ServerMetricsResultsProcessor(BaseMetricsProcessor):
    """Process individual ServerMetricsRecord objects into hierarchical storage.

    This processor accumulates server metrics data from Prometheus endpoints
    and provides summarization similar to TelemetryResultsProcessor.

    Supports:
    - Gauge metrics: Point-in-time values, aggregated with statistics
    - Counter metrics: Cumulative values with delta calculation from reference point
    - Histogram metrics: Bucket distributions with delta calculation
    - Summary metrics: Pre-computed quantiles with delta sum/count

    Time filtering:
    - Warmup period exclusion via start_ns in TimeRangeFilter
    - End buffer exclusion via end_ns in TimeRangeFilter
    - Reference point for deltas is the last snapshot before start_ns
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any):
        super().__init__(user_config=user_config, **kwargs)

        self._server_metrics_hierarchy = ServerMetricsHierarchy()
        self._time_filter: TimeRangeFilter | None = None

    def get_server_metrics_hierarchy(self) -> ServerMetricsHierarchy:
        """Get the accumulated server metrics hierarchy."""
        return self._server_metrics_hierarchy

    def set_time_filter(self, time_filter: TimeRangeFilter | None) -> None:
        """Set the time filter for aggregation.

        The time filter controls:
        - Which snapshots are included in aggregation (within start_ns to end_ns)
        - The reference point for delta calculations (last snapshot before start_ns)

        Args:
            time_filter: TimeRangeFilter with start_ns (post-warmup) and end_ns (pre-flush-buffer)
        """
        self._time_filter = time_filter

    def get_time_filter(self) -> TimeRangeFilter | None:
        """Get the current time filter."""
        return self._time_filter

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record into hierarchical storage.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics and metadata
        """
        self._server_metrics_hierarchy.add_record(record)

    async def summarize(
        self, time_filter: TimeRangeFilter | None = None
    ) -> list[MetricResult]:
        """Generate MetricResult list for display and final export.

        Note: This method's output is not used for export. The actual export uses
        pre-computed endpoint_summaries generated in records_manager. This method
        is only called for error detection during processing.

        Returns:
            Empty list (results not used for export)
        """
        # Return empty list - actual export uses pre-computed endpoint_summaries
        # This method is only called for error detection, output is discarded
        return []
