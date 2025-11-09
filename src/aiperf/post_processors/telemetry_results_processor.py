# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.processor_summary_results import TelemetrySummaryResult
from aiperf.common.models.telemetry_models import TelemetryHierarchy, TelemetryRecord
from aiperf.common.protocols import (
    TelemetryResultsProcessorProtocol,
)
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(TelemetryResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.TELEMETRY_RESULTS)
class TelemetryResultsProcessor(BaseMetricsProcessor):
    """Process individual TelemetryRecord objects into hierarchical storage."""

    def __init__(self, user_config: UserConfig, **kwargs: Any):
        super().__init__(user_config=user_config, **kwargs)

        self._telemetry_hierarchy = TelemetryHierarchy()

    def get_telemetry_hierarchy(self) -> TelemetryHierarchy:
        """Get the accumulated telemetry hierarchy."""
        return self._telemetry_hierarchy

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        """Process individual telemetry record into hierarchical storage.

        Args:
            record: TelemetryRecord containing GPU metrics and hierarchical metadata
        """
        self._telemetry_hierarchy.add_record(record)

    async def summarize(self) -> TelemetrySummaryResult:
        """Generate summary with telemetry hierarchy.

        This method is called by RecordsManager for:
        1. Final results generation when profiling completes
        2. Real-time dashboard updates when --gpu-telemetry dashboard is enabled

        The telemetry hierarchy contains all GPU data in its native structured format.
        MetricResult objects can be generated on-demand from the hierarchy when needed
        for display or export.

        Returns:
            TelemetrySummaryResult containing the telemetry hierarchy.
        """
        # Get endpoints tested and successful from hierarchy
        endpoints = list(self._telemetry_hierarchy.dcgm_endpoints.keys())

        return TelemetrySummaryResult(
            telemetry_data=self._telemetry_hierarchy,
            endpoints_tested=endpoints,
            endpoints_successful=endpoints,
            error_summary=[],
        )
