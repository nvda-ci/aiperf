# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    ErrorDetails,
    MetricResult,
    ProcessServerMetricsResult,
    ServerMetricRecord,
)
from aiperf.common.types import MessageTypeT


class ServerMetricRecordsMessage(BaseServiceMessage):
    """Message from the server metrics data collector to the records manager to notify it
    of the metric records for a batch of server samples."""

    message_type: MessageTypeT = MessageType.SERVER_METRIC_RECORDS

    collector_id: str = Field(
        ...,
        description="The ID of the server metrics data collector that collected the records.",
    )
    server_url: str = Field(
        ...,
        description="The server metrics endpoint URL that was contacted (e.g., 'http://frontend:8080/metrics')",
    )
    records: list[ServerMetricRecord] = Field(
        ..., description="The metric records collected from server monitoring"
    )
    error: ErrorDetails | None = Field(
        default=None, description="The error details if metric collection failed."
    )

    @property
    def valid(self) -> bool:
        """Whether the metric collection was valid."""

        return self.error is None and len(self.records) > 0


class ProcessServerMetricsResultMessage(BaseServiceMessage):
    """Message containing processed server metrics results - mirrors ProcessTelemetryResultMessage."""

    message_type: MessageTypeT = MessageType.PROCESS_SERVER_METRICS_RESULT

    server_metrics_result: ProcessServerMetricsResult = Field(
        description="The processed server metrics results"
    )


class ServerMetricsStatusMessage(BaseServiceMessage):
    """Message from ServerMetricsManager to SystemController indicating server metrics availability."""

    message_type: MessageTypeT = MessageType.SERVER_METRICS_STATUS

    enabled: bool = Field(
        description="Whether server metrics collection is enabled and will produce results"
    )
    reason: str | None = Field(
        default=None,
        description="Reason why server metrics are disabled (if enabled=False)",
    )
    endpoints_configured: list[str] = Field(
        default_factory=list,
        description="List of server endpoint URLs in the configured scope for display",
    )
    endpoints_reachable: list[str] = Field(
        default_factory=list,
        description="List of server endpoint URLs that were reachable and will provide data",
    )


class RealtimeServerMetricsMessage(BaseServiceMessage):
    """Message from the records manager to show real-time server metrics."""

    message_type: MessageTypeT = MessageType.REALTIME_SERVER_METRICS

    metrics: list[MetricResult] = Field(
        ..., description="The current real-time server metrics."
    )
