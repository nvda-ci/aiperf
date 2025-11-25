# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.enums.worker_enums import WorkerStatus
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.common.types import MessageTypeT


class WorkerHealthMessage(BaseServiceMessage):
    """Message for a worker health check."""

    message_type: MessageTypeT = MessageType.WORKER_HEALTH

    health: ProcessHealth = Field(..., description="The health of the worker process")

    # Worker specific fields
    task_stats: WorkerTaskStats = Field(
        ...,
        description="Stats for the tasks that have been sent to the worker",
    )

    @property
    def error_rate(self) -> float:
        """The error rate of the worker."""
        if self.task_stats.total == 0:
            return 0
        return self.task_stats.failed / self.task_stats.total


class WorkerReadyMessage(BaseServiceMessage):
    """
    Message sent by worker via DEALER socket to announce readiness.

    Sent immediately after connecting to ROUTER.
    service_id from BaseServiceMessage contains worker_id
    """

    message_type: MessageTypeT = MessageType.WORKER_READY


class WorkerShutdownMessage(BaseServiceMessage):
    """
    Message sent by worker via DEALER socket before graceful shutdown.

    service_id from BaseServiceMessage contains worker_id
    """

    message_type: MessageTypeT = MessageType.WORKER_SHUTDOWN


class WorkerStatusSummaryMessage(BaseServiceMessage):
    """Message for a worker status summary."""

    message_type: MessageTypeT = MessageType.WORKER_STATUS_SUMMARY

    worker_statuses: dict[str, WorkerStatus] = Field(
        ...,
        description="A mapping of worker IDs to their status",
    )
