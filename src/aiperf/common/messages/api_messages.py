# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API command and response messages.

These messages are used by the APIService to request data from other services
via the command pattern.
"""

from pydantic import Field

from aiperf.common.enums import CommandType, LifecycleState
from aiperf.common.messages.command_messages import (
    CommandMessage,
    CommandSuccessResponse,
)
from aiperf.common.types import CommandTypeT
from aiperf.plugin.enums import ServiceType


class GetAPIStatusCommand(CommandMessage):
    """Command to request current benchmark status from SystemController."""

    command: CommandTypeT = CommandType.GET_API_STATUS
    target_service_type: ServiceType | None = Field(  # type: ignore[assignment]
        default=ServiceType.SYSTEM_CONTROLLER,
        description="Target service for this command",
    )


class GetAPIStatusResponse(CommandSuccessResponse):
    """Response containing current benchmark status."""

    command: CommandTypeT = CommandType.GET_API_STATUS
    state: LifecycleState = Field(
        default=LifecycleState.CREATED,
        description="Current state of the benchmark",
    )
    phase: str | None = Field(
        default=None,
        description="Current phase of the benchmark (warmup, profiling, processing)",
    )
    profile_id: str | None = Field(
        default=None,
        description="Current profile ID if running",
    )
    error: str | None = Field(
        default=None,
        description="Error message if benchmark failed",
    )
