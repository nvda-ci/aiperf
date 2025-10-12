# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from pydantic import ConfigDict, Field

from aiperf.common.enums import (
    LifecycleState,
    ServiceRegistrationStatus,
)
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.types import ServiceTypeT


class ServiceRunInfo(AIPerfBaseModel):
    """Base model for tracking service run information."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    service_type: ServiceTypeT = Field(
        ...,
        description="The type of service",
    )
    service_id: str = Field(
        ...,
        description="The ID of the service",
    )
    registration_status: ServiceRegistrationStatus = Field(
        default=ServiceRegistrationStatus.UNREGISTERED,
        description="The registration status of the service",
    )
    first_seen: int | None = Field(
        default_factory=time.time_ns,
        description="The first time the service was seen",
    )
    last_seen: int | None = Field(
        default_factory=time.time_ns,
        description="The last time the service was seen",
    )
    state: LifecycleState = Field(
        default=LifecycleState.CREATED,
        description="The current state of the service",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the service failed",
    )
    required: bool = Field(
        default=False,
        description="Whether the service is required to be running, such that if it fails, the system will stop",
    )
