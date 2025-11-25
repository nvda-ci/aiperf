# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

from pydantic import Field

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import AIPerfBaseModel, ErrorDetails
from aiperf.common.types import MessageTypeT


class Credit(AIPerfBaseModel):
    """Credit representing the right to make a single request to an inference server."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this credit.",
    )
    phase: CreditPhase = Field(
        ..., description="The type of credit phase, such as warmup or profiling."
    )
    num: int = Field(
        ...,
        ge=0,
        description="The sequential number of the credit in the credit phase. "
        "This is used to track the order that requests are sent in.",
    )
    cancel_after_ns: int | None = Field(
        default=None,
        ge=0,
        description="Delay in nanoseconds after which the request should be cancelled. Only applicable if should_cancel is True.",
    )
    conversation_id: str = Field(
        ...,
        description="The ID of the conversation that this credit belongs to.",
    )
    x_correlation_id: str = Field(
        ...,
        description="Conversation instance ID. Shared across all turns. Used by StickyCreditRouter for sticky routing.",
    )
    turn_index: int = Field(
        ...,
        ge=0,
        description="Turn index within conversation (0-based).",
    )
    is_final_turn: bool = Field(
        default=False,
        description="True if this is the last turn of the conversation. Used for cache eviction and assignment cleanup.",
    )

    @property
    def should_cancel(self) -> bool:
        """Whether this credit should be cancelled."""
        return self.cancel_after_ns is not None


class CreditDropMessage(BaseServiceMessage):
    """Message indicating that a credit has been dropped.

    This message is sent by the TimingManager to the Worker to indicate that some work needs to be done.
    """

    message_type: MessageTypeT = MessageType.CREDIT_DROP

    credit: Credit = Field(
        ...,
        description="The credit data.",
    )


class CreditReturnMessage(BaseServiceMessage):
    """Message indicating that a credit has been returned.

    This message is sent by a worker to the timing manager to indicate that work has
    been completed.
    """

    message_type: MessageTypeT = MessageType.CREDIT_RETURN

    credit: Credit = Field(
        ...,
        description="The original credit data.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )


class CreditContext(AIPerfBaseModel):
    """Context for a credit."""

    credit: Credit = Field(
        ...,
        description="The credit data.",
    )
    drop_perf_ns: int = Field(
        ...,
        ge=0,
        description="The perf_ns timestamp when the credit was dropped. This is the time the credit was received by the worker.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )

    def to_return_message(self, service_id: str) -> CreditReturnMessage:
        """Convert the credit context to a credit return message."""
        return CreditReturnMessage(
            service_id=service_id,
            credit=self.credit,
            error=self.error,
        )


class CreditPhaseStartMessage(BaseServiceMessage):
    """Message for credit phase start. Sent by the TimingManager to report that a credit phase has started."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_START
    phase: CreditPhase = Field(..., description="The type of credit phase")
    start_ns: int = Field(
        ge=1,
        description="The start time of the credit phase in nanoseconds.",
    )
    total_expected_requests: int | None = Field(
        default=None,
        ge=1,
        description="The total number of expected requests. If None, the phase is not request count based.",
    )
    expected_duration_sec: float | None = Field(
        default=None,
        ge=1,
        description="The expected duration of the credit phase in seconds. If None, the phase is not time based.",
    )


class CreditPhaseProgressMessage(BaseServiceMessage):
    """Sent by the TimingManager to report the progress of a credit phase."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_PROGRESS
    phase: CreditPhase = Field(..., description="The type of credit phase")
    sent: int = Field(
        ...,
        ge=0,
        description="The number of sent credits",
    )
    completed: int = Field(
        ...,
        ge=0,
        description="The number of completed credits (returned from the workers)",
    )


class CreditPhaseSendingCompleteMessage(BaseServiceMessage):
    """Message for credit phase sending complete. Sent by the TimingManager to report that a credit phase has completed sending."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_SENDING_COMPLETE
    phase: CreditPhase = Field(..., description="The type of credit phase")
    sent_end_ns: int = Field(
        ...,
        ge=1,
        description="The time of the last sent credit in nanoseconds.",
    )
    sent: int = Field(
        ...,
        ge=0,
        description="The final number of sent credits.",
    )


class CreditPhaseCompleteMessage(BaseServiceMessage):
    """Message for credit phase complete. Sent by the TimingManager to report that a credit phase has completed."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_COMPLETE
    phase: CreditPhase = Field(..., description="The type of credit phase")
    completed: int = Field(
        ...,
        ge=0,
        description="The number of completed credits (returned from the workers). This is the final count of completed credits.",
    )
    end_ns: int = Field(
        ...,
        ge=1,
        description="The time in which the last credit was returned from the workers in nanoseconds",
    )
    timeout_triggered: bool = Field(
        default=False,
        description="Whether this phase completed because a timeout was triggered",
    )
    final_request_count: int = Field(
        ...,
        ge=0,
        description="The total number of requests sent.",
    )


class CreditsCompleteMessage(BaseServiceMessage):
    """Credits complete message sent by the TimingManager to the System controller to signify all Credit Phases
    have been completed."""

    message_type: MessageTypeT = MessageType.CREDITS_COMPLETE
