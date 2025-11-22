# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import Conversation, DatasetMetadata, Turn
from aiperf.common.types import MessageTypeT


class ConversationRequestMessage(BaseServiceMessage):
    """Message to request a full conversation by ID."""

    message_type: MessageTypeT = MessageType.CONVERSATION_REQUEST

    conversation_id: str = Field(..., description="The session ID of the conversation")
    credit_phase: CreditPhase | None = Field(
        default=None,
        description="The type of credit phase (either warmup or profiling). If not provided, the dataset manager will use the default credit phase.",
    )


class ConversationResponseMessage(BaseServiceMessage):
    """Message containing a full conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_RESPONSE
    conversation: Conversation = Field(..., description="The conversation data")


class ConversationTurnRequestMessage(BaseServiceMessage):
    """Message to request a single turn from a conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_TURN_REQUEST

    conversation_id: str = Field(
        ...,
        description="The ID of the conversation.",
    )
    turn_index: int = Field(
        ...,
        ge=0,
        description="The index of the turn in the conversation.",
    )


class ConversationTurnResponseMessage(BaseServiceMessage):
    """Message containing a single turn from a conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_TURN_RESPONSE

    turn: Turn = Field(..., description="The turn data")


class DatasetConfiguredNotification(BaseServiceMessage):
    """Notification sent to notify other services that the dataset has been configured."""

    message_type: MessageTypeT = MessageType.DATASET_CONFIGURED_NOTIFICATION

    metadata: DatasetMetadata = Field(
        ...,
        description="The metadata of the dataset.",
    )
