# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar

from pydantic import Field

from aiperf.common.enums import DatasetSamplingStrategy, MediaType
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import MediaTypeT


class Media(AIPerfBaseModel):
    """Base class for all media fields. Contains name and contents of the media data."""

    name: str = Field(default="", description="Name of the media field.")

    contents: list[str] = Field(
        default=[],
        description="List of media contents. Supports batched media payload in a single turn.",
    )


class Text(Media):
    """Media that contains text/prompt data."""

    media_type: ClassVar[MediaTypeT] = MediaType.TEXT


class Image(Media):
    """Media that contains image data."""

    media_type: ClassVar[MediaTypeT] = MediaType.IMAGE


class Audio(Media):
    """Media that contains audio data."""

    media_type: ClassVar[MediaTypeT] = MediaType.AUDIO


class Video(Media):
    """Media that contains video data."""

    media_type: ClassVar[MediaTypeT] = MediaType.VIDEO


class TurnMetadata(AIPerfBaseModel):
    """Metadata of a turn."""

    timestamp_ms: int | float | None = Field(
        default=None,
        description="The absolute timestamp of the turn in milliseconds.",
    )
    delay_ms: int | float | None = Field(
        default=None,
        description="The delay of the turn in the conversation (in milliseconds).",
    )


class Turn(TurnMetadata):
    """A dataset representation of a single turn within a conversation.

    A turn is a single interaction between a user and an AI assistant,
    and it contains timestamp, delay, and raw data that user sends in each turn.
    """

    model: str | None = Field(default=None, description="Model name used for the turn.")
    role: str | None = Field(default=None, description="Role of the turn.")
    timestamp: int | float | None = Field(
        default=None,
        description="The absolute timestamp of the turn in milliseconds.",
    )
    delay: int | float | None = Field(
        default=None,
        description="The delay of the turn in the conversation (in milliseconds).",
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate for this turn."
    )
    texts: list[Text] = Field(
        default=[], description="Collection of text data in each turn."
    )
    images: list[Image] = Field(
        default=[], description="Collection of image data in each turn."
    )
    audios: list[Audio] = Field(
        default=[], description="Collection of audio data in each turn."
    )
    videos: list[Video] = Field(
        default=[], description="Collection of video data in each turn."
    )

    def metadata(self) -> TurnMetadata:
        """Get the metadata of the turn."""
        return TurnMetadata(
            timestamp_ms=self.timestamp,
            delay_ms=self.delay,
        )


class ConversationMetadata(AIPerfBaseModel):
    """Metadata of a conversation."""

    conversation_id: str = Field(
        ...,
        description="The ID of the conversation.",
    )
    turns: list[TurnMetadata] = Field(
        default_factory=list,
        description="The metadata of the turns in the conversation.",
    )


class DatasetMetadata(AIPerfBaseModel):
    """Metadata of a dataset."""

    conversations: list[ConversationMetadata] = Field(
        default_factory=list,
        description="The conversation metadata of the dataset.",
    )
    sampling_strategy: DatasetSamplingStrategy = Field(
        ...,
        description="The sampling strategy to use when choosing conversations from the dataset.",
    )
    has_timing_data: bool = Field(
        default=False,
        description="Whether the dataset has timing data.",
    )

    @property
    def total_turn_count(self) -> int:
        """Get the total number of turns in the dataset."""
        return sum(len(conversation.turns) for conversation in self.conversations)


class Conversation(AIPerfBaseModel):
    """A dataset representation of a full conversation.

    A conversation is a sequence of turns between a user and an endpoint,
    and it contains the session ID and all the turns that consists the conversation.
    """

    session_id: str = Field(default="", description="Session ID of the conversation.")
    turns: list[Turn] = Field(
        default=[], description="List of turns in the conversation."
    )

    def metadata(self) -> ConversationMetadata:
        """Get the metadata of the conversation."""
        return ConversationMetadata(
            conversation_id=self.session_id,
            turns=[turn.metadata() for turn in self.turns],
        )


class SessionPayloads(AIPerfBaseModel):
    """A single session, with its session ID and a list of formatted payloads (one per turn)."""

    session_id: str | None = Field(
        default=None, description="Session ID of the conversation."
    )
    payloads: list[dict[str, Any]] = Field(
        default=[],
        description="List of formatted payloads in the session (one per turn). These have been formatted for the model and endpoint.",
    )


class InputsFile(AIPerfBaseModel):
    """A list of all dataset sessions. Each session contains a list of formatted payloads (one per turn).
    This is similar to the format used by GenAI-Perf for the inputs.json file.
    """

    data: list[SessionPayloads] = Field(
        default=[], description="List of all dataset sessions."
    )
