# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.models.dataset_models import Conversation, Text, Turn


class TestConversationValidation:
    """Test Conversation model validation."""

    def test_conversation_requires_session_id(self):
        """Test that Conversation requires a session_id."""
        with pytest.raises(ValidationError) as exc_info:
            Conversation()

        assert "session_id" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)

    def test_conversation_rejects_empty_session_id(self):
        """Test that Conversation rejects empty string session_id."""
        with pytest.raises(ValidationError) as exc_info:
            Conversation(session_id="")

        assert "session_id must be a non-empty string" in str(exc_info.value)

    def test_conversation_rejects_whitespace_session_id(self):
        """Test that Conversation rejects whitespace-only session_id."""
        with pytest.raises(ValidationError) as exc_info:
            Conversation(session_id="   ")

        assert "session_id must be a non-empty string" in str(exc_info.value)

    def test_conversation_rejects_tab_whitespace_session_id(self):
        """Test that Conversation rejects tab/newline whitespace session_id."""
        with pytest.raises(ValidationError) as exc_info:
            Conversation(session_id="\t\n  ")

        assert "session_id must be a non-empty string" in str(exc_info.value)

    def test_conversation_accepts_valid_session_id(self):
        """Test that Conversation accepts valid session_id."""
        conversation = Conversation(session_id="session_000001")
        assert conversation.session_id == "session_000001"
        assert conversation.turns == []

    def test_conversation_accepts_uuid_session_id(self):
        """Test that Conversation accepts UUID session_id."""
        uuid_id = "a1b2c3d4-5678-90ab-cdef-1234567890ab"
        conversation = Conversation(session_id=uuid_id)
        assert conversation.session_id == uuid_id

    def test_conversation_with_turns(self):
        """Test that Conversation can be created with turns."""
        turn = Turn(
            texts=[Text(contents=["Hello"])],
            role="user",
            model="test-model",
        )
        conversation = Conversation(session_id="test_session", turns=[turn])
        assert conversation.session_id == "test_session"
        assert len(conversation.turns) == 1
        assert conversation.turns[0].texts[0].contents == ["Hello"]
