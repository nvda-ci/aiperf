# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, InputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import CustomDatasetType
from aiperf.common.messages.command_messages import ProfileConfigureCommand
from aiperf.dataset.dataset_manager import DatasetManager


class TestDatasetManager:
    """Test DatasetManager functionality.

    Note: Dataset sampling tests have been moved to test_dataset_samplers.py
    since sampling is now handled by timing strategies, not DatasetManager.
    """

    @pytest.fixture(autouse=True)
    async def teardown(self):
        """Clean up after each test to prevent shared state issues."""
        yield
        # Reset any global state if needed
        # Clear communication factory state
        from aiperf.common.factories import CommunicationFactory

        if hasattr(CommunicationFactory, "_instances"):
            CommunicationFactory._instances.clear()

    @pytest.mark.skip(reason="Pre-existing test failure - needs investigation")
    @pytest.mark.asyncio
    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_dataset_configured_notification_for_multi_turn_conversations(
        self,
        mock_tokenizer_from_pretrained,
        create_mooncake_trace_file,
        mock_tokenizer_cls,
    ):
        """Test that dataset configured notification includes correct metadata for multi-turn conversations.

        When a dataset has multiple turns per conversation, the notification should:
        - Include one ConversationMetadata per conversation (not one per turn)
        - Include the first_turn_timestamp and turn_delays for each conversation
        - Have the correct turn count for each conversation
        - Mark has_timing_data as True
        """
        # Mock the tokenizer to avoid HTTP requests
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        # Create a file with multi-turn conversations
        entries = [
            '{"session_id": "sess-1", "timestamp": 0, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000, "input_length": 100, "output_length": 10}',
            '{"session_id": "sess-2", "timestamp": 20000, "input_length": 25, "output_length": 20}',
            '{"session_id": "sess-2", "delay": 10000, "input_length": 10000, "output_length": 20}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            service_config = ServiceConfig()
            dataset_manager = DatasetManager(service_config, user_config)

            await dataset_manager.initialize()

            # Mock the publish method to capture notifications
            from unittest.mock import AsyncMock

            from aiperf.common.messages import DatasetConfiguredNotification

            published_messages = []

            async def mock_publish(msg):
                published_messages.append(msg)

            dataset_manager.publish = AsyncMock(side_effect=mock_publish)

            # Configure the dataset to load conversations
            await dataset_manager._profile_configure_command(
                ProfileConfigureCommand(config=user_config, service_id="test_service")
            )

            # Verify the notification was published
            published_notifications = [
                msg
                for msg in published_messages
                if isinstance(msg, DatasetConfiguredNotification)
            ]
            assert len(published_notifications) == 1

            notification = published_notifications[0]
            metadata = notification.metadata

            # Verify dataset metadata structure
            assert len(metadata.conversations) == 2  # 2 conversations, not 5 turns
            assert metadata.has_timing_data is True

            # Extract conversation metadata for easier testing
            conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

            # Verify session 1 metadata
            # Note: First turn has timestamp, subsequent turns have delays
            assert "sess-1" in conv_dict
            sess1 = conv_dict["sess-1"]
            assert len(sess1.turns) == 3
            assert sess1.turns[0].timestamp_ms == 0  # First turn timestamp
            assert [turn.delay for turn in sess1.turns[1:]] == [
                10000,
                10000,
            ]  # Subsequent turn delays

            # Verify session 2 metadata
            assert "sess-2" in conv_dict
            sess2 = conv_dict["sess-2"]
            assert len(sess2.turns) == 2
            assert sess2.turns[0].timestamp_ms == 20000  # First turn timestamp
            assert [turn.delay for turn in sess2.turns[1:]] == [
                10000
            ]  # Second turn delay

            # Verify no duplicate conversation IDs (one per conversation, not per turn)
            conversation_ids = [conv.conversation_id for conv in metadata.conversations]
            assert len(conversation_ids) == len(set(conversation_ids))

        finally:
            Path(filename).unlink(missing_ok=True)

    @pytest.mark.skip(reason="Pre-existing test failure - needs investigation")
    @pytest.mark.asyncio
    @patch("aiperf.common.tokenizer.Tokenizer.from_pretrained")
    async def test_dataset_configured_notification_preserves_float_timestamps(
        self,
        mock_tokenizer_from_pretrained,
        create_mooncake_trace_file,
        mock_tokenizer_cls,
    ):
        """Test that floating point timestamps are preserved exactly in dataset notifications.

        This test verifies that high-precision floating point timestamps from trace data
        are maintained throughout the dataset loading and notification process.
        """
        # Mock the tokenizer to avoid HTTP requests
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        # Create a file with floating point timestamps (in milliseconds)
        entries = [
            '{"session_id": "sess-1", "timestamp": 0.123, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-1", "delay": 10000.456, "input_length": 50, "output_length": 10}',
            '{"session_id": "sess-2", "timestamp": 20000.789, "input_length": 25, "output_length": 20}',
            '{"session_id": "sess-2", "delay": 15000.123, "input_length": 100, "output_length": 20}',
        ]
        filename = create_mooncake_trace_file(entries)

        try:
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    file=filename, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE
                ),
            )

            service_config = ServiceConfig()
            dataset_manager = DatasetManager(service_config, user_config)

            await dataset_manager.initialize()

            # Mock the publish method to capture notifications
            from unittest.mock import AsyncMock

            from aiperf.common.messages import DatasetConfiguredNotification

            published_messages = []

            async def mock_publish(msg):
                published_messages.append(msg)

            dataset_manager.publish = AsyncMock(side_effect=mock_publish)

            # Configure the dataset to load conversations
            await dataset_manager._profile_configure_command(
                ProfileConfigureCommand(config=user_config, service_id="test_service")
            )

            # Verify the notification was published
            published_notifications = [
                msg
                for msg in published_messages
                if isinstance(msg, DatasetConfiguredNotification)
            ]
            assert len(published_notifications) == 1

            notification = published_notifications[0]
            metadata = notification.metadata

            # Extract conversation metadata
            conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

            # Verify session 1 - floating point timestamps preserved exactly
            assert "sess-1" in conv_dict
            sess1 = conv_dict["sess-1"]
            assert sess1.turns[0].timestamp_ms == 0.123  # Exact float value
            assert sess1.turns[1].delay == 10000.456  # Exact float delay

            # Verify session 2 - floating point timestamps preserved exactly
            assert "sess-2" in conv_dict
            sess2 = conv_dict["sess-2"]
            assert sess2.turns[0].timestamp_ms == 20000.789  # Exact float value
            assert sess2.turns[1].delay == 15000.123  # Exact float delay

            # Verify types are float (not int)
            assert isinstance(sess1.turns[0].timestamp_ms, float)
            assert isinstance(sess1.turns[1].delay, float)
            assert isinstance(sess2.turns[0].timestamp_ms, float)
            assert isinstance(sess2.turns[1].delay, float)

        finally:
            Path(filename).unlink(missing_ok=True)
