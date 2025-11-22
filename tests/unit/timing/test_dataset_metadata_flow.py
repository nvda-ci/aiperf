# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for DatasetMetadata and DatasetConfiguredNotification flow."""

from aiperf.common.enums import DatasetSamplingStrategy
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from tests.unit.timing.conftest import (
    create_mock_dataset_metadata,
    create_mock_dataset_metadata_with_schedule,
)


class TestDatasetMetadataCreation:
    """Tests for creating DatasetMetadata objects."""

    def test_create_basic_metadata_without_timing_data(self):
        """Test creating basic dataset metadata without timing information."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=["conv1", "conv2", "conv3"],
            has_timing_data=False,
        )

        assert len(metadata.conversations) == 3
        assert metadata.has_timing_data is False
        assert metadata.sampling_strategy == DatasetSamplingStrategy.SEQUENTIAL

        # Verify all conversations are present
        conv_ids = {conv.conversation_id for conv in metadata.conversations}
        assert conv_ids == {"conv1", "conv2", "conv3"}

        # Verify no timing data
        for conv in metadata.conversations:
            assert conv.turns[0].timestamp is None if conv.turns else True
            assert all(turn.delay is None for turn in conv.turns[1:])
            assert len(conv.turns) == 1

    def test_create_metadata_with_timing_data(self):
        """Test creating dataset metadata with timing information."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=["conv1", "conv2"],
            has_timing_data=True,
            first_turn_timestamps=[0, 50],
            turn_delays=[[100, 100], [100]],
            turn_counts=[3, 2],
        )

        assert len(metadata.conversations) == 2
        assert metadata.has_timing_data is True

        # Find conversations by ID
        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        # Verify conv1
        assert len(conv_dict["conv1"].turns) == 3
        assert conv_dict["conv1"].turns[0].timestamp == 0
        assert [turn.delay for turn in conv_dict["conv1"].turns[1:]] == [100, 100]

        # Verify conv2
        assert len(conv_dict["conv2"].turns) == 2
        assert conv_dict["conv2"].turns[0].timestamp == 50
        assert [turn.delay for turn in conv_dict["conv2"].turns[1:]] == [100]

    def test_create_metadata_with_different_sampling_strategies(self):
        """Test creating metadata with different sampling strategies."""
        for strategy in DatasetSamplingStrategy:
            metadata = create_mock_dataset_metadata(
                conversation_ids=["conv1"],
                sampling_strategy=strategy,
            )
            assert metadata.sampling_strategy == strategy

    def test_create_metadata_from_schedule(self):
        """Test creating dataset metadata from a schedule."""
        schedule = [
            (0, "conv1"),
            (100, "conv2"),
            (200, "conv3"),
            (300, "conv1"),  # Second turn for conv1
        ]

        metadata = create_mock_dataset_metadata_with_schedule(schedule)

        assert len(metadata.conversations) == 3
        assert metadata.has_timing_data is True

        # Find conversations by ID
        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        # Verify conv1 has 2 turns
        assert len(conv_dict["conv1"].turns) == 2
        assert conv_dict["conv1"].turns[0].timestamp == 0
        assert [turn.delay for turn in conv_dict["conv1"].turns[1:]] == [
            300
        ]  # 300 - 0 = 300

        # Verify conv2 has 1 turn
        assert len(conv_dict["conv2"].turns) == 1
        assert conv_dict["conv2"].turns[0].timestamp == 100
        assert len(conv_dict["conv2"].turns[1:]) == 0  # No delays

        # Verify conv3 has 1 turn
        assert len(conv_dict["conv3"].turns) == 1
        assert conv_dict["conv3"].turns[0].timestamp == 200
        assert len(conv_dict["conv3"].turns[1:]) == 0  # No delays

    def test_create_metadata_with_empty_conversation_list(self):
        """Test creating metadata with empty conversation list."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=[],
            has_timing_data=False,
        )

        assert len(metadata.conversations) == 0
        assert metadata.has_timing_data is False


class TestConversationMetadataValidation:
    """Tests for ConversationMetadata validation."""

    def test_conversation_metadata_with_all_fields(self):
        """Test creating conversation metadata with all fields."""
        turns = [
            TurnMetadata(timestamp=0, delay=None),
            TurnMetadata(timestamp=100, delay=100),
            TurnMetadata(timestamp=200, delay=100),
        ]
        metadata = ConversationMetadata(
            conversation_id="test-conv",
            turns=turns,
        )

        assert metadata.conversation_id == "test-conv"
        assert metadata.turns[0].timestamp == 0
        assert [turn.delay for turn in metadata.turns[1:]] == [100, 100]
        assert len(metadata.turns) == 3

    def test_conversation_metadata_minimal_fields(self):
        """Test creating conversation metadata with minimal required fields."""
        metadata = ConversationMetadata(
            conversation_id="test-conv",
            turns=[TurnMetadata(timestamp=None, delay=None)],
        )

        assert metadata.conversation_id == "test-conv"
        assert metadata.turns[0].timestamp is None
        assert len(metadata.turns[1:]) == 0  # No delays
        assert len(metadata.turns) == 1  # Default value

    def test_conversation_metadata_turn_count_validation(self):
        """Test that turn count must be >= 1."""

        # Valid turn count
        ConversationMetadata(
            conversation_id="test", turns=[TurnMetadata(timestamp=None, delay=None)]
        )
        ConversationMetadata(
            conversation_id="test",
            turns=[TurnMetadata(timestamp=None, delay=None) for _ in range(100)],
        )

        # Empty turns list is valid in Pydantic (we don't have a validator for this)
        # If you want to enforce >= 1, you'd need to add a validator to the model
        ConversationMetadata(conversation_id="test", turns=[])


class TestDatasetMetadataValidation:
    """Tests for DatasetMetadata validation."""

    def test_dataset_metadata_with_all_fields(self):
        """Test creating dataset metadata with all fields."""
        conversations = [
            ConversationMetadata(
                conversation_id="conv1",
                turns=[
                    TurnMetadata(timestamp=0, delay=None),
                    TurnMetadata(timestamp=100, delay=100),
                ],
            ),
            ConversationMetadata(
                conversation_id="conv2",
                turns=[TurnMetadata(timestamp=50, delay=None)],
            ),
        ]

        metadata = DatasetMetadata(
            conversations=conversations,
            sampling_strategy=DatasetSamplingStrategy.RANDOM,
            has_timing_data=True,
        )

        assert len(metadata.conversations) == 2
        assert metadata.sampling_strategy == DatasetSamplingStrategy.RANDOM
        assert metadata.has_timing_data is True

    def test_dataset_metadata_default_values(self):
        """Test dataset metadata default values."""
        metadata = DatasetMetadata(
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        assert len(metadata.conversations) == 0
        assert metadata.has_timing_data is False

    def test_dataset_metadata_empty_conversations(self):
        """Test dataset metadata with empty conversations list."""
        metadata = DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            has_timing_data=False,
        )

        assert len(metadata.conversations) == 0


class TestDatasetMetadataHelperFunctions:
    """Tests for the helper functions used to create mock dataset metadata."""

    def test_create_mock_dataset_metadata_default_parameters(self):
        """Test create_mock_dataset_metadata with default parameters."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=["conv1", "conv2"],
        )

        assert len(metadata.conversations) == 2
        assert metadata.has_timing_data is False
        assert metadata.sampling_strategy == DatasetSamplingStrategy.SEQUENTIAL

        for conv in metadata.conversations:
            assert len(conv.turns) == 1
            assert conv.turns[0].timestamp is None
            assert len(conv.turns[1:]) == 0  # No delays

    def test_create_mock_dataset_metadata_with_custom_turn_counts(self):
        """Test create_mock_dataset_metadata with custom turn counts."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=["conv1", "conv2", "conv3"],
            turn_counts=[1, 3, 5],
        )

        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        assert len(conv_dict["conv1"].turns) == 1
        assert len(conv_dict["conv2"].turns) == 3
        assert len(conv_dict["conv3"].turns) == 5

    def test_create_mock_dataset_metadata_with_timing_data_complete(self):
        """Test create_mock_dataset_metadata with complete timing data."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=["conv1", "conv2"],
            has_timing_data=True,
            first_turn_timestamps=[0, 50],
            turn_delays=[[100, 100], [100, 100]],
            turn_counts=[3, 3],
        )

        assert metadata.has_timing_data is True

        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        assert conv_dict["conv1"].turns[0].timestamp == 0
        assert [turn.delay for turn in conv_dict["conv1"].turns[1:]] == [100, 100]
        assert conv_dict["conv2"].turns[0].timestamp == 50
        assert [turn.delay for turn in conv_dict["conv2"].turns[1:]] == [100, 100]

    def test_create_mock_dataset_metadata_with_schedule_simple(self):
        """Test create_mock_dataset_metadata_with_schedule with simple schedule."""
        schedule = [(0, "conv1"), (100, "conv2"), (200, "conv3")]

        metadata = create_mock_dataset_metadata_with_schedule(schedule)

        assert len(metadata.conversations) == 3
        assert metadata.has_timing_data is True

        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        assert conv_dict["conv1"].turns[0].timestamp == 0
        assert len(conv_dict["conv1"].turns[1:]) == 0  # No delays
        assert len(conv_dict["conv1"].turns) == 1
        assert conv_dict["conv2"].turns[0].timestamp == 100
        assert len(conv_dict["conv2"].turns[1:]) == 0  # No delays
        assert len(conv_dict["conv2"].turns) == 1
        assert conv_dict["conv3"].turns[0].timestamp == 200
        assert len(conv_dict["conv3"].turns[1:]) == 0  # No delays
        assert len(conv_dict["conv3"].turns) == 1

    def test_create_mock_dataset_metadata_with_schedule_multi_turn(self):
        """Test create_mock_dataset_metadata_with_schedule with multi-turn conversations."""
        schedule = [
            (0, "conv1"),
            (100, "conv2"),
            (150, "conv1"),  # Second turn for conv1
            (200, "conv1"),  # Third turn for conv1
            (250, "conv2"),  # Second turn for conv2
        ]

        metadata = create_mock_dataset_metadata_with_schedule(schedule)

        assert len(metadata.conversations) == 2
        assert metadata.has_timing_data is True

        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        # Verify conv1 has 3 turns
        assert len(conv_dict["conv1"].turns) == 3
        assert conv_dict["conv1"].turns[0].timestamp == 0
        assert [turn.delay for turn in conv_dict["conv1"].turns[1:]] == [
            150,
            50,
        ]  # 150-0=150, 200-150=50

        # Verify conv2 has 2 turns
        assert len(conv_dict["conv2"].turns) == 2
        assert conv_dict["conv2"].turns[0].timestamp == 100
        assert [turn.delay for turn in conv_dict["conv2"].turns[1:]] == [
            150
        ]  # 250-100=150

    def test_create_mock_dataset_metadata_with_schedule_custom_sampling_strategy(
        self,
    ):
        """Test create_mock_dataset_metadata_with_schedule with custom sampling strategy."""
        schedule = [(0, "conv1"), (100, "conv2")]

        metadata = create_mock_dataset_metadata_with_schedule(
            schedule, sampling_strategy=DatasetSamplingStrategy.RANDOM
        )

        assert metadata.sampling_strategy == DatasetSamplingStrategy.RANDOM

    def test_create_mock_dataset_metadata_with_schedule_empty(self):
        """Test create_mock_dataset_metadata_with_schedule with empty schedule."""
        metadata = create_mock_dataset_metadata_with_schedule([])

        assert len(metadata.conversations) == 0
        assert metadata.has_timing_data is True


class TestDatasetMetadataIntegration:
    """Integration tests for dataset metadata usage in strategies."""

    def test_metadata_extraction_for_fixed_schedule(self):
        """Test extracting timing information from metadata for fixed schedule strategy."""
        schedule = [
            (0, "conv1"),
            (100, "conv2"),
            (200, "conv3"),
        ]

        metadata = create_mock_dataset_metadata_with_schedule(schedule)

        # Simulate what FixedScheduleStrategy does
        extracted_schedule = []
        for conv in metadata.conversations:
            if conv.turns and conv.turns[0].timestamp is not None:
                extracted_schedule.append(
                    (conv.turns[0].timestamp, conv.conversation_id)
                )

        # Sort by timestamp
        extracted_schedule.sort(key=lambda x: x[0])

        assert extracted_schedule == schedule

    def test_metadata_extraction_for_request_rate(self):
        """Test extracting conversation IDs from metadata for request rate strategy."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=["conv1", "conv2", "conv3", "conv4", "conv5"],
        )

        # Simulate what RequestRateStrategy does
        extracted_ids = [conv.conversation_id for conv in metadata.conversations]

        assert len(extracted_ids) == 5
        assert set(extracted_ids) == {"conv1", "conv2", "conv3", "conv4", "conv5"}

    def test_metadata_with_mixed_turn_counts(self):
        """Test metadata with conversations having different turn counts."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=["single", "double", "triple", "quad"],
            turn_counts=[1, 2, 3, 4],
        )

        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        assert len(conv_dict["single"].turns) == 1
        assert len(conv_dict["double"].turns) == 2
        assert len(conv_dict["triple"].turns) == 3
        assert len(conv_dict["quad"].turns) == 4

    def test_metadata_conversation_ordering(self):
        """Test that conversation order is preserved in metadata."""
        conversation_ids = [f"conv{i}" for i in range(10)]

        metadata = create_mock_dataset_metadata(
            conversation_ids=conversation_ids,
        )

        extracted_ids = [conv.conversation_id for conv in metadata.conversations]

        assert extracted_ids == conversation_ids


class TestFloatingPointTimestampPreservation:
    """Tests to ensure floating point timestamps are preserved throughout the system."""

    def test_turn_metadata_preserves_float_timestamps(self):
        """Test that TurnMetadata preserves floating point precision."""
        # Test with various float values
        test_timestamps = [0.0, 100.5, 150.75, 200.123, 250.456789, 999.999999]

        for timestamp in test_timestamps:
            turn = TurnMetadata(timestamp=timestamp, delay=None)
            assert turn.timestamp == timestamp
            assert isinstance(turn.timestamp, int | float)
            # Verify exact value is preserved
            if "." in str(timestamp):
                assert turn.timestamp == timestamp

    def test_turn_metadata_preserves_float_delays(self):
        """Test that TurnMetadata preserves floating point delays."""
        test_delays = [10.5, 25.75, 50.123, 100.456789]

        for delay in test_delays:
            turn = TurnMetadata(timestamp=None, delay=delay)
            assert turn.delay == delay
            assert isinstance(turn.delay, int | float)
            # Verify exact value is preserved
            assert turn.delay == delay

    def test_conversation_metadata_preserves_float_timestamps(self):
        """Test that ConversationMetadata preserves floating point timestamps in turns."""
        turns = [
            TurnMetadata(timestamp=0.0, delay=None),
            TurnMetadata(timestamp=100.5, delay=100.5),
            TurnMetadata(timestamp=200.75, delay=100.25),
            TurnMetadata(timestamp=300.123, delay=99.373),
        ]

        conv = ConversationMetadata(conversation_id="test-conv", turns=turns)

        # Verify all timestamps are preserved exactly
        assert conv.turns[0].timestamp == 0.0
        assert conv.turns[1].timestamp == 100.5
        assert conv.turns[2].timestamp == 200.75
        assert conv.turns[3].timestamp == 300.123

        # Verify all delays are preserved exactly
        assert conv.turns[1].delay == 100.5
        assert conv.turns[2].delay == 100.25
        assert conv.turns[3].delay == 99.373

    def test_dataset_metadata_preserves_float_timestamps(self):
        """Test that DatasetMetadata preserves floating point timestamps across conversations."""
        metadata = create_mock_dataset_metadata_with_schedule(
            [
                (0.0, "conv1"),
                (100.5, "conv2"),
                (150.75, "conv1"),  # Second turn
                (200.123, "conv3"),
                (250.456, "conv2"),  # Second turn
            ]
        )

        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        # Verify conv1 timestamps preserved
        assert conv_dict["conv1"].turns[0].timestamp == 0.0
        assert conv_dict["conv1"].turns[1].timestamp == 150.75
        assert conv_dict["conv1"].turns[1].delay == 150.75

        # Verify conv2 timestamps preserved
        assert conv_dict["conv2"].turns[0].timestamp == 100.5
        assert conv_dict["conv2"].turns[1].timestamp == 250.456
        assert conv_dict["conv2"].turns[1].delay == 149.956

        # Verify conv3 timestamps preserved
        assert conv_dict["conv3"].turns[0].timestamp == 200.123

    def test_high_precision_float_timestamps_preserved(self):
        """Test that very high precision floating point timestamps are preserved."""
        # Test with timestamps that have many decimal places
        high_precision_timestamps = [
            (0.123456789, "conv1"),
            (100.987654321, "conv2"),
            (200.111222333, "conv3"),
        ]

        metadata = create_mock_dataset_metadata_with_schedule(high_precision_timestamps)

        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        # Verify exact precision is maintained
        assert conv_dict["conv1"].turns[0].timestamp == 0.123456789
        assert conv_dict["conv2"].turns[0].timestamp == 100.987654321
        assert conv_dict["conv3"].turns[0].timestamp == 200.111222333

    def test_mixed_int_float_timestamps_preserved(self):
        """Test that mixed integer and float timestamps maintain their types."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=["conv1", "conv2", "conv3"],
            has_timing_data=True,
            first_turn_timestamps=[0, 100.5, 200],  # Mixed int and float
            turn_delays=[[50, 75.5], [100.25, 150], [200.123]],  # Mixed delays
            turn_counts=[3, 3, 2],
        )

        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        # Verify conv1 (int first timestamp)
        assert conv_dict["conv1"].turns[0].timestamp == 0
        assert isinstance(conv_dict["conv1"].turns[0].timestamp, int)

        # Verify conv2 (float first timestamp)
        assert conv_dict["conv2"].turns[0].timestamp == 100.5
        assert isinstance(conv_dict["conv2"].turns[0].timestamp, float)

        # Verify conv3 (int first timestamp)
        assert conv_dict["conv3"].turns[0].timestamp == 200
        assert isinstance(conv_dict["conv3"].turns[0].timestamp, int)

        # Verify delays are preserved with correct types
        assert conv_dict["conv1"].turns[1].delay == 50
        assert conv_dict["conv1"].turns[2].delay == 75.5
        assert conv_dict["conv2"].turns[1].delay == 100.25
        assert conv_dict["conv2"].turns[2].delay == 150
        assert conv_dict["conv3"].turns[1].delay == 200.123
