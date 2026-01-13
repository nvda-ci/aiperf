# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ConversationSource - behavior focused.

Tests sampling behavior, metadata access, iteration, and multi-turn helpers.
"""

import pytest

from aiperf.common.enums import DatasetSamplingStrategy
from aiperf.common.factories import DatasetSamplingStrategyFactory
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.timing.conversation_source import ConversationSource, SampledSession
from tests.unit.timing.conftest import make_credit


@pytest.fixture
def sample_dataset():
    """Create sample dataset with 3 conversations."""
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="conv1",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(delay_ms=100.0),
                ],
            ),
            ConversationMetadata(
                conversation_id="conv2",
                turns=[TurnMetadata(timestamp_ms=50.0)],
            ),
            ConversationMetadata(
                conversation_id="conv3",
                turns=[
                    TurnMetadata(timestamp_ms=100.0),
                    TurnMetadata(delay_ms=50.0),
                    TurnMetadata(delay_ms=75.0),
                ],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


@pytest.fixture
def conversation_source(sample_dataset):
    """Create ConversationSource with sequential sampling."""
    sampler = DatasetSamplingStrategyFactory.create_instance(
        sample_dataset.sampling_strategy,
        conversation_ids=[c.conversation_id for c in sample_dataset.conversations],
    )
    return ConversationSource(sample_dataset, sampler)


class TestConversationSampling:
    """Test conversation sampling behavior."""

    def test_next_returns_sampled_session(self, conversation_source):
        """Should return SampledSession with metadata."""
        sampled = conversation_source.next()

        assert isinstance(sampled, SampledSession)
        assert sampled.conversation_id in ["conv1", "conv2", "conv3"]
        assert sampled.metadata is not None
        assert isinstance(sampled.x_correlation_id, str)
        assert len(sampled.x_correlation_id) == 36  # UUID format

    def test_next_generates_unique_correlation_ids(self, conversation_source):
        """Should generate unique correlation IDs for each sample."""
        sampled1 = conversation_source.next()
        sampled2 = conversation_source.next()

        assert sampled1.x_correlation_id != sampled2.x_correlation_id

    def test_sequential_sampling_order(self, sample_dataset):
        """Should sample conversations in sequential order."""
        sampler = DatasetSamplingStrategyFactory.create_instance(
            DatasetSamplingStrategy.SEQUENTIAL,
            conversation_ids=["conv1", "conv2", "conv3"],
        )
        source = ConversationSource(sample_dataset, sampler)

        sampled1 = source.next()
        sampled2 = source.next()
        sampled3 = source.next()

        assert sampled1.conversation_id == "conv1"
        assert sampled2.conversation_id == "conv2"
        assert sampled3.conversation_id == "conv3"


class TestMetadataAccess:
    """Test metadata access behavior."""

    def test_get_metadata_returns_conversation(self, conversation_source):
        """Should return metadata for valid conversation ID."""
        metadata = conversation_source.get_metadata("conv1")

        assert metadata.conversation_id == "conv1"
        assert len(metadata.turns) == 2

    def test_get_metadata_raises_for_invalid_id(self, conversation_source):
        """Should raise KeyError for invalid conversation ID."""
        with pytest.raises(KeyError, match="No metadata for conversation invalid_id"):
            conversation_source.get_metadata("invalid_id")

    def test_get_metadata_matches_sampled_metadata(self, conversation_source):
        """Metadata from get_metadata should match sampled metadata."""
        sampled = conversation_source.next()
        metadata = conversation_source.get_metadata(sampled.conversation_id)

        assert metadata == sampled.metadata


# =============================================================================
# Multi-turn Helper Tests (merged from ConversationContext)
# =============================================================================


@pytest.fixture
def multi_turn_test_dataset(three_turn_conversation):
    """Dataset with single three-turn conversation."""
    return DatasetMetadata(
        conversations=[three_turn_conversation],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


@pytest.fixture
def multi_turn_source(multi_turn_test_dataset):
    """ConversationSource for multi-turn tests."""
    sampler = DatasetSamplingStrategyFactory.create_instance(
        multi_turn_test_dataset.sampling_strategy,
        conversation_ids=[
            c.conversation_id for c in multi_turn_test_dataset.conversations
        ],
    )
    return ConversationSource(multi_turn_test_dataset, sampler)


class TestGetNextTurnMetadata:
    """Tests for get_next_turn_metadata method."""

    def test_returns_next_turn_metadata(self, multi_turn_source):
        """Returns metadata for turn_index + 1."""
        credit = make_credit(
            conversation_id="conv-3turn", turn_index=0, is_final_turn=False
        )

        result = multi_turn_source.get_next_turn_metadata(credit)

        assert result.delay_ms == 50.0  # Turn 1's delay

    def test_returns_correct_turn_for_different_indices(self, multi_turn_source):
        """Works for different turn indices."""
        credit = make_credit(
            conversation_id="conv-3turn", turn_index=1, is_final_turn=False
        )

        result = multi_turn_source.get_next_turn_metadata(credit)

        assert result.delay_ms == 100.0  # Turn 2's delay

    def test_raises_when_no_next_turn(self, multi_turn_source):
        """Raises ValueError when no next turn exists."""
        credit = make_credit(
            conversation_id="conv-3turn", turn_index=2, is_final_turn=True
        )

        with pytest.raises(ValueError, match="No turn 3"):
            multi_turn_source.get_next_turn_metadata(credit)
