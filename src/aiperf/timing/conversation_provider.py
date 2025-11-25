# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections import deque

from aiperf.common.models import DatasetMetadata
from aiperf.common.protocols import DatasetSamplingStrategyProtocol


class BaseConversationProvider(ABC):
    """Base class for conversation providers."""

    def __init__(
        self,
        dataset_metadata: DatasetMetadata,
        dataset_sampler: DatasetSamplingStrategyProtocol,
    ):
        """Initialize with dataset metadata.

        Args:
            dataset_metadata: Dataset metadata to sample from
        """
        self._metadata = dataset_metadata
        self._dataset_sampler = dataset_sampler

    @abstractmethod
    def next_conversation_id(self) -> str:
        """Get the next conversation ID."""
        raise NotImplementedError

    @abstractmethod
    def has_more_conversation_ids(self) -> bool:
        """Check if more conversation IDs are available."""
        raise NotImplementedError


class PreSampledConversationProvider(BaseConversationProvider):
    """Uses pre-sampled queue for count-based mode with num_sessions."""

    def __init__(
        self,
        dataset_metadata: DatasetMetadata,
        dataset_sampler: DatasetSamplingStrategyProtocol,
    ):
        """Initialize with dataset metadata.

        Args:
            dataset_metadata: Dataset metadata to sample from
        """
        super().__init__(dataset_metadata, dataset_sampler)
        self._pre_sampled_queue: deque[str] = deque()

    def pre_sample_conversation_ids(self, num_conversations: int) -> int:
        """Pre-sample conversation IDs.

        Args:
            num_conversations: Number of conversations to pre-sample

        Returns:
            Total number of turns in the pre-sampled conversations (total_expected_requests)
        """
        if num_conversations <= 0:
            raise ValueError("num_conversations must be greater than 0")

        conversations = {
            conversation.conversation_id: conversation
            for conversation in self._metadata.conversations
        }
        self._pre_sampled_queue = deque()
        total_expected_requests = 0
        for _ in range(num_conversations):
            conversation_id = self._dataset_sampler.next_conversation_id()
            conversation = conversations[conversation_id]
            total_expected_requests += len(conversation.turns)
            self._pre_sampled_queue.append(conversation_id)
        return total_expected_requests

    def next_conversation_id(self) -> str:
        """Get next conversation ID from pre-sampled queue."""
        if not self._pre_sampled_queue:
            raise StopIteration("No more pre-sampled conversation IDs")
        return self._pre_sampled_queue.popleft()

    def has_more_conversation_ids(self) -> bool:
        """Check if queue has more conversation IDs."""
        return len(self._pre_sampled_queue) > 0


class LiveConversationProvider(BaseConversationProvider):
    """Provides conversation IDs on-demand for duration-based and single-turn count-based modes."""

    def next_conversation_id(self) -> str:
        """Sample next conversation ID on-demand using dataset sampling strategy."""
        return self._dataset_sampler.next_conversation_id()

    def has_more_conversation_ids(self) -> bool:
        """Always returns True for live sampling (no predetermined limit on conversation IDs)."""
        return True
