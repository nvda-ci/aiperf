# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import DatasetSamplingStrategy
from aiperf.common.factories import ModelSelectionStrategyFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.output_tokens_sampler import OutputTokensSampler


class BaseDatasetLoader(AIPerfLoggerMixin, ABC):
    """Base class for all dataset loaders (synthetic, file, remote).

    Provides common utilities shared by ALL loaders:
    - session_id_generator: Generates unique session IDs
    - model_selector: Selects model names based on strategy
    - output_tokens_sampler: Samples output token counts

    Subclasses must implement:
    - load(): Returns list[Conversation]
    - can_load(): Class method to check if loader can handle config
    - get_preferred_sampling_strategy(): Returns preferred sampling strategy

    Args:
        config: The user configuration
        tokenizer: The tokenizer instance
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        super().__init__(config=config, tokenizer=tokenizer)

        # Session ID generator (deterministic when seed is set)
        self.session_id_generator = SessionIDGenerator(seed=config.input.random_seed)

        # Model selector for conversation model assignment
        self.model_selector = ModelSelectionStrategyFactory.create_instance(
            config.endpoint.model_selection_strategy,
            user_config=config,
        )

        # Output tokens sampler for max_tokens generation
        self.output_tokens_sampler = OutputTokensSampler(
            mean=config.input.prompt.output_tokens.mean,
            stddev=config.input.prompt.output_tokens.stddev,
        )

    @abstractmethod
    def load(self) -> list[Conversation]:
        """Load conversations from any source (synthetic, file, remote).

        Returns:
            List of Conversation objects ready for benchmarking.
        """
        ...

    @classmethod
    @abstractmethod
    def can_load(cls, config: UserConfig) -> bool:
        """Check if this loader can handle the given config.

        Args:
            config: The user configuration to check.

        Returns:
            True if this loader can handle the config, False otherwise.
        """
        ...

    @abstractmethod
    def get_preferred_sampling_strategy(self) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy for this loader.

        Returns:
            The preferred DatasetSamplingStrategy enum value.
        """
        ...

    # Shared utility methods

    def assign_model(self, conversation: Conversation) -> None:
        """Assign model to conversation using model selector.

        Args:
            conversation: The conversation to assign a model to.
        """
        conversation.model = self.model_selector.select_model()

    def sample_max_tokens(self) -> int:
        """Sample max_tokens using output tokens sampler.

        Returns:
            Sampled max_tokens value.
        """
        return self.output_tokens_sampler.sample()
