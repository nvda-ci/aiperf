# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from aiperf.common.config.user_config import UserConfig
from aiperf.common.factories import ModelSelectionStrategyFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.dataset.loader.models import CustomDatasetT
from aiperf.dataset.output_tokens_sampler import OutputTokensSampler


class BaseLoader(AIPerfLoggerMixin, ABC):
    """Base class for loading data.

    This abstract class provides a base implementation for loading data.
    Subclasses must implement the load_dataset and convert_to_conversations methods.

    Provides common utilities for dataset loaders:
    - session_id_generator: Generates unique session IDs
    - model_selector: Selects model names based on strategy
    - output_tokens_sampler: Samples output token counts for synthetic data

    Args:
        user_config: The user configuration.
        **kwargs: Additional arguments to pass to the base class.
    """

    def __init__(self, *, user_config: UserConfig, **kwargs):
        self.user_config = user_config
        super().__init__(user_config=user_config, **kwargs)

        # Session ID generator (deterministic when seed is set)
        self.session_id_generator = SessionIDGenerator(
            seed=user_config.input.random_seed
        )

        # Model selector for turn model assignment
        self.model_selector = ModelSelectionStrategyFactory.create_instance(
            user_config.endpoint.model_selection_strategy,
            user_config=user_config,
        )

        # Output tokens sampler for synthetic max_tokens generation
        self.output_tokens_sampler = OutputTokensSampler(
            mean=user_config.input.prompt.output_tokens.mean,
            stddev=user_config.input.prompt.output_tokens.stddev,
        )

    @abstractmethod
    def load_dataset(self) -> dict[str, list[CustomDatasetT]]: ...

    @abstractmethod
    def convert_to_conversations(
        self, custom_data: dict[str, list[CustomDatasetT]]
    ) -> list[Conversation]: ...
