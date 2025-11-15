# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model selection strategies for AIPerf benchmarking.

Provides flexible, extensible model selection strategies using Protocol pattern.
Strategies can examine turn content (modalities, tokens, etc.) to make decisions.

Example:
    ```python
    from aiperf.common.factories import ModelSelectionStrategyFactory
    from aiperf.common.enums import ModelSelectionStrategy

    # Create strategy using factory
    strategy = ModelSelectionStrategyFactory.create_instance(
        ModelSelectionStrategy.ROUND_ROBIN,
        user_config=user_config,
    )
    model = strategy.select(turn)
    ```
"""

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.enums.model_enums import ModelSelectionStrategy
from aiperf.common.factories import ModelSelectionStrategyFactory
from aiperf.common.models import Turn


@ModelSelectionStrategyFactory.register(ModelSelectionStrategy.ROUND_ROBIN)
class RoundRobinModelSelectionStrategy:
    """Round-robin model selection strategy.

    Cycles through available models in order, wrapping around to the beginning.
    Ignores turn content - selection is purely sequential.

    Example:
        ```python
        strategy = RoundRobinModelSelectionStrategy(user_config=user_config)
        model = strategy.select(turn)  # "model-1"
        ```
    """

    def __init__(self, user_config: UserConfig, **kwargs):
        """Initialize round-robin strategy.

        Args:
            user_config: The user configuration.
        """
        self.user_config = user_config
        self._index = 0

    def select(self, turn: Turn) -> str:
        """Select next model in round-robin order.

        Args:
            turn: Turn object (unused by this strategy).

        Returns:
            Next model name in sequence.
        """
        model_name = self.user_config.endpoint.model_names[
            self._index % len(self.user_config.endpoint.model_names)
        ]
        self._index += 1
        return model_name

    def reset(self):
        """Reset index to 0 (used for testing)."""
        self._index = 0


@ModelSelectionStrategyFactory.register(ModelSelectionStrategy.RANDOM)
class RandomModelSelectionStrategy:
    """Random model selection strategy with replacement.

    Randomly selects from available models with uniform distribution.
    Ignores turn content - selection is purely random.

    Example:
        ```python
        strategy = RandomModelSelectionStrategy(user_config=user_config)
        model = strategy.select(turn)  # Randomly "model-1" or "model-2"
        ```
    """

    def __init__(self, user_config: UserConfig, **kwargs):
        """Initialize random strategy.

        Args:
            user_config: The user configuration.

        """
        self.user_config = user_config
        self._rng = rng.derive("dataset.model_select.random")

    def select(self, turn: Turn) -> str:
        """Randomly select a model.

        Args:
            turn: Turn object (unused by this strategy).

        Returns:
            Randomly selected model name.
        """
        return self._rng.choice(self.user_config.endpoint.model_names)


@ModelSelectionStrategyFactory.register(ModelSelectionStrategy.SHUFFLE)
class ShuffleModelSelectionStrategy:
    """Shuffle model selection strategy.

    Shuffles the list of models and cycles through them without replacement.
    Re-shuffles after the end of the list.

    Example:
        ```python
        strategy = ShuffleModelSelectionStrategy(user_config=user_config)
        model = strategy.select(turn)  # Randomly "model-1" or "model-2"
        ```
    """

    def __init__(self, user_config: UserConfig, **kwargs):
        """Initialize shuffle strategy."""
        self.user_config = user_config
        self._rng = rng.derive("dataset.model_select.shuffle")
        self._rng.shuffle(self.user_config.endpoint.model_names)
        self._index = 0

    def select(self, turn: Turn) -> str:
        """Shuffle the list of models and cycle through them without replacement.

        Args:
            turn: Turn object (unused by this strategy).
        """
        if self._index >= len(self.user_config.endpoint.model_names):
            self._rng.shuffle(self.user_config.endpoint.model_names)
            self._index = 0
        model_name = self.user_config.endpoint.model_names[self._index]
        self._index += 1
        return model_name

    def reset(self):
        """Reset index to 0 (used for testing)."""
        self._index = 0
