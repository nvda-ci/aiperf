# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator import (
    AudioGenerator,
    ImageGenerator,
    PromptGenerator,
    VideoGenerator,
)
from aiperf.dataset.loader.base import BaseDatasetLoader


class BaseSyntheticLoader(BaseDatasetLoader, ABC):
    """Base class for synthetic data generators.

    Provides shared functionality for loaders that generate synthetic conversations:
    - All media generators (prompt, image, audio, video)
    - ISL/OSL distribution handling for consistent prompt/response pairing
    - Turn sequence caching to ensure same ISL/OSL pair used within a turn

    Subclasses must implement:
    - load(): Generate and return synthetic conversations
    - can_load(): Check if this loader should be used based on config
    - get_preferred_sampling_strategy(): Return preferred strategy

    Args:
        config: The user configuration
        tokenizer: The tokenizer instance
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer):
        super().__init__(config, tokenizer)

        # Media generators (all synthetic loaders need these)
        self.prompt_generator = PromptGenerator(
            config.input.prompt,
            tokenizer,
        )
        self.image_generator = ImageGenerator(config.input.image)
        self.audio_generator = AudioGenerator(config.input.audio)
        self.video_generator = VideoGenerator(config.input.video)

        # Sequence length distribution (for ISL/OSL pairing)
        self._seq_distribution = config.input.prompt.get_sequence_distribution()

        # Cache for turn-level sequence lengths to ensure ISL/OSL consistency
        # Key: id(turn), Value: (input_seq_len, output_seq_len)
        self._turn_sequence_cache: dict[int, tuple[int, int]] = {}

    def get_max_tokens_for_turn(self, turn: Turn) -> int | None:
        """Get max_tokens for a turn with ISL/OSL pairing consistency.

        If max_tokens is already set on the turn, returns it unchanged.
        Otherwise, uses cached sequence distribution if available, or samples
        from the output tokens sampler.

        This ensures that for a given turn, the same ISL/OSL pair is used
        for both prompt generation and max_tokens assignment.

        Args:
            turn: The turn object to get max_tokens for.

        Returns:
            The max_tokens value, or None if no sampling is configured.
        """
        if turn.max_tokens is not None:
            return turn.max_tokens

        if self._seq_distribution is not None:
            turn_id = id(turn)
            _, osl = self._get_turn_sequence_lengths(turn_id)
            return osl

        return self.sample_max_tokens()

    def _get_turn_sequence_lengths(self, turn_id: int) -> tuple[int, int]:
        """Get or sample ISL/OSL pair for a specific turn, ensuring consistency.

        Caches the sequence lengths per turn to ensure that the same ISL/OSL
        pair is used for both prompt generation and max_tokens setting.

        Args:
            turn_id: Unique identifier for the turn (typically id(turn))

        Returns:
            Tuple of (input_seq_len, output_seq_len)
        """
        if turn_id in self._turn_sequence_cache:
            return self._turn_sequence_cache[turn_id]

        if self._seq_distribution is None:
            # Fallback to mean values if no distribution configured
            seq_lengths = (
                self.config.input.prompt.input_tokens.mean,
                self.config.input.prompt.output_tokens.mean
                or max(128, self.config.input.prompt.input_tokens.mean // 2),
            )
        else:
            # Sample from the distribution
            seq_lengths = self._seq_distribution.sample()

        self._turn_sequence_cache[turn_id] = seq_lengths
        return seq_lengths

    def clear_turn_cache(self, turn: Turn) -> None:
        """Clear cached sequence lengths for a specific turn.

        Should be called after a conversation is fully created to avoid
        memory leaks from the cache growing indefinitely.

        Args:
            turn: Turn object whose cache entry should be cleared.
        """
        turn_id = id(turn)
        self._turn_sequence_cache.pop(turn_id, None)

    @property
    def prefix_prompt_enabled(self) -> bool:
        """Check if prefix prompt is enabled in the configuration.

        Returns:
            True if prefix prompt length > 0, False otherwise.
        """
        return self.config.input.prompt.prefix_prompt.length > 0
