# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.enums import DatasetLoaderType, DatasetSamplingStrategy, EndpointType
from aiperf.common.factories import DatasetLoaderFactory
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.synthetic.base import BaseSyntheticLoader


@DatasetLoaderFactory.register(DatasetLoaderType.SYNTHETIC_RANKINGS)
class SyntheticRankingsLoader(BaseSyntheticLoader):
    """Generates synthetic data for Rankings endpoints.

    Each conversation contains one turn with:
    - One query (Text with name="query")
    - Multiple passages (Text with name="passages")

    Number of passages is configurable with mean and standard deviation.

    Args:
        config: The user configuration
        tokenizer: The tokenizer instance
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer):
        super().__init__(config, tokenizer)

        # Derived RNG for sampling number of passages
        self._passages_rng = rng.derive("loader.rankings.passages")

        # Validate prompt generation is enabled
        if config.input.prompt.input_tokens.mean <= 0:
            raise ValueError(
                "Synthetic rankings data generation requires text prompts to be enabled. "
                "Please set --prompt-input-tokens-mean > 0."
            )

    @classmethod
    def can_load(cls, config: UserConfig) -> bool:
        """Check if this loader should be used.

        Returns True if:
        - No file specified
        - No public dataset specified
        - Endpoint type is RANKINGS

        Args:
            config: The user configuration to check.

        Returns:
            True if this loader should generate synthetic rankings data.
        """
        return (
            config.input.file is None
            and config.input.public_dataset_type is None
            and config.endpoint.endpoint_type == EndpointType.RANKINGS
        )

    def load(self) -> list[Conversation]:
        """Generate synthetic ranking conversations.

        Each conversation contains one turn with one query and N passages.

        Returns:
            List of Conversation objects with ranking data.
        """
        conversations: list[Conversation] = []

        num_entries = self.config.input.conversation.num_dataset_entries
        num_passages_mean = self.config.input.rankings_passages_mean
        num_passages_stddev = self.config.input.rankings_passages_stddev

        for _ in range(num_entries):
            session_id = self.session_id_generator.next()

            # Sample number of passages for this entry
            num_passages = self._passages_rng.sample_positive_normal_integer(
                num_passages_mean, num_passages_stddev
            )

            # Create turn with query + passages
            turn = self._create_ranking_turn(num_passages)

            conversation = Conversation(session_id=session_id, turns=[turn])
            conversations.append(conversation)

        return conversations

    def _create_ranking_turn(self, num_passages: int) -> Turn:
        """Create a single ranking turn with query and passages.

        Args:
            num_passages: Number of passages to generate.

        Returns:
            Turn object with query and passages.
        """
        turn = Turn()

        # Generate query text
        query_text = self.prompt_generator.generate(
            mean=self.config.input.prompt.input_tokens.mean,
            stddev=self.config.input.prompt.input_tokens.stddev,
        )
        query = Text(name="query", contents=[query_text])

        # Generate passages
        passages = Text(name="passages")
        for _ in range(num_passages):
            passage_text = self.prompt_generator.generate(
                mean=self.config.input.prompt.input_tokens.mean,
                stddev=self.config.input.prompt.input_tokens.stddev,
            )
            passages.contents.append(passage_text)

        turn.texts.extend([query, passages])

        # Set turn metadata
        turn.model = self.model_selector.select(turn)
        turn.max_tokens = self.get_max_tokens_for_turn(turn)

        # Clear cached sequence lengths to free memory
        self.clear_turn_cache(turn)

        self.debug(
            lambda num_passages=num_passages,
            query_text=query_text: f"[rankings] query_len={len(query_text)} chars, passages={num_passages}"
        )

        return turn

    def get_preferred_sampling_strategy(self) -> DatasetSamplingStrategy:
        """Return preferred sampling strategy for rankings data.

        Returns:
            DatasetSamplingStrategy.RANDOM for variety in rankings.
        """
        return DatasetSamplingStrategy.RANDOM
