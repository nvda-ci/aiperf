# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pydantic import ValidationError

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import DatasetLoaderType, DatasetSamplingStrategy
from aiperf.common.factories import DatasetLoaderFactory
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import load_json_str
from aiperf.dataset.loader.file.base import BaseFileLoader
from aiperf.dataset.loader.models import ShareGPT


@DatasetLoaderFactory.register(DatasetLoaderType.SHAREGPT)
class ShareGPTLoader(BaseFileLoader):
    """ShareGPT format loader (pure parsing logic).

    This loader parses ShareGPT format files. Dataset metadata (URLs, etc.)
    are handled by ShareGPTDataset in the public_datasets module.

    ShareGPT format consists of multi-turn conversations with alternating
    human/assistant messages. This loader currently uses only the first
    two messages (human + gpt) for single-turn conversations.

    The loader filters conversations based on:
    - Minimum conversation length (at least 2 turns required)
    - Sequence length validation for prompt and completion tokens
    - Configurable max prompt length and total sequence length

    Example ShareGPT format:
        {
            "conversations": [
                {"from": "human", "value": "What is AI?"},
                {"from": "gpt", "value": "AI stands for..."}
            ]
        }

    Args:
        config: The user configuration
        tokenizer: The tokenizer instance
        filename: Path to ShareGPT format file
    """

    def __init__(
        self,
        config: UserConfig,
        tokenizer: Tokenizer,
        filename: str,
    ):
        """Initialize ShareGPT loader.

        Args:
            config: User configuration
            tokenizer: Tokenizer for sequence length validation
            filename: Path to ShareGPT format file
        """
        super().__init__(config, tokenizer, filename)
        self.output_tokens_mean = config.input.prompt.output_tokens.mean

    @classmethod
    def can_load_file(cls, path: Path) -> bool:
        """Check if file is ShareGPT format by validating structure.

        Args:
            path: Path to the file to check.

        Returns:
            True if the file matches ShareGPT format, False otherwise.
        """
        try:
            with open(path) as f:
                content = f.read()

            data = load_json_str(content)

            # ShareGPT is a JSON array (not JSONL)
            if not isinstance(data, list):
                return False

            if len(data) == 0:
                return False

            # Validate first entry
            ShareGPT.model_validate(data[0])
            return True

        except (FileNotFoundError, ValidationError, ValueError, KeyError):
            return False

    def parse_and_validate(self) -> list[ShareGPT]:
        """Parse ShareGPT JSON file and validate entries.

        Returns:
            List of validated ShareGPT objects.

        Raises:
            ValueError: If file format is invalid.
        """
        with open(self.filename) as f:
            content = f.read()

        raw_data = load_json_str(content)

        # ShareGPT file is typically a JSON array, not JSONL
        if not isinstance(raw_data, list):
            raise ValueError(
                f"ShareGPT file must contain a JSON array, got {type(raw_data)}"
            )

        data: list[ShareGPT] = []
        for entry in raw_data:
            try:
                sharegpt_entry = ShareGPT.model_validate(entry)
                data.append(sharegpt_entry)
            except ValidationError as e:
                self.debug(lambda e=e: f"Skipping invalid entry: {e!r}")
                continue

        return data

    def convert_to_conversations(self, data: list[ShareGPT]) -> list[Conversation]:
        """Convert ShareGPT data to Conversation objects.

        Filters conversations based on sequence length validation and
        converts valid entries to internal Conversation format.

        Args:
            data: List of validated ShareGPT objects.

        Returns:
            List of filtered and validated Conversation objects.
        """
        self.info("Validating ShareGPT dataset and constructing conversations")

        filtered_conversations = []
        skipped_entries = 0
        total_entries = len(data)

        for entry in data:
            conversations = entry.conversations

            # Ensure minimum 2 turns
            if len(conversations) < 2:
                skipped_entries += 1
                continue

            # Get first prompt and completion for validation
            prompt = conversations[0].value
            completion = conversations[1].value

            # Tokenize and validate sequence lengths
            prompt_length = len(self.tokenizer.encode(prompt))
            completion_length = len(self.tokenizer.encode(completion))

            if not self._is_valid_sequence(
                prompt_len=prompt_length,
                output_len=completion_length,
            ):
                skipped_entries += 1
                continue

            # Create conversation with first turn only
            # TODO: Support full multi-turn conversations in the future
            turn = Turn(
                texts=[Text(contents=[prompt])],
                max_tokens=completion_length,
            )
            turn.model = self.model_selector.select(turn)

            filtered_conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[turn],
                )
            )

        self.debug(
            lambda: f"Filtered to {len(filtered_conversations)} conversations "
            f"out of {total_entries} (skipped {skipped_entries})"
        )

        return filtered_conversations

    def _is_valid_sequence(
        self,
        prompt_len: int,
        output_len: int,
        min_seq_len: int = 4,
        max_prompt_len: int = 1024,
        max_total_len: int = 2048,
    ) -> bool:
        """Validate sequence lengths based on hardcoded limits.

        This validation is adopted from vllm/benchmarks/benchmark_dataset.py
        and uses the same defaults as the original ShareGPT loader.

        Args:
            prompt_len: Length of the prompt in tokens.
            output_len: Length of the output in tokens.
            min_seq_len: Minimum sequence length (default: 4).
            max_prompt_len: Maximum prompt length (default: 1024).
            max_total_len: Maximum total sequence length (default: 2048).

        Returns:
            True if the sequence is valid, False otherwise.
        """
        # Check for invalid conditions
        prompt_too_short = prompt_len < min_seq_len
        prompt_too_long = prompt_len > max_prompt_len

        # Skip min output length check if synthetic output will be used
        skip_min_output_len_check = self.output_tokens_mean is not None
        output_too_short = (not skip_min_output_len_check) and (
            output_len < min_seq_len
        )

        combined_too_long = (prompt_len + output_len) > max_total_len

        return not (
            prompt_too_short or output_too_short or prompt_too_long or combined_too_long
        )

    def get_preferred_sampling_strategy(self) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy for ShareGPT.

        Returns:
            DatasetSamplingStrategy.SEQUENTIAL for ordered iteration.
        """
        return DatasetSamplingStrategy.SEQUENTIAL
