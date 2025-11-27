# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import TypeAlias

from pydantic import ValidationError

from aiperf.common import random_generator as rng
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import DatasetLoaderType, DatasetSamplingStrategy, MediaType
from aiperf.common.factories import DatasetLoaderFactory
from aiperf.common.models import Conversation, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.file.base import BaseFileLoader
from aiperf.dataset.loader.mixins import MediaConversionMixin
from aiperf.dataset.loader.models import RandomPool

# Type aliases
Filename: TypeAlias = str


@DatasetLoaderFactory.register(DatasetLoaderType.RANDOM_POOL)
class RandomPoolDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads data from a single file or a directory.

    Each line in the file represents single-turn conversation data,
    and files create individual pools for random sampling:
      - Single file: All lines form one single pool (to be randomly sampled from)
      - Directory: Each file becomes a separate pool, then pools are randomly sampled
                   and merged into conversations later.

    The random pool custom dataset
      - supports multi-modal data (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch size > 1)
      - supports named fields for each modality (e.g. text_field_a, text_field_b, etc.)
      - DOES NOT support multi-turn or its features (e.g. delay, sessions, etc.)

    Example:

    1. Single file
    ```jsonl
    {"text": "Who are you?", "image": "/path/to/image1.png"}
    {"text": "Explain what is the meaning of life.", "image": "/path/to/image2.png"}
    ...
    ```
    The file will form a single pool of text and image data that will be used
    to generate conversations.

    2. Directory

    Directory will be useful if user wants to
      - create multiple pools of different modalities separately (e.g. text, image)
      - specify different field names for the same modality.

    data/queries.jsonl
    ```jsonl
    {"texts": [{"name": "query", "contents": ["Who are you?"]}]}
    {"texts": [{"name": "query", "contents": ["What is the meaning of life?"]}]}
    ...
    ```

    data/passages.jsonl
    ```jsonl
    {"texts": [{"name": "passage", "contents": ["I am a cat."]}]}
    {"texts": [{"name": "passage", "contents": ["I am a dog."]}]}
    ...
    ```

    The loader will create two separate pools for each file: queries and passages.
    Each pool is a text dataset with a different field name (e.g. query, passage),
    and loader will later sample from these two pools to create conversations.
    """

    def __init__(
        self,
        config: UserConfig,
        tokenizer: Tokenizer,
        filename: str,
        **kwargs,
    ):
        super().__init__(config, tokenizer, filename, **kwargs)
        self._rng = rng.derive("dataset.loader.random_pool")
        self.num_conversations = config.input.conversation.num or 1
        # Store filename->pool mapping for convert_to_conversations
        self._pool_mapping: dict[Filename, list[RandomPool]] = {}

    @staticmethod
    def _validate_path(path: Path) -> int:
        """Validate all files and directories recursively against the RandomPool model.

        Args:
            path: The path to the file or directory to validate.

        Returns:
            int: Count of files with at least one valid line.

        Raises:
            ValidationError: If any file contains invalid data.
        """
        valid_count = 0

        if path.is_dir():
            # if path is a directory, recursively call this function for each child
            # if any child fails validation, it will exit early with an exception
            for file in path.iterdir():
                valid_count += RandomPoolDatasetLoader._validate_path(file)

        elif path.is_file():
            # if path is a file, validate the first non-empty line against the RandomPool model
            # if the line is valid, increment the valid count and break the loop,
            # otherwise a ValidationError will be raised and the function will exit early
            with open(path) as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    RandomPool.model_validate_json(line)
                    valid_count += 1
                    break

        return valid_count

    @classmethod
    def can_load_file(cls, path: Path) -> bool:
        """Check if this loader can handle the given file.

        RandomPool format is ambiguous with SingleTurn (both have modality fields),
        so we only match on explicit type field in the data. Regular files without
        explicit type field will not match here.

        Args:
            path: Path to the file to check.

        Returns:
            True if file has explicit RandomPool type field, False otherwise.
        """
        if not path.is_file():
            return False

        try:
            with open(path) as f:
                for line in f:
                    if (line := line.strip()) == "":
                        continue  # Skip empty lines
                    # Check if data has explicit type field
                    import orjson

                    data = orjson.loads(line)
                    if data.get("type") == "random_pool":
                        RandomPool.model_validate(data)
                        return True
                    return False  # First line doesn't have explicit type
            return False  # File is empty or has only empty lines
        except (ValidationError, Exception):
            return False

    @classmethod
    def can_load_directory(cls, path: Path) -> bool:
        """Check if this loader can handle the given directory.

        RandomPool is the only loader that supports directory inputs.
        Validates that directory contains at least one valid file.

        Args:
            path: Path to the directory to check.

        Returns:
            True if directory contains at least one valid file, False otherwise.
        """
        if not path.is_dir():
            return False

        try:
            valid_count = cls._validate_path(path)
            return valid_count > 0
        except (ValidationError, Exception):
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for RandomPool."""
        return DatasetSamplingStrategy.SHUFFLE

    def parse_and_validate(self) -> list[RandomPool]:
        """Parse and validate random pool data from a file or directory.

        If filename is a file, reads and parses using the RandomPool model.
        If filename is a directory, reads each file in the directory and stores
        the filename->pool mapping for later sampling.

        Returns:
            A flat list of RandomPool objects (for interface compliance).
            The filename->pool mapping is stored in self._pool_mapping.
        """
        path = Path(self.filename)

        if path.is_file():
            dataset_pool = self._load_dataset_from_file(path)
            self._pool_mapping = {path.name: dataset_pool}
            return dataset_pool

        self._pool_mapping = self._load_dataset_from_dir(path)
        # Return flat list for interface compliance
        return [item for pool in self._pool_mapping.values() for item in pool]

    def _load_dataset_from_file(self, file_path: Path) -> list[RandomPool]:
        """Load random pool data from a single file.

        Args:
            file_path: The path to the file containing the data.

        Returns:
            A list of RandomPool objects.
        """
        dataset_pool: list[RandomPool] = []

        with open(file_path) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                random_pool_data = RandomPool.model_validate_json(line)
                dataset_pool.append(random_pool_data)

        return dataset_pool

    def _load_dataset_from_dir(
        self, dir_path: Path
    ) -> dict[Filename, list[RandomPool]]:
        """Load random pool data from all files in a directory.

        Args:
            dir_path: The path to the directory containing the files.

        Returns:
            A dictionary mapping filename to list of RandomPool objects.
        """
        data: dict[Filename, list[RandomPool]] = defaultdict(list)

        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file():
                dataset_pool = self._load_dataset_from_file(file_path)
                data[file_path.name].extend(dataset_pool)

        return data

    def convert_to_conversations(self, data: list[RandomPool]) -> list[Conversation]:
        """Convert random pool data to conversation objects.

        Uses self._pool_mapping to sample from each file's pool and merge turns.

        Args:
            data: A flat list of RandomPool objects (unused, uses self._pool_mapping).

        Returns:
            A list of conversations.
        """
        # Use the pool_mapping stored during parse_and_validate
        conversations = [
            Conversation(session_id=self.session_id_generator.next())
            for _ in range(self.num_conversations)
        ]

        # F x N (F: num of files, N: num of conversations)
        sampled_dataset: dict[Filename, list[Turn]] = {}

        # Randomly sample (with replacement) from each dataset pool
        for filename, dataset_pool in self._pool_mapping.items():
            samples = self._rng.choices(dataset_pool, k=self.num_conversations)
            turns: list[Turn] = []
            for sample in samples:
                media = self.convert_to_media_objects(sample, name=Path(filename).stem)
                turns.append(
                    Turn(
                        texts=media[MediaType.TEXT],
                        images=media[MediaType.IMAGE],
                        audios=media[MediaType.AUDIO],
                        videos=media[MediaType.VIDEO],
                    )
                )
            sampled_dataset[filename] = turns

        # Merge turns for each conversation
        for i, batched_turns in enumerate(zip(*sampled_dataset.values(), strict=False)):
            turn = self._merge_turns(batched_turns)
            conversations[i].turns.append(turn)

        return conversations

    def _merge_turns(self, turns: list[Turn]) -> Turn:
        """Merge turns into a single turn.

        Args:
            turns: A list of turns.

        Returns:
            A single turn.
        """
        merged_turn = Turn(
            texts=[text for turn in turns for text in turn.texts],
            images=[image for turn in turns for image in turn.images],
            audios=[audio for turn in turns for audio in turn.audios],
            videos=[video for turn in turns for video in turn.videos],
        )
        return merged_turn
