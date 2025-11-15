# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from aiperf.common.enums import DatasetLoaderType, DatasetSamplingStrategy, MediaType
from aiperf.common.factories import DatasetLoaderFactory
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.file.base import BaseFileLoader
from aiperf.dataset.loader.mixins import MediaConversionMixin
from aiperf.dataset.loader.models import MultiTurn


@DatasetLoaderFactory.register(DatasetLoaderType.MULTI_TURN)
class MultiTurnDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads multi-turn data from a file.

    The multi-turn type
      - supports multi-modal data (e.g. text, image, audio)
      - supports multi-turn features (e.g. delay, sessions, etc.)
      - supports client-side batching for each data (e.g. batch_size > 1)

    NOTE: If the user specifies multiple multi-turn entries with same session ID,
    the loader will group them together. If the timestamps are specified, they will
    be sorted in ascending order later in the timing manager.

    Examples:
    1. Simple version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"text": "Hello", "image": "url", "delay": 0},
            {"text": "Hi there", "delay": 1000}
        ]
    }
    ```

    2. Batched version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"texts": ["Who are you?", "Hello world"], "images": ["/path/1.png", "/path/2.png"]},
            {"texts": ["What is in the image?", "What is AI?"], "images": ["/path/3.png", "/path/4.png"]}
        ]
    }
    ```

    3. Fixed schedule version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"timestamp": 0, "text": "What is deep learning?"},
            {"timestamp": 1000, "text": "Who are you?"}
        ]
    }
    ```

    4. Time delayed version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"delay": 0, "text": "What is deep learning?"},
            {"delay": 1000, "text": "Who are you?"}
        ]
    }
    ```

    5. full-featured version (multi-batch, multi-modal, multi-fielded, session-based, etc.)
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {
                "timestamp": 1234,
                "texts": [
                    {"name": "text_field_a", "contents": ["hello", "world"]},
                    {"name": "text_field_b", "contents": ["hi there"]}
                ],
                "images": [
                    {"name": "image_field_a", "contents": ["/path/1.png", "/path/2.png"]},
                    {"name": "image_field_b", "contents": ["/path/3.png"]}
                ]
            }
        ]
    }
    ```
    """

    @classmethod
    def can_load_file(cls, path: Path) -> bool:
        """Check if this loader can handle the given file.

        For multi-turn data, validates first non-empty line against the MultiTurn model.
        This will handle all of the validation logic for the different input combinations.

        Args:
            path: Path to the file to check.

        Returns:
            True if this loader can handle the file, False otherwise.
        """
        if not path.is_file():
            return False

        try:
            with open(path) as f:
                for line in f:
                    if (line := line.strip()) == "":
                        continue  # Skip empty lines
                    MultiTurn.model_validate_json(line)
                    return True  # Successfully validated first line
            return False  # File is empty or has only empty lines
        except (ValidationError, Exception):
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for MultiTurn."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def parse_and_validate(self) -> list[MultiTurn]:
        """Parse and validate multi-turn data from a JSONL file.

        Each line represents a complete multi-turn conversation with its own
        session_id and multiple turns.

        Returns:
            A list of validated MultiTurn objects.
        """
        data: list[MultiTurn] = []

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                multi_turn_data = MultiTurn.model_validate_json(line)
                data.append(multi_turn_data)

        return data

    def convert_to_conversations(self, data: list[MultiTurn]) -> list[Conversation]:
        """Convert multi-turn data to conversation objects.

        Groups MultiTurn entries by session_id and creates conversations.

        Args:
            data: A list of MultiTurn objects.

        Returns:
            A list of conversations.
        """
        # Group by session_id
        grouped_data: dict[str, list[MultiTurn]] = defaultdict(list)
        for multi_turn in data:
            session_id = multi_turn.session_id or self.session_id_generator.next()
            grouped_data[session_id].append(multi_turn)

        # Convert grouped data to conversations
        conversations = []
        for session_id, multi_turns in grouped_data.items():
            conversation = Conversation(session_id=session_id)

            # Process all MultiTurn objects for this session
            for multi_turn in multi_turns:
                for single_turn in multi_turn.turns:
                    media = self.convert_to_media_objects(single_turn)
                    conversation.turns.append(
                        Turn(
                            texts=media[MediaType.TEXT],
                            images=media[MediaType.IMAGE],
                            audios=media[MediaType.AUDIO],
                            videos=media[MediaType.VIDEO],
                            timestamp=single_turn.timestamp,
                            delay=single_turn.delay,
                            role=single_turn.role,
                        )
                    )
            conversations.append(conversation)
        return conversations
