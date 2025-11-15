# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pydantic import ValidationError

from aiperf.common.enums import DatasetLoaderType, DatasetSamplingStrategy, MediaType
from aiperf.common.factories import DatasetLoaderFactory
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.file.base import BaseFileLoader
from aiperf.dataset.loader.mixins import MediaConversionMixin
from aiperf.dataset.loader.models import SingleTurn


@DatasetLoaderFactory.register(DatasetLoaderType.SINGLE_TURN)
class SingleTurnDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads single turn data from a file.

    The single turn type
      - supports multi-modal data (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch_size > 1)
      - DOES NOT support multi-turn features (e.g. delay, sessions, etc.)

    Examples:
    1. Single-batch, text only
    ```json
    {"text": "What is deep learning?"}
    ```

    2. Single-batch, multi-modal
    ```json
    {"text": "What is in the image?", "image": "/path/to/image.png"}
    ```

    3. Multi-batch, multi-modal
    ```json
    {"texts": ["Who are you?", "Hello world"], "images": ["/path/to/image.png", "/path/to/image2.png"]}
    ```

    4. Fixed schedule version
    ```json
    {"timestamp": 0, "text": "What is deep learning?"},
    {"timestamp": 1000, "text": "Who are you?"},
    {"timestamp": 2000, "text": "What is AI?"}
    ```

    5. Time delayed version
    ```json
    {"delay": 0, "text": "What is deep learning?"},
    {"delay": 1234, "text": "Who are you?"}
    ```

    6. Full-featured version (Multi-batch, multi-modal, multi-fielded)
    ```json
    {
        "texts": [
            {"name": "text_field_A", "contents": ["Hello", "World"]},
            {"name": "text_field_B", "contents": ["Hi there"]}
        ],
        "images": [
            {"name": "image_field_A", "contents": ["/path/1.png", "/path/2.png"]},
            {"name": "image_field_B", "contents": ["/path/3.png"]}
        ]
    }
    ```
    """

    @classmethod
    def can_load_file(cls, path: Path) -> bool:
        """Check if this loader can handle the given file.

        SingleTurn format has modality fields (text/texts, image/images, etc.)
        but does NOT have a "turns" field. Validates first non-empty line against SingleTurn model.

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
                    SingleTurn.model_validate_json(line)
                    return True  # Successfully validated first line
            return False  # File is empty or has only empty lines
        except (ValidationError, Exception):
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for SingleTurn."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def parse_and_validate(self) -> list[SingleTurn]:
        """Parse and validate single-turn data from a JSONL file.

        Each line represents a single turn conversation.

        Returns:
            A list of validated SingleTurn objects.
        """
        data: list[SingleTurn] = []

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                single_turn_data = SingleTurn.model_validate_json(line)
                data.append(single_turn_data)

        return data

    def convert_to_conversations(self, data: list[SingleTurn]) -> list[Conversation]:
        """Convert single turn data to conversation objects.

        Each SingleTurn becomes a separate Conversation with a unique session ID.

        Args:
            data: A list of SingleTurn objects.

        Returns:
            A list of conversations.
        """
        conversations = []
        for single_turn in data:
            session_id = self.session_id_generator.next()
            conversation = Conversation(session_id=session_id)
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
