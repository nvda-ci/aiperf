# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.base import BaseDatasetLoader
from aiperf.dataset.loader.models import CustomDatasetT


class BaseFileLoader(BaseDatasetLoader, ABC):
    """Base class for file-based dataset loaders.

    Provides a two-stage loading process:
    1. parse_and_validate(): Parse file and validate against Pydantic models
    2. convert_to_conversations(): Convert validated models to Conversation objects

    This ensures data validation before conversion, catching format errors early.

    Subclasses must implement:
    - parse_and_validate(): Return list[CustomDatasetT] (validated Pydantic models)
    - convert_to_conversations(): Convert models to list[Conversation]
    - can_load_file(): Class method to check if loader can handle a file
    - can_load_directory(): Class method to check if loader can handle a directory (optional)
    - get_preferred_sampling_strategy(): Return preferred strategy

    Args:
        config: The user configuration
        tokenizer: The tokenizer instance
        filename: Path to the file or directory to load
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer, filename: str):
        super().__init__(config, tokenizer)
        self.filename = filename

    def load(self) -> list[Conversation]:
        """Load conversations from file using two-stage process.

        Returns:
            List of Conversation objects.
        """
        validated_data = self.parse_and_validate()
        conversations = self.convert_to_conversations(validated_data)
        return conversations

    @classmethod
    def can_load(cls, config: UserConfig) -> bool:
        """Check if this file loader can handle the config.

        Checks if config.input.file is set and whether this loader can handle
        the file/directory format.

        Args:
            config: The user configuration to check.

        Returns:
            True if this loader can handle the file, False otherwise.
        """
        if config.input.file is None:
            return False

        path = Path(config.input.file)
        if path.is_dir():
            return cls.can_load_directory(path)
        else:
            return cls.can_load_file(path)

    @classmethod
    @abstractmethod
    def can_load_file(cls, path: Path) -> bool:
        """Check if this loader can handle the given file.

        Should inspect the file structure/content to determine compatibility.

        Args:
            path: Path to the file to check.

        Returns:
            True if this loader can handle the file, False otherwise.
        """
        ...

    @classmethod
    def can_load_directory(cls, path: Path) -> bool:
        """Check if this loader can handle the given directory.

        Default implementation returns False. Override if loader supports directories.

        Args:
            path: Path to the directory to check.

        Returns:
            True if this loader can handle the directory, False otherwise.
        """
        return False

    @abstractmethod
    def parse_and_validate(self) -> list[CustomDatasetT]:
        """Parse file and validate against Pydantic models.

        Returns:
            List of validated CustomDatasetT models.

        Raises:
            ValidationError: If file format is invalid.
        """
        ...

    @abstractmethod
    def convert_to_conversations(
        self, data: list[CustomDatasetT]
    ) -> list[Conversation]:
        """Convert validated data to Conversation objects.

        Args:
            data: List of validated CustomDatasetT models.

        Returns:
            List of Conversation objects ready for benchmarking.
        """
        ...
