# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import mmap
import weakref
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import orjson
from pydantic import Field, field_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.models import AIPerfBaseModel, Conversation

_logger = AIPerfLogger(__name__)


class MemoryMapDatasetError(Exception):
    """Base exception for memory-mapped dataset operations."""

    pass


class MemoryMapSerializationError(MemoryMapDatasetError):
    """Exception raised during serialization/deserialization operations."""

    pass


class MemoryMapFileOperationError(MemoryMapDatasetError):
    """Exception raised during file operations."""

    pass


class MemoryMapFileType(CaseInsensitiveStrEnum):
    """Enumeration for memory-mapped file types."""

    DATA = "data"
    INDEX = "index"


@dataclass(frozen=True)
class MemoryMapConstants:
    """Constants for memory-mapped dataset operations."""

    TEMP_DIR_NAME: Final[str] = "aiperf_memory_map"
    DATA_FILE_SUFFIX: Final[str] = ".dat"
    INDEX_FILE_SUFFIX: Final[str] = ".dat"


# Global instance for easy access
MEMORY_MAP_CONSTANTS = MemoryMapConstants()


class ConversationOffset(AIPerfBaseModel):
    """Offset information for a single conversation in the memory-mapped file."""

    offset: int = Field(ge=0, description="Byte offset where conversation data starts")
    size: int = Field(ge=0, description="Size of the conversation data in bytes")


class MemoryMapDatasetIndex(AIPerfBaseModel):
    """Index structure for the memory-mapped dataset.

    All data is stored as uncompressed JSON bytes serialized with orjson.
    """

    session_ids: list[str] = Field(
        default_factory=list, description="List of all session IDs in the dataset"
    )
    offsets: dict[str, ConversationOffset] = Field(
        default_factory=dict,
        description="Mapping of session IDs to their byte offsets and sizes",
    )
    total_size: int = Field(
        default=0, ge=0, description="Total size of the serialized dataset in bytes"
    )

    @field_validator("session_ids")
    @classmethod
    def validate_session_ids(cls, v: list[str]) -> list[str]:
        """Validate that session_ids contains unique strings."""
        if len(v) != len(set(v)):
            raise ValueError("session_ids must contain unique values")
        return v


class MemoryMapDatasetSerializer:
    """Utilities for deserializing conversations from memory-mapped files.

    Note: Serialization (writing) is handled by DatasetBackingStore implementations
    which stream conversations directly to files. This class only handles reading.
    """

    @staticmethod
    def deserialize_single_conversation(data: bytes) -> Conversation:
        """Deserialize a single conversation from bytes.

        Args:
            data: Serialized conversation data bytes (uncompressed JSON from orjson)

        Returns:
            Conversation object

        Raises:
            MemoryMapSerializationError: If deserialization fails
        """
        try:
            # Use orjson for fast deserialization - accepts bytes directly
            conv_data = orjson.loads(data)
        except orjson.JSONDecodeError as e:
            raise MemoryMapSerializationError(
                f"Failed to decode conversation data: {e}"
            ) from e

        try:
            return Conversation.model_validate(conv_data)
        except Exception as e:
            raise MemoryMapSerializationError(
                f"Failed to validate conversation data: {e}"
            ) from e


class MemoryMapDatasetClient:
    """
    Client for accessing memory-mapped dataset from worker processes.

    This class provides an interface for workers to access conversation data
    from memory-mapped files without making network requests.

    Supports context manager protocol for automatic resource cleanup.
    """

    def __init__(self, data_file_path: str, index_file_path: str) -> None:
        """Initialize the MemoryMapDatasetClient.

        Args:
            data_file_path: Path to the memory-mapped data file
            index_file_path: Path to the memory-mapped index file

        Raises:
            MemoryMapFileOperationError: If files cannot be opened
            MemoryMapSerializationError: If index data is invalid
        """
        self.data_file_path = Path(data_file_path)
        self.index_file_path = Path(index_file_path)

        # Validate file existence
        if not self.data_file_path.exists():
            raise MemoryMapFileOperationError(f"Data file not found: {data_file_path}")
        if not self.index_file_path.exists():
            raise MemoryMapFileOperationError(
                f"Index file not found: {index_file_path}"
            )

        try:
            # Open and map the files
            self.data_file = self.data_file_path.open("rb")
            self.data_mmap = mmap.mmap(
                self.data_file.fileno(), 0, access=mmap.ACCESS_READ
            )

            self.index_file = self.index_file_path.open("rb")
            self.index_mmap = mmap.mmap(
                self.index_file.fileno(), 0, access=mmap.ACCESS_READ
            )

            # Load and validate the index using Pydantic
            index_data = self.index_mmap.read()
            self.index = MemoryMapDatasetIndex.model_validate_json(index_data)

        except OSError as e:
            self._cleanup_resources()
            raise MemoryMapFileOperationError(
                f"Failed to open memory-mapped files: {e}"
            ) from e
        except (ValueError, orjson.JSONDecodeError) as e:
            self._cleanup_resources()
            raise MemoryMapSerializationError(f"Invalid index data: {e}") from e

        self._finalizer = weakref.finalize(
            self,
            self._cleanup_finalizer,
            self.data_mmap,
            self.index_mmap,
            self.data_file,
            self.index_file,
        )

        _logger.debug(
            lambda: "MemoryMapDatasetClient initialized successfully",
            extra={
                "component": "MemoryMapDatasetClient",
                "data_file_path": str(self.data_file_path),
                "index_file_path": str(self.index_file_path),
                "conversations_count": len(self.index.session_ids),
                "total_size_bytes": self.index.total_size,
            },
        )

    def __enter__(self) -> "MemoryMapDatasetClient":
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

    @staticmethod
    def _cleanup_finalizer(
        data_mmap: mmap.mmap | None,
        index_mmap: mmap.mmap | None,
        data_file: Any | None,
        index_file: Any | None,
    ) -> None:
        """Resource cleanup method called during garbage collection."""
        resources = [data_mmap, index_mmap, data_file, index_file]
        for resource in resources:
            with suppress(Exception):
                if resource is not None:
                    resource.close()
                    _logger.debug("Finalizer cleaned up resource")

    def _cleanup_resources(self) -> None:
        """Clean up partially opened resources during error recovery."""
        for attr in ["data_mmap", "index_mmap", "data_file", "index_file"]:
            if hasattr(self, attr):
                obj = getattr(self, attr)
                if obj:
                    with suppress(Exception):
                        obj.close()

    def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a conversation by ID without loading all conversations into memory.

        Uses the offset index to read only the specific conversation data from the
        memory-mapped file, minimizing memory usage in workers.

        Args:
            conversation_id: Specific conversation ID to retrieve

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id is not found
            MemoryMapSerializationError: If conversation data is corrupted
        """
        # Check if conversation exists in index
        if conversation_id not in self.index.offsets:
            raise KeyError(f"Conversation '{conversation_id}' not found in dataset")

        # Get offset information for this conversation
        offset_info = self.index.offsets[conversation_id]

        try:
            # Seek to the conversation's offset and read only its data
            self.data_mmap.seek(offset_info.offset)
            conv_bytes = self.data_mmap.read(offset_info.size)

            _logger.debug(
                lambda: f"Loading conversation '{conversation_id}' from memory-mapped file",
                extra={
                    "component": "MemoryMapDatasetClient",
                    "operation": "get_conversation",
                    "conversation_id": conversation_id,
                    "offset": offset_info.offset,
                    "size_bytes": offset_info.size,
                },
            )

            # Deserialize only this conversation using orjson
            conversation = MemoryMapDatasetSerializer.deserialize_single_conversation(
                conv_bytes
            )

            return conversation

        except (OSError, MemoryMapSerializationError) as e:
            _logger.error(
                f"Failed to load conversation '{conversation_id}' from memory-mapped file",
                extra={
                    "component": "MemoryMapDatasetClient",
                    "conversation_id": conversation_id,
                    "error": str(e),
                    "data_file_path": str(self.data_file_path),
                },
            )
            raise

    def close(self) -> None:
        """Close the memory-mapped files and associated resources.

        This method is safe to call multiple times.
        """
        resources = [
            (self.data_mmap, "data_mmap", MemoryMapFileType.DATA),
            (self.index_mmap, "index_mmap", MemoryMapFileType.INDEX),
            (self.data_file, "data_file", MemoryMapFileType.DATA),
            (self.index_file, "index_file", MemoryMapFileType.INDEX),
        ]

        for resource, attr_name, file_type in resources:
            if hasattr(self, attr_name) and resource:
                try:
                    resource.close()
                    _logger.debug(f"Closed {file_type.value} {attr_name}")
                except Exception as e:
                    _logger.warning(f"Error closing {file_type.value} {attr_name}: {e}")
                finally:
                    setattr(self, attr_name, None)
