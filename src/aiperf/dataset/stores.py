# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset store implementations for different storage backends."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import aiofiles
import orjson

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DatasetBackingStoreType, DatasetClientStoreType
from aiperf.common.factories import (
    DatasetBackingStoreFactory,
    DatasetClientStoreFactory,
)
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import Conversation
from aiperf.common.protocols import (
    DatasetBackingStoreProtocol,
    DatasetClientStoreProtocol,
)
from aiperf.dataset.memory_map_utils import (
    MEMORY_MAP_CONSTANTS,
    ConversationOffset,
    MemoryMapDatasetClient,
    MemoryMapDatasetIndex,
)

# ==================== IN-MEMORY STORES ====================


@implements_protocol(DatasetBackingStoreProtocol)
@DatasetBackingStoreFactory.register(DatasetBackingStoreType.IN_MEMORY)
class InMemoryDatasetBackingStore(AIPerfLifecycleMixin):
    """In-memory dataset storage for maximum speed (DatasetManager side).

    Insertion order is preserved. Workers access via ZMQ_REMOTE client store.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize in-memory storage.

        Args:
            **kwargs: Additional configuration (unused for in-memory)
        """
        super().__init__()
        self._dataset: dict[str, Conversation] = {}
        self._finalized = False

    @on_init
    async def _setup(self) -> None:
        """Initialize in-memory storage."""
        self.info("In-memory backing store initialized")

    async def add_conversation(
        self, conversation_id: str, conversation: Conversation
    ) -> None:
        """Add a single conversation to in-memory dict.

        Args:
            conversation_id: Session ID of the conversation
            conversation: Conversation object to add

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Cannot add conversations after finalization")

        self._dataset[conversation_id] = conversation

        # Log progress periodically
        count = len(self._dataset)
        if count % 1000 == 0:
            self.debug(f"Added {count} conversations")

    async def add_conversations(self, conversations: dict[str, Conversation]) -> None:
        """Add multiple conversations to in-memory dict.

        Args:
            conversations: Dictionary mapping session IDs to Conversation objects

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Cannot add conversations after finalization")

        self._dataset.update(conversations)
        self.debug(
            f"Added {len(conversations)} conversations (total: {len(self._dataset)})"
        )

    async def finalize(self) -> None:
        """Finalize and make ready for workers.

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Already finalized")

        self._finalized = True
        self.info(
            f"In-memory backing store finalized with {len(self._dataset)} conversations"
        )

    def get_client_metadata(self) -> dict[str, Any]:
        """Return metadata for client initialization.

        Returns:
            Dict with store type and conversation count

        Raises:
            RuntimeError: If not finalized
        """
        if not self._finalized:
            raise RuntimeError(
                "Cannot get metadata before finalization. Call finalize() first."
            )

        return {
            "store_type": "in_memory",
            "conversation_count": len(self._dataset),
            "dataset": self._dataset,  # Pass the actual dataset for in-memory access
        }

    @on_stop
    async def _cleanup(self) -> None:
        """No cleanup needed for in-memory storage."""
        self.debug("In-memory backing store cleanup complete")


# ==================== MEMORY-MAPPED STORES ====================


@implements_protocol(DatasetBackingStoreProtocol)
@DatasetBackingStoreFactory.register(DatasetBackingStoreType.MEMORY_MAP)
class MemoryMapDatasetBackingStore(AIPerfLifecycleMixin):
    """Memory-mapped file storage for efficient access with lower memory usage (DatasetManager side).

    TRUE STREAMING: Writes conversations to file immediately as they arrive.
    No memory accumulation - constant memory usage regardless of dataset size!
    Insertion order is preserved. Local files only (single machine).
    """

    def __init__(self, **kwargs) -> None:
        """Initialize memory-mapped storage.

        Args:
            **kwargs: Additional configuration (unused for local mmap)
        """
        super().__init__()
        self._finalized = False

        # Streaming state
        self._data_file = None
        self._current_offset = 0
        self._offsets: dict[str, ConversationOffset] = {}
        self._session_ids: list[str] = []  # Maintain insertion order

        # File paths (created in temp dir)
        temp_dir = Path(tempfile.gettempdir()) / MEMORY_MAP_CONSTANTS.TEMP_DIR_NAME
        unique_suffix = f"{os.getpid()}_{id(self)}"
        self._data_path = str(
            temp_dir / f"dataset_{unique_suffix}{MEMORY_MAP_CONSTANTS.DATA_FILE_SUFFIX}"
        )
        self._index_path = str(
            temp_dir / f"index_{unique_suffix}{MEMORY_MAP_CONSTANTS.INDEX_FILE_SUFFIX}"
        )

    @on_init
    async def _setup(self) -> None:
        """Initialize streaming to data file."""
        # Ensure temp directory exists
        temp_dir = Path(self._data_path).parent
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Open data file for writing
        self._data_file = await aiofiles.open(self._data_path, "wb")
        self.info(
            f"Memory-mapped backing store initialized (streaming to {self._data_path})"
        )

    async def add_conversation(
        self, conversation_id: str, conversation: Conversation
    ) -> None:
        """Add a single conversation (written immediately to file).

        Args:
            conversation_id: Session ID of the conversation
            conversation: Conversation object to add

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Cannot add conversations after finalization")

        # Serialize conversation
        conv_data = conversation.model_dump(mode="json")
        conv_bytes = orjson.dumps(conv_data)

        # Write to file immediately
        await self._data_file.write(conv_bytes)

        # Track offset for index
        self._offsets[conversation_id] = ConversationOffset(
            offset=self._current_offset, size=len(conv_bytes)
        )
        self._session_ids.append(conversation_id)
        self._current_offset += len(conv_bytes)

        # Conversation is now garbage collected! No memory accumulation!

        # Log progress periodically
        if len(self._session_ids) % 1000 == 0:
            self.debug(
                f"Streamed {len(self._session_ids)} conversations ({self._current_offset} bytes)"
            )

    async def add_conversations(self, conversations: dict[str, Conversation]) -> None:
        """Add multiple conversations (written immediately to file).

        Args:
            conversations: Dictionary mapping session IDs to Conversation objects

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Cannot add conversations after finalization")

        # Stream each conversation
        for conversation_id, conversation in conversations.items():
            await self.add_conversation(conversation_id, conversation)

    async def finalize(self) -> None:
        """Finalize by closing data file and writing index.

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Already finalized")

        # Close data file
        await self._data_file.close()
        self.info(
            f"Data file finalized: {len(self._session_ids)} conversations, {self._current_offset} bytes"
        )

        # Create and write index
        index = MemoryMapDatasetIndex(
            session_ids=self._session_ids,
            offsets=self._offsets,
            total_size=self._current_offset,
        )

        async with aiofiles.open(self._index_path, "wb") as f:
            index_data = index.model_dump_json(by_alias=True).encode("utf-8")
            await f.write(index_data)

        self._finalized = True
        self.info(f"Index file created: {self._index_path}")

    def get_client_metadata(self) -> dict[str, Any]:
        """Return file paths for client initialization.

        Returns:
            Dict with file paths and conversation count

        Raises:
            RuntimeError: If not finalized
        """
        if not self._finalized:
            raise RuntimeError(
                "Cannot get metadata before finalization. Call finalize() first."
            )

        return {
            "store_type": "memory_map",
            "data_file_path": self._data_path,
            "index_file_path": self._index_path,
            "conversation_count": len(self._session_ids),
            "total_size_bytes": self._current_offset,
        }

    @on_stop
    async def _cleanup(self) -> None:
        """Clean up memory-mapped files."""
        # Close data file if still open
        if self._data_file and not self._data_file.closed:
            await self._data_file.close()

        # Remove temp files
        for file_path in [self._data_path, self._index_path]:
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    self.debug(f"Removed file: {file_path}")
                except OSError as e:
                    self.warning(f"Error removing file {file_path}: {e}")

        self.debug("Memory-mapped backing store cleanup complete")


@implements_protocol(DatasetClientStoreProtocol)
@DatasetClientStoreFactory.register(DatasetClientStoreType.MEMORY_MAP)
class MemoryMapDatasetClientStore(AIPerfLifecycleMixin):
    """Memory-mapped file access for workers (Worker side)."""

    def __init__(self, client_metadata: dict[str, Any], **kwargs) -> None:
        """Initialize from metadata provided by backing store.

        Args:
            client_metadata: Metadata from MemoryMapDatasetBackingStore.get_client_metadata()
            **kwargs: Additional configuration
        """
        super().__init__()
        self._data_path = client_metadata["data_file_path"]
        self._index_path = client_metadata["index_file_path"]
        self._client: MemoryMapDatasetClient | None = None

    @on_init
    async def _setup(self) -> None:
        """Open memory-mapped files (read-only)."""
        self.info(
            f"Opening memory-mapped files: data={self._data_path}, index={self._index_path}"
        )

        self._client = MemoryMapDatasetClient(self._data_path, self._index_path)

        self.info(
            f"Memory-mapped client store initialized with "
            f"{len(self._client.index.session_ids)} conversations"
        )

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Retrieve conversation from memory-mapped file.

        Args:
            conversation_id: Session ID of the conversation

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id not found
        """
        if self._client is None:
            raise RuntimeError("Client store not initialized. Call initialize() first.")

        # Run blocking mmap read in executor to keep it async
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._client.get_conversation, conversation_id
        )

    @on_stop
    async def _cleanup(self) -> None:
        """Close memory-mapped files."""
        if self._client:
            self.info("Closing memory-mapped files")
            self._client.close()
            self.debug("Memory-mapped client store cleanup complete")


# ==================== ZMQ REMOTE CLIENT STORE ====================


@implements_protocol(DatasetClientStoreProtocol)
@DatasetClientStoreFactory.register(DatasetClientStoreType.ZMQ_REMOTE)
class ZMQRemoteDatasetClientStore(AIPerfLifecycleMixin):
    """Remote dataset access via ZMQ request-reply to DatasetManager (Worker side).

    Used when workers cannot access local files (distributed deployment, cloud).
    DatasetManager handles requests via @on_request(MessageType.CONVERSATION_REQUEST).
    """

    def __init__(self, client_metadata: dict[str, Any], **kwargs) -> None:
        """Initialize ZMQ remote client store.

        Args:
            client_metadata: Metadata from backing store (unused - ZMQ uses existing infrastructure)
            **kwargs: Additional configuration
        """
        super().__init__()
        # ZMQ_REMOTE uses existing DatasetManager request-reply infrastructure
        # No additional setup needed - workers already have access via send_request()

    @on_init
    async def _setup(self) -> None:
        """Initialize ZMQ remote access."""
        self.info(
            "ZMQ remote client store initialized (using existing request infrastructure)"
        )

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Retrieve conversation via ZMQ request to DatasetManager.

        Uses existing infrastructure - DatasetManager handles via
        @on_request(MessageType.CONVERSATION_REQUEST).

        Args:
            conversation_id: Session ID of the conversation

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id not found
        """
        # Note: This requires the worker to have RequestClient capability
        # Workers should use self.send_request(ConversationRequestMessage(conversation_id)) directly
        raise NotImplementedError(
            "ZMQ_REMOTE client store requires integration with Worker's request client. "
            "Workers should use send_request(ConversationRequestMessage) directly."
        )

    @on_stop
    async def _cleanup(self) -> None:
        """No cleanup needed for ZMQ remote."""
        self.debug("ZMQ remote client store cleanup complete")
