# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-mapped dataset for zero-copy conversation access.

Eliminates the DatasetManager network bottleneck at high QPS by letting workers
read conversations directly from shared files in O(1) time.

Flow:
    1. DatasetManager writes conversations to disk via MemoryMapDatasetBackingStore
    2. Workers read via mmap (zero-copy) through MemoryMapDatasetClientStore

For Kubernetes: set AIPERF_DATASET_MMAP_BASE_PATH to a shared PVC mount.
"""

import asyncio
import mmap
import os
import tempfile
import weakref
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any

import aiofiles
import orjson
from pydantic import Field, field_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.user_config import UserConfig
from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CreditPhase,
    DatasetBackingStoreType,
    DatasetClientStoreType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import (
    MemoryMapFileOperationError,
    MemoryMapSerializationError,
)
from aiperf.common.factories import (
    DatasetBackingStoreFactory,
    DatasetClientStoreFactory,
    EndpointFactory,
)
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    AIPerfBaseModel,
    BaseMemoryMapClientMetadata,
    Conversation,
    MemoryMapConversationClientMetadata,
    MemoryMapPayloadClientMetadata,
    ModelEndpointInfo,
    RequestInfo,
)
from aiperf.common.protocols import (
    DatasetBackingStoreProtocol,
    DatasetClientStoreProtocol,
)

_logger = AIPerfLogger(__name__)


@implements_protocol(DatasetBackingStoreProtocol)
class BaseMemoryMapDatasetBackingStore(AIPerfLifecycleMixin, ABC):
    """Streams conversations to disk as they arrive (DatasetManager side).

    Writes each conversation immediately — constant memory usage regardless of dataset size.
    Preserves insertion order.

    Directory Structure::

        {base_path}/aiperf_mmap_{benchmark_id}/
        ├── dataset.dat   # Serialized conversation data (JSON bytes)
        └── index.dat     # Byte offset index for O(1) lookups

    For Kubernetes with shared storage, set AIPERF_DATASET_MMAP_BASE_PATH to a PVC mount
    so workers across pods can access the same files.
    """

    def __init__(self, user_config: UserConfig, **kwargs) -> None:
        """Initialize memory-mapped storage.

        Args:
            user_config: User configuration
            **kwargs: Additional configuration (unused for local mmap)
        """
        super().__init__()
        self._finalized = False
        self._user_config = user_config

        # Streaming state
        self._data_file = None
        self._current_offset = 0
        self._offsets: dict[str, ConversationOffset] = {}
        self._conversation_ids: list[str] = []  # Maintain insertion order

        # File paths (configurable base path for k8s mounted volumes)
        # Directory structure: {base_path}/aiperf_mmap_{benchmark_id}/
        base_path = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())
        dir_suffix = user_config.benchmark_id or f"{os.getpid()}_{id(self)}"
        mmap_dir = base_path / f"aiperf_mmap_{dir_suffix}"
        self._data_path: Path = mmap_dir / "dataset.dat"
        self._index_path: Path = mmap_dir / "index.dat"

    @on_init
    async def _setup(self) -> None:
        """Create output directory and open data file for streaming writes."""
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_file = await aiofiles.open(self._data_path, "wb")
        self.info(
            f"Memory-mapped backing store initialized (streaming to {self._data_path})"
        )

    async def _write_conversation_bytes(
        self, conversation_id: str, conv_bytes: bytes
    ) -> None:
        """Write conversation bytes to the data file."""
        if self._finalized:
            raise RuntimeError("Cannot add conversations after finalization")

        await self._data_file.write(conv_bytes)

        self._offsets[conversation_id] = ConversationOffset(
            offset=self._current_offset, size=len(conv_bytes)
        )
        self._conversation_ids.append(conversation_id)
        self._current_offset += len(conv_bytes)

        if len(self._conversation_ids) % 1000 == 0:
            self.debug(
                f"Streamed {len(self._conversation_ids)} conversations ({self._current_offset} bytes)"
            )

    @abstractmethod
    async def add_conversation(
        self, conversation_id: str, conversation: Conversation
    ) -> None:
        """Add a single conversation (written immediately to file).

        Args:
            conversation_id: Unique ID of the conversation
            conversation: Conversation object to add

        Raises:
            RuntimeError: If already finalized
        """
        pass

    async def add_conversations(self, conversations: dict[str, Conversation]) -> None:
        """Add multiple conversations (written immediately to file).

        Args:
            conversations: Dictionary mapping session IDs to Conversation objects

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Cannot add conversations after finalization")
        for conversation_id, conversation in conversations.items():
            await self.add_conversation(conversation_id, conversation)

    async def finalize(self) -> None:
        """Finalize by closing data file and writing index.

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Already finalized")

        await self._data_file.close()
        self.info(
            f"Data file finalized: {len(self._conversation_ids)} conversations, {self._current_offset / BYTES_PER_MIB:,.2f} MB"
        )

        index = MemoryMapDatasetIndex(
            conversation_ids=self._conversation_ids,
            offsets=self._offsets,
            total_size=self._current_offset,
        )
        async with aiofiles.open(self._index_path, "wb") as f:
            await f.write(index.model_dump_json(by_alias=True).encode("utf-8"))

        self._finalized = True
        self.info(f"Index file created: {self._index_path}")

    @abstractmethod
    def get_client_metadata(self) -> BaseMemoryMapClientMetadata:
        """Return file paths for client initialization.

        Returns:
            MemoryMapClientMetadata with file paths and stats

        Raises:
            RuntimeError: If not finalized
        """
        pass

    @on_stop
    async def _cleanup(self) -> None:
        """Close file handle and delete temp files."""
        if self._data_file and not self._data_file.closed:
            await self._data_file.close()

        for path in [self._data_path, self._index_path]:
            if path.exists():
                try:
                    path.unlink()
                    self.debug(f"Removed file: {path}")
                except OSError as e:
                    self.warning(f"Error removing file {path}: {e}")

        self.debug("Memory-mapped backing store cleanup complete")


@implements_protocol(DatasetBackingStoreProtocol)
@DatasetBackingStoreFactory.register(DatasetBackingStoreType.MEMORY_MAP_CONVERSATION)
class MemoryMapConversationDatasetBackingStore(BaseMemoryMapDatasetBackingStore):
    """Write conversations to memory-mapped files (DatasetManager side).

    Uses mmap for zero-copy writes — the OS pages data into memory as needed.
    """

    async def add_conversation(
        self, conversation_id: str, conversation: Conversation
    ) -> None:
        """Add a single conversation (written immediately to file).

        Args:
            conversation_id: Unique ID of the conversation
            conversation: Conversation object to add
        """
        if self._finalized:
            raise RuntimeError("Cannot add conversations after finalization")

        conv_bytes = orjson.dumps(
            conversation.model_dump(exclude_none=True, mode="json")
        )
        await self._write_conversation_bytes(conversation_id, conv_bytes)

    def get_client_metadata(self) -> BaseMemoryMapClientMetadata:
        """Return file paths for client initialization.

        Returns:
            MemoryMapClientMetadata with file paths and stats

        Raises:
            RuntimeError: If not finalized
        """
        if not self._finalized:
            raise RuntimeError(
                "Cannot get metadata before finalization. Call finalize() first."
            )

        return MemoryMapConversationClientMetadata(
            data_file_path=self._data_path,
            index_file_path=self._index_path,
            conversation_count=len(self._conversation_ids),
            total_size_bytes=self._current_offset,
        )


@implements_protocol(DatasetBackingStoreProtocol)
@DatasetBackingStoreFactory.register(DatasetBackingStoreType.MEMORY_MAP_PAYLOAD)
class MemoryMapPayloadDatasetBackingStore(BaseMemoryMapDatasetBackingStore):
    """Write payloads to memory-mapped files (DatasetManager side).

    Uses mmap for zero-copy writes — the OS pages data into memory as needed.
    """

    def __init__(self, user_config: UserConfig, **kwargs) -> None:
        """Initialize memory-mapped storage.

        Args:
            user_config: User configuration
            **kwargs: Additional configuration (unused for local mmap)
        """
        super().__init__(user_config=user_config, **kwargs)
        self._model_endpoint = ModelEndpointInfo.from_user_config(user_config)
        self._endpoint = EndpointFactory.create_instance(
            self._model_endpoint.endpoint.type,
            model_endpoint=self._model_endpoint,
        )

    async def add_conversation(
        self, conversation_id: str, conversation: Conversation
    ) -> None:
        """Add a single payload (written immediately to file).

        Args:
            conversation_id: Unique ID of the conversation
            conversation: Conversation object to add
        """
        if self._finalized:
            raise RuntimeError("Cannot add payloads after finalization")

        if len(conversation.turns) != 1:
            raise ValueError(
                "Raw Payload DatasetBackingStore must have exactly one turn"
            )

        payload = self._endpoint.format_payload(
            RequestInfo(
                model_endpoint=self._model_endpoint,
                turns=conversation.turns,
                turn_index=0,
                credit_num=0,
                credit_phase=CreditPhase.PROFILING,
                x_request_id="",
                x_correlation_id="",
                conversation_id=conversation_id,
            )
        )
        conv_bytes = orjson.dumps(payload)
        await self._write_conversation_bytes(conversation_id, conv_bytes)

    def get_client_metadata(self) -> BaseMemoryMapClientMetadata:
        """Return file paths for client initialization.

        Returns:
            MemoryMapClientMetadata with file paths and stats

        Raises:
            RuntimeError: If not finalized
        """
        if not self._finalized:
            raise RuntimeError(
                "Cannot get metadata before finalization. Call finalize() first."
            )

        return MemoryMapPayloadClientMetadata(
            data_file_path=self._data_path,
            index_file_path=self._index_path,
            conversation_count=len(self._conversation_ids),
            total_size_bytes=self._current_offset,
        )


@implements_protocol(DatasetClientStoreProtocol)
class BaseMemoryMapDatasetClientStore(AIPerfLifecycleMixin, ABC):
    """Reads conversations from memory-mapped files (Worker side).

    Uses mmap for zero-copy reads — the OS pages data into memory as needed.
    """

    def __init__(self, client_metadata: BaseMemoryMapClientMetadata, **kwargs) -> None:
        """Initialize from metadata provided by backing store.

        Args:
            client_metadata: Typed metadata from BaseMemoryMapDatasetBackingStore.get_client_metadata()
        """
        super().__init__(**kwargs)
        self._data_path: Path = client_metadata.data_file_path
        self._index_path: Path = client_metadata.index_file_path
        self._client: BaseMemoryMapDatasetClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @abstractmethod
    def _create_client(self) -> "BaseMemoryMapDatasetClient":
        """Create a new instance of the memory-mapped client."""
        pass

    @on_init
    async def _setup(self) -> None:
        """Open memory-mapped files (read-only)."""
        self._loop = asyncio.get_running_loop()
        self.debug(
            lambda: f"Opening memory-mapped files: data={self._data_path}, index={self._index_path}"
        )
        self._client = self._create_client()
        self.debug(
            lambda: f"Memory-mapped client store initialized with "
            f"{len(self._client.index.conversation_ids)} conversations"
        )

    async def get_conversation(self, conversation_id: str) -> Conversation | bytes:
        """Retrieve conversation from memory-mapped file.

        Runs in executor since mmap reads can block on page faults.

        Args:
            conversation_id: Session ID of the conversation

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id not found
        """
        if self._client is None or self._loop is None:
            raise RuntimeError("Client store not initialized. Call initialize() first.")
        return await self._loop.run_in_executor(
            None, self._client.get_conversation, conversation_id
        )

    @on_stop
    async def _cleanup(self) -> None:
        """Close memory-mapped files."""
        if self._client:
            self.debug("Closing memory-mapped files")
            self._client.close()
            self.debug("Memory-mapped client store cleanup complete")


@implements_protocol(DatasetClientStoreProtocol)
@DatasetClientStoreFactory.register(DatasetClientStoreType.MEMORY_MAP_CONVERSATION)
class MemoryMapConversationClientStore(BaseMemoryMapDatasetClientStore):
    """Read conversations from memory-mapped files (Worker side).

    Uses mmap for zero-copy reads — the OS pages data into memory as needed.
    """

    def _create_client(self) -> "BaseMemoryMapDatasetClient":
        return MemoryMapConversationClient(self._data_path, self._index_path)


@implements_protocol(DatasetClientStoreProtocol)
@DatasetClientStoreFactory.register(DatasetClientStoreType.MEMORY_MAP_PAYLOAD)
class MemoryMapPayloadClientStore(BaseMemoryMapDatasetClientStore):
    """Read payloads from memory-mapped files (Worker side).

    Uses mmap for zero-copy reads — the OS pages data into memory as needed.
    """

    def _create_client(self) -> "BaseMemoryMapDatasetClient":
        return MemoryMapPayloadClient(self._data_path, self._index_path)


class ConversationOffset(AIPerfBaseModel):
    """Offset information for a single conversation in the memory-mapped file."""

    offset: int = Field(ge=0, description="Byte offset where conversation data starts")
    size: int = Field(ge=0, description="Size of the conversation data in bytes")


class MemoryMapDatasetIndex(AIPerfBaseModel):
    """Index structure for the memory-mapped dataset.

    All data is stored as uncompressed JSON bytes serialized with orjson.
    """

    conversation_ids: list[str] = Field(
        default_factory=list, description="List of all conversation IDs in the dataset"
    )
    offsets: dict[str, ConversationOffset] = Field(
        default_factory=dict,
        description="Mapping of conversation IDs to their byte offsets and sizes",
    )
    total_size: int = Field(
        default=0, ge=0, description="Total size of the serialized dataset in bytes"
    )

    @field_validator("conversation_ids")
    @classmethod
    def validate_conversation_ids(cls, v: list[str]) -> list[str]:
        """Ensure conversation_ids are unique."""
        if len(v) != len(set(v)):
            raise ValueError("conversation_ids must contain unique values")
        return v


class BaseMemoryMapDatasetClient(ABC):
    """Low-level mmap client for reading conversations.

    Use as context manager or call close() explicitly.
    """

    def __init__(self, data_file_path: Path | str, index_file_path: Path | str) -> None:
        """Open memory-mapped files and load the index.

        Args:
            data_file_path: Path to the memory-mapped data file
            index_file_path: Path to the memory-mapped index file

        Raises:
            MemoryMapFileOperationError: If files cannot be opened
            MemoryMapSerializationError: If index data is invalid
        """
        self.data_file_path = (
            Path(data_file_path) if isinstance(data_file_path, str) else data_file_path
        )
        self.index_file_path = (
            Path(index_file_path)
            if isinstance(index_file_path, str)
            else index_file_path
        )

        if not self.data_file_path.exists():
            raise MemoryMapFileOperationError(f"Data file not found: {data_file_path}")
        if not self.index_file_path.exists():
            raise MemoryMapFileOperationError(
                f"Index file not found: {index_file_path}"
            )

        try:
            self.data_file = self.data_file_path.open("rb")
            self.data_mmap = mmap.mmap(
                self.data_file.fileno(), 0, access=mmap.ACCESS_READ
            )

            self.index_file = self.index_file_path.open("rb")
            self.index_mmap = mmap.mmap(
                self.index_file.fileno(), 0, access=mmap.ACCESS_READ
            )

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

        # Safety net: closes resources when object is garbage collected if close() wasn't called.
        # weakref.finalize holds a weak ref to self, and the callback receives the resources
        # as args (not self) so cleanup can run even after self is gone.
        self._finalizer = weakref.finalize(
            self,
            self._cleanup_finalizer,
            self.data_mmap,
            self.index_mmap,
            self.data_file,
            self.index_file,
        )

        _logger.debug(
            lambda: f"{self.__class__.__name__} initialized successfully: data_file={self.data_file_path}, index_file={self.index_file_path}, conversations={len(self.index.conversation_ids)}, size={self.index.total_size} bytes"
        )

    def __enter__(self) -> "BaseMemoryMapDatasetClient":
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

    _RESOURCE_ATTRS = ("data_mmap", "index_mmap", "data_file", "index_file")

    @staticmethod
    def _cleanup_finalizer(
        data_mmap: mmap.mmap | None,
        index_mmap: mmap.mmap | None,
        data_file: Any | None,
        index_file: Any | None,
    ) -> None:
        """Called by weakref.finalize during GC to close leaked resources."""
        for resource in (data_mmap, index_mmap, data_file, index_file):
            if resource is not None:
                with suppress(Exception):
                    resource.close()
                    _logger.debug("Finalizer cleaned up resource")

    def _cleanup_resources(self) -> None:
        """Close partially opened resources during __init__ error recovery."""
        for attr in self._RESOURCE_ATTRS:
            if (obj := getattr(self, attr, None)) is not None:
                with suppress(Exception):
                    obj.close()

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Conversation | bytes:
        """Get a conversation by ID. O(1) lookup using byte offset index.

        Args:
            conversation_id: Specific conversation ID to retrieve

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id is not found
            MemoryMapSerializationError: If conversation data is corrupted
        """
        pass

    def close(self) -> None:
        """Close the memory-mapped files and associated resources.

        This method is safe to call multiple times.
        """
        for attr_name in ["data_mmap", "index_mmap", "data_file", "index_file"]:
            resource = getattr(self, attr_name, None)
            if resource is not None:
                try:
                    resource.close()
                except Exception as e:
                    _logger.warning(f"Error closing {attr_name}: {e}")
                finally:
                    setattr(self, attr_name, None)


class MemoryMapConversationClient(BaseMemoryMapDatasetClient):
    """Low-level mmap client for reading conversations.

    Use as context manager or call close() explicitly.
    """

    def _deserialize_conversation(self, data: bytes) -> Conversation:
        """Deserialize a single conversation from bytes.

        Args:
            data: Serialized conversation data bytes (JSON format)

        Returns:
            Conversation object

        Raises:
            MemoryMapSerializationError: If deserialization fails
        """
        try:
            return Conversation.model_validate_json(data)
        except Exception as e:
            raise MemoryMapSerializationError(
                f"Failed to decode conversation data: {e}"
            ) from e

    def get_conversation(self, conversation_id: str) -> Conversation | bytes:
        """Get a conversation by ID. O(1) lookup using byte offset index.

        Args:
            conversation_id: Specific conversation ID to retrieve

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id is not found
            MemoryMapSerializationError: If conversation data is corrupted
            OSError: If the file cannot be opened
        """
        if conversation_id not in self.index.offsets:
            raise KeyError(f"Conversation '{conversation_id}' not found in dataset")

        offset_info = self.index.offsets[conversation_id]

        try:
            self.data_mmap.seek(offset_info.offset)
            conv_bytes = self.data_mmap.read(offset_info.size)

            _logger.debug(
                lambda: f"Loading conversation '{conversation_id}': offset={offset_info.offset}, size={offset_info.size} bytes"
            )

            return self._deserialize_conversation(conv_bytes)

        except (OSError, MemoryMapSerializationError) as e:
            _logger.error(
                f"Failed to load conversation '{conversation_id}' from {self.data_file_path}: {e}"
            )
            raise e


class MemoryMapPayloadClient(BaseMemoryMapDatasetClient):
    """Low-level mmap client for reading payloads.

    Use as context manager or call close() explicitly.
    """

    def get_conversation(self, conversation_id: str) -> Conversation | bytes:
        """Get a conversation by ID. O(1) lookup using byte offset index.

        Args:
            conversation_id: Specific conversation ID to retrieve

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id is not found
            OSError: If the file cannot be opened
        """
        if conversation_id not in self.index.offsets:
            raise KeyError(f"Conversation '{conversation_id}' not found in dataset")

        offset_info = self.index.offsets[conversation_id]

        try:
            self.data_mmap.seek(offset_info.offset)
            payload_bytes = self.data_mmap.read(offset_info.size)

            _logger.debug(
                lambda: f"Loading conversation '{conversation_id}': offset={offset_info.offset}, size={offset_info.size} bytes"
            )
            return payload_bytes

        except OSError as e:
            _logger.error(
                f"Failed to load conversation '{conversation_id}' from {self.data_file_path}: {e}"
            )
            raise e
