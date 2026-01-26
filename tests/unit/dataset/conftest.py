# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for dataset manager testing.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aiperf.endpoints  # noqa: F401  # Import to register endpoints
import aiperf.transports  # noqa: F401  # Import to register transports
from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.models import Conversation
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import EndpointType


def _mock_communication_init(self, service_config, **kwargs):
    """Mock CommunicationMixin.__init__ to avoid real ZMQ connections."""
    from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin

    AIPerfLifecycleMixin.__init__(self, service_config=service_config, **kwargs)
    self.service_config = service_config
    self.comms = MagicMock()
    # Set was_initialized to False to avoid publish calls in _on_state_change
    self.comms.was_initialized = False

    # Create mock sub_client and pub_client with proper async methods
    mock_sub_client = MagicMock()
    mock_sub_client.subscribe_all = AsyncMock()
    mock_sub_client.subscribe = AsyncMock()

    mock_pub_client = MagicMock()
    mock_pub_client.publish = AsyncMock()

    mock_reply_client = MagicMock()
    mock_reply_client.register_request_handler = AsyncMock()

    # Make comms return our mock clients when creating them
    self.comms.create_sub_client.return_value = mock_sub_client
    self.comms.create_pub_client.return_value = mock_pub_client
    self.comms.create_reply_client.return_value = mock_reply_client

    # Note: We do NOT mock logging methods here (debug, info, etc.) so that
    # tests using caplog can capture real log messages


@pytest.fixture(autouse=True)
def mock_communication_layer():
    """Autouse fixture to mock communication layer for all dataset tests."""
    with patch(
        "aiperf.common.mixins.CommunicationMixin.__init__", _mock_communication_init
    ):
        yield


@pytest.fixture
def user_config(tmp_path: Path) -> UserConfig:
    """Create a UserConfig for testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
            url="http://localhost:8000",
        ),
        output=OutputConfig(artifact_directory=tmp_path),
    )


@pytest.fixture
def empty_dataset_manager(
    user_config: UserConfig,
) -> DatasetManager:
    """Create a DatasetManager instance with empty dataset."""
    manager = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )
    manager.dataset = {}
    return manager


@pytest.fixture
def populated_dataset_manager(
    user_config: UserConfig,
    sample_conversations: dict[str, Conversation],
) -> DatasetManager:
    """Create a DatasetManager instance with sample data."""
    manager = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )
    manager.dataset = sample_conversations
    return manager


@pytest.fixture
def capture_file_writes():
    """Provide a fixture to capture file write operations for testing purposes."""

    class FileWriteCapture:
        def __init__(self):
            self.written_content = ""

        def write_bytes(self, data: bytes):
            self.written_content = data.decode("utf-8")

    capture = FileWriteCapture()

    def mock_write_bytes(self, data):
        capture.write_bytes(data)

    with patch("pathlib.Path.write_bytes", mock_write_bytes):
        yield capture


@pytest.fixture
def conversation_ids() -> list[str]:
    """Standard list of conversation IDs for sampler testing."""
    return ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
