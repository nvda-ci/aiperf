# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import (
    ConversationConfig,
    EndpointConfig,
    InputConfig,
    InputTokensConfig,
    PromptConfig,
    ServiceConfig,
    UserConfig,
)
from aiperf.dataset.dataset_manager import DatasetManager


@pytest.fixture
def create_jsonl_file():
    """Create a temporary JSONL file with custom content."""
    filename = None

    def _create_file(content_lines):
        nonlocal filename
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in content_lines:
                f.write(line + "\n")
            filename = f.name
        return filename

    yield _create_file

    # Cleanup all created files
    if filename:
        Path(filename).unlink(missing_ok=True)


@pytest.fixture
def create_user_config_and_dataset_manager(
    mock_tokenizer_from_pretrained, mock_tokenizer_cls
):
    """Create a UserConfig and DatasetManager for testing."""

    def _create(file_path="test_data.jsonl"):
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig.model_construct(
                file=file_path,
                conversation=ConversationConfig(num=5),
                prompt=PromptConfig(
                    input_tokens=InputTokensConfig(mean=10, stddev=2),
                ),
            ),
        )
        # Mock the tokenizer to avoid HTTP requests
        mock_tokenizer_from_pretrained.return_value = (
            mock_tokenizer_cls.from_pretrained("test-model")
        )

        service_config = ServiceConfig()
        dataset_manager = DatasetManager(service_config, config)
        return config, dataset_manager

    return _create


@pytest.fixture
def default_user_config() -> UserConfig:
    """Create a default UserConfig for testing."""
    return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
