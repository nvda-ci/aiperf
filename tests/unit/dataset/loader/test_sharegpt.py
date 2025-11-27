# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from aiperf.common.models import Conversation
from aiperf.dataset.loader.file.sharegpt import ShareGPTLoader
from aiperf.dataset.loader.models import ShareGPT


class TestShareGPTLoader:
    """Test suite for ShareGPTLoader class"""

    @pytest.fixture
    def sharegpt_file(self, tmp_path):
        """Create a temporary ShareGPT format JSON file for testing"""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello how are you"},
                    {"from": "gpt", "value": "This is test output"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "This is test output"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Hello how are you"},
                    {"from": "gpt", "value": "This"},
                ]
            },
        ]
        file_path = tmp_path / "sharegpt_test.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return str(file_path)

    @pytest.fixture
    def sharegpt_loader(self, user_config, mock_tokenizer_cls, sharegpt_file):
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        return ShareGPTLoader(
            config=user_config, tokenizer=tokenizer, filename=sharegpt_file
        )

    def test_initialization(self, sharegpt_loader: ShareGPTLoader):
        """Test initialization of ShareGPTLoader"""
        assert sharegpt_loader.tokenizer is not None
        assert sharegpt_loader.config is not None
        assert isinstance(sharegpt_loader.filename, str)

    def test_can_load_valid_data(self, tmp_path):
        """Test can_load_file method with valid ShareGPT file"""
        valid_data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there"},
                ]
            }
        ]
        file_path = tmp_path / "valid_sharegpt.json"
        with open(file_path, "w") as f:
            json.dump(valid_data, f)
        assert ShareGPTLoader.can_load_file(file_path) is True

    def test_can_load_invalid_data(self, tmp_path):
        """Test can_load_file method with invalid data"""
        invalid_data = {"messages": [{"from": "human", "value": "Hello"}]}
        file_path = tmp_path / "invalid_sharegpt.json"
        with open(file_path, "w") as f:
            json.dump(invalid_data, f)
        assert ShareGPTLoader.can_load_file(file_path) is False

    def test_get_preferred_sampling_strategy(self, sharegpt_loader: ShareGPTLoader):
        """Test get_preferred_sampling_strategy method"""
        strategy = sharegpt_loader.get_preferred_sampling_strategy()
        from aiperf.common.enums import DatasetSamplingStrategy

        assert strategy == DatasetSamplingStrategy.SEQUENTIAL

    def test_load_dataset(self, sharegpt_loader: ShareGPTLoader):
        """Test loading dataset from file"""
        data = sharegpt_loader.parse_and_validate()
        assert isinstance(data, list)
        assert len(data) == 3
        assert all(isinstance(entry, ShareGPT) for entry in data)

    def test_convert_to_conversations(self, sharegpt_loader: ShareGPTLoader):
        """Test converting loaded dataset to conversations"""
        data = sharegpt_loader.parse_and_validate()
        conversations = sharegpt_loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)
        turn = conversations[0].turns[0]
        assert turn.texts[0].contents[0] == "Hello how are you"
        assert turn.max_tokens == len(["This", "is", "test", "output"])
        assert turn.model == "test-model"

    def test_end_to_end_workflow(self, sharegpt_loader: ShareGPTLoader):
        """Test complete workflow: load -> convert"""
        data = sharegpt_loader.parse_and_validate()
        assert len(data) == 3
        conversations = sharegpt_loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)
        assert len(conversations[0].turns) == 1
