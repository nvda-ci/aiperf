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
        assert sharegpt_loader.turn_count == 0
        assert isinstance(sharegpt_loader.filename, str)

    def test_can_load_valid_data(self):
        """Test can_load method with valid ShareGPT data"""
        valid_data = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there"},
            ]
        }
        assert ShareGPTLoader.can_load(data=valid_data) is True

    def test_can_load_invalid_data(self):
        """Test can_load method with invalid data"""
        # Missing conversations field
        invalid_data = {"messages": [{"from": "human", "value": "Hello"}]}
        assert ShareGPTLoader.can_load(data=invalid_data) is False

        # Too few conversations
        invalid_data = {"conversations": [{"from": "human", "value": "Hello"}]}
        assert ShareGPTLoader.can_load(data=invalid_data) is False

    def test_get_preferred_sampling_strategy(self):
        """Test get_preferred_sampling_strategy method"""
        strategy = ShareGPTLoader.get_preferred_sampling_strategy()
        from aiperf.common.enums import DatasetSamplingStrategy

        assert strategy == DatasetSamplingStrategy.SEQUENTIAL

    def test_load_dataset(self, sharegpt_loader: ShareGPTLoader):
        """Test loading dataset from file"""
        dataset = sharegpt_loader.parse_and_validate()

        assert "default" in dataset
        assert len(dataset["default"]) == 3
        assert all(isinstance(entry, ShareGPT) for entry in dataset["default"])

    def test_convert_to_conversations(self, sharegpt_loader: ShareGPTLoader):
        """Test converting loaded dataset to conversations"""
        dataset = sharegpt_loader.parse_and_validate()
        conversations = sharegpt_loader.convert_to_conversations(dataset)

        # Should only include 1 valid conversation (others filtered out by validation)
        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)

        turn = conversations[0].turns[0]
        assert turn.texts[0].contents[0] == "Hello how are you"
        assert turn.max_tokens == len(["This", "is", "test", "output"])
        assert turn.model == "test-model"

    def test_end_to_end_workflow(self, sharegpt_loader: ShareGPTLoader):
        """Test complete workflow: load -> convert"""
        # Load the dataset
        dataset = sharegpt_loader.parse_and_validate()
        assert len(dataset["default"]) == 3

        # Convert to conversations
        conversations = sharegpt_loader.convert_to_conversations(dataset)

        # Verify filtering worked (only 1 valid entry)
        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)
        assert len(conversations[0].turns) == 1
