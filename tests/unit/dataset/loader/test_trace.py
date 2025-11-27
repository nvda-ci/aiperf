# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from aiperf.common.config import EndpointConfig, InputConfig, UserConfig
from aiperf.common.enums import CustomDatasetType
from aiperf.dataset import MooncakeTrace, MooncakeTraceDatasetLoader


class TestMooncakeTrace:
    """Basic functionality tests for MooncakeTrace model."""

    def test_create_with_input_length(self):
        """Test creating MooncakeTrace with input_length."""
        data = MooncakeTrace(input_length=100, hash_ids=[123, 456, 789], timestamp=1000)
        assert data.input_length == 100
        assert data.output_length is None
        assert data.text_input is None
        assert data.hash_ids == [123, 456, 789]
        assert data.timestamp == 1000
        assert data.type == CustomDatasetType.MOONCAKE_TRACE

    def test_create_with_text_input(self):
        """Test creating MooncakeTrace with text_input."""
        data = MooncakeTrace(text_input="This is test input text", timestamp=1000)
        assert data.text_input == "This is test input text"
        assert data.input_length is None
        assert data.output_length is None
        assert data.hash_ids is None
        assert data.timestamp == 1000

    def test_create_with_both_input_fields_and_hash_ids(self):
        """Test that input_length and text_input cannot be provided together."""
        with pytest.raises(
            ValidationError,
            match="'input_length' and 'text_input' cannot be provided together",
        ):
            MooncakeTrace(
                input_length=100,
                text_input="This is test input text",
                hash_ids=[123],
                timestamp=1000,
            )

    def test_create_with_optional_output_length(self):
        """Test creating MooncakeTrace with optional output_length."""
        data = MooncakeTrace(
            input_length=100, output_length=50, hash_ids=[123], timestamp=1000
        )
        assert data.output_length == 50

    def test_validation_missing_input_fields_errors(self):
        """Test validation errors when neither input_length nor text_input provided."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError,
            match="Either 'input_length' or 'text_input' must be provided",
        ):
            MooncakeTrace(hash_ids=[123], timestamp=1000)

    def test_validation_missing_required_fields_errors(self):
        """Test validation errors for MooncakeTrace missing other required fields."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError,
            match="Either 'input_length' or 'text_input' must be provided",
        ):
            MooncakeTrace(hash_ids=[123], timestamp=1000)
        data = MooncakeTrace(text_input="test input")
        assert data.text_input == "test input"
        assert data.hash_ids is None

    def test_validation_hash_ids_requires_input_length(self):
        """Test that hash_ids is only allowed with input_length, not text_input."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError,
            match="'hash_ids' is only allowed when 'input_length' is provided, not when 'text_input' is provided",
        ):
            MooncakeTrace(text_input="test input", hash_ids=[123], timestamp=1000)


class TestMooncakeTraceDatasetLoader:
    """Basic functionality tests for MooncakeTraceDatasetLoader."""

    @pytest.fixture
    def mock_prompt_generator(self):
        """Create a mock prompt generator for testing."""
        generator = Mock()
        generator.generate.return_value = "Generated prompt text"
        return generator

    @pytest.fixture
    def default_user_config(self):
        """Create a default UserConfig for testing."""
        return UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

    def make_user_config(
        self, start_offset: int | None = None, end_offset: int | None = None
    ):
        """Create a UserConfig for testing."""
        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                fixed_schedule_start_offset=start_offset,
                fixed_schedule_end_offset=end_offset,
            ),
        )

    def test_load_dataset_basic_functionality(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test basic JSONL file loading."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123, 456], "timestamp": 1000}',
            '{"input_length": 200, "output_length": 75, "hash_ids": [789], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filename
        )
        traces = loader.parse_and_validate()
        assert isinstance(traces, list)
        assert len(traces) == 2
        assert isinstance(traces[0], MooncakeTrace)
        assert isinstance(traces[1], MooncakeTrace)
        assert traces[0].input_length == 100
        assert traces[0].output_length == 50
        assert traces[0].hash_ids == [123, 456]
        assert traces[0].timestamp == 1000
        assert traces[1].input_length == 200
        assert traces[1].output_length == 75
        assert traces[1].hash_ids == [789]
        assert traces[1].timestamp == 2000

    def test_load_dataset_with_text_input(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test loading JSONL file with text_input fields."""
        content = [
            '{"text_input": "This is the first test input", "timestamp": 1000}',
            '{"text_input": "This is the second test input", "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filename
        )
        traces = loader.parse_and_validate()
        assert len(traces) == 2
        assert traces[0].text_input == "This is the first test input"
        assert traces[0].input_length is None
        assert traces[1].text_input == "This is the second test input"
        assert traces[1].input_length is None

    def test_load_dataset_mixed_input_types(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test loading JSONL file with mixed input_length and text_input entries (but not both in same entry)."""
        content = [
            '{"input_length": 100, "hash_ids": [123], "timestamp": 1000}',
            '{"text_input": "Mixed input test", "timestamp": 2000}',
            '{"input_length": 200, "output_length": 50, "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filename
        )
        traces = loader.parse_and_validate()
        assert len(traces) == 3
        assert traces[0].input_length == 100
        assert traces[0].text_input is None
        assert traces[0].hash_ids == [123]
        assert traces[1].input_length is None
        assert traces[1].text_input == "Mixed input test"
        assert traces[2].input_length == 200
        assert traces[2].output_length == 50
        assert traces[2].text_input is None

    def test_load_dataset_skips_empty_lines(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test that empty lines are skipped."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            "",
            '{"input_length": 200, "output_length": 75, "hash_ids": [456], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filename
        )
        result = loader.parse_and_validate()
        assert len(result) == 2

    def test_load_dataset_with_timestamps(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test loading dataset with timestamp fields."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            '{"input_length": 200, "output_length": 75, "hash_ids": [456], "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filename
        )
        traces = loader.parse_and_validate()
        assert traces[0].timestamp == 1000
        assert traces[1].timestamp == 2000

    @pytest.mark.parametrize(
        "start_offset,end_offset,expected_count,description",
        [
            (None, None, 4, "no filtering"),
            (1500, None, 3, "start offset only - keeps timestamps >= 1500"),
            (None, 2500, 3, "end offset only - keeps timestamps <= 2500"),
            (1500, 2500, 2, "both offsets - keeps timestamps in range [1500, 2500]"),
        ],
    )
    def test_load_dataset_with_offset_filtering(
        self,
        create_jsonl_file,
        start_offset,
        end_offset,
        expected_count,
        description,
        mock_tokenizer_cls,
    ):
        """Test dataset loading with start and end offset filtering."""
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            '{"input_length": 150, "output_length": 60, "hash_ids": [456], "timestamp": 2000}',
            '{"input_length": 200, "output_length": 70, "hash_ids": [789], "timestamp": 2500}',
            '{"input_length": 250, "output_length": 80, "hash_ids": [111], "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)
        user_config = self.make_user_config(start_offset, end_offset)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=user_config, tokenizer=tokenizer, filename=filename
        )
        traces = loader.parse_and_validate()
        assert len(traces) == expected_count, f"Failed for {description}"

    @pytest.mark.parametrize(
        "start_offset,end_offset,expected_skipped", [(2500, None, 2), (None, 1500, 2)]
    )
    def test_load_dataset_logs_skipped_traces(
        self,
        create_jsonl_file,
        caplog,
        start_offset,
        end_offset,
        expected_skipped,
        mock_tokenizer_cls,
    ):
        """Test that skipped traces are properly logged."""
        caplog.set_level(logging.INFO)
        content = [
            '{"input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            '{"input_length": 150, "output_length": 60, "hash_ids": [456], "timestamp": 2000}',
            '{"input_length": 200, "output_length": 70, "hash_ids": [789], "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)
        user_config = self.make_user_config(start_offset, end_offset)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=user_config, tokenizer=tokenizer, filename=filename
        )
        loader.parse_and_validate()
        assert f"Skipped {expected_skipped:,} traces" in caplog.text

    def test_convert_to_conversations(self, default_user_config, mock_tokenizer_cls):
        """Test conversion of trace data to conversations."""
        trace_data = [
            MooncakeTrace(
                session_id="session-1",
                text_input="Test prompt 1",
                output_length=50,
                timestamp=1000,
            ),
            MooncakeTrace(
                session_id="session-2",
                text_input="Test prompt 2",
                output_length=100,
                timestamp=2000,
            ),
            MooncakeTrace(
                session_id="session-3",
                text_input="Test prompt 3",
                output_length=75,
                timestamp=3000,
            ),
        ]
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename="dummy.jsonl"
        )
        conversations = loader.convert_to_conversations(trace_data)
        assert len(conversations) == 3
        # Find conversations by session_id
        conv_map = {c.session_id: c for c in conversations}
        conv1 = conv_map["session-1"]
        assert len(conv1.turns) == 1
        assert conv1.turns[0].timestamp == 1000
        assert conv1.turns[0].texts[0].contents[0] == "Test prompt 1"
        conv2 = conv_map["session-2"]
        assert len(conv2.turns) == 1
        assert conv2.turns[0].timestamp == 2000
        assert conv2.turns[0].texts[0].contents[0] == "Test prompt 2"
        conv3 = conv_map["session-3"]
        assert len(conv3.turns) == 1
        assert conv3.turns[0].timestamp == 3000
        assert conv3.turns[0].texts[0].contents[0] == "Test prompt 3"

    def test_convert_to_conversations_empty_data(
        self, default_user_config, mock_tokenizer_cls
    ):
        """Test conversion with empty trace data."""
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename="dummy.jsonl"
        )
        conversations = loader.convert_to_conversations([])
        assert len(conversations) == 0

    def test_convert_to_conversations_with_text_input(
        self, default_user_config, mock_tokenizer_cls
    ):
        """Test conversion uses text_input when provided - covers 'if trace.text_input is not None' line."""
        trace_data = [
            MooncakeTrace(
                session_id="session1", text_input="Hello, how are you?", timestamp=1000
            ),
            MooncakeTrace(
                session_id="session1",
                text_input="What is the weather like?",
                timestamp=2000,
            ),
        ]
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename="dummy.jsonl"
        )
        conversations = loader.convert_to_conversations(trace_data)
        assert len(conversations) == 1
        conversation = conversations[0]
        assert len(conversation.turns) == 2
        assert conversation.turns[0].texts[0].contents[0] == "Hello, how are you?"
        assert conversation.turns[1].texts[0].contents[0] == "What is the weather like?"

    def test_load_dataset_with_session_ids(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test loading JSONL file with session_id fields."""
        content = [
            '{"session_id": "session-1", "input_length": 100, "output_length": 50, "hash_ids": [123], "timestamp": 1000}',
            '{"session_id": "session-1", "input_length": 150, "output_length": 60, "hash_ids": [456], "timestamp": 2000}',
            '{"session_id": "session-2", "text_input": "This is session 2 input", "timestamp": 3000}',
        ]
        filename = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filename
        )
        traces = loader.parse_and_validate()
        assert len(traces) == 3
        # Verify session-1 traces
        session1_traces = [t for t in traces if t.session_id == "session-1"]
        assert len(session1_traces) == 2
        assert session1_traces[0].input_length == 100
        assert session1_traces[1].input_length == 150
        # Verify session-2 traces
        session2_traces = [t for t in traces if t.session_id == "session-2"]
        assert len(session2_traces) == 1
        assert session2_traces[0].text_input == "This is session 2 input"

    def test_load_dataset_with_delay_field(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test loading JSONL file with delay fields."""
        content = [
            '{"session_id": "abc", "input_length": 100, "output_length": 50, "delay": 500}',
            '{"session_id": "def", "text_input": "This is test input", "delay": 1000}',
        ]
        filename = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = MooncakeTraceDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filename
        )
        traces = loader.parse_and_validate()
        assert len(traces) == 2
        assert traces[0].delay == 500
        assert traces[1].delay == 1000
