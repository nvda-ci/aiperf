# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.dataset.loader.models import (
    MooncakeTrace,
    MultiTurn,
    RandomPool,
    SingleTurn,
)
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader
from aiperf.dataset.loader.random_pool import RandomPoolDatasetLoader
from aiperf.dataset.loader.single_turn import SingleTurnDatasetLoader


def _validate_model(model_cls, data: dict | None) -> bool:
    """Helper to validate data against a Pydantic model."""
    if data is None:
        return False
    try:
        model_cls.model_validate(data)
        return True
    except ValidationError:
        return False


def _create_temp_file(data: dict | None) -> Path | None:
    """Create a temp file with JSON data for can_load_file testing."""
    if data is None:
        return None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(orjson.dumps(data).decode() + "\n")
        return Path(f.name)


class TestSingleTurnModelValidation:
    """Tests for SingleTurn Pydantic model validation.

    Note: Loaders use pydantic model validation which includes type field validation.
    The 'type' field must match the loader's expected type or be omitted (defaults to correct type).
    """

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"text": "Hello world"}, True, id="text_field"),
            param({"texts": ["Hello", "World"]}, True, id="texts_field"),
            param({"image": "/path/to/image.png"}, True, id="image_field"),
            param({"images": ["/path/1.png", "/path/2.png"]}, True, id="images_field"),
            param({"audio": "/path/to/audio.wav"}, True, id="audio_field"),
            param({"audios": ["/path/1.wav", "/path/2.wav"]}, True, id="audios_field"),
            param({"text": "Describe this", "image": "/path.png", "audio": "/audio.wav"}, True, id="multimodal"),
            param({"type": "single_turn", "text": "Hello"}, True, id="with_type_field"),
            param({"type": "random_pool", "text": "Hello"}, False, id="wrong_type_rejected"),
            param({"turns": [{"text": "Hello"}]}, False, id="has_turns_field"),
            param({"session_id": "123", "metadata": "test"}, False, id="no_modality"),
            param(None, False, id="none_data"),
        ],
    )  # fmt: skip
    def test_model_validation(self, data, expected):
        """Test various data formats for SingleTurn pydantic validation."""
        assert _validate_model(SingleTurn, data) is expected


class TestSingleTurnCanLoadFile:
    """Tests for SingleTurnDatasetLoader.can_load_file() method."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"text": "Hello world"}, True, id="text_field"),
            param({"type": "single_turn", "text": "Hello"}, True, id="with_type_field"),
            param({"type": "random_pool", "text": "Hello"}, False, id="wrong_type_rejected"),
            param({"turns": [{"text": "Hello"}]}, False, id="has_turns_field"),
        ],
    )  # fmt: skip
    def test_can_load_file(self, data, expected):
        """Test can_load_file with various data formats."""
        path = _create_temp_file(data)
        try:
            assert SingleTurnDatasetLoader.can_load_file(path) is expected
        finally:
            if path:
                path.unlink(missing_ok=True)

    def test_can_load_file_nonexistent(self):
        """Test can_load_file returns False for nonexistent file."""
        assert (
            SingleTurnDatasetLoader.can_load_file(Path("/nonexistent.jsonl")) is False
        )


class TestMultiTurnModelValidation:
    """Tests for MultiTurn Pydantic model validation.

    Note: Loaders use pydantic model validation which includes type field validation.
    The 'type' field must match the loader's expected type or be omitted (defaults to correct type).
    """

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"turns": [{"text": "Turn 1"}, {"text": "Turn 2"}]}, True, id="turns_list"),
            param({"session_id": "session_123", "turns": [{"text": "Hello"}]}, True, id="with_session_id"),
            param({"type": "multi_turn", "turns": [{"text": "Hello"}]}, True, id="with_type_field"),
            param({"text": "Hello world"}, False, id="no_turns_field"),
            param({"turns": "not a list"}, False, id="turns_not_list_string"),
            param({"turns": {"text": "Hello"}}, False, id="turns_not_list_dict"),
            param(None, False, id="none_data"),
        ],
    )  # fmt: skip
    def test_model_validation(self, data, expected):
        """Test various data formats for MultiTurn pydantic validation."""
        assert _validate_model(MultiTurn, data) is expected


class TestMultiTurnCanLoadFile:
    """Tests for MultiTurnDatasetLoader.can_load_file() method."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"turns": [{"text": "Turn 1"}, {"text": "Turn 2"}]}, True, id="turns_list"),
            param({"type": "multi_turn", "turns": [{"text": "Hello"}]}, True, id="with_type_field"),
            param({"text": "Hello world"}, False, id="no_turns_field"),
        ],
    )  # fmt: skip
    def test_can_load_file(self, data, expected):
        """Test can_load_file with various data formats."""
        path = _create_temp_file(data)
        try:
            assert MultiTurnDatasetLoader.can_load_file(path) is expected
        finally:
            if path:
                path.unlink(missing_ok=True)


class TestRandomPoolModelValidation:
    """Tests for RandomPool Pydantic model validation.

    Note: Loaders use pydantic model validation. RandomPool requires either:
    1. Data with explicit type="random_pool" and valid modality fields, OR
    2. A directory/file path with at least one valid data entry
    """

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"text": "Hello"}, True, id="text_field"),
            param({"type": "random_pool", "text": "Query"}, True, id="explicit_type_validates"),
        ],
    )  # fmt: skip
    def test_model_validation(self, data, expected):
        """Test content-based validation for RandomPool."""
        assert _validate_model(RandomPool, data) is expected


class TestRandomPoolCanLoadFile:
    """Tests for RandomPoolDatasetLoader.can_load_file() method."""

    def test_can_load_file_requires_explicit_type(self):
        """Test that RandomPool requires explicit type field to match via can_load_file."""
        # Without explicit type, ambiguous with SingleTurn
        data_no_type = {"text": "Hello"}
        path = _create_temp_file(data_no_type)
        try:
            assert RandomPoolDatasetLoader.can_load_file(path) is False
        finally:
            path.unlink(missing_ok=True)

        # With explicit type, matches
        data_with_type = {"type": "random_pool", "text": "Query"}
        path = _create_temp_file(data_with_type)
        try:
            assert RandomPoolDatasetLoader.can_load_file(path) is True
        finally:
            path.unlink(missing_ok=True)

    def test_can_load_directory(self):
        """Test detection with directory path containing valid files (unique to RandomPool)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "data.jsonl"
            file_path.write_text('{"text": "Hello"}\n')
            assert RandomPoolDatasetLoader.can_load_directory(temp_path) is True

    def test_cannot_load_empty_directory(self):
        """Test that empty directory returns False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assert RandomPoolDatasetLoader.can_load_directory(temp_path) is False


class TestMooncakeTraceModelValidation:
    """Tests for MooncakeTrace Pydantic model validation.

    Note: Loaders use pydantic model validation. MooncakeTrace requires either:
    - input_length (with optional hash_ids), OR
    - text_input (hash_ids not allowed with text_input)
    """

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"input_length": 100, "output_length": 50}, True, id="input_length_with_output"),
            param({"input_length": 100}, True, id="input_length_only"),
            param({"input_length": 100, "hash_ids": [123, 456]}, True, id="input_length_with_hash_ids"),
            param({"type": "mooncake_trace", "input_length": 100}, True, id="with_type_field"),
            param({"text_input": "Hello world", "hash_ids": [123, 456]}, False, id="text_input_with_hash_ids_invalid"),
            param({"text_input": "Hello world"}, True, id="text_input_only"),
            param({"timestamp": 1000, "session_id": "abc"}, False, id="no_required_fields"),
            param({"output_length": 50}, False, id="only_output_length"),
            param(None, False, id="none_data"),
        ],
    )  # fmt: skip
    def test_model_validation(self, data, expected):
        """Test various data formats for MooncakeTrace pydantic validation."""
        assert _validate_model(MooncakeTrace, data) is expected


class TestMooncakeTraceCanLoadFile:
    """Tests for MooncakeTraceDatasetLoader.can_load_file() method."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"input_length": 100, "output_length": 50}, True, id="input_length_with_output"),
            param({"text_input": "Hello world"}, True, id="text_input_only"),
            param({"timestamp": 1000, "session_id": "abc"}, False, id="no_required_fields"),
        ],
    )  # fmt: skip
    def test_can_load_file(self, data, expected):
        """Test can_load_file with various data formats."""
        path = _create_temp_file(data)
        try:
            assert MooncakeTraceDatasetLoader.can_load_file(path) is expected
        finally:
            if path:
                path.unlink(missing_ok=True)


class TestAutoDetection:
    """Tests for loader auto-detection via can_load_file."""

    @pytest.mark.parametrize(
        "content,expected_loader",
        [
            param('{"text": "Hello world"}', SingleTurnDatasetLoader, id="single_turn_text"),
            param('{"image": "/path.png"}', SingleTurnDatasetLoader, id="single_turn_image"),
            param('{"turns": [{"text": "Turn 1"}, {"text": "Turn 2"}]}', MultiTurnDatasetLoader, id="multi_turn"),
            param('{"type": "random_pool", "text": "Query"}', RandomPoolDatasetLoader, id="random_pool_explicit"),
            param('{"input_length": 100, "output_length": 50}', MooncakeTraceDatasetLoader, id="mooncake_input_length"),
            param('{"text_input": "Hello"}', MooncakeTraceDatasetLoader, id="mooncake_text_input"),
        ],
    )  # fmt: skip
    def test_loader_detection(self, content, expected_loader):
        """Test that the correct loader is detected for various file formats."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(content + "\n")
            path = Path(f.name)

        try:
            loaders = [
                SingleTurnDatasetLoader,
                MultiTurnDatasetLoader,
                RandomPoolDatasetLoader,
                MooncakeTraceDatasetLoader,
            ]
            matching = [l for l in loaders if l.can_load_file(path)]
            assert len(matching) == 1, (
                f"Expected exactly one match, got {[l.__name__ for l in matching]}"
            )
            assert matching[0] == expected_loader
        finally:
            path.unlink(missing_ok=True)


class TestDetectionPriorityAndAmbiguity:
    """Tests for detection priority and handling of ambiguous cases."""

    def test_explicit_type_handled_by_validation(self):
        """Test that explicit type field is validated by loaders via pydantic."""
        data = {"type": "random_pool", "text": "Hello"}
        path = _create_temp_file(data)
        try:
            # SingleTurn rejects because type doesn't match
            assert SingleTurnDatasetLoader.can_load_file(path) is False
            # RandomPool validates with pydantic and returns True
            assert RandomPoolDatasetLoader.can_load_file(path) is True
        finally:
            path.unlink(missing_ok=True)

    def test_multi_turn_takes_priority_over_single_turn(self):
        """Test that MultiTurn is correctly detected over SingleTurn."""
        data = {"turns": [{"text": "Hello"}]}
        path = _create_temp_file(data)
        try:
            assert MultiTurnDatasetLoader.can_load_file(path) is True
            assert SingleTurnDatasetLoader.can_load_file(path) is False
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.parametrize(
        "loader,should_match",
        [
            param(MooncakeTraceDatasetLoader, True, id="mooncake"),
            param(SingleTurnDatasetLoader, False, id="single_turn"),
            param(MultiTurnDatasetLoader, False, id="multi_turn"),
            param(RandomPoolDatasetLoader, False, id="random_pool"),
        ],
    )  # fmt: skip
    def test_mooncake_trace_distinct_from_others(self, loader, should_match):
        """Test that MooncakeTrace is distinct from other types."""
        data = {"input_length": 100}
        path = _create_temp_file(data)
        try:
            assert loader.can_load_file(path) is should_match
        finally:
            path.unlink(missing_ok=True)

    def test_directory_path_uniquely_identifies_random_pool(self):
        """Test that directory path with valid files uniquely identifies RandomPool."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "data.jsonl"
            file_path.write_text('{"text": "Hello"}\n')

            # Only RandomPool can load directories
            assert RandomPoolDatasetLoader.can_load_directory(temp_path) is True
            assert SingleTurnDatasetLoader.can_load_directory(temp_path) is False
            assert MultiTurnDatasetLoader.can_load_directory(temp_path) is False
            assert MooncakeTraceDatasetLoader.can_load_directory(temp_path) is False
