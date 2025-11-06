# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path

import pytest

from aiperf.common.enums import CustomDatasetType
from aiperf.common.models import Image, Text
from aiperf.dataset import SingleTurn
from aiperf.dataset.loader.single_turn import SingleTurnDatasetLoader


class TestSingleTurn:
    """Basic functionality tests for SingleTurn model."""

    def test_create_with_text_only(self):
        """Test creating SingleTurn with text."""
        data = SingleTurn(text="What is deep learning?")

        assert data.text == "What is deep learning?"
        assert data.texts is None
        assert data.type == CustomDatasetType.SINGLE_TURN

    def test_create_with_multimodal_data(self):
        """Test creating SingleTurn with text and image."""
        data = SingleTurn(
            text="What is in the image?",
            image="/path/to/image.png",
            audio="/path/to/audio.wav",
        )

        assert data.text == "What is in the image?"
        assert data.texts is None
        assert data.image == "/path/to/image.png"
        assert data.images is None
        assert data.audio == "/path/to/audio.wav"
        assert data.audios is None

    def test_create_with_batched_inputs(self):
        """Test creating SingleTurn with batched inputs."""
        data = SingleTurn(
            texts=["What is the weather today?", "What is deep learning?"],
            images=["/path/to/image1.png", "/path/to/image2.png"],
        )

        assert data.texts == ["What is the weather today?", "What is deep learning?"]
        assert data.images == ["/path/to/image1.png", "/path/to/image2.png"]
        assert data.audios is None

    def test_create_with_fixed_schedule(self):
        """Test creating SingleTurn with fixed schedule (timestamp)."""
        data = SingleTurn(text="What is deep learning?", timestamp=1000)

        assert data.text == "What is deep learning?"
        assert data.timestamp == 1000
        assert data.delay is None

    def test_create_with_delay(self):
        """Test creating SingleTurn with delay."""
        data = SingleTurn(text="Who are you?", delay=1234)

        assert data.text == "Who are you?"
        assert data.delay == 1234
        assert data.timestamp is None

    def test_create_with_full_featured_version(self):
        """Test creating SingleTurn with full-featured version."""
        data = SingleTurn(
            texts=[
                Text(name="text_field_A", contents=["Hello", "World"]),
                Text(name="text_field_B", contents=["Hi there"]),
            ],
            images=[
                Image(name="image_field_A", contents=["/path/1.png", "/path/2.png"]),
                Image(name="image_field_B", contents=["/path/3.png"]),
            ],
        )

        assert len(data.texts) == 2
        assert len(data.images) == 2
        assert data.audios is None

        assert data.texts[0].name == "text_field_A"
        assert data.texts[0].contents == ["Hello", "World"]
        assert data.texts[1].name == "text_field_B"
        assert data.texts[1].contents == ["Hi there"]

        assert data.images[0].name == "image_field_A"
        assert data.images[0].contents == ["/path/1.png", "/path/2.png"]
        assert data.images[1].name == "image_field_B"
        assert data.images[1].contents == ["/path/3.png"]

    def test_validation_errors(self):
        """Test that at least one modality must be provided."""
        # No modality provided
        with pytest.raises(ValueError):
            SingleTurn()

    @pytest.mark.parametrize(
        "text, texts, image, images, audio, audios, timestamp, delay",
        [
            ("foo", ["bar"], None, None, None, None, None, None),  # text and texts
            (None, None, "foo", ["bar"], None, None, None, None),  # image and images
            (None, None, None, None, "foo", ["bar"], None, None),  # audio and audios
            (None, None, None, None, None, None, 1000, 500),  # timestamp and delay
        ],
    )
    def test_validation_mutual_exclusion(
        self, text, texts, image, images, audio, audios, timestamp, delay
    ):
        """Test that mutual exclusion among fields is preserved."""
        with pytest.raises(ValueError):
            SingleTurn(
                text=text,
                texts=texts,
                image=image,
                images=images,
                audio=audio,
                audios=audios,
                timestamp=timestamp,
                delay=delay,
            )


class TestSingleTurnDatasetLoader:
    """Basic functionality tests for SingleTurnDatasetLoader."""

    def test_load_dataset_basic_functionality(
        self, create_jsonl_file, default_user_config
    ):
        """Test basic JSONL file loading."""
        content = [
            '{"text": "What is deep learning?"}',
            '{"text": "What is in the image?", "image": "/path/to/image.png"}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert isinstance(dataset, dict)
        assert len(dataset) == 2

        # Check that each session has single turn
        for _, turns in dataset.items():
            assert len(turns) == 1

        turn1, turn2 = list(dataset.values())
        assert turn1[0].text == "What is deep learning?"
        assert turn1[0].image is None
        assert turn1[0].audio is None

        assert turn2[0].text == "What is in the image?"
        assert turn2[0].image == "/path/to/image.png"
        assert turn2[0].audio is None

    def test_load_dataset_skips_empty_lines(
        self, create_jsonl_file, default_user_config
    ):
        """Test that empty lines are skipped."""
        content = [
            '{"text": "Hello"}',
            "",  # Empty line
            '{"text": "World"}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2  # Should skip empty line

    def test_load_dataset_with_batched_inputs(
        self, create_jsonl_file, default_user_config
    ):
        """Test loading dataset with batched inputs."""
        content = [
            '{"texts": ["What is the weather?", "What is AI?"], "images": ["/path/1.png", "/path/2.png"]}',
            '{"texts": ["Summarize the podcast", "What is audio about?"], "audios": ["/path/3.wav", "/path/4.wav"]}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        # Check that there are two sessions
        assert len(dataset) == 2

        turn1, turn2 = list(dataset.values())
        assert turn1[0].texts == ["What is the weather?", "What is AI?"]
        assert turn1[0].images == ["/path/1.png", "/path/2.png"]
        assert turn1[0].audios is None

        assert turn2[0].texts == ["Summarize the podcast", "What is audio about?"]
        assert turn2[0].images is None
        assert turn2[0].audios == ["/path/3.wav", "/path/4.wav"]

    def test_load_dataset_with_timestamp(self, create_jsonl_file, default_user_config):
        """Test loading dataset with timestamp field."""
        content = [
            '{"text": "What is deep learning?", "timestamp": 1000}',
            '{"text": "Who are you?", "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2

        turn1, turn2 = list(dataset.values())
        assert turn1[0].text == "What is deep learning?"
        assert turn1[0].timestamp == 1000
        assert turn1[0].delay is None

        assert turn2[0].text == "Who are you?"
        assert turn2[0].timestamp == 2000
        assert turn2[0].delay is None

    def test_load_dataset_with_delay(self, create_jsonl_file, default_user_config):
        """Test loading dataset with delay field."""
        content = [
            '{"text": "What is deep learning?", "delay": 0}',
            '{"text": "Who are you?", "delay": 1234}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2

        turn1, turn2 = list(dataset.values())
        assert turn1[0].text == "What is deep learning?"
        assert turn1[0].delay == 0
        assert turn1[0].timestamp is None

        assert turn2[0].text == "Who are you?"
        assert turn2[0].delay == 1234
        assert turn2[0].timestamp is None

    def test_load_dataset_with_full_featured_version(
        self, create_jsonl_file, default_user_config
    ):
        """Test loading dataset with full-featured version."""

        content = [
            json.dumps(
                {
                    "texts": [
                        {"name": "text_field_A", "contents": ["Hello", "World"]},
                        {"name": "text_field_B", "contents": ["Hi there"]},
                    ],
                    "images": [
                        {
                            "name": "image_field_A",
                            "contents": ["/path/1.png", "/path/2.png"],
                        },
                        {"name": "image_field_B", "contents": ["/path/3.png"]},
                    ],
                }
            )
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 1

        turn = list(dataset.values())[0]
        assert len(turn[0].texts) == 2
        assert len(turn[0].images) == 2

        assert turn[0].texts[0].name == "text_field_A"
        assert turn[0].texts[0].contents == ["Hello", "World"]
        assert turn[0].texts[1].name == "text_field_B"
        assert turn[0].texts[1].contents == ["Hi there"]

        assert turn[0].images[0].name == "image_field_A"
        assert turn[0].images[0].contents == ["/path/1.png", "/path/2.png"]
        assert turn[0].images[1].name == "image_field_B"
        assert turn[0].images[1].contents == ["/path/3.png"]


class TestSingleTurnDatasetLoaderConvertToConversations:
    """Test convert_to_conversations method for SingleTurnDatasetLoader."""

    def test_convert_simple_text_data(self, default_user_config):
        """Test converting simple text data to conversations."""
        loader = SingleTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        data = {
            "session_1": [SingleTurn(text="Hello world")],
            "session_2": [SingleTurn(text="How are you?")],
        }

        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        assert conversations[0].session_id == "session_1"
        assert len(conversations[0].turns) == 1
        assert conversations[0].turns[0].texts[0].contents == ["Hello world"]

        assert conversations[1].session_id == "session_2"
        assert len(conversations[1].turns) == 1
        assert conversations[1].turns[0].texts[0].contents == ["How are you?"]

    def test_convert_multimodal_data(self, default_user_config):
        """Test converting multimodal data to conversations."""
        loader = SingleTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        data = {
            "session_1": [
                SingleTurn(
                    text="What's in this image?",
                    image="https://example.com/image.png",
                    audio="https://example.com/audio.wav",
                )
            ]
        }

        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 1
        assert turn.texts[0].contents == ["What's in this image?"]
        assert len(turn.images) == 1
        assert turn.images[0].contents == ["https://example.com/image.png"]
        assert len(turn.audios) == 1
        assert turn.audios[0].contents == ["https://example.com/audio.wav"]

    def test_convert_batched_data(self, default_user_config):
        """Test converting batched data to conversations."""
        loader = SingleTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        data = {
            "session_1": [
                SingleTurn(
                    texts=["First message", "Second message"],
                    images=["https://example.com/1.png", "https://example.com/2.png"],
                )
            ]
        }

        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 1
        assert turn.texts[0].contents == ["First message", "Second message"]
        assert len(turn.images) == 1
        assert turn.images[0].contents == [
            "https://example.com/1.png",
            "https://example.com/2.png",
        ]

    def test_convert_with_timing_data(self, default_user_config):
        """Test converting data with timestamp and delay."""
        loader = SingleTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        data = {
            "session_1": [
                SingleTurn(text="First", timestamp=1000),
                SingleTurn(text="Second", delay=500, role="user"),
            ]
        }

        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert len(conversations[0].turns) == 2

        first_turn = conversations[0].turns[0]
        assert first_turn.timestamp == 1000
        assert first_turn.delay is None
        assert first_turn.role is None

        second_turn = conversations[0].turns[1]
        assert second_turn.timestamp is None
        assert second_turn.delay == 500
        assert second_turn.role == "user"

    def test_convert_structured_text_objects(self, default_user_config):
        """Test converting data with structured Text objects."""
        loader = SingleTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        text_objects = [
            Text(name="query", contents=["What is AI?"]),
            Text(name="context", contents=["AI stands for artificial intelligence"]),
        ]
        data = {"session_1": [SingleTurn(texts=text_objects)]}

        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 2
        assert turn.texts[0].name == "query"
        assert turn.texts[0].contents == ["What is AI?"]
        assert turn.texts[1].name == "context"
        assert turn.texts[1].contents == ["AI stands for artificial intelligence"]


class TestSingleTurnDatasetLoaderImageEncoding:
    """Test base64 encoding for local image files."""

    def test_convert_local_image_to_base64(self, create_jsonl_file):
        """Test that local image files are encoded to base64 data URLs."""
        # Use an actual test image from the source_images directory
        test_image = Path(
            "src/aiperf/dataset/generator/assets/source_images/0bfd8fdf-457f-43c8-9253-a2346d37d26a_1024.jpg"
        )

        # Skip if the image doesn't exist
        if not test_image.exists():
            pytest.skip("Test image not found")

        content = [
            json.dumps({"text": "What is in this image?", "image": str(test_image)})
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # Check that the image was encoded
        assert len(turn.images) == 1
        image_content = turn.images[0].contents[0]

        # Verify it's a data URL with base64 encoding
        assert image_content.startswith("data:image/")
        assert ";base64," in image_content

        # Extract and verify the base64 content is valid
        base64_part = image_content.split(";base64,")[1]
        try:
            base64.b64decode(base64_part)
        except Exception as e:
            pytest.fail(f"Invalid base64 encoding: {e}")

    def test_url_images_not_encoded(self, create_jsonl_file):
        """Test that URLs are not encoded and passed through as-is."""
        content = [
            json.dumps(
                {"text": "What is this?", "image": "https://example.com/image.png"}
            )
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # URL should remain unchanged
        assert turn.images[0].contents[0] == "https://example.com/image.png"

    def test_data_url_not_reencoded(self, create_jsonl_file):
        """Test that existing data URLs are not re-encoded."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        content = [json.dumps({"text": "Already encoded", "image": data_url})]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # Data URL should remain unchanged
        assert turn.images[0].contents[0] == data_url

    def test_multiple_images_encoded(self, create_jsonl_file):
        """Test that multiple local images are all encoded."""
        test_images = [
            Path(
                "src/aiperf/dataset/generator/assets/source_images/0bfd8fdf-457f-43c8-9253-a2346d37d26a_1024.jpg"
            ),
            Path(
                "src/aiperf/dataset/generator/assets/source_images/119544eb-9bbf-47d1-8d93-a51de6370295_861.jpg"
            ),
        ]

        # Skip if images don't exist
        for img in test_images:
            if not img.exists():
                pytest.skip("Test images not found")

        content = [
            json.dumps(
                {
                    "text": "What are in these images?",
                    "images": [str(img) for img in test_images],
                }
            )
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # Check that both images were encoded
        assert len(turn.images) == 1
        assert len(turn.images[0].contents) == 2

        for image_content in turn.images[0].contents:
            assert image_content.startswith("data:image/")
            assert ";base64," in image_content

    def test_mixed_image_sources(self, create_jsonl_file):
        """Test handling mixed image sources (local files, URLs, data URLs)."""
        test_image = Path(
            "src/aiperf/dataset/generator/assets/source_images/0bfd8fdf-457f-43c8-9253-a2346d37d26a_1024.jpg"
        )

        if not test_image.exists():
            pytest.skip("Test image not found")

        url = "https://example.com/image.png"
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        content = [
            json.dumps(
                {"text": "Mixed sources", "images": [str(test_image), url, data_url]}
            )
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # Check that we have 3 images
        assert len(turn.images) == 1
        assert len(turn.images[0].contents) == 3

        # First one (local file) should be encoded
        assert turn.images[0].contents[0].startswith("data:image/")
        assert ";base64," in turn.images[0].contents[0]

        # Second one (URL) should be unchanged
        assert turn.images[0].contents[1] == url

        # Third one (data URL) should be unchanged
        assert turn.images[0].contents[2] == data_url

    def test_invalid_image_path_raises_error(self, create_jsonl_file):
        """Test that invalid local file paths raise appropriate errors."""
        content = [
            json.dumps(
                {"text": "Invalid image", "image": "/nonexistent/path/image.png"}
            )
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()

        # The error should be raised during conversion
        with pytest.raises(FileNotFoundError):
            loader.convert_to_conversations(data)


class TestSingleTurnDatasetLoaderAudioEncoding:
    """Test base64 encoding for local audio files."""

    def test_convert_local_audio_to_base64(self, create_jsonl_file, create_test_audio):
        """Test that local audio files are encoded to format,base64 strings."""
        test_audio = create_test_audio("test_audio.wav")

        content = [json.dumps({"text": "Transcribe this audio", "audio": test_audio})]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # Check that the audio was encoded
        assert len(turn.audios) == 1
        audio_content = turn.audios[0].contents[0]

        # Verify it's in "format,base64" format
        assert "," in audio_content
        format_part, base64_part = audio_content.split(",", 1)
        assert format_part == "wav"

        # Verify the base64 content is valid
        try:
            base64.b64decode(base64_part)
        except Exception as e:
            pytest.fail(f"Invalid base64 encoding: {e}")

    def test_audio_url_not_encoded(self, create_jsonl_file):
        """Test that audio URLs are not encoded and passed through as-is."""
        content = [
            json.dumps(
                {"text": "Audio from URL", "audio": "https://example.com/audio.wav"}
            )
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # URL should remain unchanged
        assert turn.audios[0].contents[0] == "https://example.com/audio.wav"

    def test_audio_already_encoded_not_reencoded(self, create_jsonl_file):
        """Test that existing format,base64 audio strings are not re-encoded."""
        encoded_audio = "wav,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        content = [json.dumps({"text": "Already encoded", "audio": encoded_audio})]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # Should remain unchanged
        assert turn.audios[0].contents[0] == encoded_audio


class TestSingleTurnDatasetLoaderVideoEncoding:
    """Test base64 encoding for local video files."""

    def test_convert_local_video_to_base64(self, create_jsonl_file, create_test_video):
        """Test that local video files are encoded to data URL format."""
        test_video = create_test_video("test_video.mp4")

        content = [json.dumps({"text": "Describe this video", "video": test_video})]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # Check that the video was encoded
        assert len(turn.videos) == 1
        video_content = turn.videos[0].contents[0]

        # Verify it's a data URL with base64 encoding
        assert video_content.startswith("data:video/")
        assert ";base64," in video_content

        # Extract and verify the base64 content is valid
        base64_part = video_content.split(";base64,")[1]
        try:
            base64.b64decode(base64_part)
        except Exception as e:
            pytest.fail(f"Invalid base64 encoding: {e}")

    def test_video_url_not_encoded(self, create_jsonl_file):
        """Test that video URLs are not encoded and passed through as-is."""
        content = [
            json.dumps(
                {"text": "Video from URL", "video": "https://example.com/video.mp4"}
            )
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # URL should remain unchanged
        assert turn.videos[0].contents[0] == "https://example.com/video.mp4"

    def test_video_data_url_not_reencoded(self, create_jsonl_file):
        """Test that existing data URL videos are not re-encoded."""
        data_url = (
            "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMQAAAAAABW1kYXQ="
        )
        content = [json.dumps({"text": "Already encoded", "video": data_url})]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        data = loader.load_dataset()
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]

        # Data URL should remain unchanged
        assert turn.videos[0].contents[0] == data_url
