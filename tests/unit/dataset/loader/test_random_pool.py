# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path

import pytest

from aiperf.common.enums import CustomDatasetType
from aiperf.common.models import Text
from aiperf.dataset.loader.models import RandomPool
from aiperf.dataset.loader.random_pool import RandomPoolDatasetLoader


class TestRandomPool:
    """Tests for RandomPool model validation and functionality."""

    def test_create_with_text_only(self):
        """Test creating RandomPool with simple text."""
        data = RandomPool(text="What is machine learning?")
        assert data.text == "What is machine learning?"
        assert data.texts is None
        assert data.type == CustomDatasetType.RANDOM_POOL

    def test_create_with_multimodal_data(self):
        """Test creating RandomPool with multiple modalities."""
        data = RandomPool(
            text="Describe this audio",
            image="/path/to/chart.png",
            audio="/path/to/recording.wav",
        )
        assert data.text == "Describe this audio"
        assert data.texts is None
        assert data.image == "/path/to/chart.png"
        assert data.images is None
        assert data.audio == "/path/to/recording.wav"
        assert data.audios is None

    def test_create_with_batched_inputs(self):
        """Test creating RandomPool with batched content."""
        data = RandomPool(
            texts=["What is AI?", "Explain neural networks"],
            images=["/path/image1.jpg", "/path/image2.jpg"],
            audios=["/path/audio1.wav", "/path/audio2.wav"],
        )
        assert data.text is None
        assert data.texts == ["What is AI?", "Explain neural networks"]
        assert data.image is None
        assert data.images == ["/path/image1.jpg", "/path/image2.jpg"]
        assert data.audio is None
        assert data.audios == ["/path/audio1.wav", "/path/audio2.wav"]

    def test_validation_at_least_one_modality_required(self):
        """Test that at least one modality must be provided."""
        with pytest.raises(ValueError):
            RandomPool()

    @pytest.mark.parametrize(
        "text,texts,image,images,audio,audios",
        [
            ("hello", ["world"], None, None, None, None),
            (None, None, "img.png", ["img2.png"], None, None),
            (None, None, None, None, "audio.wav", ["audio2.wav"]),
        ],
    )
    def test_validation_mutually_exclusive_fields(
        self, text, texts, image, images, audio, audios
    ):
        """Test that mutually exclusive fields cannot be set together."""
        with pytest.raises(ValueError):
            RandomPool(
                text=text,
                texts=texts,
                image=image,
                images=images,
                audio=audio,
                audios=audios,
            )


class TestRandomPoolDatasetLoader:
    """Tests for RandomPoolDatasetLoader functionality."""

    def test_load_simple_single_file(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test loading from a single file with simple content."""
        content = [
            '{"text": "What is deep learning?"}',
            '{"text": "Explain neural networks", "image": "/chart.png"}',
        ]
        filepath = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = RandomPoolDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filepath
        )
        data = loader.parse_and_validate()
        # parse_and_validate returns a list
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0].text == "What is deep learning?"
        assert data[1].text == "Explain neural networks"
        assert data[1].image == "/chart.png"

    def test_load_multimodal_single_file(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test loading multimodal content from single file."""
        content = [
            '{"text": "Analyze this image", "image": "/data.png"}',
            '{"text": "Transcribe audio", "audio": "/recording.wav"}',
            '{"texts": ["Query 1", "Query 2"], "images": ["/img1.jpg", "/img2.jpg"]}',
        ]
        filepath = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = RandomPoolDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filepath
        )
        data = loader.parse_and_validate()
        assert len(data) == 3
        assert data[0].text == "Analyze this image"
        assert data[0].image == "/data.png"
        assert data[1].audio == "/recording.wav"
        assert data[2].texts == ["Query 1", "Query 2"]
        assert data[2].images == ["/img1.jpg", "/img2.jpg"]

    def test_load_dataset_skips_empty_lines(
        self, create_jsonl_file, default_user_config, mock_tokenizer_cls
    ):
        """Test that empty lines are skipped during loading."""
        content = [
            '{"text": "First entry"}',
            "",
            '{"text": "Second entry"}',
            "   ",
            '{"text": "Third entry"}',
        ]
        filepath = create_jsonl_file(content)
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = RandomPoolDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename=filepath
        )
        data = loader.parse_and_validate()
        assert len(data) == 3

    def test_load_directory_with_multiple_files(
        self, default_user_config, mock_tokenizer_cls
    ):
        """Test loading from directory with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            queries_file = temp_path / "queries.jsonl"
            with open(queries_file, "w") as f:
                f.write(
                    '{"texts": [{"name": "query", "contents": ["Who are you?"]}]}\n'
                )
                f.write('{"texts": [{"name": "query", "contents": ["What is AI?"]}]}\n')
            passages_file = temp_path / "passages.jsonl"
            with open(passages_file, "w") as f:
                f.write(
                    '{"texts": [{"name": "passage", "contents": ["I am an AI assistant."]}]}\n'
                )
                f.write(
                    '{"texts": [{"name": "passage", "contents": ["AI is artificial intelligence."]}]}\n'
                )
            images_file = temp_path / "images.jsonl"
            with open(images_file, "w") as f:
                f.write(
                    '{"images": [{"name": "image", "contents": ["/path/to/image1.png"]}]}\n'
                )
                f.write(
                    '{"images": [{"name": "image", "contents": ["/path/to/image2.png"]}]}\n'
                )
            tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
            loader = RandomPoolDatasetLoader(
                filename=str(temp_path), config=default_user_config, tokenizer=tokenizer
            )
            data = loader.parse_and_validate()
            # For directories, parse_and_validate returns a flat list but stores mapping in _pool_mapping
            dataset = loader._pool_mapping
            assert len(dataset) == 3
            assert "queries.jsonl" in dataset
            assert "passages.jsonl" in dataset
            assert "images.jsonl" in dataset
            queries_pool = dataset["queries.jsonl"]
            assert len(queries_pool) == 2
            assert all(item.texts[0].name == "query" for item in queries_pool)
            assert queries_pool[0].texts[0].contents == ["Who are you?"]
            assert queries_pool[1].texts[0].contents == ["What is AI?"]
            passages_pool = dataset["passages.jsonl"]
            assert len(passages_pool) == 2
            assert all(item.texts[0].name == "passage" for item in passages_pool)
            assert passages_pool[0].texts[0].contents == ["I am an AI assistant."]
            assert passages_pool[1].texts[0].contents == [
                "AI is artificial intelligence."
            ]
            images_pool = dataset["images.jsonl"]
            assert len(images_pool) == 2
            assert all(item.images[0].name == "image" for item in images_pool)
            assert images_pool[0].images[0].contents == ["/path/to/image1.png"]
            assert images_pool[1].images[0].contents == ["/path/to/image2.png"]

    def test_convert_simple_pool_data(self, default_user_config, mock_tokenizer_cls):
        """Test converting simple random pool data to conversations."""
        data = {"file1.jsonl": [RandomPool(text="Hello world")]}
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = RandomPoolDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename="dummy.jsonl"
        )
        loader._pool_mapping = data
        conversations = loader.convert_to_conversations([])
        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1
        assert conversations[0].turns[0].texts[0].contents == ["Hello world"]

    def test_convert_multimodal_pool_data(
        self, default_user_config, mock_tokenizer_cls
    ):
        """Test converting multimodal random pool data."""
        data = {
            "multimodal.jsonl": [
                RandomPool(
                    text="What's in this image?",
                    image="/path/to/image.png",
                    audio="/path/to/audio.wav",
                )
            ]
        }
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = RandomPoolDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename="dummy.jsonl"
        )
        loader._pool_mapping = data
        conversations = loader.convert_to_conversations([])
        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 1
        assert turn.texts[0].contents == ["What's in this image?"]
        assert len(turn.images) == 1
        assert turn.images[0].contents == ["/path/to/image.png"]
        assert len(turn.audios) == 1
        assert turn.audios[0].contents == ["/path/to/audio.wav"]

    def test_convert_batched_pool_data(self, default_user_config, mock_tokenizer_cls):
        """Test converting pool data with batched content."""
        data = {
            "batched.jsonl": [
                RandomPool(
                    texts=["First question", "Second question"],
                    images=["/image1.png", "/image2.png"],
                )
            ]
        }
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = RandomPoolDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename="dummy.jsonl"
        )
        loader._pool_mapping = data
        conversations = loader.convert_to_conversations([])
        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 1
        assert turn.texts[0].contents == ["First question", "Second question"]
        assert len(turn.images) == 1
        assert turn.images[0].contents == ["/image1.png", "/image2.png"]

    def test_convert_multiple_files_no_name_specified(
        self, default_user_config, mock_tokenizer_cls
    ):
        """Test converting data from multiple files without name specified."""
        data = {
            "queries.jsonl": [RandomPool(text="What is AI?")],
            "contexts.jsonl": [RandomPool(text="AI is artificial intelligence")],
        }
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = RandomPoolDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename="dummy_dir"
        )
        loader._pool_mapping = data
        conversations = loader.convert_to_conversations([])
        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 2
        assert turn.texts[0].name == "queries"
        assert turn.texts[0].contents == ["What is AI?"]
        assert turn.texts[1].name == "contexts"
        assert turn.texts[1].contents == ["AI is artificial intelligence"]

    def test_convert_multiple_files_with_name_specified(
        self, default_user_config, mock_tokenizer_cls
    ):
        """Test converting data from multiple files with name specified."""
        data = {
            "queries.jsonl": [
                RandomPool(texts=[Text(name="abc123", contents=["What is AI?"])])
            ],
            "contexts.jsonl": [
                RandomPool(
                    texts=[
                        Text(name="def456", contents=["AI is artificial intelligence"])
                    ]
                )
            ],
        }
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        loader = RandomPoolDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename="dummy_dir"
        )
        loader._pool_mapping = data
        conversations = loader.convert_to_conversations([])
        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 2
        assert turn.texts[0].name == "abc123"
        assert turn.texts[0].contents == ["What is AI?"]
        assert turn.texts[1].name == "def456"
        assert turn.texts[1].contents == ["AI is artificial intelligence"]

    def test_convert_multiple_files_with_multiple_samples(
        self, default_user_config, mock_tokenizer_cls
    ):
        """Test converting data from multiple files with multiple samples."""
        data = {
            "queries.jsonl": [
                RandomPool(text="text1", image="image1.png"),
                RandomPool(text="text2", image="image2.png"),
            ],
            "contexts.jsonl": [
                RandomPool(text="text3", image="image3.png"),
                RandomPool(text="text4", image="image4.png"),
            ],
        }
        tokenizer = mock_tokenizer_cls.from_pretrained("test-model")
        default_user_config.input.conversation.num = 2
        loader = RandomPoolDatasetLoader(
            config=default_user_config, tokenizer=tokenizer, filename="dummy_dir"
        )
        loader._pool_mapping = data
        conversations = loader.convert_to_conversations([])
        assert len(conversations) == 2
        conv1, conv2 = conversations
        assert len(conv1.turns) == 1
        assert len(conv2.turns) == 1
        turn1, turn2 = (conv1.turns[0], conv2.turns[0])
        assert len(turn1.texts) == 2
        assert len(turn1.images) == 2
        assert len(turn2.texts) == 2
        assert len(turn2.images) == 2
        possible_text_contents = {
            ("text1", "text3"),
            ("text1", "text4"),
            ("text2", "text3"),
            ("text2", "text4"),
        }
        possible_image_contents = {
            ("image1.png", "image3.png"),
            ("image1.png", "image4.png"),
            ("image2.png", "image3.png"),
            ("image2.png", "image4.png"),
        }
        text_contents = tuple(t.contents[0] for t in turn1.texts)
        image_contents = tuple(i.contents[0] for i in turn1.images)
        assert text_contents in possible_text_contents
        assert image_contents in possible_image_contents
        text_contents = tuple(t.contents[0] for t in turn2.texts)
        image_contents = tuple(i.contents[0] for i in turn2.images)
        assert text_contents in possible_text_contents
        assert image_contents in possible_image_contents
