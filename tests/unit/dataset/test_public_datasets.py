# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from aiperf.common.enums import DatasetLoaderType, PublicDatasetType
from aiperf.dataset.public_datasets import PublicDataset


class TestPublicDatasetFilenameDetection:
    """Test automatic filename detection from URLs."""

    def test_filename_explicitly_provided(self):
        """Test that explicit remote_filename is used when provided."""
        dataset = PublicDataset(
            dataset_type=PublicDatasetType.SHAREGPT,
            name="Test Dataset",
            url="https://example.com/path/to/data.json",
            loader_type=DatasetLoaderType.SHAREGPT,
            remote_filename="custom_name.json",
        )

        assert dataset.get_cache_filename() == "custom_name.json"

    def test_filename_extracted_from_url(self):
        """Test that filename is extracted from URL when not provided."""
        dataset = PublicDataset(
            dataset_type=PublicDatasetType.SHAREGPT,
            name="Test Dataset",
            url="https://example.com/datasets/my_dataset.json",
            loader_type=DatasetLoaderType.SHAREGPT,
        )

        assert dataset.get_cache_filename() == "my_dataset.json"

    def test_filename_from_complex_url(self):
        """Test filename extraction from complex URL with query params."""
        dataset = PublicDataset(
            dataset_type=PublicDatasetType.SHAREGPT,
            name="Test Dataset",
            url="https://example.com/api/v1/datasets/data.json?version=1&format=json",
            loader_type=DatasetLoaderType.SHAREGPT,
        )

        assert dataset.get_cache_filename() == "data.json"

    def test_filename_fallback_to_dataset_name(self):
        """Test fallback to sanitized dataset name when URL has no filename."""
        dataset = PublicDataset(
            dataset_type=PublicDatasetType.SHAREGPT,
            name="My Test Dataset",
            url="https://example.com/api/",
            loader_type=DatasetLoaderType.SHAREGPT,
        )

        assert dataset.get_cache_filename() == "my_test_dataset.json"

    def test_filename_from_huggingface_url(self):
        """Test filename extraction from real HuggingFace URL."""
        dataset = PublicDataset(
            dataset_type=PublicDatasetType.SHAREGPT,
            name="ShareGPT",
            url="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
            loader_type=DatasetLoaderType.SHAREGPT,
        )

        assert (
            dataset.get_cache_filename() == "ShareGPT_V3_unfiltered_cleaned_split.json"
        )

    def test_empty_remote_filename_uses_detection(self):
        """Test that empty string for remote_filename triggers detection."""
        dataset = PublicDataset(
            dataset_type=PublicDatasetType.SHAREGPT,
            name="Test",
            url="https://example.com/dataset.json",
            loader_type=DatasetLoaderType.SHAREGPT,
            remote_filename="",
        )

        assert dataset.get_cache_filename() == "dataset.json"


class TestPublicDatasetDownload:
    """Test the download_public_dataset function."""

    @patch("aiperf.dataset.public_datasets.asyncio.run")
    @patch("aiperf.dataset.public_datasets.Path.exists")
    def test_uses_cached_file_when_exists(self, mock_exists, mock_async_run):
        """Test that cached file is used when it exists."""
        from aiperf.dataset.public_datasets import download_public_dataset

        mock_exists.return_value = True

        dataset = PublicDataset(
            dataset_type=PublicDatasetType.SHAREGPT,
            name="Test",
            url="https://example.com/test.json",
            loader_type=DatasetLoaderType.SHAREGPT,
        )

        result = download_public_dataset(dataset)

        # Should not download
        mock_async_run.assert_not_called()
        assert result.name == "test.json"

    @patch("aiperf.dataset.public_datasets.asyncio.run")
    @patch("aiperf.dataset.public_datasets.Path.exists")
    @patch("aiperf.dataset.public_datasets._save_to_cache")
    def test_downloads_when_not_cached(self, mock_save, mock_exists, mock_async_run):
        """Test that file is downloaded when not in cache."""
        from aiperf.dataset.public_datasets import download_public_dataset

        mock_exists.return_value = False
        # Close the coroutine to avoid "coroutine was never awaited" warning
        mock_async_run.side_effect = lambda coro: (coro.close(), '{"data": "test"}')[1]

        dataset = PublicDataset(
            dataset_type=PublicDatasetType.SHAREGPT,
            name="Test",
            url="https://example.com/test.json",
            loader_type=DatasetLoaderType.SHAREGPT,
        )

        result = download_public_dataset(dataset)

        # Should download
        mock_async_run.assert_called_once()
        mock_save.assert_called_once()
        assert result.name == "test.json"

    @patch("aiperf.dataset.public_datasets.asyncio.run")
    @patch("aiperf.dataset.public_datasets.Path.exists")
    @patch("aiperf.dataset.public_datasets._save_to_cache")
    def test_uses_detected_filename_for_cache_path(
        self, mock_save, mock_exists, mock_async_run
    ):
        """Test that detected filename is used for cache path."""
        from aiperf.dataset.public_datasets import download_public_dataset

        mock_exists.return_value = False
        # Close the coroutine to avoid "coroutine was never awaited" warning
        mock_async_run.side_effect = lambda coro: (coro.close(), '{"data": "test"}')[1]

        dataset = PublicDataset(
            dataset_type=PublicDatasetType.SHAREGPT,
            name="Test",
            url="https://example.com/datasets/my_data.json",
            loader_type=DatasetLoaderType.SHAREGPT,
            # No explicit remote_filename - should auto-detect
        )

        result = download_public_dataset(dataset)

        # Should use detected filename
        assert result.name == "my_data.json"
        # Cache save should be called with correct path
        call_args = mock_save.call_args[0]
        assert call_args[0].name == "my_data.json"


class TestSHAREGPTInstance:
    """Test the built-in SHAREGPT dataset instance."""

    def test_sharegpt_has_correct_metadata(self):
        """Test that SHAREGPT has correct metadata."""
        from aiperf.dataset.public_datasets import SHAREGPT

        assert SHAREGPT.name == "ShareGPT"
        assert "huggingface.co" in SHAREGPT.url
        assert SHAREGPT.loader_type == DatasetLoaderType.SHAREGPT
        assert SHAREGPT.dataset_type == PublicDatasetType.SHAREGPT

    def test_sharegpt_auto_detects_filename(self):
        """Test that SHAREGPT uses filename auto-detection."""
        from aiperf.dataset.public_datasets import SHAREGPT

        # Should have empty remote_filename (triggers auto-detection)
        assert SHAREGPT.remote_filename == ""

        # Should extract filename from URL
        filename = SHAREGPT.get_cache_filename()
        assert filename == "ShareGPT_V3_unfiltered_cleaned_split.json"
