# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.enums import DatasetLoaderType, PublicDatasetType
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.factories import PublicDatasetFactory
from aiperf.common.models import RequestRecord
from aiperf.transports.aiohttp_client import AioHttpClient

if TYPE_CHECKING:
    from aiperf.common.protocols import PublicDatasetProtocol


@dataclass(frozen=True)
class PublicDataset:
    """Public dataset metadata (pure data, auto-registering).

    Simple dataclass containing everything needed to download and parse
    a public dataset. Automatically registers itself with PublicDatasetFactory
    on instantiation.

    Attributes:
        dataset_type: PublicDatasetType enum (used for auto-registration)
        name: Human-readable name for the dataset
        url: Remote URL to download from
        loader_type: Which DatasetLoaderType to use for parsing
        remote_filename: Optional filename for local caching (auto-detected from URL if not provided)
    """

    dataset_type: PublicDatasetType
    """Dataset type enum (for factory registration)"""

    name: str
    """Human-readable name for the dataset"""

    url: str
    """URL to download the dataset from"""

    loader_type: DatasetLoaderType
    """Which loader to use for parsing this dataset"""

    remote_filename: str = ""
    """Optional filename for local caching (extracted from URL if empty)"""

    def __post_init__(self) -> None:
        """Auto-register this instance with PublicDatasetFactory."""
        PublicDatasetFactory.register_instance(self.dataset_type, self)

    def get_cache_filename(self) -> str:
        """Get the filename to use for caching.

        Returns the explicit remote_filename if provided, otherwise extracts
        the filename from the URL.

        Returns:
            Filename to use for local caching.
        """
        if self.remote_filename:
            return self.remote_filename

        # Extract filename from URL
        from urllib.parse import urlparse

        parsed = urlparse(self.url)
        filename = Path(parsed.path).name

        # Check if filename has an extension (likely a real file)
        if not filename or "." not in filename:
            # Fallback: use dataset name with .json extension
            filename = f"{self.name.lower().replace(' ', '_')}.json"

        return filename


# Hard-coded dataset instances
# Each instance auto-registers on creation via __post_init__!

SHAREGPT = PublicDataset(
    dataset_type=PublicDatasetType.SHAREGPT,  # Auto-registers with factory!
    name="ShareGPT",
    url=(
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    ),
    loader_type=DatasetLoaderType.SHAREGPT,
    # remote_filename auto-detected from URL: "ShareGPT_V3_unfiltered_cleaned_split.json"
)


def download_public_dataset(dataset: "PublicDatasetProtocol") -> Path:
    """Download a public dataset or use cached version.

    This utility function handles downloading public datasets based on their
    metadata. It checks for cached versions first, then downloads if needed.

    The filename for caching is determined by:
    1. Explicit remote_filename if provided
    2. Otherwise extracted from the URL path
    3. Fallback to sanitized dataset name with .json extension

    Args:
        dataset: PublicDataset metadata instance (must implement PublicDatasetProtocol)

    Returns:
        Path to the local cached file.

    Raises:
        DatasetLoaderError: If download or caching fails.

    Example:
        dataset_class = PublicDatasetFactory.get_class_from_type(PublicDatasetType.SHAREGPT)
        file_path = download_public_dataset(dataset_class)
    """
    from aiperf.common.environment import Environment

    cache_filename = dataset.get_cache_filename()
    cache_path = Path(Environment.DATASET.CACHE_DIR) / cache_filename

    if cache_path.exists():
        print(f"Using cached dataset: {cache_path}")
        return cache_path

    print(f"No local cache found, downloading {dataset.name} from {dataset.url}")

    # Download asynchronously
    async def download() -> str:
        http_client = AioHttpClient(timeout=Environment.DATASET.PUBLIC_DATASET_TIMEOUT)
        record: RequestRecord = await http_client.get_request(dataset.url)
        await http_client.close()
        return record.responses[0].get_text()

    dataset_text = asyncio.run(download())

    # Save to cache
    _save_to_cache(cache_path, dataset_text, dataset.name)

    print(f"Downloaded {dataset.name} to {cache_path}")
    return cache_path


def _save_to_cache(cache_path: Path, content: str, dataset_name: str) -> None:
    """Save dataset content to local cache.

    Args:
        cache_path: Path to save the content to
        content: The dataset content to save
        dataset_name: Name of the dataset (for error messages)

    Raises:
        DatasetLoaderError: If saving fails.
    """
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(content)
    except Exception as e:
        raise DatasetLoaderError(
            f"Error saving {dataset_name} dataset to cache: {e}"
        ) from e
