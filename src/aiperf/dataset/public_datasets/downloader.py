# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import RequestRecord
from aiperf.transports.aiohttp_client import AioHttpClient

if TYPE_CHECKING:
    from aiperf.common.protocols import PublicDatasetProtocol

# Cache directory for downloaded datasets
AIPERF_DATASET_CACHE_DIR = Path(".cache/aiperf/datasets")


def download_public_dataset(dataset: "PublicDatasetProtocol") -> Path:
    """Download a public dataset or use cached version.

    This utility function handles downloading public datasets based on their
    metadata. It checks for cached versions first, then downloads if needed.

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
    cache_path = AIPERF_DATASET_CACHE_DIR / dataset.remote_filename

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
