# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from aiperf.common.enums import DatasetLoaderType, PublicDatasetType
from aiperf.common.factories import PublicDatasetFactory


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
        remote_filename: Filename for local caching
        loader_type: Which DatasetLoaderType to use for parsing
    """

    dataset_type: PublicDatasetType
    """Dataset type enum (for factory registration)"""

    name: str
    """Human-readable name for the dataset"""

    url: str
    """URL to download the dataset from"""

    remote_filename: str
    """Filename to use for local caching"""

    loader_type: DatasetLoaderType
    """Which loader to use for parsing this dataset"""

    def __post_init__(self) -> None:
        """Auto-register this instance with PublicDatasetFactory."""
        PublicDatasetFactory.register_instance(self.dataset_type, self)


# Hard-coded dataset instances
# Each instance auto-registers on creation via __post_init__!

SHAREGPT = PublicDataset(
    dataset_type=PublicDatasetType.SHAREGPT,  # Auto-registers with factory!
    name="ShareGPT",
    url=(
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    ),
    remote_filename="ShareGPT_V3_unfiltered_cleaned_split.json",
    loader_type=DatasetLoaderType.SHAREGPT,
)
