# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class PublicDatasetType(CaseInsensitiveStrEnum):
    SHAREGPT = "sharegpt"


class ComposerType(CaseInsensitiveStrEnum):
    SYNTHETIC = "synthetic"
    CUSTOM = "custom"
    PUBLIC_DATASET = "public_dataset"
    SYNTHETIC_RANKINGS = "synthetic_rankings"


class CustomDatasetType(CaseInsensitiveStrEnum):
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    RANDOM_POOL = "random_pool"
    MOONCAKE_TRACE = "mooncake_trace"


class ImageFormat(CaseInsensitiveStrEnum):
    PNG = "png"
    JPEG = "jpeg"
    RANDOM = "random"


class AudioFormat(CaseInsensitiveStrEnum):
    WAV = "wav"
    MP3 = "mp3"


class VideoFormat(CaseInsensitiveStrEnum):
    MP4 = "mp4"
    WEBM = "webm"


class VideoSynthType(CaseInsensitiveStrEnum):
    MOVING_SHAPES = "moving_shapes"
    GRID_CLOCK = "grid_clock"


class PromptSource(CaseInsensitiveStrEnum):
    SYNTHETIC = "synthetic"
    FILE = "file"
    PAYLOAD = "payload"


class DatasetSamplingStrategy(CaseInsensitiveStrEnum):
    SEQUENTIAL = "sequential"
    """Iterate through the dataset sequentially, then wrap around to the beginning."""

    RANDOM = "random"
    """Randomly select a conversation from the dataset. Will randomly sample with replacement."""

    SHUFFLE = "shuffle"
    """Shuffle the dataset and iterate through it. Will randomly sample without replacement.
    Once the end of the dataset is reached, shuffle the dataset again and start over."""


class DatasetBackingStoreType(CaseInsensitiveStrEnum):
    """Types of dataset backing stores (DatasetManager side).

    Defines how DatasetManager stores and manages the dataset.
    """

    IN_MEMORY = "in_memory"
    """Store dataset in memory only (workers use ZMQ_REMOTE to access)"""

    MEMORY_MAP = "memory_map"
    """Store dataset in local memory-mapped files (single machine, workers use MEMORY_MAP to access)"""


class DatasetClientStoreType(CaseInsensitiveStrEnum):
    """Types of dataset client stores (Worker side).

    Defines how Workers access the dataset created by DatasetManager.
    """

    MEMORY_MAP = "memory_map"
    """Read from local memory-mapped files (single machine)"""

    ZMQ_REMOTE = "zmq_remote"
    """Request conversations via ZMQ from DatasetManager"""
