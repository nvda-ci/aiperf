# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class PublicDatasetType(CaseInsensitiveStrEnum):
    """Public datasets available for benchmarking."""

    SHAREGPT = "sharegpt"
    """ShareGPT dataset from HuggingFace. Multi-turn conversational dataset with user/assistant exchanges."""


class ImageFormat(CaseInsensitiveStrEnum):
    """Image file formats for synthetic image generation."""

    PNG = "png"
    """PNG format. Lossless compression, larger file sizes, best quality."""

    JPEG = "jpeg"
    """JPEG format. Lossy compression, smaller file sizes, good for photos."""

    RANDOM = "random"
    """Randomly select PNG or JPEG for each image."""


class AudioFormat(CaseInsensitiveStrEnum):
    """Audio file formats for synthetic audio generation."""

    WAV = "wav"
    """WAV format. Uncompressed audio, larger file sizes, best quality."""

    MP3 = "mp3"
    """MP3 format. Compressed audio, smaller file sizes, good quality."""


class VideoFormat(CaseInsensitiveStrEnum):
    """Video container formats for synthetic video generation."""

    MP4 = "mp4"
    """MP4 container. Widely compatible, good for H.264/H.265 codecs."""

    WEBM = "webm"
    """WebM container. Open format, optimized for web, good for VP9 codec."""


class VideoSynthType(CaseInsensitiveStrEnum):
    MOVING_SHAPES = "moving_shapes"
    """Generate videos with animated geometric shapes moving across the frame"""

    GRID_CLOCK = "grid_clock"
    """Generate videos with a grid pattern and timestamp overlay for frame-accurate verification"""


class PromptSource(CaseInsensitiveStrEnum):
    SYNTHETIC = "synthetic"
    FILE = "file"
    PAYLOAD = "payload"
