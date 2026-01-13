# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for different media format support (JPEG, PNG, MP3, WAV)."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestMediaFormats:
    """Tests for different media format support (JPEG, PNG, MP3, WAV)."""

    @pytest.mark.parametrize("image_format", ["jpeg", "png"])
    def test_image_formats(self, cli: AIPerfCLI, image_format: str):
        """Test different image format support."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --endpoint-type chat \
                --request-count 2 \
                --concurrency 2 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --image-format {image_format} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 2
        assert result.has_input_images

    @pytest.mark.parametrize("audio_format", ["mp3", "wav"])
    def test_audio_formats(self, cli: AIPerfCLI, audio_format: str):
        """Test different audio format support."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --endpoint-type chat \
                --request-count 2 \
                --concurrency 2 \
                --audio-length-mean 0.1 \
                --audio-format {audio_format} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 2
        assert result.has_input_audio
