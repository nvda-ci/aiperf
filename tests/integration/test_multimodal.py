# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal inputs (images, audio)."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultimodal:
    """Tests for multimodal inputs (images, audio)."""

    async def test_images(self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer):
        """Chat with image inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_input_images

    async def test_audio(self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer):
        """Chat with audio inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --audio-length-mean 0.1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_input_audio

    async def test_images_and_audio(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Chat with combined image and audio inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_input_images
        assert result.has_input_audio

    async def test_image_batch_size(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that --batch-size-image produces correct number of images per turn."""
        batch_size = 3
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --batch-size-image {batch_size} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_input_images

        # Verify inputs.json contains the correct number of images per turn
        assert result.inputs is not None, "inputs.json should exist"
        assert result.inputs.data, "inputs.json should contain data"

        for session in result.inputs.data:
            assert session.payloads, "session should have payloads"
            for payload in session.payloads:
                # Check OpenAI message format
                messages = payload.get("messages", [])
                assert messages, "payload should have messages"

                for message in messages:
                    content = message.get("content", [])
                    if isinstance(content, list):
                        # Count image_url entries in the content array
                        image_count = sum(
                            1
                            for item in content
                            if isinstance(item, dict)
                            and item.get("type") == "image_url"
                        )
                        # Each turn should have exactly batch_size images
                        assert image_count == batch_size, (
                            f"Expected {batch_size} images per turn, got {image_count}"
                        )
