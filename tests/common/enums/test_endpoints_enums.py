# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory


class TestEndpointType:
    """Test class for EndpointType enum."""

    @pytest.mark.parametrize(
        "endpoint_type,expected_tag,expected_streaming,expected_tokens,expected_path,expected_title",
        [
            (
                EndpointType.CHAT,
                "chat",
                True,
                True,
                "/v1/chat/completions",
                "LLM Metrics",
            ),
            (
                EndpointType.COMPLETIONS,
                "completions",
                True,
                True,
                "/v1/completions",
                "LLM Metrics",
            ),
            (
                EndpointType.EMBEDDINGS,
                "embeddings",
                False,
                False,
                "/v1/embeddings",
                "Embeddings Metrics",
            ),
            (
                EndpointType.RANKINGS,
                "rankings",
                False,
                False,
                "/v1/ranking",
                "Rankings Metrics",
            ),
        ],
    )
    def test_endpoint_type_metadata(
        self,
        endpoint_type,
        expected_tag,
        expected_streaming,
        expected_tokens,
        expected_path,
        expected_title,
    ):
        """Test EndpointType metadata is retrieved from factory."""
        assert str(endpoint_type) == expected_tag
        assert endpoint_type.value == expected_tag

        # Get metadata from factory
        metadata = EndpointFactory.metadata(endpoint_type)
        assert metadata.supports_streaming == expected_streaming
        assert metadata.produces_tokens == expected_tokens
        assert metadata.endpoint_path == expected_path
        assert metadata.metrics_title == expected_title

    @pytest.mark.parametrize(
        "tag_value",
        ["chat", "completions", "embeddings", "rankings"],
    )
    def test_enum_string_comparison(self, tag_value):
        """Test that enum values can be compared with strings."""
        endpoint_type = EndpointType(tag_value)
        assert endpoint_type == tag_value
        assert str(endpoint_type) == tag_value

    def test_endpoint_type_case_insensitive(self):
        """Test case insensitive enum behavior."""
        assert EndpointType("CHAT") == EndpointType.CHAT
        assert EndpointType("Chat") == EndpointType.CHAT
        assert EndpointType("chat") == EndpointType.CHAT

    def test_all_endpoint_types_have_valid_metadata(self):
        """Test that all endpoint types have valid metadata via factory."""
        for endpoint_type in EndpointType:
            metadata = EndpointFactory.metadata(endpoint_type)
            assert metadata.endpoint_path is not None
            assert len(metadata.endpoint_path) > 0
            assert isinstance(metadata.supports_streaming, bool)
            assert isinstance(metadata.produces_tokens, bool)
            assert metadata.metrics_title is not None
            assert len(metadata.metrics_title) > 0

    def test_endpoint_factory_get_metadata(self):
        """Test EndpointFactory.metadata() returns correct metadata."""
        metadata = EndpointFactory.metadata(EndpointType.CHAT)

        assert metadata.endpoint_path == "/v1/chat/completions"
        assert metadata.supports_streaming is True
        assert metadata.produces_tokens is True
        assert metadata.supports_audio is True
        assert metadata.supports_images is True
        assert metadata.supports_videos is True
        assert metadata.metrics_title == "LLM Metrics"

    def test_endpoint_factory_get_metadata_string_input(self):
        """Test EndpointFactory.metadata() works with string input."""
        metadata = EndpointFactory.metadata("chat")

        assert metadata.endpoint_path == "/v1/chat/completions"
        assert metadata.metrics_title == "LLM Metrics"
