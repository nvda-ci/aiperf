# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import EndpointType
from aiperf.common.models.record_models import EmbeddingResponseData, RequestInfo
from aiperf.endpoints.nvclip import NVClipEndpoint
from tests.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_multimodal_request_info,
)


@pytest.fixture
def nvclip_model_endpoint():
    """Create a test ModelEndpointInfo for nvclip."""
    return create_model_endpoint(EndpointType.NVCLIP, model_name="nvclip-model")


@pytest.fixture
def nvclip_endpoint(nvclip_model_endpoint):
    """Create a NVClipEndpoint instance."""
    return create_endpoint_with_mock_transport(NVClipEndpoint, nvclip_model_endpoint)


class TestNVClipFormatPayload:
    """Tests for NVClipEndpoint format_payload functionality."""

    @pytest.mark.parametrize(
        "texts,images,expected_input",
        [
            (["text1"], None, ["text1"]),
            (None, ["image1"], ["image1"]),
            (["text1"], ["image1"], ["text1", "image1"]),
            (
                ["text1", "text2"],
                ["image1", "image2"],
                ["text1", "text2", "image1", "image2"],
            ),
        ],
    )
    def test_multimodal_inputs(
        self, nvclip_endpoint, nvclip_model_endpoint, texts, images, expected_input
    ):
        """Test various combinations of text and image inputs."""
        request_info = create_multimodal_request_info(
            nvclip_model_endpoint, texts=texts, images=images, model="nvclip-model"
        )

        payload = nvclip_endpoint.format_payload(request_info)

        assert payload["model"] == "nvclip-model"
        assert payload["input"] == expected_input

    def test_filters_empty_inputs(self, nvclip_endpoint, nvclip_model_endpoint):
        """Test that empty strings are filtered from inputs."""
        request_info = create_multimodal_request_info(
            nvclip_model_endpoint,
            texts=["Valid", "", "Another"],
            images=["img1", ""],
            model="nvclip-model",
        )

        payload = nvclip_endpoint.format_payload(request_info)

        assert payload["input"] == ["Valid", "Another", "img1"]

    def test_no_inputs_raises_error(self, nvclip_endpoint, nvclip_model_endpoint):
        """Test that empty inputs raise ValueError."""
        request_info = create_multimodal_request_info(
            nvclip_model_endpoint, model="nvclip-model"
        )

        with pytest.raises(
            ValueError, match="requires at least one text or image input"
        ):
            nvclip_endpoint.format_payload(request_info)

    def test_model_fallback(self, nvclip_endpoint, nvclip_model_endpoint):
        """Test fallback to endpoint model when turn model is None."""
        request_info = create_multimodal_request_info(
            nvclip_model_endpoint, texts=["Test"]
        )

        payload = nvclip_endpoint.format_payload(request_info)

        assert payload["model"] == nvclip_model_endpoint.primary_model_name

    def test_extra_parameters(self):
        """Test extra parameters are included in payload."""
        model_endpoint = create_model_endpoint(
            EndpointType.NVCLIP,
            model_name="nvclip-model",
            extra=[("encoding_format", "base64")],
        )
        endpoint = create_endpoint_with_mock_transport(NVClipEndpoint, model_endpoint)
        request_info = create_multimodal_request_info(
            model_endpoint, texts=["Test"], images=["img1"], model="nvclip-model"
        )

        payload = endpoint.format_payload(request_info)

        assert payload["encoding_format"] == "base64"

    def test_only_one_turn_supported(self, nvclip_endpoint, nvclip_model_endpoint):
        """Test that multiple turns raise ValueError."""
        from aiperf.common.models import Text, Turn

        turns = [
            Turn(texts=[Text(contents=["Turn 1"])]),
            Turn(texts=[Text(contents=["Turn 2"])]),
        ]
        request_info = RequestInfo(model_endpoint=nvclip_model_endpoint, turns=turns)

        with pytest.raises(ValueError, match="only supports one turn"):
            nvclip_endpoint.format_payload(request_info)


class TestNVClipParseResponse:
    """Tests for NVClipEndpoint parse_response functionality."""

    @pytest.mark.parametrize(
        "embeddings_data,expected_count",
        [
            ([{"object": "embedding", "embedding": [0.1, 0.2]}], 1),
            (
                [
                    {"object": "embedding", "embedding": [0.1, 0.2]},
                    {"object": "embedding", "embedding": [0.3, 0.4]},
                ],
                2,
            ),
            (
                [
                    {"object": "embedding", "embedding": [0.1, 0.2]},
                    {"object": "embedding", "embedding": None},
                    {"object": "embedding", "embedding": [0.3, 0.4]},
                ],
                2,
            ),
        ],
    )
    def test_parse_embeddings(self, nvclip_endpoint, embeddings_data, expected_count):
        """Test parsing valid embedding responses."""
        response = create_mock_response({"data": embeddings_data})

        parsed = nvclip_endpoint.parse_response(response)

        assert parsed is not None
        assert isinstance(parsed.data, EmbeddingResponseData)
        assert len(parsed.data.embeddings) == expected_count

    @pytest.mark.parametrize(
        "json_data",
        [
            None,
            {},
            {"data": []},
            {"data": [{"object": "embedding"}]},
            {"data": [{"object": "embedding", "embedding": None}]},
        ],
    )
    def test_returns_none_for_invalid_data(self, nvclip_endpoint, json_data):
        """Test that invalid or missing data returns None."""
        response = create_mock_response(json_data)

        parsed = nvclip_endpoint.parse_response(response)

        assert parsed is None

    @pytest.mark.parametrize(
        "embeddings_data",
        [
            [
                {"object": "embedding", "embedding": [0.1]},
                {"object": "wrong_type", "embedding": [0.2]},
            ],
            [{"object": "embedding", "embedding": [0.1]}, "not a dict"],
        ],
    )
    def test_invalid_object_type_raises(self, nvclip_endpoint, embeddings_data):
        """Test that invalid object types raise ValueError."""
        response = create_mock_response({"data": embeddings_data})

        with pytest.raises(ValueError, match="invalid list"):
            nvclip_endpoint.parse_response(response)
