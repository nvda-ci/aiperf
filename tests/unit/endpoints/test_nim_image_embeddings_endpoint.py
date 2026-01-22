# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NIMImageEmbeddingsEndpoint."""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models import Image, Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import NIMImageEmbeddingResponseData
from aiperf.common.protocols import InferenceServerResponse
from aiperf.endpoints.nim_image_embeddings import NIMImageEmbeddingsEndpoint
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


class TestNIMImageEmbeddingsEndpointFormatPayload:
    """Tests for NIMImageEmbeddingsEndpoint format_payload functionality."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for NIM image embeddings."""
        return create_model_endpoint(
            EndpointType.NIM_IMAGE_EMBEDDINGS, model_name="nvidia/c-radio-v3-h"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a NIMImageEmbeddingsEndpoint instance."""
        return create_endpoint_with_mock_transport(
            NIMImageEmbeddingsEndpoint, model_endpoint
        )

    def test_format_payload_single_text_query(self, endpoint, model_endpoint):
        """Test single text input results in query request_type."""
        turn = Turn(
            texts=[Text(contents=["Describe this image"])],
            model="nvidia/c-radio-v3-h",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "nvidia/c-radio-v3-h"
        assert payload["input"] == "Describe this image"
        assert payload["request_type"] == "query"

    def test_format_payload_multiple_texts_bulk(self, endpoint, model_endpoint):
        """Test multiple text inputs result in bulk_text request_type."""
        turn = Turn(
            texts=[Text(contents=["Text one", "Text two", "Text three"])],
            model="nvidia/c-radio-v3-h",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == ["Text one", "Text two", "Text three"]
        assert payload["request_type"] == "bulk_text"

    def test_format_payload_image_input(self, endpoint, model_endpoint):
        """Test single image input results in query request_type."""
        base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        turn = Turn(
            images=[Image(contents=[base64_image])],
            model="nvidia/c-radio-v3-h",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == base64_image
        assert payload["request_type"] == "query"

    def test_format_payload_multiple_images(self, endpoint, model_endpoint):
        """Test multiple image inputs."""
        images = ["base64_image_1", "base64_image_2"]
        turn = Turn(
            images=[Image(contents=images)],
            model="nvidia/c-radio-v3-h",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == images
        assert payload["request_type"] == "bulk_image"

    def test_format_payload_images_take_precedence(self, endpoint, model_endpoint):
        """Test that single image takes precedence over text when both are present."""
        turn = Turn(
            texts=[Text(contents=["Some text"])],
            images=[Image(contents=["base64_image"])],
            model="nvidia/c-radio-v3-h",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == "base64_image"
        assert payload["request_type"] == "query"

    def test_format_payload_empty_input(self, endpoint, model_endpoint):
        """Test empty input defaults to query with empty string."""
        turn = Turn(
            texts=[],
            images=[],
            model="nvidia/c-radio-v3-h",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == ""
        assert payload["request_type"] == "query"

    def test_format_payload_filters_empty_contents(self, endpoint, model_endpoint):
        """Test that empty strings are filtered from inputs."""
        turn = Turn(
            texts=[Text(contents=["Valid text", "", "Another valid", ""])],
            model="nvidia/c-radio-v3-h",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == ["Valid text", "Another valid"]
        assert payload["request_type"] == "bulk_text"

    def test_format_payload_max_tokens_warning(self, endpoint, model_endpoint, caplog):
        """Test that max_tokens triggers an error log for image embeddings."""
        turn = Turn(
            images=[Image(contents=["base64_image"])],
            model="nvidia/c-radio-v3-h",
            max_tokens=100,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with caplog.at_level(logging.ERROR):
            endpoint.format_payload(request_info)

        assert "not supported for image embeddings" in caplog.text

    def test_format_payload_model_fallback(self, endpoint, model_endpoint):
        """Test fallback to endpoint model when turn model is None."""
        turn = Turn(
            texts=[Text(contents=["Test"])],
            model=None,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name

    def test_format_payload_extra_parameters(self):
        """Test extra parameters (pyramid, max_pixels) are included."""
        extra_params = [
            ("pyramid", [[1, 1], [3, 3], [5, 5]]),
            ("max_pixels", 50176),
            ("encoding_format", "float"),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.NIM_IMAGE_EMBEDDINGS,
            model_name="nvidia/c-radio-v3-h",
            extra=extra_params,
        )
        endpoint = create_endpoint_with_mock_transport(
            NIMImageEmbeddingsEndpoint, model_endpoint
        )

        turn = Turn(
            images=[Image(contents=["base64_image"])],
            model="nvidia/c-radio-v3-h",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["pyramid"] == [[1, 1], [3, 3], [5, 5]]
        assert payload["max_pixels"] == 50176
        assert payload["encoding_format"] == "float"

    def test_format_payload_multiple_turns_raises(self, endpoint, model_endpoint):
        """Test that multiple turns raises ValueError."""
        turn1 = Turn(texts=[Text(contents=["Text 1"])], model="model1")
        turn2 = Turn(texts=[Text(contents=["Text 2"])], model="model2")

        request_info = create_request_info(
            model_endpoint=model_endpoint, turn_index=0, turns=[turn1, turn2]
        )

        with pytest.raises(
            ValueError, match="NIM Image Embeddings endpoint only supports one turn"
        ):
            endpoint.format_payload(request_info)


class TestNIMImageEmbeddingsEndpointParseResponse:
    """Tests for NIMImageEmbeddingsEndpoint parse_response functionality."""

    @pytest.fixture
    def endpoint(self):
        """Create a NIMImageEmbeddingsEndpoint instance for parsing tests."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="nvidia/c-radio-v3-h")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.NIM_IMAGE_EMBEDDINGS,
                base_url="http://localhost:8000",
            ),
        )
        with patch(
            "aiperf.common.factories.TransportFactory.create_instance"
        ) as mock_transport:
            mock_transport.return_value = MagicMock()
            return NIMImageEmbeddingsEndpoint(model_endpoint=model_endpoint)

    def test_parse_response_single_embedding(self, endpoint):
        """Test parsing response with single embedding."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "index": 0,
                }
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, NIMImageEmbeddingResponseData)
        assert len(parsed.data.embeddings) == 1
        assert parsed.data.embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert parsed.data.patch_metadata is None

    def test_parse_response_with_patch_metadata(self, endpoint):
        """Test parsing response with patch_metadata for pyramidal patching."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0,
                    "patch_metadata": {
                        "pyramid_level": 0,
                        "patch_coords": [0, 0],
                    },
                },
                {
                    "object": "embedding",
                    "embedding": [0.4, 0.5, 0.6],
                    "index": 1,
                    "patch_metadata": {
                        "pyramid_level": 1,
                        "patch_coords": [0, 0],
                    },
                },
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.embeddings) == 2
        assert parsed.data.patch_metadata is not None
        assert len(parsed.data.patch_metadata) == 2
        assert parsed.data.patch_metadata[0]["pyramid_level"] == 0
        assert parsed.data.patch_metadata[1]["pyramid_level"] == 1

    def test_parse_response_with_usage(self, endpoint):
        """Test parsing response with usage information."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10,
                "num_images": 1,
                "num_patches": 25,
            },
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.num_images == 1
        assert parsed.usage.num_patches == 25

    def test_parse_response_multiple_embeddings(self, endpoint):
        """Test parsing response with multiple embeddings."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
                {"object": "embedding", "embedding": [0.7, 0.8, 0.9], "index": 2},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.embeddings) == 3
        assert parsed.data.embeddings[0] == [0.1, 0.2, 0.3]
        assert parsed.data.embeddings[1] == [0.4, 0.5, 0.6]
        assert parsed.data.embeddings[2] == [0.7, 0.8, 0.9]

    def test_parse_response_empty_data_array(self, endpoint):
        """Test parsing response with empty data array."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {"data": []}

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_no_data_key(self, endpoint):
        """Test parsing response without data key."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {}

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_no_json(self, endpoint):
        """Test parsing when get_json returns None."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = None
        mock_response.get_raw.return_value = "not json"

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_missing_embedding_field(self, endpoint):
        """Test parsing when embedding field is missing."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [{"object": "embedding", "index": 0}]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_null_embedding(self, endpoint):
        """Test parsing when embedding is None."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [{"object": "embedding", "embedding": None, "index": 0}]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_invalid_object_type_raises(self, endpoint):
        """Test that invalid object type raises ValueError."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2]},
                {"object": "wrong_type", "embedding": [0.3, 0.4]},
            ]
        }

        with pytest.raises(ValueError, match="invalid list"):
            endpoint.parse_response(mock_response)

    def test_parse_response_high_dimensional_embedding(self, endpoint):
        """Test parsing high-dimensional embedding vectors (C-RADIO produces 1280-dim)."""
        embedding_vector = [float(i) / 1000 for i in range(1280)]
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [{"object": "embedding", "embedding": embedding_vector, "index": 0}]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.embeddings[0]) == 1280

    def test_parse_response_filters_null_embeddings(self, endpoint):
        """Test that None embeddings are filtered out."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                {"object": "embedding", "embedding": None, "index": 1},
                {"object": "embedding", "embedding": [0.3, 0.4], "index": 2},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.embeddings) == 2
        assert parsed.data.embeddings[0] == [0.1, 0.2]
        assert parsed.data.embeddings[1] == [0.3, 0.4]


class TestNIMImageEmbeddingsEndpointMetadata:
    """Tests for NIMImageEmbeddingsEndpoint metadata."""

    def test_metadata_values(self):
        """Test endpoint metadata has correct values."""
        metadata = NIMImageEmbeddingsEndpoint.metadata()

        assert metadata.endpoint_path == "/v1/embeddings"
        assert metadata.supports_streaming is False
        assert metadata.produces_tokens is False
        assert metadata.tokenizes_input is False
        assert metadata.metrics_title == "NIM Image Embeddings Metrics"
