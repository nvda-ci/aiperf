# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import ImageRetrievalResponseData
from aiperf.endpoints.nim_image_retrieval import ImageRetrievalEndpoint


class TestImageRetrievalEndpointParseResponse:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="image-retrieval-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.IMAGE_RETRIEVAL,
                base_url="http://localhost:8000",
            ),
        )
        with patch(
            "aiperf.common.factories.TransportFactory.create_instance"
        ) as mock_transport:
            mock_transport.return_value = MagicMock()
            return ImageRetrievalEndpoint(model_endpoint=model_endpoint)

    def test_parse_response_basic(self, endpoint):
        """Test basic parse_response with valid bounding box data."""
        mock_response = Mock()
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {
                    "index": 0,
                    "bounding_boxes": {
                        "chart": [
                            {
                                "x_min": 10,
                                "y_min": 20,
                                "x_max": 100,
                                "y_max": 120,
                                "confidence": 0.95,
                            }
                        ]
                    },
                }
            ],
            "usage": {"images_size_mb": 0.5},
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, ImageRetrievalResponseData)
        assert len(parsed.data.data) == 1
        assert "chart" in parsed.data.data[0]["bounding_boxes"]

    def test_parse_response_invalid(self, endpoint):
        """Test parse_response returns None for invalid/empty data."""
        mock_response = Mock()
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = None

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None
