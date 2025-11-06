# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import EndpointType
from aiperf.common.models import Image, Turn
from aiperf.common.models.record_models import RequestInfo
from aiperf.endpoints.nim_image_retrieval import ImageRetrievalEndpoint
from tests.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)

BASE64_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class TestImageRetrievalEndpoint:
    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.IMAGE_RETRIEVAL, model_name="image-retrieval-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            ImageRetrievalEndpoint, model_endpoint
        )

    def test_format_payload_basic(self, endpoint, model_endpoint):
        """Test basic format_payload with valid image."""
        turn = Turn(
            images=[Image(contents=[BASE64_PNG])], model="image-retrieval-model"
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 1
        assert payload["input"][0]["type"] == "image_url"
        assert payload["input"][0]["url"] == BASE64_PNG

    def test_format_payload_validation_error(self, endpoint, model_endpoint):
        """Test that empty images raises ValueError."""
        turn = Turn(images=[], model="image-retrieval-model")
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(
            ValueError, match="Image Retrieval request requires at least one image"
        ):
            endpoint.format_payload(request_info)
