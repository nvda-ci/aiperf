# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory
from aiperf.common.models import ParsedResponse
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.record_models import (
    NIMImageEmbeddingResponseData,
    RequestInfo,
)
from aiperf.common.models.usage_models import Usage
from aiperf.common.protocols import EndpointProtocol, InferenceServerResponse
from aiperf.common.types import RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint


@implements_protocol(EndpointProtocol)
@EndpointFactory.register(EndpointType.NIM_IMAGE_EMBEDDINGS)
class NIMImageEmbeddingsEndpoint(BaseEndpoint):
    """NVIDIA NIM Image Embeddings endpoint.

    Generates vector embeddings for images using NVIDIA C-RADIO NIM.
    Supports pyramidal patching for multi-scale image representations.

    Accepts both text and image inputs, with automatic request_type detection:
    - query: Single text or single image input
    - bulk_text: Multiple text inputs
    - bulk_image: Multiple base64-encoded images
    """

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return NIM Image Embeddings endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/v1/embeddings",
            supports_streaming=False,
            produces_tokens=False,
            tokenizes_input=False,
            supports_images=True,
            metrics_title="NIM Image Embeddings Metrics",
        )

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format payload for a NIM image embeddings request.

        Supports three request types:
        - query: Single text or single image input
        - bulk_text: Multiple text inputs
        - bulk_image: Multiple base64-encoded images

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            NIM Image Embeddings API payload
        """
        if len(request_info.turns) != 1:
            raise ValueError("NIM Image Embeddings endpoint only supports one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        if turn.max_tokens:
            self.error(
                "Max_tokens is provided but is not supported for image embeddings."
            )

        # Collect text inputs
        text_inputs = [
            content for text in turn.texts for content in text.contents if content
        ]

        # Collect image inputs (base64 encoded)
        image_inputs = [
            content for image in turn.images for content in image.contents if content
        ]

        # Determine request type and input based on available data
        if len(image_inputs) == 1:
            # Single image query mode
            request_type = "query"
            input_data = image_inputs[0]
        elif image_inputs:
            # Bulk image embedding mode
            request_type = "bulk_image"
            input_data = image_inputs
        elif len(text_inputs) == 1:
            # Single text query mode
            request_type = "query"
            input_data = text_inputs[0]
        elif text_inputs:
            # Bulk text embedding mode
            request_type = "bulk_text"
            input_data = text_inputs
        else:
            # Default to empty query
            request_type = "query"
            input_data = ""

        payload: dict[str, Any] = {
            "model": turn.model or model_endpoint.primary_model_name,
            "input": input_data,
            "request_type": request_type,
        }

        # Add extra parameters from endpoint config
        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse NIM Image Embeddings response.

        Handles extended response format with patch_metadata for pyramidal patching.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted embeddings and patch metadata
        """
        json_obj = response.get_json()
        if not json_obj:
            self.debug(
                lambda: f"No JSON object found in response: {response.get_raw()}"
            )
            return None

        data = json_obj.get("data", [])
        if not data:
            self.debug(lambda: f"No data found in response: {json_obj}")
            return None

        # Validate and extract embeddings
        if not all(
            isinstance(item, dict) and item.get("object") == "embedding"
            for item in data
        ):
            raise ValueError(f"Received invalid list in response: {json_obj}")

        embeddings = [
            item.get("embedding") for item in data if item.get("embedding") is not None
        ]
        if not embeddings:
            return None

        # Extract patch_metadata if present (for pyramidal patching)
        patch_metadata = [
            item.get("patch_metadata") for item in data if "patch_metadata" in item
        ]

        # Extract usage information
        usage = None
        usage_data = json_obj.get("usage")
        if usage_data:
            usage = Usage(usage_data)

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=NIMImageEmbeddingResponseData(
                embeddings=embeddings,
                patch_metadata=patch_metadata if patch_metadata else None,
            ),
            usage=usage,
        )
