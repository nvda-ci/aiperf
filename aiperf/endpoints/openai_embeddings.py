# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.enums.openai_enums import OpenAIObjectType
from aiperf.common.models import (
    BaseResponseData,
    ParsedResponse,
)
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.record_models import EmbeddingResponseData, RequestInfo
from aiperf.common.protocols import InferenceServerResponse
from aiperf.common.types import JsonObject, RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint


# @EndpointFactory.register(EndpointType.EMBEDDINGS)
class EmbeddingsEndpoint(BaseEndpoint):
    """OpenAI Embeddings endpoint.

    Generates vector embeddings for text inputs.
    """

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return Embeddings endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/v1/embeddings",
            supports_streaming=False,
            produces_tokens=False,
            metrics_title="Embeddings Metrics",
        )

    async def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format payload for an embeddings request.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Embeddings API payload
        """
        # Use first turn (hardcoded for now)
        turn = request_info.turns[0]

        if turn.max_tokens:
            self.error("Max_tokens is provided but is not supported for embeddings.")

        # Extract all text contents as input
        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]
        model_endpoint = request_info.model_endpoint

        payload: dict[str, Any] = {
            "model": turn.model or model_endpoint.primary_model_name,
            "input": prompts,
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.debug(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Embeddings response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted embeddings
        """
        if not (json_obj := response.get_json()):
            return None

        # Extract embeddings inline
        if data := self._extract_from_json(json_obj):
            return ParsedResponse(perf_ns=response.perf_ns, data=data)

        return None

    def _extract_from_json(self, json_obj: JsonObject) -> BaseResponseData | None:
        """Extract embeddings from OpenAI response.

        Args:
            json_obj: Deserialized OpenAI response

        Returns:
            Embedding response data or None if invalid
        """
        # Get data array from response
        data = json_obj.get("data", [])

        # Validate all items are embedding objects
        if not all(
            isinstance(item, dict)
            and item.get("object") == str(OpenAIObjectType.EMBEDDING)
            for item in data
        ):
            raise ValueError(f"Received invalid list in response: {json_obj}")

        # Extract embeddings inline
        if not data:
            self.warning("No data found in response")
            return None

        embeddings = [
            embedding
            for item in data
            if (embedding := item.get("embedding")) is not None
        ]

        return EmbeddingResponseData(embeddings=embeddings) if embeddings else None
