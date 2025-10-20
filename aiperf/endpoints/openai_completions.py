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
from aiperf.common.models.record_models import RequestInfo, TextResponseData
from aiperf.common.protocols import InferenceServerResponse
from aiperf.common.types import JsonObject, RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint


# @EndpointFactory.register(EndpointType.COMPLETIONS)
class CompletionsEndpoint(BaseEndpoint):
    """OpenAI Completions endpoint.

    Supports text completions with streaming.
    """

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return Completions endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/v1/completions",
            supports_streaming=True,
            produces_tokens=True,
            metrics_title="LLM Metrics",
        )

    async def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format payload for a completions request.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Completions API payload
        """
        # Use first turn (hardcoded for now)
        turn = request_info.turns[0]

        # Extract all text contents as prompts
        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]
        model_endpoint = request_info.model_endpoint

        payload: dict[str, Any] = {
            "prompt": prompts,
            "model": turn.model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turn.max_tokens:
            payload["max_tokens"] = turn.max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.debug(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Completions response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text content
        """
        if not (json_obj := response.get_json()):
            return None

        # Extract content inline with pattern matching
        if data := self._extract_from_json(json_obj):
            return ParsedResponse(perf_ns=response.perf_ns, data=data)

        return None

    def _extract_from_json(self, json_obj: JsonObject) -> BaseResponseData | None:
        """Extract content from OpenAI Completions JSON response.

        Handles both text.completion and completion object types.

        Args:
            json_obj: Deserialized OpenAI response

        Returns:
            Extracted text data or None if no content
        """
        # Pattern match on OpenAI object type
        match json_obj.get("object"):
            case str(OpenAIObjectType.COMPLETION) | str(
                OpenAIObjectType.TEXT_COMPLETION
            ):
                # Extract text from choices array
                if text := json_obj.get("choices", [{}])[0].get("text"):
                    return TextResponseData(text=text)
                return None

            case _:
                self.warning(f"Unsupported object type: {json_obj.get('object')}")
                return None
