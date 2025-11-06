# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory
from aiperf.common.models import ParsedResponse
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.record_models import RequestInfo
from aiperf.common.protocols import EndpointProtocol, InferenceServerResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint


@implements_protocol(EndpointProtocol)
@EndpointFactory.register(EndpointType.OLLAMA_GENERATE)
class OllamaGenerateEndpoint(BaseEndpoint):
    """Ollama Generate endpoint.

    Supports both streaming and non-streaming text generation using Ollama's
    /api/generate endpoint. This endpoint is designed for single-turn text
    generation with optional system prompts and advanced parameters.
    """

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return Ollama Generate endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/api/generate",
            supports_streaming=True,
            produces_tokens=True,
            tokenizes_input=True,
            metrics_title="LLM Metrics",
        )

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format payload for Ollama Generate request.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            Ollama Generate API payload
        """
        if not request_info.turns:
            raise ValueError("Ollama Generate endpoint requires at least one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        prompt = " ".join(
            [content for text in turn.texts for content in text.contents if content]
        )

        payload: dict[str, Any] = {
            "model": turn.model or model_endpoint.primary_model_name,
            "prompt": prompt,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turn.max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = turn.max_tokens

        if model_endpoint.endpoint.extra:
            extra = dict(model_endpoint.endpoint.extra)
            extra_options = extra.pop("options", {})

            payload.update(extra)

            if extra_options:
                payload.setdefault("options", {}).update(extra_options)

        self.debug(lambda: f"Formatted Ollama Generate payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Ollama Generate response.

        Handles both streaming and non-streaming modes. In streaming mode,
        each chunk contains incremental response text. In non-streaming mode,
        the complete response is returned at once.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text and usage data
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        text = json_obj.get("response")
        if not text:
            self.debug(lambda: f"No 'response' field in Ollama response: {json_obj}")
            return None

        data = self.make_text_response_data(text)

        usage = None
        if json_obj.get("done"):
            prompt_eval_count = json_obj.get("prompt_eval_count")
            eval_count = json_obj.get("eval_count")

            if prompt_eval_count is not None or eval_count is not None:
                usage = {
                    "prompt_tokens": prompt_eval_count,
                    "completion_tokens": eval_count,
                }
                if prompt_eval_count is not None and eval_count is not None:
                    usage["total_tokens"] = prompt_eval_count + eval_count

        return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)
