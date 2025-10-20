# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aiperf.common.enums import EndpointType
from aiperf.common.enums.openai_enums import OpenAIObjectType
from aiperf.common.factories import EndpointFactory
from aiperf.common.models import (
    BaseResponseData,
    ParsedResponse,
    TextResponseData,
    Turn,
)
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.record_models import ReasoningResponseData, RequestInfo
from aiperf.common.protocols import InferenceServerResponse
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint

_DEFAULT_ROLE: str = "user"


@EndpointFactory.register(EndpointType.CHAT)
class ChatEndpoint(BaseEndpoint):
    """OpenAI Chat Completions endpoint.

    Supports multi-modal inputs (text, images, audio, video) and both
    streaming and non-streaming responses.
    """

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return Chat Completions endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/v1/chat/completions",
            supports_streaming=True,
            produces_tokens=True,
            supports_audio=True,
            supports_images=True,
            supports_videos=True,
            metrics_title="LLM Metrics",
        )

    async def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Chat Completions request payload from RequestInfo.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Chat Completions API payload
        """
        # Use first turn (hardcoded for now)
        turn = request_info.turns[0]
        messages = self._create_messages(turn)
        model_endpoint = request_info.model_endpoint

        payload: dict[str, Any] = {
            "messages": messages,
            "model": turn.model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turn.max_tokens is not None:
            payload["max_completion_tokens"] = turn.max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        return payload

    def _create_messages(self, turn: Turn) -> Sequence[dict[str, Any]]:
        """Create OpenAI message array from Turn data.

        Handles both simple text-only messages and complex multi-modal messages
        with images, audio, and video.

        Args:
            turn: Input data containing text, images, audio, and/or video

        Returns:
            List of OpenAI message objects

        Raises:
            ValueError: If audio content is not in 'format,b64_audio' format
        """
        message: dict[str, Any] = {"role": turn.role or _DEFAULT_ROLE}

        # Fast path: simple text-only message
        if (
            len(turn.texts) == 1
            and len(turn.texts[0].contents) == 1
            and not turn.images
            and not turn.audios
            and not turn.videos
        ):
            message["name"] = turn.texts[0].name
            message["content"] = (
                turn.texts[0].contents[0] if turn.texts[0].contents else ""
            )
            return [message]

        # Complex path: multi-modal content array
        message_content: list[dict[str, Any]] = []

        # Add text content
        for text in turn.texts:
            message_content.extend(
                {"type": "text", "text": content}
                for content in text.contents
                if content
            )

        # Add image content
        for image in turn.images:
            message_content.extend(
                {"type": "image_url", "image_url": {"url": content}}
                for content in image.contents
                if content
            )

        # Add audio content
        for audio in turn.audios:
            for content in audio.contents:
                if not content:
                    continue
                if "," not in content:
                    raise ValueError(
                        "Audio content must be in format 'format,b64_audio'."
                    )
                audio_format, b64_audio = content.split(",", 1)
                message_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64_audio, "format": audio_format},
                    }
                )

        # Add video content
        for video in turn.videos:
            message_content.extend(
                {"type": "video_url", "video_url": {"url": content}}
                for content in video.contents
                if content
            )

        message["content"] = message_content
        return [message]

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Chat Completions response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text/reasoning content
        """
        if not (json_obj := response.get_json()):
            return None

        if data := self._extract_from_json(json_obj):
            return ParsedResponse(perf_ns=response.perf_ns, data=data)

        return None

    def _extract_from_json(self, json_obj: JsonObject) -> BaseResponseData | None:
        """Extract content from OpenAI JSON response.

        Handles both streaming (chat.completion.chunk) and non-streaming
        (chat.completion) formats using pattern matching.

        Args:
            json_obj: Deserialized OpenAI response

        Returns:
            Extracted response data or None if no content
        """
        # Pattern match on OpenAI object type
        match json_obj.get("object"):
            case str(OpenAIObjectType.CHAT_COMPLETION):
                # Non-streaming: extract from "message"
                message = json_obj.get("choices", [{}])[0].get("message", {})
                return self._extract_content(message)

            case str(OpenAIObjectType.CHAT_COMPLETION_CHUNK):
                # Streaming: extract from "delta"
                delta = json_obj.get("choices", [{}])[0].get("delta", {})
                return self._extract_content(delta)

            case _:
                return None

    def _extract_content(self, obj: Mapping[str, Any]) -> BaseResponseData | None:
        """Extract content from message or delta object.

        Handles both standard text content and reasoning content (o1 models).

        Args:
            obj: Message or delta object from choices array

        Returns:
            Text or reasoning response data, or None if empty
        """
        content = obj.get("content")
        reasoning = obj.get("reasoning_content") or obj.get("reasoning")

        if not content and not reasoning:
            return None
        if not reasoning:
            return TextResponseData(text=content) if content else None

        return ReasoningResponseData(content=content, reasoning=reasoning)
