# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory
from aiperf.common.models import (
    ParsedResponse,
)
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.record_models import RankingsResponseData, RequestInfo
from aiperf.common.protocols import InferenceServerResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint


@EndpointFactory.register(EndpointType.RANKINGS)
class RankingsEndpoint(BaseEndpoint):
    """NIM Rankings endpoint.

    Ranks passages against a query.

    Expected input format:
    - 'query': Text object containing the query to rank against
    - 'passages': Text object containing passages to be ranked
    """

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return Rankings endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/v1/ranking",
            supports_streaming=False,
            produces_tokens=False,
            metrics_title="Rankings Metrics",
        )

    async def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format payload for a rankings request.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            NIM Rankings API payload

        Raises:
            ValueError: If query is missing
        """
        # Use first turn (hardcoded for now)
        turn = request_info.turns[0]

        if turn.max_tokens:
            self.warning("Max_tokens is provided but is not supported for rankings.")

        # Separate query and passage texts by name using pattern matching
        query_texts: list[str] = []
        passage_texts: list[str] = []

        for text in turn.texts:
            match text.name:
                case "query":
                    query_texts.extend(text.contents)
                case "passages":
                    passage_texts.extend(text.contents)
                case _:
                    self.warning(
                        f"Ignoring text with name '{text.name}' - rankings expects 'query' and 'passages'"
                    )

        # Validate query
        if not query_texts:
            raise ValueError(
                "Rankings request requires a text with name 'query'. "
                "Provide a Text object with name='query' containing the search query."
            )

        if len(query_texts) > 1:
            self.warning(
                f"Multiple query texts found, using the first one. Found {len(query_texts)} queries."
            )

        # Warn if no passages
        if not passage_texts:
            self.warning(
                "Rankings request has query but no passages to rank. "
                "Consider adding a Text object with name='passages' containing texts to rank."
            )

        # Build payload
        model_endpoint = request_info.model_endpoint
        payload: dict[str, Any] = {
            "model": turn.model or model_endpoint.primary_model_name,
            "query": {"text": query_texts[0]},
            "passages": [{"text": passage} for passage in passage_texts],
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.debug(lambda: f"Formatted rankings payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse NIM Rankings response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted rankings
        """
        if not (json_obj := response.get_json()):
            return None

        if rankings := json_obj.get("rankings"):
            return ParsedResponse(
                perf_ns=response.perf_ns, data=RankingsResponseData(rankings=rankings)
            )

        self.warning(f"No rankings found in response: {json_obj}")
        return None
