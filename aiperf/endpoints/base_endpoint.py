# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from aiperf.common.decorators import implements_protocol
from aiperf.common.factories import TransportFactory
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import (
    ParsedResponse,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.protocols import (
    EndpointProtocol,
    InferenceServerResponse,
    RequestOutputT,
)
from aiperf.common.types import RequestInputT


@implements_protocol(EndpointProtocol)
class BaseEndpoint(AIPerfLifecycleMixin, ABC):
    """Base for all endpoints.

    Endpoints handle API-specific formatting and parsing.
    They work with any transport - just format JSON and parse responses.
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs):
        super().__init__(**kwargs)
        self.model_endpoint = model_endpoint
        self.transport = TransportFactory.create_instance(model_endpoint.transport)
        self.attach_child_lifecycle(self.transport)

    @classmethod
    @abstractmethod
    def metadata(cls) -> EndpointMetadata:
        """Return endpoint metadata."""

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint headers (auth + user custom). Override to customize."""
        headers = {}

        cfg = self.model_endpoint.endpoint
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        if cfg.headers:
            headers.update(dict(cfg.headers))

        return headers

    def get_endpoint_params(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint URL query params (e.g., api-version). Override to customize."""
        params = {}

        cfg = self.model_endpoint.endpoint
        if cfg.url_params:
            params.update(cfg.url_params)

        return params

    @abstractmethod
    async def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format request payload from RequestInfo.

        Uses request_info.turns[0] as the turn data (currently hardcoded to first turn).
        """

    @abstractmethod
    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse response. Return None to skip."""

    async def extract_response_data(
        self, record: RequestRecord
    ) -> Sequence[ParsedResponse]:
        """Extract parsed data from record.

        Args:
            record: Request record containing responses to parse

        Returns:
            Sequence of successfully parsed responses
        """
        results: list[ParsedResponse] = []
        for response in record.responses:
            if parsed := self.parse_response(response):
                results.append(parsed)
        return results

    async def send_request(
        self, request_info: RequestInfo, payload: RequestInputT
    ) -> RequestRecord:
        """Send request via transport."""
        request_info.endpoint_headers = self.get_endpoint_headers(request_info)
        request_info.endpoint_params = self.get_endpoint_params(request_info)
        return await self.transport.send_request(request_info, payload)
