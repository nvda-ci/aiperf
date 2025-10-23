# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base HTTP transport providing shared functionality for HTTP-based transports."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

import orjson

from aiperf.common.models import ErrorDetails, RequestInfo, RequestRecord
from aiperf.common.protocols import HTTPClientProtocol
from aiperf.transports.base_transports import BaseTransport


class BaseHTTPTransport(BaseTransport):
    """Base class for HTTP-based transports providing URL construction, header building,
    and request serialization.

    Subclasses must implement get_http_client() property and metadata() classmethod.
    """

    @property
    @abstractmethod
    def get_http_client(self) -> HTTPClientProtocol | None:
        """Get the HTTP client instance implementing HTTPClientProtocol, or None if not initialized."""
        ...

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Build HTTP headers (Content-Type and Accept) based on streaming mode."""
        accept = (
            "text/event-stream"
            if request_info.model_endpoint.endpoint.streaming
            else "application/json"
        )
        return {"Content-Type": "application/json", "Accept": accept}

    def get_url(self, request_info: RequestInfo) -> str:
        """Build complete HTTP URL from base_url and endpoint path (from metadata or custom endpoint)."""
        endpoint_info = request_info.model_endpoint.endpoint

        # Start with base URL
        base_url = endpoint_info.base_url.rstrip("/")

        # Determine the endpoint path
        if endpoint_info.custom_endpoint:
            # Use custom endpoint path if provided
            path = endpoint_info.custom_endpoint.lstrip("/")
            url = f"{base_url}/{path}"
        else:
            # Get endpoint path from endpoint metadata
            from aiperf.common.factories import EndpointFactory

            endpoint_metadata = EndpointFactory.get_metadata(endpoint_info.type)
            if not endpoint_metadata.endpoint_path:
                # No endpoint path, just use base URL
                url = base_url
            else:
                path = endpoint_metadata.endpoint_path.lstrip("/")
                # Handle /v1 base URL with v1/ path prefix to avoid duplication
                if base_url.endswith("/v1") and path.startswith("v1/"):
                    path = path.removeprefix("v1/")
                url = f"{base_url}/{path}"

        return url if url.startswith("http") else f"http://{url}"

    async def send_request(
        self, request_info: RequestInfo, payload: dict[str, Any]
    ) -> RequestRecord:
        """Send HTTP POST request with JSON payload.

        Args:
            request_info: Request context and metadata
            payload: JSON-serializable request payload

        Returns:
            RequestRecord with responses, timing, and any errors

        Raises:
            NotInitializedError: If transport not initialized before calling
        """
        http_client = self.get_http_client
        if http_client is None:
            from aiperf.common.exceptions import NotInitializedError

            raise NotInitializedError(
                f"{self.__class__.__name__} send_request failed: Client not initialized. Call initialize() before send_request()."
            )

        start_perf_ns = time.perf_counter_ns()
        try:
            url = self.build_url(request_info)
            headers = self.build_headers(request_info)

            # Serialize with orjson for performance
            json_str = orjson.dumps(payload).decode("utf-8")

            record = await http_client.post_request(url, json_str, headers)

        except Exception as e:
            # Capture all exceptions with timing and error details
            record = RequestRecord(
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )
            self.exception(f"{self.__class__.__name__} send_request failed: {e!r}")

        return record
