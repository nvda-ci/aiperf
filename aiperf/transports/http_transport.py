# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import orjson

from aiperf.common.enums import TransportType
from aiperf.common.factories import TransportFactory
from aiperf.common.hooks import on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails, RequestInfo, RequestRecord
from aiperf.transports.aiohttp_client import AioHttpClient
from aiperf.transports.base_transports import BaseTransport, TransportMetadata


@TransportFactory.register(TransportType.HTTP)
class HttpTransport(BaseTransport, AIPerfLifecycleMixin):
    """HTTP/1.1 transport implementation using aiohttp.

    Provides high-performance async HTTP client with:
    - Connection pooling and TCP optimization
    - SSE (Server-Sent Events) streaming support
    - Automatic error handling and timing
    - Custom TCP connector configuration
    """

    def __init__(
        self, tcp_kwargs: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Initialize HTTP transport with optional TCP configuration.

        Args:
            tcp_kwargs: TCP connector configuration (socket options, timeouts, etc.)
            **kwargs: Additional arguments passed to parent classes
        """
        super().__init__(**kwargs)
        self.aiohttp_client = AioHttpClient(tcp_kwargs=tcp_kwargs, **kwargs)

    @on_stop
    async def _close_aiohttp_client(self) -> None:
        """Cleanup hook to close aiohttp session on stop."""
        await self.aiohttp_client.close()

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return HTTP transport metadata."""
        return TransportMetadata(
            transport_type=TransportType.HTTP,
            url_schemes=["http", "https"],
        )

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Build HTTP-specific headers based on streaming mode.

        Args:
            request_info: Request context with endpoint configuration

        Returns:
            HTTP headers (Content-Type and Accept)
        """
        accept = (
            "text/event-stream"
            if request_info.model_endpoint.endpoint.streaming
            else "application/json"
        )
        return {"Content-Type": "application/json", "Accept": accept}

    def get_url(self, request_info: RequestInfo) -> str:
        """Build HTTP URL, adding http:// prefix if missing.

        Args:
            request_info: Request context with model endpoint URL

        Returns:
            Complete HTTP URL with scheme
        """
        url = request_info.model_endpoint.url
        # Add http:// prefix if no scheme specified
        return url if url.startswith("http") else f"http://{url}"

    async def send_request(
        self, request_info: RequestInfo, payload: dict[str, Any]
    ) -> RequestRecord:
        """Send HTTP POST request with JSON payload.

        Args:
            request_info: Request context and metadata
            payload: JSON-serializable request payload

        Returns:
            Request record with responses, timing, and any errors
        """
        start_perf_ns = time.perf_counter_ns()
        try:
            url = self.build_url(request_info)
            headers = self.build_headers(request_info)

            # Serialize with orjson for performance
            json_str = orjson.dumps(payload).decode("utf-8")

            record = await self.aiohttp_client.post_request(url, json_str, headers)

        except Exception as e:
            # Capture all exceptions with timing and error details
            record = RequestRecord(
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails(type=e.__class__.__name__, message=str(e)),
            )
            self.exception(f"HTTP request failed: {e}")

        return record
