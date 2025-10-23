# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiperf.common.enums import TransportType
from aiperf.common.factories import TransportFactory
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.protocols import HTTPClientProtocol
from aiperf.transports.aiohttp_client import AioHttpClient
from aiperf.transports.base_http_transport import BaseHTTPTransport
from aiperf.transports.base_transports import TransportMetadata


@TransportFactory.register(TransportType.HTTP)
class AioHttpTransport(BaseHTTPTransport):
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
        self.tcp_kwargs = tcp_kwargs
        self.aiohttp_client = None

    @on_init
    async def _init_aiohttp_client(self) -> None:
        """Initialize the AioHttpClient."""
        self.aiohttp_client = AioHttpClient(
            timeout=self.model_endpoint.endpoint.timeout, tcp_kwargs=self.tcp_kwargs
        )

    @on_stop
    async def _close_aiohttp_client(self) -> None:
        """Cleanup hook to close aiohttp session on stop."""
        if self.aiohttp_client:
            await self.aiohttp_client.close()
            self.aiohttp_client = None

    @property
    def get_http_client(self) -> HTTPClientProtocol | None:
        """Get the AioHttp client instance.

        Returns:
            AioHttpClient instance or None if not initialized
        """
        return self.aiohttp_client

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return HTTP transport metadata."""
        return TransportMetadata(
            transport_type=TransportType.HTTP,
            url_schemes=["http", "https"],
        )
