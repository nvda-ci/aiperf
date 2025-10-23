# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from aiperf.common.enums import TransportType
from aiperf.common.factories import TransportFactory
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.protocols import HTTPClientProtocol
from aiperf.transports.base_http_transport import BaseHTTPTransport
from aiperf.transports.base_transports import TransportMetadata
from aiperf.transports.httpcore_client import HttpCoreClient


@TransportFactory.register(TransportType.HTTP2)
class HttpCoreTransport(BaseHTTPTransport):
    """HTTP/2 transport implementation using httpcore.

    This module provides a production-grade HTTP transport with HTTP/2 multiplexing support,
    offering significantly higher concurrency than HTTP/1.1-only implementations.

    Key Features:
        - HTTP/2 multiplexing: 100 streams per connection
        - Connection pooling: 25 connections (configurable)
        - Total capacity: 2,500+ concurrent requests
        - Automatic HTTP/1.1 fallback for compatibility
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize HTTP/2 transport.

        Args:
            **kwargs: Additional arguments passed to parent BaseTransport
        """
        super().__init__(**kwargs)
        self.httpcore_client = None

    @on_init
    async def _init_httpcore_client(self) -> None:
        """Initialize the HttpCoreClientMixin with HTTP/2 support."""
        self.httpcore_client = HttpCoreClient(
            model_endpoint=self.model_endpoint,
        )

    @on_stop
    async def _close_httpcore_client(self) -> None:
        """Cleanup hook to close httpcore connection pool on stop."""
        if self.httpcore_client:
            await self.httpcore_client.close()
            self.httpcore_client = None

    @property
    def get_http_client(self) -> HTTPClientProtocol | None:
        """Get the httpcore client instance.

        Returns:
            HttpCoreClientMixin instance or None if not initialized
        """
        return self.httpcore_client

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return HTTP/2 transport metadata.

        Returns:
            Metadata describing transport type and supported URL schemes
        """
        return TransportMetadata(
            transport_type=TransportType.HTTP2,
            url_schemes=["http", "https"],
        )
