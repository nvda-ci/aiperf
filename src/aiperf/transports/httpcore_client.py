# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ssl
import time
from typing import Any

import httpcore

from aiperf.common.constants import AIPERF_HTTP_CONNECTION_LIMIT, AIPERF_TLS_VERIFY
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    ErrorDetails,
    ModelEndpointInfo,
    RequestRecord,
    TextResponse,
)
from aiperf.transports.http_defaults import HttpCoreDefaults, SocketDefaults
from aiperf.transports.sse_utils import AsyncSSEStreamReader

################################################################################
# HTTPCore Client with HTTP/2 Support
################################################################################


class HttpCoreClient(AIPerfLoggerMixin):
    """High-performance HTTP client using httpcore with HTTP/2 multiplexing support.

    Supports multiple concurrent streams over a single TCP connection, with automatic
    protocol negotiation (HTTP/2 or fallback to HTTP/1.1).
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs: Any) -> None:
        """Initialize the httpcore client with HTTP/2 support and connection pooling.

        Args:
            model_endpoint: Model endpoint configuration containing URL and timeout
            **kwargs: Additional arguments passed to parent AIPerfLoggerMixin
        """
        self.model_endpoint = model_endpoint
        super().__init__(model_endpoint=model_endpoint, **kwargs)

        # Calculate connection pool size for HTTP/2 multiplexing
        # Strategy: AIPERF_HTTP_CONNECTION_LIMIT / streams_per_connection
        max_connections = HttpCoreDefaults.calculate_max_connections()

        self.debug(
            lambda: f"Initializing httpcore client: {max_connections} connections, "
            f"~{max_connections * HttpCoreDefaults.STREAMS_PER_CONNECTION} stream capacity"
        )

        # Setup SSL context for HTTPS connections
        ssl_context = ssl.create_default_context()

        # Disable certificate verification if configured
        # WARNING: This is insecure and should only be used for testing
        if not AIPERF_TLS_VERIFY:
            self.warning("TLS certificate verification is DISABLED - this is insecure!")
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        # Create async connection pool with HTTP/2 support
        # Note: http2=True enables HTTP/2 for both HTTPS (via ALPN) and plain HTTP (via prior knowledge/h2c)
        # if the server doesn't support HTTP/2, it will automatically fallback to HTTP/1.1
        self.pool = httpcore.AsyncConnectionPool(
            http1=HttpCoreDefaults.HTTP1,
            http2=HttpCoreDefaults.HTTP2,
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            keepalive_expiry=HttpCoreDefaults.KEEPALIVE_EXPIRY,
            retries=HttpCoreDefaults.RETRIES,
            socket_options=SocketDefaults.build_socket_options(),
            ssl_context=ssl_context,
        )

        # Store timeout for request-level timeout configuration
        # This is used in extensions for each request
        self.timeout_seconds = self.model_endpoint.endpoint.timeout

        self.debug(lambda: "httpcore client initialized successfully")

    async def close(self) -> None:
        """Close the connection pool and cleanup all resources."""
        if self.pool:
            self.debug(lambda: "Closing httpcore connection pool")
            await self.pool.aclose()
            self.pool = None
            self.debug(lambda: "httpcore connection pool closed")

    async def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: str | None = None,
        **kwargs: Any,
    ) -> RequestRecord:
        """Execute HTTP requests with nanosecond-precision timing and error handling.

        Automatically detects and handles SSE streams. All exceptions are caught and
        converted to ErrorDetails in the returned RequestRecord.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL with scheme (e.g., https://example.com/path)
            headers: Request headers dict
            data: Optional request body string
            **kwargs: Additional arguments for future extension

        Returns:
            RequestRecord with status, timing data, responses, and optional error
        """
        self.debug(lambda: f"Sending {method} request to {url}")

        # Initialize request record with start timestamp
        record = RequestRecord(start_perf_ns=time.perf_counter_ns())

        try:
            # Detect if this is an SSE streaming request
            # SSE detection: Check if Accept header requests text/event-stream
            is_sse_request = headers.get("Accept", "").startswith("text/event-stream")

            # Convert headers dict to httpcore format: list of (bytes, bytes) tuples
            # httpcore requires headers as [(b"name", b"value"), ...]
            httpcore_headers = [
                (name.encode("utf-8"), value.encode("utf-8"))
                for name, value in headers.items()
            ]

            # Convert request body to bytes if provided
            # httpcore expects bytes or async iterable for content
            content: bytes | None = data.encode("utf-8") if data else None

            # Configure request-level timeout using extensions
            # httpcore uses extensions dict for per-request configuration
            # Timeout format: {"timeout": {"connect": float, "read": float, "write": float, "pool": float}}
            #
            # Timeout types:
            #   - connect: Time to establish TCP connection
            #   - read: Time to read response data
            #   - write: Time to send request data
            #   - pool: Time to acquire connection from pool
            extensions: dict[str, Any] = {
                "timeout": {
                    "connect": self.timeout_seconds,  # Connection timeout
                    "read": self.timeout_seconds,  # Read timeout
                    "write": self.timeout_seconds,  # Write timeout
                    "pool": 60.0,  # Pool acquisition timeout (generous for high concurrency)
                }
            }

            # Make HTTP request using httpcore's streaming API
            # Using stream() instead of request() for incremental processing
            # This allows us to:
            #   1. Capture timestamp of first byte received
            #   2. Process response body incrementally (memory efficient)
            #   3. Parse SSE messages as they arrive
            async with self.pool.stream(
                method=method.encode("utf-8"),
                url=url.encode("utf-8"),
                headers=httpcore_headers,
                content=content,
                extensions=extensions,
            ) as response:
                # Capture status code
                # httpcore returns status as int
                record.status = response.status

                # Capture first byte timestamp (start of response)
                # This is set before reading any response data
                record.recv_start_perf_ns = time.perf_counter_ns()

                self.debug(
                    lambda: f"Response status: {record.status}, "
                    f"HTTP version: HTTP/{response.extensions.get('http_version', b'').decode()}"
                )

                # Check for HTTP error status codes
                # We don't raise exceptions, but capture errors in RequestRecord
                if record.status != 200:
                    # Read error response body for detailed error message
                    error_body = b""
                    async for chunk in response.aiter_stream():
                        error_body += chunk

                    error_text = error_body.decode("utf-8", errors="replace")

                    record.error = ErrorDetails(
                        code=record.status,
                        type=f"HTTP {record.status}",
                        message=error_text or f"HTTP {record.status} error",
                    )
                    record.end_perf_ns = time.perf_counter_ns()

                    self.debug(
                        lambda: f"HTTP error {record.status}: {error_text[:100]}"
                    )

                    return record

                # Parse response headers to detect SSE content type
                # httpcore returns headers as list of (bytes, bytes) tuples
                response_headers = {
                    name.decode("utf-8").lower(): value.decode("utf-8")
                    for name, value in response.headers
                }
                content_type = response_headers.get("content-type", "")

                # Handle Server-Sent Events (SSE) streaming
                # SSE is detected by both request Accept header and response Content-Type
                if is_sse_request and content_type.startswith("text/event-stream"):
                    self.debug(lambda: "Processing SSE stream")

                    # Parse SSE stream incrementally
                    # Each message gets its own timestamp for accurate TTFT/TPOT measurements
                    async for message in AsyncSSEStreamReader(response.aiter_stream()):
                        record.responses.append(message)

                    self.debug(lambda: f"Parsed {len(record.responses)} SSE messages")

                # Handle regular (non-streaming) response
                else:
                    self.debug(lambda: "Processing regular response")

                    # Read complete response body
                    # aiter_stream() provides raw bytes as they arrive
                    response_body = b""
                    async for chunk in response.aiter_stream():
                        response_body += chunk

                    # Decode response body to string
                    # Use 'replace' error handling to avoid UnicodeDecodeError
                    response_text = response_body.decode("utf-8", errors="replace")

                    # Capture end timestamp after reading all data
                    record.end_perf_ns = time.perf_counter_ns()

                    # Create TextResponse object
                    record.responses.append(
                        TextResponse(
                            perf_ns=record.end_perf_ns,
                            content_type=content_type,
                            text=response_text,
                        )
                    )

                    self.debug(lambda: f"Response complete: {len(response_text)} bytes")

                # Set final end timestamp if not already set
                # (SSE sets this in the streaming loop)
                if not record.end_perf_ns:
                    record.end_perf_ns = time.perf_counter_ns()

        # Handle httpcore timeout exceptions
        # httpcore has specific timeout exception types for different timeout scenarios
        except httpcore.ConnectTimeout as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Connection timeout: {e!r}")
            record.error = ErrorDetails(
                type="ConnectTimeout",
                message=f"Connection to {url} timed out after {self.timeout_seconds}s",
            )

        except httpcore.ReadTimeout as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Read timeout: {e!r}")
            record.error = ErrorDetails(
                type="ReadTimeout",
                message=f"Reading response from {url} timed out after {self.timeout_seconds}s",
            )

        except httpcore.WriteTimeout as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Write timeout: {e!r}")
            record.error = ErrorDetails(
                type="WriteTimeout",
                message=f"Sending request to {url} timed out after {self.timeout_seconds}s",
            )

        except httpcore.PoolTimeout as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Pool timeout: {e!r}")
            record.error = ErrorDetails(
                type="PoolTimeout",
                message=(
                    f"No available connection in pool after 60s. "
                    f"Consider increasing AIPERF_HTTP_CONNECTION_LIMIT (current: {AIPERF_HTTP_CONNECTION_LIMIT})"
                ),
            )

        except httpcore.TimeoutException as e:
            # Generic timeout (catch-all for timeout types not handled above)
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Request timeout: {e!r}")
            record.error = ErrorDetails(
                type="TimeoutError",
                message=f"Request to {url} timed out: {e!r}",
            )

        # Handle connection errors
        except httpcore.ConnectError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Connection error: {e!r}")
            record.error = ErrorDetails(
                type="ConnectError",
                message=f"Failed to connect to {url}: {e!r}",
            )

        # Handle HTTP/2 protocol errors
        except httpcore.RemoteProtocolError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Remote protocol error: {e!r}")
            record.error = ErrorDetails(
                type="RemoteProtocolError",
                message=f"Server sent invalid HTTP/2 frames: {e!r}",
            )

        except httpcore.LocalProtocolError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Local protocol error: {e!r}")
            record.error = ErrorDetails(
                type="LocalProtocolError",
                message=f"Client attempted invalid HTTP/2 operation: {e!r}",
            )

        except httpcore.ProtocolError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Protocol error: {e!r}")
            record.error = ErrorDetails(
                type="ProtocolError",
                message=f"HTTP/2 protocol error: {e!r}",
            )

        # Handle any other unexpected errors
        except Exception as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Unexpected error in HTTP request: {e!r}")
            record.error = ErrorDetails.from_exception(e)

        return record

    async def post_request(
        self,
        url: str,
        payload: str,
        headers: dict[str, str],
        **kwargs: Any,
    ) -> RequestRecord:
        """Send an HTTP POST request with optional SSE streaming support.

        Args:
            url: Target URL with scheme (e.g., https://api.example.com/v1/chat)
            payload: Request body string
            headers: HTTP headers dict
            **kwargs: Additional arguments passed to _request()

        Returns:
            RequestRecord with status, timing data, responses (TextResponse or SSEMessage), and optional error
        """
        return await self._request("POST", url, headers, data=payload, **kwargs)

    async def get_request(
        self, url: str, headers: dict[str, str], **kwargs: Any
    ) -> RequestRecord:
        """Send an HTTP GET request.

        Args:
            url: Target URL with scheme (e.g., https://api.example.com/health)
            headers: HTTP headers dict
            **kwargs: Additional arguments passed to _request()

        Returns:
            RequestRecord with status, timing data, responses (TextResponse), and optional error
        """
        return await self._request("GET", url, headers, **kwargs)
