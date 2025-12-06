# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base mixin for async HTTP metrics data collectors.

This mixin provides common functionality for collecting metrics from HTTP endpoints,
used by both GPU telemetry and server metrics systems.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import aiohttp

from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails


@dataclass
class HttpTraceTiming:
    """Timing data captured from aiohttp TraceConfig.

    Provides precise timestamps for HTTP request lifecycle events,
    enabling accurate correlation between client requests and server metrics.
    """

    request_start_ns: int | None = None
    first_byte_ns: int | None = None
    request_end_ns: int | None = None

    @property
    def transfer_time_ns(self) -> int | None:
        """Time to transfer response body after first byte received."""
        if self.first_byte_ns is not None and self.request_end_ns is not None:
            return self.request_end_ns - self.first_byte_ns
        return None


@dataclass
class FetchResult:
    """Result of fetching metrics from an HTTP endpoint."""

    text: str
    trace_timing: HttpTraceTiming


# Type variables for records returned by collectors
TRecord = TypeVar("TRecord")
TRecordCallback = TypeVar(
    "TRecordCallback", bound=Callable[[list[TRecord], str], Awaitable[None]]
)
TErrorCallback = TypeVar(
    "TErrorCallback", bound=Callable[[ErrorDetails, str], Awaitable[None]]
)


class BaseMetricsCollectorMixin(AIPerfLifecycleMixin, ABC, Generic[TRecord]):
    """Mixin providing async HTTP collection for metrics endpoints.

    This mixin encapsulates the pattern of periodically fetching metrics from
    HTTP endpoints, parsing them, and delivering them via callbacks.

    Common patterns:
        - aiohttp session management
        - Reachability testing
        - Background collection task with error handling
        - Callback-based delivery

    Used by:
        - TelemetryDataCollector (DCGM metrics)
        - ServerMetricsDataCollector (Prometheus metrics)
    """

    def __init__(
        self,
        endpoint_url: str,
        collection_interval: float,
        reachability_timeout: float,
        record_callback: TRecordCallback | None = None,
        error_callback: TErrorCallback | None = None,
        **kwargs,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            endpoint_url: URL of the metrics endpoint
            collection_interval: Interval in seconds between collections
            reachability_timeout: Timeout in seconds for reachability checks
            record_callback: Optional callback to receive collected records
            error_callback: Optional callback to receive collection errors
            **kwargs: Additional arguments passed to super().__init__()
        """
        self._endpoint_url = endpoint_url
        self._collection_interval = collection_interval
        self._reachability_timeout = reachability_timeout
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._session: aiohttp.ClientSession | None = None
        # Storage for trace timing data (keyed by trace_request_ctx)
        self._trace_timing: dict[object, HttpTraceTiming] = {}
        super().__init__(**kwargs)

    @property
    def endpoint_url(self) -> str:
        """Get the metrics endpoint URL."""
        return self._endpoint_url

    @property
    def collection_interval(self) -> float:
        """Get the collection interval in seconds."""
        return self._collection_interval

    @on_init
    async def _initialize_http_client(self) -> None:
        """Initialize the aiohttp client session with trace config.

        Called automatically during initialization phase.
        Creates an aiohttp ClientSession with appropriate timeout settings.
        Uses connect timeout only (no total timeout) to allow long-running scrapes.
        Configures TraceConfig to capture HTTP timing events for precise correlation.
        """
        timeout = aiohttp.ClientTimeout(
            total=None,  # No total timeout for ongoing scrapes
            connect=self._reachability_timeout,  # Fast connection timeout only
        )
        trace_config = self._create_trace_config()
        self._session = aiohttp.ClientSession(
            timeout=timeout, trace_configs=[trace_config]
        )

    def _create_trace_config(self) -> aiohttp.TraceConfig:
        """Create TraceConfig for HTTP timing capture.

        Captures:
        - request_start: When HTTP request headers are sent
        - response_chunk_received: First byte of response (TTFB proxy)
        - request_end: When response is fully received

        Returns:
            Configured TraceConfig instance
        """
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self._on_request_start)
        trace_config.on_response_chunk_received.append(self._on_response_chunk_received)
        trace_config.on_request_end.append(self._on_request_end)
        return trace_config

    async def _on_request_start(
        self,
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceRequestStartParams,
    ) -> None:
        """Capture timestamp when HTTP request starts."""
        ctx = trace_config_ctx.trace_request_ctx
        if ctx is not None:
            self._trace_timing[ctx] = HttpTraceTiming(request_start_ns=time.time_ns())

    async def _on_response_chunk_received(
        self,
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceResponseChunkReceivedParams,
    ) -> None:
        """Capture timestamp when first response byte is received (TTFB)."""
        ctx = trace_config_ctx.trace_request_ctx
        if ctx is not None and ctx in self._trace_timing:
            timing = self._trace_timing[ctx]
            # Only capture first byte (first chunk)
            if timing.first_byte_ns is None:
                timing.first_byte_ns = time.time_ns()

    async def _on_request_end(
        self,
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceRequestEndParams,
    ) -> None:
        """Capture timestamp when response is fully received."""
        ctx = trace_config_ctx.trace_request_ctx
        if ctx is not None and ctx in self._trace_timing:
            self._trace_timing[ctx].request_end_ns = time.time_ns()

    @on_stop
    async def _cleanup_http_client(self) -> None:
        """Clean up the aiohttp client session.

        Called automatically during shutdown phase.
        """
        if self._session:
            await self._session.close()
            self._session = None

    async def is_url_reachable(self) -> bool:
        """Check if metrics endpoint is accessible.

        Attempts HEAD request first for efficiency, falls back to GET if HEAD is not supported.
        Uses existing session if available, otherwise creates a temporary session.

        Returns:
            True if endpoint responds with HTTP 200, False otherwise
        """
        if not self._endpoint_url:
            return False

        # Use existing session if available, otherwise create a temporary one
        if self._session:
            return await self._check_reachability_with_session(self._session)
        else:
            # Create a temporary session for reachability check
            timeout = aiohttp.ClientTimeout(total=self._reachability_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as temp_session:
                return await self._check_reachability_with_session(temp_session)

    async def _check_reachability_with_session(
        self, session: aiohttp.ClientSession
    ) -> bool:
        """Check reachability using a specific session.

        Args:
            session: aiohttp session to use for the check

        Returns:
            True if endpoint is reachable with HTTP 200
        """
        try:
            # Try HEAD first for efficiency
            async with session.head(
                self._endpoint_url, allow_redirects=False
            ) as response:
                if response.status == 200:
                    return True
            # Fall back to GET if HEAD is not supported
            async with session.get(self._endpoint_url) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    @background_task(immediate=True, interval=lambda self: self.collection_interval)
    async def _collect_metrics_loop(self) -> None:
        """Background task for collecting metrics at regular intervals.

        This uses the @background_task decorator which automatically handles
        lifecycle management and stopping when the collector is stopped.
        """
        self.execute_async(self._collect_metrics_task())

    async def _collect_metrics_task(self) -> None:
        """Collect metrics from the endpoint."""
        try:
            await self._collect_and_process_metrics()
        except Exception as e:
            if self._error_callback:
                try:
                    await self._error_callback(
                        ErrorDetails.from_exception(e),
                        self.id,
                    )
                except Exception as callback_error:
                    self.error(f"Failed to send error via callback: {callback_error}")
            else:
                self.error(f"Metrics collection error: {e}")

    @abstractmethod
    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from endpoint and process them into records.

        Subclasses must implement this to:
        1. Fetch raw metrics data from the endpoint
        2. Parse data into record objects
        3. Send records via callback (if configured)
        """
        pass

    async def _fetch_metrics_text(self) -> FetchResult:
        """Fetch raw metrics text from the HTTP endpoint with trace timing.

        Performs safety checks before making HTTP request:
        - Verifies stop_requested flag to allow graceful shutdown
        - Checks session is initialized and not closed
        - Handles concurrent session closure gracefully

        Uses aiohttp TraceConfig to capture precise HTTP timing:
        - request_start_ns: When request headers were sent
        - first_byte_ns: Time-to-first-byte (TTFB) - best proxy for server snapshot time
        - transfer_time_ns: Time to transfer response body

        Returns:
            FetchResult containing raw metrics text and trace timing data

        Raises:
            RuntimeError: If HTTP session is not initialized
            aiohttp.ClientError: If HTTP request fails
            asyncio.CancelledError: If collector is being stopped or session is closed
        """
        if self.stop_requested:
            raise asyncio.CancelledError

        # Snapshot session to avoid race with _cleanup_http_client setting it to None
        session = self._session
        if not session:
            raise RuntimeError("HTTP session not initialized")

        # Create unique context for this request's trace timing
        trace_ctx = object()

        try:
            if session.closed:
                raise asyncio.CancelledError

            async with session.get(
                self._endpoint_url, trace_request_ctx=trace_ctx
            ) as response:
                response.raise_for_status()
                text = await response.text()

            # Retrieve and clean up trace timing data
            timing = self._trace_timing.pop(trace_ctx, HttpTraceTiming())
            return FetchResult(text=text, trace_timing=timing)
        except (aiohttp.ClientConnectionError, RuntimeError) as e:
            # Convert connection errors during shutdown to CancelledError
            if self.stop_requested or session.closed:
                raise asyncio.CancelledError from e
            raise
        finally:
            # Ensure cleanup for any exception (TimeoutError, CancelledError, etc.)
            # Safe to call even after success path pop - just returns None
            self._trace_timing.pop(trace_ctx, None)

    async def _send_records_via_callback(self, records: list[TRecord]) -> None:
        """Send records to the callback if configured.

        Args:
            records: List of records to send
        """
        if records and self._record_callback:
            try:
                await self._record_callback(records, self.id)
            except Exception as e:
                self.error(f"Failed to send records via callback: {e!r}", exc_info=True)
