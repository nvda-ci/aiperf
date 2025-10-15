# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aiohttp trace configuration factory.

This module provides factory functions for creating aiohttp TraceConfig objects
with comprehensive timestamp tracking callbacks.
"""

import time
import typing

import aiohttp

from aiperf.common.models import AioHttpTraceTimestamps

################################################################################
# AioHTTP Trace Config Factory
################################################################################


def create_trace_config(timestamps: AioHttpTraceTimestamps) -> aiohttp.TraceConfig:
    """Create a TraceConfig with comprehensive timestamp tracking callbacks.

    This function sets up all available tracing callbacks to capture detailed timing
    information about the HTTP request lifecycle, including connection pooling,
    DNS resolution, connection creation, and data transfer. Also extracts metadata
    from the trace params for enhanced observability.

    Args:
        timestamps: The AioHttpTraceTimestamps instance to populate with timing data.

    Returns:
        A configured aiohttp.TraceConfig instance with all callbacks registered.
    """
    trace_config = aiohttp.TraceConfig()

    # Connection Pool Callbacks (must be async)
    async def on_connection_queued_start(*_) -> None:
        timestamps.connection_queued_start_ns = time.perf_counter_ns()

    async def on_connection_queued_end(*_) -> None:
        timestamps.connection_queued_end_ns = time.perf_counter_ns()

    async def on_connection_create_start(*_) -> None:
        timestamps.connection_create_start_ns = time.perf_counter_ns()

    async def on_connection_create_end(*_) -> None:
        timestamps.connection_create_end_ns = time.perf_counter_ns()

    async def on_connection_reuseconn(*_) -> None:
        timestamps.connection_reuseconn_ns = time.perf_counter_ns()

    # DNS callbacks with metadata extraction
    async def on_dns_resolvehost_start(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        params: aiohttp.TraceDnsResolveHostStartParams,
    ) -> None:
        """Capture DNS resolution start time and hostname."""
        timestamps.dns_resolvehost_start_ns = time.perf_counter_ns()
        timestamps.dns_host = params.host

    async def on_dns_resolvehost_end(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        _params: aiohttp.TraceDnsResolveHostEndParams,
    ) -> None:
        """Capture DNS resolution end time."""
        timestamps.dns_resolvehost_end_ns = time.perf_counter_ns()

    async def on_dns_cache_hit(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        params: aiohttp.TraceDnsCacheHitParams,
    ) -> None:
        """Capture DNS cache hit time and hostname."""
        timestamps.dns_cache_hit_ns = time.perf_counter_ns()
        if not timestamps.dns_host:
            timestamps.dns_host = params.host

    async def on_dns_cache_miss(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        params: aiohttp.TraceDnsCacheMissParams,
    ) -> None:
        """Capture DNS cache miss time and hostname."""
        timestamps.dns_cache_miss_ns = time.perf_counter_ns()
        if not timestamps.dns_host:
            timestamps.dns_host = params.host

    # Request callbacks with metadata extraction
    async def on_request_start(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        params: aiohttp.TraceRequestStartParams,
    ) -> None:
        """Capture request start time and metadata."""
        timestamps.request_start_ns = time.perf_counter_ns()
        timestamps.request_method = params.method
        timestamps.request_url = str(params.url)

    async def on_request_headers_sent(*_) -> None:
        timestamps.request_headers_sent_ns = time.perf_counter_ns()

    trace_config.on_request_headers_sent.append(on_request_headers_sent)

    async def on_request_end(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        params: aiohttp.TraceRequestEndParams,
    ) -> None:
        """Capture request end time and response metadata."""
        timestamps.request_end_ns = time.perf_counter_ns()
        timestamps.response_status = params.response.status
        timestamps.response_reason = params.response.reason
        # Convert headers to a regular dict for serialization
        timestamps.response_headers = dict(params.response.headers)
        # Try to capture the actual connection host (may include resolved IP)
        try:
            if hasattr(params.response, "host") and params.response.host:
                timestamps.connection_host = params.response.host
        except Exception:
            pass  # Silently ignore if not available

    async def on_request_chunk_sent(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        params: aiohttp.TraceRequestChunkSentParams,
    ) -> None:
        """Capture request chunk timestamp and size."""
        timestamps.request_chunk_sent_ns.append(time.perf_counter_ns())
        timestamps.request_chunk_sizes.append(len(params.chunk))

    async def on_response_chunk_received(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        params: aiohttp.TraceResponseChunkReceivedParams,
    ) -> None:
        """Capture response chunk timestamp and size."""
        timestamps.response_chunk_received_ns.append(time.perf_counter_ns())
        timestamps.response_chunk_sizes.append(len(params.chunk))

    async def on_request_redirect(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        params: aiohttp.TraceRequestRedirectParams,
    ) -> None:
        """Capture redirect timestamp, URL, and status code."""
        timestamps.request_redirect_ns.append(time.perf_counter_ns())
        timestamps.redirect_urls.append(str(params.url))
        timestamps.redirect_status_codes.append(params.response.status)

    async def on_request_exception(
        _session: aiohttp.ClientSession,
        _trace_config_ctx: typing.Any,
        params: aiohttp.TraceRequestExceptionParams,
    ) -> None:
        """Capture exception timestamp and details."""
        timestamps.request_exception_ns = time.perf_counter_ns()
        timestamps.exception_type = type(params.exception).__name__
        timestamps.exception_message = str(params.exception)

    trace_config.on_connection_queued_start.append(on_connection_queued_start)
    trace_config.on_connection_queued_end.append(on_connection_queued_end)
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)
    trace_config.on_connection_reuseconn.append(on_connection_reuseconn)
    trace_config.on_dns_resolvehost_start.append(on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_dns_cache_hit.append(on_dns_cache_hit)
    trace_config.on_dns_cache_miss.append(on_dns_cache_miss)
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
    trace_config.on_response_chunk_received.append(on_response_chunk_received)
    trace_config.on_request_redirect.append(on_request_redirect)
    trace_config.on_request_exception.append(on_request_exception)

    return trace_config
