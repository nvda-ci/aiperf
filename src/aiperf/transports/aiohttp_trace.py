# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from time import perf_counter_ns, time_ns

import aiohttp

from aiperf.common.models import AioHttpTraceData


def create_aiohttp_trace_config(
    trace_data: AioHttpTraceData,
) -> aiohttp.TraceConfig:
    """Create a TraceConfig for aiohttp with comprehensive timestamp tracking.

    This function creates a TraceConfig that captures all aiohttp trace events
    and populates the AioHttpTraceData object with detailed timing data including:
    - Connection pool wait times
    - DNS resolution timing
    - TCP connection timing
    - Request/response timing with chunk-level granularity
    - Error timestamps

    Note: For HTTPS connections, the TCP connection timing will include both TCP
    handshake and TLS handshake as aiohttp provides no way to separate them.

    Args:
        trace_data: The AioHttpTraceData instance to populate with timing data.

    Returns:
        A TraceConfig object ready to be attached to an aiohttp ClientSession.

    Example:
        >>> trace_data = AioHttpTraceData()
        >>> trace_config = create_aiohttp_trace_config(trace_data)
        >>> async with aiohttp.ClientSession(trace_configs=[trace_config]) as session:
        ...     async with session.get("https://example.com") as response:
        ...         await response.text()
        >>> print(f"Total request time: {trace_data.to_export().total_ns}ns")
    """
    trace_config = aiohttp.TraceConfig()

    trace_data.reference_time_ns = time_ns()
    trace_data.reference_perf_ns = perf_counter_ns()

    # ============================================================================
    # CONNECTION POOL EVENTS
    # ============================================================================

    async def on_connection_queued_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceConnectionQueuedStartParams,
    ) -> None:
        """Called when request starts waiting for an available connection from pool."""
        trace_data.connection_pool_wait_start_perf_ns = perf_counter_ns()

    async def on_connection_queued_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceConnectionQueuedEndParams,
    ) -> None:
        """Called when request obtained an available connection from pool."""
        trace_data.connection_pool_wait_end_perf_ns = perf_counter_ns()

    # ============================================================================
    # CONNECTION CREATION EVENTS
    # ============================================================================

    async def on_connection_create_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceConnectionCreateStartParams,
    ) -> None:
        """Called when creation of a new connection starts."""
        trace_data.tcp_connect_start_perf_ns = perf_counter_ns()

    async def on_connection_create_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceConnectionCreateEndParams,
    ) -> None:
        """Called when creation of a new connection completes."""
        trace_data.tcp_connect_end_perf_ns = perf_counter_ns()

    # ============================================================================
    # CONNECTION REUSE EVENT
    # ============================================================================

    async def on_connection_reuseconn(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceConnectionReuseconnParams,
    ) -> None:
        """Called when an existing connection is reused from pool."""
        trace_data.connection_reused_perf_ns = perf_counter_ns()

    # ============================================================================
    # DNS RESOLUTION EVENTS
    # ============================================================================

    async def on_dns_resolvehost_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceDnsResolveHostStartParams,
    ) -> None:
        """Called when DNS resolution starts."""
        trace_data.dns_lookup_start_perf_ns = perf_counter_ns()

    async def on_dns_resolvehost_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceDnsResolveHostEndParams,
    ) -> None:
        """Called when DNS resolution completes."""
        trace_data.dns_lookup_end_perf_ns = perf_counter_ns()

    async def on_dns_cache_hit(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceDnsCacheHitParams,
    ) -> None:
        """Called when DNS cache hit occurs."""
        trace_data.dns_cache_hit_perf_ns = perf_counter_ns()

    async def on_dns_cache_miss(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceDnsCacheMissParams,
    ) -> None:
        """Called when DNS cache miss occurs."""
        trace_data.dns_cache_miss_perf_ns = perf_counter_ns()

    # ============================================================================
    # REQUEST EVENTS
    # ============================================================================

    async def on_request_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceRequestStartParams,
    ) -> None:
        """Called when HTTP request starts being sent."""
        trace_data.request_send_start_perf_ns = perf_counter_ns()

    async def on_request_chunk_sent(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceRequestChunkSentParams,
    ) -> None:
        """Called when each request chunk is sent."""
        trace_data.request_write_timestamps_perf_ns.append(perf_counter_ns())
        trace_data.request_write_sizes_bytes.append(len(params.chunk))

    async def on_request_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceRequestEndParams,
    ) -> None:
        """Called when HTTP request finishes being sent.

        Note: At this point, response headers have been received, so we capture
        response metadata here (status code and headers).
        """
        trace_data.request_send_end_perf_ns = perf_counter_ns()

        # Capture response metadata (status and headers are available at this point)
        trace_data.response_status_code = params.response.status
        trace_data.response_headers = dict(params.response.headers)

    async def on_request_headers_sent(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceRequestHeadersSentParams,
    ) -> None:
        """Called when request headers finish being sent."""
        trace_data.request_headers_sent_perf_ns = perf_counter_ns()
        trace_data.request_headers = dict(params.headers)

    # ============================================================================
    # RESPONSE EVENTS
    # ============================================================================

    async def on_response_chunk_received(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceResponseChunkReceivedParams,
    ) -> None:
        """Called when each response chunk is received.

        Note: Response metadata (status/headers) is captured in on_request_end,
        not here, since TraceResponseChunkReceivedParams doesn't include the response object.
        """
        current_time = perf_counter_ns()
        trace_data.response_receive_timestamps_perf_ns.append(current_time)
        trace_data.response_receive_sizes_bytes.append(len(params.chunk))

        # Track start/end timestamps for duration calculations
        if trace_data.response_receive_start_perf_ns is None:
            trace_data.response_receive_start_perf_ns = current_time
        trace_data.response_receive_end_perf_ns = current_time

    # ============================================================================
    # EXCEPTION EVENTS
    # ============================================================================

    async def on_request_exception(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.TraceRequestStartParams,
        params: aiohttp.TraceRequestExceptionParams,
    ) -> None:
        """Called when exception occurs during request."""
        trace_data.error_timestamp_perf_ns = perf_counter_ns()

    # ============================================================================
    # REGISTER ALL CALLBACKS
    # ============================================================================

    # Connection pool events
    trace_config.on_connection_queued_start.append(on_connection_queued_start)
    trace_config.on_connection_queued_end.append(on_connection_queued_end)

    # Connection creation events
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)

    # Connection reuse event
    trace_config.on_connection_reuseconn.append(on_connection_reuseconn)

    # DNS resolution events
    trace_config.on_dns_resolvehost_start.append(on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_dns_cache_hit.append(on_dns_cache_hit)
    trace_config.on_dns_cache_miss.append(on_dns_cache_miss)

    # Request events
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_headers_sent.append(on_request_headers_sent)

    # Response events
    trace_config.on_response_chunk_received.append(on_response_chunk_received)

    # Exception events
    trace_config.on_request_exception.append(on_request_exception)

    return trace_config
