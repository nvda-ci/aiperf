# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for aiohttp trace functionality."""

import asyncio
from unittest.mock import Mock

import aiohttp
import pytest

from aiperf.common.models import AioHttpTraceData
from aiperf.transports.aiohttp_trace import create_aiohttp_trace_config


class TestCreateAioHttpTraceConfig:
    """Tests for create_aiohttp_trace_config function."""

    def test_creates_trace_config(self):
        """Function creates aiohttp TraceConfig."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        assert isinstance(trace_config, aiohttp.TraceConfig)
        assert trace_data.reference_time_ns is not None
        assert trace_data.reference_perf_ns is not None

    def test_initializes_reference_timestamps(self):
        """Trace config initialization sets reference timestamps."""
        trace_data = AioHttpTraceData()
        create_aiohttp_trace_config(trace_data)

        assert trace_data.reference_time_ns is not None
        assert trace_data.reference_perf_ns is not None
        # time_ns should be much larger than perf_counter_ns
        assert trace_data.reference_time_ns > trace_data.reference_perf_ns

    @pytest.mark.asyncio
    async def test_connection_pool_events(self):
        """Connection pool events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        # Create mock parameters
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Simulate connection pool events
        await trace_config.on_connection_queued_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.connection_pool_wait_start_perf_ns is not None

        await trace_config.on_connection_queued_end[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.connection_pool_wait_end_perf_ns is not None
        assert (
            trace_data.connection_pool_wait_end_perf_ns
            >= trace_data.connection_pool_wait_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_dns_resolution_events(self):
        """DNS resolution events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # DNS resolution
        await trace_config.on_dns_resolvehost_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.dns_lookup_start_perf_ns is not None

        await trace_config.on_dns_resolvehost_end[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.dns_lookup_end_perf_ns is not None
        assert trace_data.dns_lookup_end_perf_ns >= trace_data.dns_lookup_start_perf_ns

    @pytest.mark.asyncio
    async def test_dns_cache_events(self):
        """DNS cache hit/miss events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Cache hit
        await trace_config.on_dns_cache_hit[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.dns_cache_hit_perf_ns is not None

        # Cache miss (separate trace data to avoid conflict)
        trace_data2 = AioHttpTraceData()
        trace_config2 = create_aiohttp_trace_config(trace_data2)
        await trace_config2.on_dns_cache_miss[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data2.dns_cache_miss_perf_ns is not None

    @pytest.mark.asyncio
    async def test_tcp_connection_events(self):
        """TCP connection creation events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_trace_ctx.url = "http://example.com"  # HTTP URL
        mock_params = Mock()

        # TCP connection
        await trace_config.on_connection_create_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.tcp_connect_start_perf_ns is not None

        await trace_config.on_connection_create_end[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.tcp_connect_end_perf_ns is not None
        assert (
            trace_data.tcp_connect_end_perf_ns >= trace_data.tcp_connect_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_connection_reuse_event(self):
        """Connection reuse event is tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        await trace_config.on_connection_reuseconn[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.connection_reused_perf_ns is not None

    @pytest.mark.asyncio
    async def test_request_events(self):
        """Request phase events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Request start
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.request_send_start_perf_ns is not None

        # Request headers sent
        mock_params.headers = {"content-type": "application/json"}
        await trace_config.on_request_headers_sent[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.request_headers_sent_perf_ns is not None
        assert trace_data.request_headers == {"content-type": "application/json"}

        # Request end (now captures response metadata)
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_params.response = mock_response
        await trace_config.on_request_end[0](mock_session, mock_trace_ctx, mock_params)
        assert trace_data.request_send_end_perf_ns is not None
        assert (
            trace_data.request_send_end_perf_ns >= trace_data.request_send_start_perf_ns
        )
        # Verify response metadata was captured
        assert trace_data.response_status_code == 200
        assert trace_data.response_headers == {"content-type": "application/json"}

    @pytest.mark.asyncio
    async def test_request_chunk_sent(self):
        """Request chunk sent events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()

        # Send multiple chunks
        for chunk_data in [b"chunk1", b"chunk2", b"chunk3"]:
            mock_params = Mock()
            mock_params.chunk = chunk_data
            await trace_config.on_request_chunk_sent[0](
                mock_session, mock_trace_ctx, mock_params
            )

        # Verify all chunks are tracked
        assert len(trace_data.request_write_timestamps_perf_ns) == 3
        assert len(trace_data.request_write_sizes_bytes) == 3
        assert trace_data.request_write_sizes_bytes == [6, 6, 6]  # len("chunkN")

        # Timestamps should be in increasing order
        assert (
            trace_data.request_write_timestamps_perf_ns[0]
            <= trace_data.request_write_timestamps_perf_ns[1]
        )
        assert (
            trace_data.request_write_timestamps_perf_ns[1]
            <= trace_data.request_write_timestamps_perf_ns[2]
        )

    @pytest.mark.asyncio
    async def test_response_chunk_received(self):
        """Response chunk received events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()

        # First chunk (note: response metadata is captured in on_request_end, not here)
        mock_params = Mock()
        mock_params.chunk = b"first chunk"

        await trace_config.on_response_chunk_received[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify first chunk tracked
        assert len(trace_data.response_receive_timestamps_perf_ns) == 1
        assert trace_data.response_receive_sizes_bytes[0] == 11  # len("first chunk")
        # Verify start/end timestamps are set
        assert trace_data.response_receive_start_perf_ns is not None
        assert trace_data.response_receive_end_perf_ns is not None

        # Second chunk
        mock_params.chunk = b"second chunk"
        await trace_config.on_response_chunk_received[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify second chunk tracked
        assert len(trace_data.response_receive_timestamps_perf_ns) == 2
        assert len(trace_data.response_receive_sizes_bytes) == 2
        assert trace_data.response_receive_sizes_bytes[1] == 12  # len("second chunk")
        # Verify end timestamp updated
        assert (
            trace_data.response_receive_end_perf_ns
            >= trace_data.response_receive_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_request_exception_event(self):
        """Request exception event is tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()
        mock_params.exception = Exception("Test error")

        await trace_config.on_request_exception[0](
            mock_session, mock_trace_ctx, mock_params
        )

        assert trace_data.error_timestamp_perf_ns is not None

    @pytest.mark.asyncio
    async def test_all_callbacks_registered(self):
        """All trace callbacks are registered on TraceConfig."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        # Connection pool
        assert len(trace_config.on_connection_queued_start) == 1
        assert len(trace_config.on_connection_queued_end) == 1

        # Connection creation
        assert len(trace_config.on_connection_create_start) == 1
        assert len(trace_config.on_connection_create_end) == 1

        # Connection reuse
        assert len(trace_config.on_connection_reuseconn) == 1

        # DNS
        assert len(trace_config.on_dns_resolvehost_start) == 1
        assert len(trace_config.on_dns_resolvehost_end) == 1
        assert len(trace_config.on_dns_cache_hit) == 1
        assert len(trace_config.on_dns_cache_miss) == 1

        # Request
        assert len(trace_config.on_request_start) == 1
        assert len(trace_config.on_request_chunk_sent) == 1
        assert len(trace_config.on_request_end) == 1
        assert len(trace_config.on_request_headers_sent) == 1

        # Response
        assert len(trace_config.on_response_chunk_received) == 1

        # Exception
        assert len(trace_config.on_request_exception) == 1


# Integration and edge case tests
class TestAioHttpTraceIntegration:
    """Integration tests for aiohttp trace functionality."""

    @pytest.mark.asyncio
    async def test_complete_request_lifecycle(self):
        """Complete request lifecycle with all connection and request events."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Simulate complete lifecycle
        # 1. Connection pool
        await trace_config.on_connection_queued_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        await trace_config.on_connection_queued_end[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # 2. DNS resolution
        await trace_config.on_dns_resolvehost_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        await trace_config.on_dns_resolvehost_end[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # 3. TCP connection (includes TLS for HTTPS)
        await trace_config.on_connection_create_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        await trace_config.on_connection_create_end[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # 4. Request
        mock_request_params = Mock()
        mock_request_params.url = "https://example.com"
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_request_params
        )
        mock_params.headers = {"user-agent": "test"}
        await trace_config.on_request_headers_sent[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Set up mock response for on_request_end
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_params.response = mock_response
        await trace_config.on_request_end[0](mock_session, mock_trace_ctx, mock_params)

        # 6. Response chunks
        mock_params.chunk = b"response data"
        await trace_config.on_response_chunk_received[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify complete trace data
        assert trace_data.connection_pool_wait_start_perf_ns is not None
        assert trace_data.dns_lookup_start_perf_ns is not None
        assert trace_data.tcp_connect_start_perf_ns is not None
        assert trace_data.request_send_start_perf_ns is not None
        assert trace_data.response_status_code == 200

        # Verify export works with all computed fields
        export = trace_data.to_export()
        assert export.blocked_ns is not None
        assert export.dns_lookup_ns is not None
        assert export.connecting_ns is not None

    @pytest.mark.asyncio
    async def test_connection_reuse_scenario(self):
        """Request using reused connection skips DNS/TCP."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Connection reuse scenario
        await trace_config.on_connection_reuseconn[0](
            mock_session, mock_trace_ctx, mock_params
        )
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify connection reuse recorded
        assert trace_data.connection_reused_perf_ns is not None

        # DNS and TCP should not be set for reused connection
        assert trace_data.dns_lookup_start_perf_ns is None
        assert trace_data.tcp_connect_start_perf_ns is None

    @pytest.mark.asyncio
    async def test_error_during_request(self):
        """Error during request captures error timestamp."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Start request
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Error occurs
        mock_params.exception = aiohttp.ClientError("Connection failed")
        await trace_config.on_request_exception[0](
            mock_session, mock_trace_ctx, mock_params
        )

        assert trace_data.request_send_start_perf_ns is not None
        assert trace_data.error_timestamp_perf_ns is not None
        assert (
            trace_data.error_timestamp_perf_ns >= trace_data.request_send_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_empty_response_body(self):
        """Response with no body still captures metadata."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Request with response metadata but no body chunks
        mock_request_params = Mock()
        mock_request_params.url = "http://example.com"
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_request_params
        )

        # Set up mock response for on_request_end
        mock_response = Mock()
        mock_response.status = 204  # No Content
        mock_response.headers = {}
        mock_params.response = mock_response
        await trace_config.on_request_end[0](mock_session, mock_trace_ctx, mock_params)

        # No response chunks received (empty body)
        assert len(trace_data.response_receive_timestamps_perf_ns) == 0
        # But response status is captured in on_request_end
        assert trace_data.response_status_code == 204

    @pytest.mark.asyncio
    async def test_multiple_trace_configs_independent(self):
        """Multiple trace configs operate independently."""
        trace_data1 = AioHttpTraceData()
        trace_data2 = AioHttpTraceData()

        trace_config1 = create_aiohttp_trace_config(trace_data1)
        trace_config2 = create_aiohttp_trace_config(trace_data2)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Trigger events on first config
        await trace_config1.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify only first trace data is affected
        assert trace_data1.request_send_start_perf_ns is not None
        assert trace_data2.request_send_start_perf_ns is None

        # Trigger events on second config
        await trace_config2.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Both should now have data
        assert trace_data1.request_send_start_perf_ns is not None
        assert trace_data2.request_send_start_perf_ns is not None

        # But values should be different (different timestamps)
        assert (
            trace_data1.request_send_start_perf_ns
            != trace_data2.request_send_start_perf_ns
        )


# Real-world async integration tests
@pytest.mark.integration
class TestRealWorldAsyncIntegration:
    """Real-world async integration tests using actual HTTP requests."""

    @pytest.mark.asyncio
    async def test_https_request_with_connection_timing(self):
        """Real HTTPS request tracks all timing (TCP+TLS combined in connection timing)."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        async with aiohttp.ClientSession(trace_configs=[trace_config]) as session:
            async with session.get("https://httpbin.org/get") as response:
                await response.text()

        # Verify all timing data was captured
        assert trace_data.request_send_start_perf_ns is not None
        assert trace_data.request_send_end_perf_ns is not None
        assert trace_data.response_status_code == 200
        assert len(trace_data.response_receive_timestamps_perf_ns) > 0

        # Verify TCP connection timing (includes TLS for HTTPS)
        assert trace_data.tcp_connect_start_perf_ns is not None
        assert trace_data.tcp_connect_end_perf_ns is not None
        assert trace_data.tcp_connect_end_perf_ns > trace_data.tcp_connect_start_perf_ns

        # Verify export works
        export = trace_data.to_export()
        assert export.total_ns is not None
        assert export.connecting_ns is not None  # TCP+TLS combined

    @pytest.mark.asyncio
    async def test_http_request_without_tls(self):
        """Real HTTP request tracks basic timing."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        async with aiohttp.ClientSession(trace_configs=[trace_config]) as session:
            async with session.get("http://httpbin.org/get") as response:
                await response.text()

        # Verify basic timing data
        assert trace_data.request_send_start_perf_ns is not None
        assert trace_data.response_status_code == 200

    @pytest.mark.asyncio
    async def test_multiple_concurrent_https_requests(self):
        """Multiple concurrent HTTPS requests each track their own timing."""
        urls = [
            "https://httpbin.org/delay/0",
            "https://httpbin.org/delay/0",
            "https://httpbin.org/delay/0",
        ]

        trace_data_list = []
        tasks = []

        for url in urls:
            trace_data = AioHttpTraceData()
            trace_config = create_aiohttp_trace_config(trace_data)
            trace_data_list.append(trace_data)

            async def make_request(url, trace_config):
                async with aiohttp.ClientSession(
                    trace_configs=[trace_config]
                ) as session:
                    async with session.get(url) as response:
                        return await response.text()

            tasks.append(make_request(url, trace_config))

        # Execute all requests concurrently
        await asyncio.gather(*tasks)

        # Verify each request has independent timing data
        for trace_data in trace_data_list:
            assert trace_data.request_send_start_perf_ns is not None
            assert trace_data.response_status_code == 200

        # Verify timing data is unique for each request
        start_times = [td.request_send_start_perf_ns for td in trace_data_list]
        assert len(set(start_times)) == len(start_times)  # All unique

    @pytest.mark.asyncio
    async def test_connection_reuse_in_session(self):
        """Connection reuse in a session shows different timing patterns."""
        trace_data1 = AioHttpTraceData()

        trace_config1 = create_aiohttp_trace_config(trace_data1)

        # First request - new connection
        async with aiohttp.ClientSession(trace_configs=[trace_config1]) as session:
            async with session.get("https://httpbin.org/get") as response:
                await response.text()

            # Second request in same session - may reuse connection
            async with session.get("https://httpbin.org/get") as response:
                # Note: Both requests use trace_config1, so trace_data1 captures
                # cumulative data from both requests
                await response.text()

        # First request should have full timing
        assert trace_data1.tcp_connect_start_perf_ns is not None
        assert trace_data1.request_send_start_perf_ns is not None

        # Note: For proper per-request connection reuse testing, we'd need
        # separate trace configs, but aiohttp doesn't support per-request traces

    @pytest.mark.asyncio
    async def test_large_response_chunk_tracking(self):
        """Large response chunk tracking verifies total bytes received."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        async with aiohttp.ClientSession(trace_configs=[trace_config]) as session:
            # Request a larger response (10KB)
            async with session.get("https://httpbin.org/bytes/10240") as response:
                await response.read()

        # Verify response chunks are tracked (may be 1 or more chunks depending on buffering)
        assert len(trace_data.response_receive_timestamps_perf_ns) >= 1
        assert len(trace_data.response_receive_sizes_bytes) >= 1
        assert sum(trace_data.response_receive_sizes_bytes) == 10240

        # Verify timestamps and sizes match
        assert len(trace_data.response_receive_timestamps_perf_ns) == len(
            trace_data.response_receive_sizes_bytes
        )

        # Verify export computed fields
        export = trace_data.to_export()
        assert export.receiving_ns is not None
        assert export.receiving_ns >= 0  # May be 0 if response is fast

    @pytest.mark.asyncio
    async def test_post_request_with_body(self):
        """POST request with body tracks request write timing."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        payload = {"test": "data" * 1000}  # Larger payload

        async with aiohttp.ClientSession(trace_configs=[trace_config]) as session:
            async with session.post(
                "https://httpbin.org/post", json=payload
            ) as response:
                await response.text()

        # Should capture request timing
        assert trace_data.request_send_start_perf_ns is not None
        assert trace_data.request_send_end_perf_ns is not None
        assert trace_data.response_status_code == 200

        # Verify export
        export = trace_data.to_export()
        assert export.sending_ns is not None
        assert export.sending_ns > 0
