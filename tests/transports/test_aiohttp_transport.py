# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import TransportType
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.transports.conftest import create_model_endpoint_info


class TestAioHttpTransport:
    """Tests for AioHttp-specific transport functionality."""

    @pytest.fixture
    def transport(self, model_endpoint_non_streaming):
        """Create an AioHttpTransport instance."""
        return AioHttpTransport(model_endpoint=model_endpoint_non_streaming)

    @pytest.fixture
    def transport_with_tcp_kwargs(self, model_endpoint_non_streaming):
        """Create an AioHttpTransport with custom TCP settings."""
        tcp_kwargs = {"limit": 200, "limit_per_host": 50}
        return AioHttpTransport(
            model_endpoint=model_endpoint_non_streaming, tcp_kwargs=tcp_kwargs
        )

    @pytest.mark.asyncio
    async def test_init_with_default_tcp_kwargs(self, transport):
        """Test initialization with default TCP kwargs."""
        assert transport.tcp_kwargs is None
        assert transport.aiohttp_client is None

    @pytest.mark.asyncio
    async def test_init_with_custom_tcp_kwargs(self, transport_with_tcp_kwargs):
        """Test initialization with custom TCP kwargs."""
        assert transport_with_tcp_kwargs.tcp_kwargs is not None
        assert transport_with_tcp_kwargs.tcp_kwargs["limit"] == 200
        assert transport_with_tcp_kwargs.tcp_kwargs["limit_per_host"] == 50

    @pytest.mark.asyncio
    async def test_init_hook_creates_aiohttp_client(self, transport):
        """Test that lifecycle initialize creates AioHttpClient."""
        await transport.initialize()
        assert transport.aiohttp_client is not None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_hook_closes_aiohttp_client(self, transport):
        """Test that lifecycle stop closes AioHttpClient."""
        await transport.initialize()
        assert transport.aiohttp_client is not None

        await transport.stop()
        assert transport.aiohttp_client is None

    @pytest.mark.asyncio
    async def test_stop_hook_handles_none_client(self, transport):
        """Test that stop hook handles None client gracefully."""
        await transport.stop()
        assert transport.aiohttp_client is None

    def test_metadata(self, transport):
        """Test metadata returns correct AioHttp transport info."""
        metadata = transport.metadata()
        assert metadata.transport_type == TransportType.HTTP
        assert "http" in metadata.url_schemes
        assert "https" in metadata.url_schemes

    def test_get_http_client_property(self, transport):
        """Test get_http_client property returns aiohttp_client."""
        assert transport.get_http_client is None

    @pytest.mark.asyncio
    async def test_get_http_client_after_init(self, transport):
        """Test get_http_client property returns aiohttp_client after init."""
        await transport.initialize()
        assert transport.get_http_client is not None
        assert transport.get_http_client is transport.aiohttp_client
        await transport.stop()


class TestAioHttpTransportLifecycle:
    """Test lifecycle management of AioHttpTransport."""

    @pytest.mark.asyncio
    async def test_init_creates_client(self, model_endpoint_non_streaming):
        """Test that init creates aiohttp client."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        assert transport.aiohttp_client is None

        await transport.initialize()
        assert transport.aiohttp_client is not None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_client(self, model_endpoint_non_streaming):
        """Test that stop closes aiohttp client."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.initialize()

        client = transport.aiohttp_client
        assert client is not None

        await transport.stop()
        assert transport.aiohttp_client is None

    @pytest.mark.asyncio
    async def test_multiple_init_calls(self, model_endpoint_non_streaming):
        """Test that multiple init calls are handled correctly."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)

        await transport.initialize()
        first_client = transport.aiohttp_client

        await transport.initialize()
        second_client = transport.aiohttp_client

        assert second_client is not None
        assert first_client is not None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_without_init(self, model_endpoint_non_streaming):
        """Test that stop works if init was never called."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.stop()
        assert transport.aiohttp_client is None

    @pytest.mark.asyncio
    async def test_client_timeout_configuration(self, model_endpoint_non_streaming):
        """Test that AioHttpClient is configured with correct timeout."""
        model_endpoint_non_streaming.endpoint.timeout = 300
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)

        await transport.initialize()

        assert transport.aiohttp_client is not None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_tcp_kwargs_passed_to_client(self, model_endpoint_non_streaming):
        """Test that tcp_kwargs are passed to AioHttpClient."""
        tcp_kwargs = {"limit": 100, "limit_per_host": 25}
        transport = AioHttpTransport(
            model_endpoint=model_endpoint_non_streaming, tcp_kwargs=tcp_kwargs
        )

        await transport.initialize()

        assert transport.aiohttp_client is not None
        await transport.stop()


class TestAioHttpTransportIntegration:
    """Integration tests for AioHttpTransport with AioHttpClient."""

    @pytest.mark.asyncio
    async def test_send_request_uses_aiohttp_client(self, model_endpoint_non_streaming):
        """Test that send_request uses the AioHttpClient."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        request_info = RequestInfo(
            model_endpoint=model_endpoint_non_streaming,
            turns=[],
            endpoint_headers={"Authorization": "Bearer token"},
            endpoint_params={},
        )
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        transport.aiohttp_client.post_request.assert_called_once()
        await transport.stop()

    @pytest.mark.asyncio
    async def test_full_request_flow_with_aiohttp_client(self):
        """Test complete request flow using AioHttpClient."""
        model_endpoint = create_model_endpoint_info(
            base_url="https://api.example.com",
            api_key="test-key",
            headers=[("Custom-Header", "value")],
        )

        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        request_info = RequestInfo(
            model_endpoint=model_endpoint,
            turns=[],
            endpoint_headers={
                "Authorization": "Bearer test-key",
                "Custom-Header": "value",
            },
            endpoint_params={"api-version": "2024-10-01"},
            x_request_id="req-123",
            x_correlation_id="corr-456",
        )

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
        }

        await transport.send_request(request_info, payload)

        assert transport.aiohttp_client.post_request.called
        call_args = transport.aiohttp_client.post_request.call_args[0]

        url, json_str, headers = call_args[0], call_args[1], call_args[2]

        assert "https://api.example.com/v1/chat/completions" in url
        assert "api-version=2024-10-01" in url
        assert "Hello" in json_str
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Custom-Header"] == "value"
        assert headers["X-Request-ID"] == "req-123"
        assert headers["X-Correlation-ID"] == "corr-456"

        await transport.stop()
