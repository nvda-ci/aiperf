# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import TransportType
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.transports.httpcore_transport import HttpCoreTransport
from tests.transports.conftest import create_model_endpoint_info


@pytest.fixture
def transport(model_endpoint_non_streaming):
    """Create an HttpCoreTransport instance."""
    return HttpCoreTransport(model_endpoint=model_endpoint_non_streaming)


@pytest.fixture
def streaming_transport(model_endpoint_streaming):
    """Create a streaming HttpCoreTransport instance."""
    return HttpCoreTransport(model_endpoint=model_endpoint_streaming)


class TestHttpCoreTransportLifecycle:
    """Test lifecycle management and initialization of HttpCoreTransport."""

    @pytest.mark.asyncio
    async def test_init_creates_none_client(self, transport):
        """Test initialization starts with None client."""
        assert transport.httpcore_client is None

    @pytest.mark.asyncio
    async def test_initialize_creates_client(self, transport):
        """Test that initialize creates HttpCoreClient with connection pool."""
        await transport.initialize()

        assert transport.httpcore_client is not None
        assert transport.httpcore_client.pool is not None
        assert transport.get_http_client is transport.httpcore_client

        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_client(self, transport):
        """Test that stop closes httpcore client and clears reference."""
        await transport.initialize()
        assert transport.httpcore_client is not None

        await transport.stop()

        assert transport.httpcore_client is None
        assert transport.get_http_client is None

    @pytest.mark.asyncio
    async def test_stop_without_init(self, transport):
        """Test that stop handles None client gracefully."""
        await transport.stop()
        assert transport.httpcore_client is None

    @pytest.mark.asyncio
    async def test_multiple_init_calls(self, model_endpoint_non_streaming):
        """Test that multiple init calls maintain valid client."""
        transport = HttpCoreTransport(model_endpoint=model_endpoint_non_streaming)

        await transport.initialize()
        first_client = transport.httpcore_client

        await transport.initialize()
        second_client = transport.httpcore_client

        assert second_client is not None
        assert first_client is not None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_client_timeout_configuration(self, model_endpoint_non_streaming):
        """Test that HttpCoreClient is configured with correct timeout."""
        model_endpoint_non_streaming.endpoint.timeout = 300
        transport = HttpCoreTransport(model_endpoint=model_endpoint_non_streaming)

        await transport.initialize()

        assert transport.httpcore_client.timeout_seconds == 300
        await transport.stop()

    def test_metadata(self, transport):
        """Test metadata returns correct HTTP/2 transport info."""
        metadata = transport.metadata()
        assert metadata.transport_type == TransportType.HTTP2
        assert "http" in metadata.url_schemes
        assert "https" in metadata.url_schemes


class TestHttpCoreTransportIntegration:
    """Integration tests for HttpCoreTransport with HttpCoreClient."""

    @pytest.mark.asyncio
    async def test_send_request_uses_httpcore_client(self, transport):
        """Test that send_request delegates to HttpCoreClient."""
        await transport.initialize()

        mock_record = RequestRecord()
        transport.httpcore_client.post_request = AsyncMock(return_value=mock_record)

        request_info = RequestInfo(
            model_endpoint=transport.model_endpoint,
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
        transport.httpcore_client.post_request.assert_called_once()
        await transport.stop()

    @pytest.mark.asyncio
    async def test_request_with_headers_and_params(self):
        """Test complete request flow with headers, params, and correlation IDs."""
        model_endpoint = create_model_endpoint_info(
            base_url="https://api.example.com",
            api_key="test-key",
            headers=[("Custom-Header", "value")],
        )

        transport = HttpCoreTransport(model_endpoint=model_endpoint)
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
        transport.httpcore_client.post_request = AsyncMock(return_value=mock_record)

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
        }

        await transport.send_request(request_info, payload)

        call_args = transport.httpcore_client.post_request.call_args[0]
        url, json_str, headers = call_args[0], call_args[1], call_args[2]

        assert "https://api.example.com/v1/chat/completions" in url
        assert "api-version=2024-10-01" in url
        assert "Hello" in json_str
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Custom-Header"] == "value"
        assert headers["X-Request-ID"] == "req-123"
        assert headers["X-Correlation-ID"] == "corr-456"

        await transport.stop()

    @pytest.mark.asyncio
    async def test_streaming_request_sets_sse_accept_header(self, streaming_transport):
        """Test that streaming requests set text/event-stream Accept header."""
        await streaming_transport.initialize()

        mock_record = RequestRecord()
        streaming_transport.httpcore_client.post_request = AsyncMock(
            return_value=mock_record
        )

        request_info = RequestInfo(
            model_endpoint=streaming_transport.model_endpoint,
            turns=[],
            endpoint_headers={},
            endpoint_params={},
        )
        payload = {"stream": True}

        await streaming_transport.send_request(request_info, payload)

        call_args = streaming_transport.httpcore_client.post_request.call_args
        headers = call_args[0][2]
        assert headers["Accept"] == "text/event-stream"

        await streaming_transport.stop()
