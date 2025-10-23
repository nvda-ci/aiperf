# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock, patch

import httpcore
import pytest

from aiperf.common.models import RequestRecord, TextResponse
from aiperf.transports.httpcore_client import HttpCoreClient
from tests.transports.conftest import create_model_endpoint_info


@pytest.fixture
def model_endpoint():
    """Create a test model endpoint."""
    return create_model_endpoint_info(
        base_url="https://api.example.com", streaming=False
    )


@pytest.fixture
def streaming_endpoint():
    """Create a streaming model endpoint."""
    return create_model_endpoint_info(
        base_url="https://api.example.com", streaming=True
    )


@pytest.fixture
def client(model_endpoint):
    """Create an HttpCoreClient instance."""
    return HttpCoreClient(model_endpoint=model_endpoint)


@pytest.fixture
def streaming_client(streaming_endpoint):
    """Create a streaming HttpCoreClient instance."""
    return HttpCoreClient(model_endpoint=streaming_endpoint)


def create_mock_response(
    status: int = 200,
    content: bytes = b'{"result": "success"}',
    content_type: bytes = b"application/json",
) -> Mock:
    """Create a mock httpcore response."""
    mock_response = Mock()
    mock_response.status = status
    mock_response.headers = [(b"content-type", content_type)]
    mock_response.extensions = {"http_version": b"2"}

    async def aiter_stream_impl():
        if content:
            yield content

    mock_response.aiter_stream = aiter_stream_impl
    return mock_response


def create_chunked_mock_response(
    content: bytes, chunk_size: int = 1000, status: int = 200
) -> Mock:
    """Create a mock response that yields content in chunks."""
    mock_response = Mock()
    mock_response.status = status
    mock_response.headers = [(b"content-type", b"text/plain")]
    mock_response.extensions = {"http_version": b"2"}

    async def aiter_stream_impl():
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    mock_response.aiter_stream = aiter_stream_impl
    return mock_response


class TestHttpCoreClientInitialization:
    """Test HttpCoreClient initialization and configuration."""

    def test_init_creates_connection_pool(self, model_endpoint):
        """Test that initialization creates HTTP/2 connection pool."""
        client = HttpCoreClient(model_endpoint=model_endpoint)

        assert client.pool is not None
        assert client.timeout_seconds == 600.0

    def test_init_with_custom_timeout(self, model_endpoint):
        """Test initialization with custom timeout."""
        model_endpoint.endpoint.timeout = 300
        client = HttpCoreClient(model_endpoint=model_endpoint)

        assert client.timeout_seconds == 300

    @pytest.mark.asyncio
    async def test_close_clears_pool(self, model_endpoint):
        """Test that close() clears the connection pool."""
        client = HttpCoreClient(model_endpoint=model_endpoint)
        assert client.pool is not None

        await client.close()
        assert client.pool is None


class TestHttpCoreClientRequests:
    """Test HTTP request functionality."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,url,payload,headers",
        [
            (
                "post",
                "https://api.example.com/v1/chat/completions",
                '{"messages": [{"role": "user", "content": "Hi"}]}',
                {"Content-Type": "application/json"},
            ),
            (
                "get",
                "https://api.example.com/health",
                None,
                {"Accept": "application/json"},
            ),
        ],
    )
    async def test_request_success(self, client, method, url, payload, headers):
        """Test successful requests with proper timing."""
        mock_response = create_mock_response()

        with patch.object(client.pool, "stream") as mock_stream:
            mock_stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.return_value.__aexit__ = AsyncMock(return_value=None)

            if method == "post":
                record = await client.post_request(url, payload, headers)
            else:
                record = await client.get_request(url, headers)

            assert isinstance(record, RequestRecord)
            assert record.status == 200
            assert record.error is None
            assert len(record.responses) == 1
            assert isinstance(record.responses[0], TextResponse)
            assert record.start_perf_ns is not None
            assert record.recv_start_perf_ns is not None
            assert record.end_perf_ns is not None
            assert (
                record.end_perf_ns >= record.recv_start_perf_ns >= record.start_perf_ns
            )


class TestHttpCoreClientSSEStreaming:
    """Test Server-Sent Events (SSE) streaming support."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,sse_content,payload",
        [
            (
                "post",
                b"data: Hello\n\ndata: World\n\ndata: [DONE]\n\n",
                '{"stream": true}',
            ),
            ("get", b"data: test\n\n", None),
        ],
    )
    async def test_sse_stream_parsing(
        self, streaming_client, method, sse_content, payload
    ):
        """Test parsing of SSE streams with different methods."""
        mock_response = create_mock_response(
            content=sse_content, content_type=b"text/event-stream"
        )

        with patch.object(streaming_client.pool, "stream") as mock_stream:
            mock_stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.return_value.__aexit__ = AsyncMock(return_value=None)

            url = "https://api.example.com/v1/chat/completions"
            headers = {"Accept": "text/event-stream"}

            if method == "post":
                record = await streaming_client.post_request(url, payload, headers)
            else:
                record = await streaming_client.get_request(url, headers)

            assert isinstance(record, RequestRecord)
            assert record.status == 200
            assert record.error is None
            assert len(record.responses) > 0


class TestHttpCoreClientErrorHandling:
    """Test error handling for various failure scenarios."""

    @pytest.mark.asyncio
    async def test_http_error_status(self, client):
        """Test handling of HTTP error status codes."""
        mock_response = create_mock_response(
            status=404, content=b'{"error": "Not found"}'
        )

        with patch.object(client.pool, "stream") as mock_stream:
            mock_stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.return_value.__aexit__ = AsyncMock(return_value=None)

            record = await client.get_request(
                "https://api.example.com/invalid", {"Accept": "application/json"}
            )

            assert record.status == 404
            assert record.error is not None
            assert record.error.code == 404
            assert record.error.type == "HTTP 404"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception,expected_type,expected_msg_contains",
        [
            (httpcore.ConnectTimeout(), "ConnectTimeout", "timed out"),
            (httpcore.ReadTimeout(), "ReadTimeout", "timed out"),
            (httpcore.WriteTimeout(), "WriteTimeout", "timed out"),
            (httpcore.PoolTimeout(), "PoolTimeout", "pool"),
            (httpcore.ConnectError(), "ConnectError", None),
            (httpcore.RemoteProtocolError(), "RemoteProtocolError", None),
            (httpcore.LocalProtocolError(), "LocalProtocolError", None),
            (httpcore.ProtocolError(), "ProtocolError", None),
            (ValueError("Unexpected error"), None, "unexpected error"),
        ],
    )
    async def test_exception_handling(
        self, client, exception, expected_type, expected_msg_contains
    ):
        """Test handling of various httpcore exceptions."""
        with patch.object(client.pool, "stream") as mock_stream:
            mock_stream.side_effect = exception

            record = await client.get_request(
                "https://api.example.com/test", {"Accept": "application/json"}
            )

            assert record.error is not None
            if expected_type:
                assert record.error.type == expected_type
            if expected_msg_contains:
                assert expected_msg_contains in record.error.message.lower()


class TestHttpCoreClientResponseParsing:
    """Test response parsing for different content types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content,content_type,expected_length",
        [
            (b'{"id": "123", "result": "success"}', b"application/json", 34),
            (b"", b"text/plain", 0),
        ],
    )
    async def test_response_parsing(
        self, client, content, content_type, expected_length
    ):
        """Test parsing of various response types."""
        mock_response = create_mock_response(content=content, content_type=content_type)

        with patch.object(client.pool, "stream") as mock_stream:
            mock_stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.return_value.__aexit__ = AsyncMock(return_value=None)

            record = await client.get_request(
                "https://api.example.com/test", {"Accept": content_type.decode()}
            )

            assert record.status == 200
            assert len(record.responses) == 1
            response = record.responses[0]
            assert isinstance(response, TextResponse)
            assert len(response.text) == expected_length
            if content:
                assert response.content_type == content_type.decode()

    @pytest.mark.asyncio
    async def test_large_response_chunked(self, client):
        """Test handling of large responses with chunked streaming."""
        large_content = b"x" * 10000
        mock_response = create_chunked_mock_response(large_content)

        with patch.object(client.pool, "stream") as mock_stream:
            mock_stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.return_value.__aexit__ = AsyncMock(return_value=None)

            record = await client.get_request(
                "https://api.example.com/large", {"Accept": "text/plain"}
            )

            assert record.status == 200
            assert len(record.responses) == 1
            assert len(record.responses[0].text) == 10000
