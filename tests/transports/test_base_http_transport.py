# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import TransportType
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.common.protocols import HTTPClientProtocol
from aiperf.transports.base_http_transport import BaseHTTPTransport
from aiperf.transports.base_transports import TransportMetadata
from tests.transports.conftest import create_model_endpoint_info


class MockHTTPTransport(BaseHTTPTransport):
    """Concrete implementation of BaseHTTPTransport for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mock_http_client = None

    async def initialize(self):
        """Initialize the mock HTTP client."""
        self.mock_http_client = AsyncMock(spec=HTTPClientProtocol)

    async def stop(self):
        """Stop the mock HTTP client."""
        self.mock_http_client = None

    @property
    def get_http_client(self) -> HTTPClientProtocol | None:
        """Get the mock HTTP client instance."""
        return self.mock_http_client

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return mock HTTP transport metadata."""
        return TransportMetadata(
            transport_type=TransportType.HTTP,
            url_schemes=["http", "https"],
        )


class TestBaseHTTPTransport:
    """Comprehensive tests for BaseHTTPTransport functionality."""

    @pytest.fixture
    def transport(self, model_endpoint_non_streaming):
        """Create a MockHTTPTransport instance."""
        return MockHTTPTransport(model_endpoint=model_endpoint_non_streaming)

    @pytest.fixture
    async def initialized_transport(self, transport):
        """Initialize transport and yield for testing."""
        await transport.initialize()
        yield transport
        await transport.stop()

    def _create_request_info(
        self,
        model_endpoint,
        endpoint_headers=None,
        endpoint_params=None,
        x_request_id=None,
        x_correlation_id=None,
    ):
        """Helper to create RequestInfo with defaults."""
        return RequestInfo(
            model_endpoint=model_endpoint,
            turns=[],
            endpoint_headers=endpoint_headers or {},
            endpoint_params=endpoint_params or {},
            x_request_id=x_request_id,
            x_correlation_id=x_correlation_id,
        )

    def _extract_call_args(self, mock_call_args):
        """Extract URL, JSON, and headers from mock call_args."""
        return {
            "url": mock_call_args[0][0],
            "json_str": mock_call_args[0][1],
            "headers": mock_call_args[0][2],
        }

    @pytest.mark.parametrize(
        "streaming,expected_accept",
        [(False, "application/json"), (True, "text/event-stream")],
        ids=["non-streaming", "streaming"],
    )
    def test_get_transport_headers(self, transport, streaming, expected_accept):
        """Test transport headers for different streaming modes."""
        model_endpoint = create_model_endpoint_info(streaming=streaming)
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[])
        headers = transport.get_transport_headers(request_info)

        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == expected_accept

    @pytest.mark.parametrize(
        "base_url,custom_endpoint,expected_url",
        [
            (
                "http://localhost:8000",
                "/v1/chat/completions",
                "http://localhost:8000/v1/chat/completions",
            ),
            ("localhost:8000", "/v1/chat", "http://localhost:8000/v1/chat"),
            ("https://api.example.com", "/v1/chat", "https://api.example.com/v1/chat"),
        ],
        ids=["http-prefix", "no-scheme", "https-prefix"],
    )
    def test_get_url(
        self, model_endpoint_non_streaming, base_url, custom_endpoint, expected_url
    ):
        """Test get_url with various base URLs and endpoints."""
        model_endpoint_non_streaming.endpoint.base_url = base_url
        model_endpoint_non_streaming.endpoint.custom_endpoint = custom_endpoint

        transport = MockHTTPTransport(model_endpoint=model_endpoint_non_streaming)
        request_info = RequestInfo(
            model_endpoint=model_endpoint_non_streaming, turns=[]
        )
        url = transport.get_url(request_info)
        assert url == expected_url

    @pytest.mark.asyncio
    async def test_send_request_success(self, transport, model_endpoint_non_streaming):
        """Test successful HTTP request."""
        await transport.initialize()

        mock_record = RequestRecord(responses=[], error=None)
        transport.mock_http_client.post_request = AsyncMock(return_value=mock_record)

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
        assert record.error is None
        transport.mock_http_client.post_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_builds_correct_url(
        self, initialized_transport, model_endpoint_non_streaming
    ):
        """Test that send_request builds URL correctly with params."""
        mock_record = RequestRecord()
        initialized_transport.mock_http_client.post_request = AsyncMock(
            return_value=mock_record
        )

        request_info = self._create_request_info(
            model_endpoint_non_streaming,
            endpoint_params={"api-version": "2024-10-01"},
        )
        payload = {"test": "data"}

        await initialized_transport.send_request(request_info, payload)

        args = self._extract_call_args(
            initialized_transport.mock_http_client.post_request.call_args
        )
        assert "api-version=2024-10-01" in args["url"]

    @pytest.mark.asyncio
    async def test_send_request_builds_correct_headers(
        self, initialized_transport, model_endpoint_non_streaming
    ):
        """Test that send_request builds headers correctly."""
        mock_record = RequestRecord()
        initialized_transport.mock_http_client.post_request = AsyncMock(
            return_value=mock_record
        )

        request_info = self._create_request_info(
            model_endpoint_non_streaming,
            endpoint_headers={"Authorization": "Bearer token123"},
            x_request_id="req-456",
        )
        payload = {"test": "data"}

        await initialized_transport.send_request(request_info, payload)

        args = self._extract_call_args(
            initialized_transport.mock_http_client.post_request.call_args
        )
        headers = args["headers"]

        assert headers["Authorization"] == "Bearer token123"
        assert headers["User-Agent"] == "aiperf/1.0"
        assert headers["X-Request-ID"] == "req-456"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    @pytest.mark.asyncio
    async def test_send_request_serializes_payload_with_orjson(
        self, initialized_transport, model_endpoint_non_streaming
    ):
        """Test that payload is serialized using orjson."""
        mock_record = RequestRecord()
        initialized_transport.mock_http_client.post_request = AsyncMock(
            return_value=mock_record
        )

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {"messages": [{"role": "user", "content": "Test"}], "model": "gpt-4"}

        await initialized_transport.send_request(request_info, payload)

        args = self._extract_call_args(
            initialized_transport.mock_http_client.post_request.call_args
        )
        json_str = args["json_str"]

        assert isinstance(json_str, str)
        assert "messages" in json_str
        assert "gpt-4" in json_str

    @pytest.mark.asyncio
    async def test_send_request_handles_exception(
        self, initialized_transport, model_endpoint_non_streaming
    ):
        """Test that exceptions are caught and recorded."""
        initialized_transport.mock_http_client.post_request = AsyncMock(
            side_effect=ValueError("Test error")
        )

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {"test": "data"}

        record = await initialized_transport.send_request(request_info, payload)

        assert record.error is not None
        assert record.error.type == "ValueError"
        assert "Test error" in record.error.message
        assert record.start_perf_ns is not None
        assert record.end_perf_ns is not None

    @pytest.mark.asyncio
    async def test_send_request_timing_on_error(
        self, initialized_transport, model_endpoint_non_streaming
    ):
        """Test that timing is recorded even on errors."""
        initialized_transport.mock_http_client.post_request = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {"test": "data"}

        record = await initialized_transport.send_request(request_info, payload)

        assert record.start_perf_ns is not None
        assert record.end_perf_ns is not None
        assert record.end_perf_ns >= record.start_perf_ns
        assert record.error is not None

    @pytest.mark.asyncio
    async def test_send_request_streaming_headers(
        self, model_endpoint_streaming, initialized_transport
    ):
        """Test correct headers for streaming requests."""
        initialized_transport.model_endpoint = model_endpoint_streaming
        mock_record = RequestRecord()
        initialized_transport.mock_http_client.post_request = AsyncMock(
            return_value=mock_record
        )

        request_info = RequestInfo(
            model_endpoint=model_endpoint_streaming,
            turns=[],
            endpoint_headers={},
            endpoint_params={},
        )
        payload = {"stream": True}

        await initialized_transport.send_request(request_info, payload)

        call_args = initialized_transport.mock_http_client.post_request.call_args
        headers = call_args[0][2]
        assert headers["Accept"] == "text/event-stream"

    @pytest.mark.asyncio
    async def test_send_request_empty_payload(
        self, initialized_transport, model_endpoint_non_streaming
    ):
        """Test send_request with empty payload."""
        mock_record = RequestRecord()
        initialized_transport.mock_http_client.post_request = AsyncMock(
            return_value=mock_record
        )

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {}

        record = await initialized_transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        args = self._extract_call_args(
            initialized_transport.mock_http_client.post_request.call_args
        )
        assert args["json_str"] == "{}"

    @pytest.mark.asyncio
    async def test_send_request_complex_payload(
        self, initialized_transport, model_endpoint_non_streaming
    ):
        """Test send_request with complex nested payload."""
        mock_record = RequestRecord()
        initialized_transport.mock_http_client.post_request = AsyncMock(
            return_value=mock_record
        )

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Response"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                },
            ],
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 500,
        }

        record = await initialized_transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        args = self._extract_call_args(
            initialized_transport.mock_http_client.post_request.call_args
        )
        json_str = args["json_str"]
        assert "messages" in json_str
        assert "image_url" in json_str
        assert "0.7" in json_str

    @pytest.mark.asyncio
    async def test_send_request_not_initialized(
        self, transport, model_endpoint_non_streaming
    ):
        """Test that send_request raises error when not initialized."""
        from aiperf.common.exceptions import NotInitializedError

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {"test": "data"}

        with pytest.raises(NotInitializedError, match="Client not initialized"):
            await transport.send_request(request_info, payload)


class TestBaseHTTPTransportURLConstruction:
    """Tests specifically for URL construction in BaseHTTPTransport."""

    @pytest.mark.parametrize(
        "base_url,custom_endpoint,expected",
        [
            ("http://localhost:8000", "/v1/chat", "http://localhost:8000/v1/chat"),
            ("http://localhost:8000/", "/v1/chat", "http://localhost:8000/v1/chat"),
            ("http://localhost:8000", "v1/chat", "http://localhost:8000/v1/chat"),
            (
                "http://localhost:8000/v1",
                "v1/chat",
                "http://localhost:8000/v1/v1/chat",
            ),
            (
                "http://localhost:8000/v1",
                "/v1/chat/completions",
                "http://localhost:8000/v1/v1/chat/completions",
            ),
        ],
        ids=[
            "standard",
            "trailing-slash-base",
            "no-leading-slash-endpoint",
            "v1-in-base-and-endpoint-custom",
            "full-path-with-v1-custom",
        ],
    )
    def test_get_url_path_construction(
        self, model_endpoint_non_streaming, base_url, custom_endpoint, expected
    ):
        """Test URL path construction with various configurations.

        Note: v1 deduplication only works with endpoint metadata paths,
        not with custom endpoints.
        """
        model_endpoint_non_streaming.endpoint.base_url = base_url
        model_endpoint_non_streaming.endpoint.custom_endpoint = custom_endpoint

        transport = MockHTTPTransport(model_endpoint=model_endpoint_non_streaming)
        request_info = RequestInfo(
            model_endpoint=model_endpoint_non_streaming, turns=[]
        )
        url = transport.get_url(request_info)
        assert url == expected

    def test_get_url_without_custom_endpoint(self, model_endpoint_non_streaming):
        """Test get_url uses endpoint metadata path when no custom endpoint."""
        model_endpoint_non_streaming.endpoint.base_url = "http://localhost:8000"
        model_endpoint_non_streaming.endpoint.custom_endpoint = None

        transport = MockHTTPTransport(model_endpoint=model_endpoint_non_streaming)
        request_info = RequestInfo(
            model_endpoint=model_endpoint_non_streaming, turns=[]
        )
        url = transport.get_url(request_info)
        assert url.startswith("http://localhost:8000")


class TestBaseHTTPTransportIntegration:
    """Integration tests for BaseHTTPTransport with full request flow."""

    @pytest.mark.asyncio
    async def test_full_request_flow_non_streaming(self):
        """Test complete request flow for non-streaming."""
        model_endpoint = create_model_endpoint_info(
            base_url="https://api.example.com",
            api_key="test-key",
            headers=[("Custom-Header", "value")],
        )

        transport = MockHTTPTransport(model_endpoint=model_endpoint)
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
        transport.mock_http_client.post_request = AsyncMock(return_value=mock_record)

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
        }

        await transport.send_request(request_info, payload)

        assert transport.mock_http_client.post_request.called
        args = {
            "url": transport.mock_http_client.post_request.call_args[0][0],
            "json_str": transport.mock_http_client.post_request.call_args[0][1],
            "headers": transport.mock_http_client.post_request.call_args[0][2],
        }

        assert "https://api.example.com/v1/chat/completions" in args["url"]
        assert "api-version=2024-10-01" in args["url"]
        assert "Hello" in args["json_str"]
        assert args["headers"]["Authorization"] == "Bearer test-key"
        assert args["headers"]["Custom-Header"] == "value"
        assert args["headers"]["X-Request-ID"] == "req-123"
        assert args["headers"]["X-Correlation-ID"] == "corr-456"
        assert args["headers"]["Accept"] == "application/json"

        await transport.stop()
