# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FakeTransport."""

import pytest
from aiperf_mock_server.config import MockServerConfig

from aiperf.common.enums import (
    CreditPhase,
    EndpointType,
    ModelSelectionStrategy,
    TransportType,
)
from aiperf.common.models import RequestInfo, RequestRecord, SSEMessage, TextResponse
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from tests.harness.fake_transport import FakeTransport as FakeTransport


def create_model_endpoint(
    endpoint_type: EndpointType = EndpointType.CHAT,
    base_url: str = "mock://localhost:8000",
    streaming: bool = True,
    model_name: str = "test-model",
) -> ModelEndpointInfo:
    """Create a ModelEndpointInfo for testing."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name=model_name)],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=endpoint_type,
            base_url=base_url,
            streaming=streaming,
        ),
    )


def create_request_info(
    model_endpoint: ModelEndpointInfo,
    x_request_id: str = "test-request-id",
) -> RequestInfo:
    """Create a RequestInfo for testing."""
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        endpoint_headers={},
        endpoint_params={},
        turn_index=0,
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        x_request_id=x_request_id,
        x_correlation_id="test-correlation-id",
        conversation_id="test-conversation-id",
    )


class TestFakeTransportMetadata:
    """Test FakeTransport metadata."""

    def test_metadata_returns_correct_type(self):
        """Test metadata returns mock transport type."""
        metadata = FakeTransport.metadata()
        assert metadata.transport_type == TransportType.HTTP

    def test_metadata_url_schemes(self):
        """Test metadata has mock URL scheme."""
        metadata = FakeTransport.metadata()
        assert "http" in metadata.url_schemes
        assert "https" in metadata.url_schemes


class TestFakeTransportInit:
    """Test FakeTransport initialization."""

    def test_init_with_default_config(self):
        """Test transport uses global config when no config provided."""
        model_endpoint = create_model_endpoint()
        transport = FakeTransport(model_endpoint=model_endpoint)
        assert transport.config is not None

    def test_init_with_custom_config(self):
        """Test transport uses provided config."""
        model_endpoint = create_model_endpoint()
        custom_config = MockServerConfig(ttft=50, itl=10)
        transport = FakeTransport(model_endpoint=model_endpoint, config=custom_config)
        assert transport.config is custom_config
        assert transport.config.ttft == 50
        assert transport.config.itl == 10

    def test_get_url(self):
        """Test get_url returns mock URL."""
        model_endpoint = create_model_endpoint(base_url="mock://test-server")
        transport = FakeTransport(model_endpoint=model_endpoint)
        request_info = create_request_info(model_endpoint)
        url = transport.get_url(request_info)
        assert url == "mock://test-server"


class TestFakeTransportChat:
    """Test FakeTransport chat completion handlers."""

    @pytest.fixture
    def fast_config(self):
        """Config with minimal latency for fast tests."""
        return MockServerConfig(ttft=1, itl=1)

    @pytest.fixture
    def streaming_endpoint(self):
        """Create streaming chat endpoint."""
        return create_model_endpoint(
            endpoint_type=EndpointType.CHAT,
            streaming=True,
        )

    @pytest.fixture
    def non_streaming_endpoint(self):
        """Create non-streaming chat endpoint."""
        return create_model_endpoint(
            endpoint_type=EndpointType.CHAT,
            streaming=False,
        )

    @pytest.mark.asyncio
    async def test_streaming_chat_returns_sse_messages(
        self, streaming_endpoint, fast_config
    ):
        """Test streaming chat returns SSEMessage responses."""
        transport = FakeTransport(model_endpoint=streaming_endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(streaming_endpoint)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        assert record.status == 200
        assert len(record.responses) > 0
        # Streaming returns SSEMessage
        assert all(isinstance(r, SSEMessage) for r in record.responses)

    @pytest.mark.asyncio
    async def test_non_streaming_chat_returns_text_response(
        self, non_streaming_endpoint, fast_config
    ):
        """Test non-streaming chat returns TextResponse."""
        transport = FakeTransport(
            model_endpoint=non_streaming_endpoint, config=fast_config
        )
        await transport.initialize()

        request_info = create_request_info(non_streaming_endpoint)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        assert record.status == 200
        assert len(record.responses) == 1
        assert isinstance(record.responses[0], TextResponse)
        assert "chat.completion" in record.responses[0].text

    @pytest.mark.asyncio
    async def test_streaming_chat_with_first_token_callback(
        self, streaming_endpoint, fast_config
    ):
        """Test FirstTokenCallback is fired during streaming."""
        transport = FakeTransport(model_endpoint=streaming_endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(streaming_endpoint)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        callback_fired = False
        callback_ttft_ns = None

        async def first_token_callback(ttft_ns: int, message: SSEMessage) -> bool:
            nonlocal callback_fired, callback_ttft_ns
            callback_fired = True
            callback_ttft_ns = ttft_ns
            return True  # Mark as meaningful

        record = await transport.send_request(
            request_info, payload, first_token_callback=first_token_callback
        )

        assert callback_fired
        assert callback_ttft_ns is not None
        assert callback_ttft_ns > 0
        assert record.status == 200


class TestFakeTransportCompletion:
    """Test FakeTransport text completion handlers."""

    @pytest.fixture
    def fast_config(self):
        """Config with minimal latency for fast tests."""
        return MockServerConfig(ttft=1, itl=1)

    @pytest.fixture
    def streaming_endpoint(self):
        """Create streaming completion endpoint."""
        return create_model_endpoint(
            endpoint_type=EndpointType.COMPLETIONS,
            streaming=True,
        )

    @pytest.fixture
    def non_streaming_endpoint(self):
        """Create non-streaming completion endpoint."""
        return create_model_endpoint(
            endpoint_type=EndpointType.COMPLETIONS,
            streaming=False,
        )

    @pytest.mark.asyncio
    async def test_streaming_completion(self, streaming_endpoint, fast_config):
        """Test streaming text completion."""
        transport = FakeTransport(model_endpoint=streaming_endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(streaming_endpoint)
        payload = {
            "model": "test-model",
            "prompt": "Once upon a time",
            "stream": True,
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        assert len(record.responses) > 0
        assert all(isinstance(r, SSEMessage) for r in record.responses)

    @pytest.mark.asyncio
    async def test_non_streaming_completion(self, non_streaming_endpoint, fast_config):
        """Test non-streaming text completion."""
        transport = FakeTransport(
            model_endpoint=non_streaming_endpoint, config=fast_config
        )
        await transport.initialize()

        request_info = create_request_info(non_streaming_endpoint)
        payload = {
            "model": "test-model",
            "prompt": "Once upon a time",
            "stream": False,
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        assert len(record.responses) == 1
        assert isinstance(record.responses[0], TextResponse)
        assert "text_completion" in record.responses[0].text


class TestFakeTransportEmbedding:
    """Test FakeTransport embedding handlers."""

    @pytest.fixture
    def fast_config(self):
        """Config with minimal latency for fast tests."""
        return MockServerConfig(
            ttft=1, itl=1, embedding_base_latency=1, embedding_per_input_latency=0
        )

    @pytest.fixture
    def embedding_endpoint(self):
        """Create embedding endpoint."""
        return create_model_endpoint(endpoint_type=EndpointType.EMBEDDINGS)

    @pytest.mark.asyncio
    async def test_embedding_single_input(self, embedding_endpoint, fast_config):
        """Test embedding with single input."""
        transport = FakeTransport(model_endpoint=embedding_endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(embedding_endpoint)
        payload = {
            "model": "test-model",
            "input": ["Hello world"],
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        assert len(record.responses) == 1
        assert isinstance(record.responses[0], TextResponse)
        assert "embedding" in record.responses[0].text

    @pytest.mark.asyncio
    async def test_embedding_multiple_inputs(self, embedding_endpoint, fast_config):
        """Test embedding with multiple inputs."""
        transport = FakeTransport(model_endpoint=embedding_endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(embedding_endpoint)
        payload = {
            "model": "test-model",
            "input": ["Hello", "World", "Test"],
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        # Response should contain all 3 embeddings
        import orjson

        response_data = orjson.loads(record.responses[0].text)
        assert len(response_data["data"]) == 3


class TestFakeTransportRanking:
    """Test FakeTransport ranking handlers."""

    @pytest.fixture
    def fast_config(self):
        """Config with minimal latency for fast tests."""
        return MockServerConfig(
            ttft=1, itl=1, ranking_base_latency=1, ranking_per_passage_latency=0
        )

    @pytest.mark.asyncio
    async def test_nim_ranking(self, fast_config):
        """Test NIM ranking endpoint."""
        endpoint = create_model_endpoint(endpoint_type=EndpointType.NIM_RANKINGS)
        transport = FakeTransport(model_endpoint=endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(endpoint)
        # NIM format uses query.text and passages[].text
        payload = {
            "model": "test-model",
            "query": {"text": "What is AI?"},
            "passages": [
                {"text": "AI is artificial intelligence"},
                {"text": "Machine learning is a subset"},
            ],
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        import orjson

        response_data = orjson.loads(record.responses[0].text)
        assert "rankings" in response_data

    @pytest.mark.asyncio
    async def test_hf_tei_ranking(self, fast_config):
        """Test HF-TEI ranking endpoint."""
        endpoint = create_model_endpoint(endpoint_type=EndpointType.HF_TEI_RANKINGS)
        transport = FakeTransport(model_endpoint=endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(endpoint)
        # HF-TEI format uses query and texts (or documents)
        payload = {
            "model": "test-model",
            "query": "What is AI?",
            "texts": ["AI is artificial intelligence", "Machine learning is a subset"],
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        import orjson

        response_data = orjson.loads(record.responses[0].text)
        assert "results" in response_data

    @pytest.mark.asyncio
    async def test_cohere_ranking(self, fast_config):
        """Test Cohere ranking endpoint."""
        endpoint = create_model_endpoint(endpoint_type=EndpointType.COHERE_RANKINGS)
        transport = FakeTransport(model_endpoint=endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(endpoint)
        # Cohere format uses query and documents
        payload = {
            "model": "test-model",
            "query": "What is AI?",
            "documents": [
                "AI is artificial intelligence",
                "Machine learning is a subset",
            ],
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        import orjson

        response_data = orjson.loads(record.responses[0].text)
        assert "results" in response_data


class TestFakeTransportImageGeneration:
    """Test FakeTransport image generation handlers."""

    @pytest.fixture
    def fast_config(self):
        """Config with minimal latency for fast tests."""
        return MockServerConfig(ttft=1, itl=1)

    @pytest.fixture
    def image_endpoint(self):
        """Create image generation endpoint."""
        return create_model_endpoint(endpoint_type=EndpointType.IMAGE_GENERATION)

    @pytest.mark.asyncio
    async def test_image_generation(self, image_endpoint, fast_config):
        """Test image generation returns base64 data."""
        transport = FakeTransport(model_endpoint=image_endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(image_endpoint)
        payload = {
            "model": "test-model",
            "prompt": "A beautiful sunset",
            "n": 1,
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        assert len(record.responses) == 1
        import orjson

        response_data = orjson.loads(record.responses[0].text)
        assert "data" in response_data
        assert len(response_data["data"]) == 1


class TestFakeTransportConfigIsolation:
    """Test that FakeTransport properly isolates config per instance."""

    @pytest.mark.asyncio
    async def test_different_configs_different_latency(self):
        """Test that different configs result in different latency behavior."""
        # This test verifies config isolation by using different TTFT values
        fast_config = MockServerConfig(ttft=1, itl=1)
        slow_config = MockServerConfig(ttft=100, itl=50)

        endpoint = create_model_endpoint(streaming=False)
        fast_transport = FakeTransport(model_endpoint=endpoint, config=fast_config)
        slow_transport = FakeTransport(model_endpoint=endpoint, config=slow_config)

        await fast_transport.initialize()
        await slow_transport.initialize()

        # Verify configs are isolated
        assert fast_transport.config.ttft == 1
        assert slow_transport.config.ttft == 100

    @pytest.mark.asyncio
    async def test_config_not_shared_between_instances(self):
        """Test that modifying one transport's config doesn't affect another."""
        config1 = MockServerConfig(ttft=10, itl=5)
        config2 = MockServerConfig(ttft=20, itl=10)

        endpoint = create_model_endpoint()
        transport1 = FakeTransport(model_endpoint=endpoint, config=config1)
        transport2 = FakeTransport(model_endpoint=endpoint, config=config2)

        # Verify they have different configs
        assert transport1.config.ttft != transport2.config.ttft
        assert transport1.config.itl != transport2.config.itl


class TestFakeTransportPayloadFormats:
    """Test FakeTransport handles different payload formats."""

    @pytest.fixture
    def fast_config(self):
        """Config with minimal latency for fast tests."""
        return MockServerConfig(ttft=1, itl=1)

    @pytest.mark.asyncio
    async def test_dict_payload(self, fast_config):
        """Test transport handles dict payload."""
        endpoint = create_model_endpoint(streaming=False)
        transport = FakeTransport(model_endpoint=endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(endpoint)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        record = await transport.send_request(request_info, payload)
        assert record.status == 200

    @pytest.mark.asyncio
    async def test_json_bytes_payload(self, fast_config):
        """Test transport handles JSON bytes payload."""
        import orjson

        endpoint = create_model_endpoint(streaming=False)
        transport = FakeTransport(model_endpoint=endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(endpoint)
        payload = orjson.dumps(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            }
        )

        record = await transport.send_request(request_info, payload)
        assert record.status == 200


class TestFakeTransportTimestamps:
    """Test FakeTransport sets proper timestamps on records."""

    @pytest.fixture
    def fast_config(self):
        """Config with minimal latency for fast tests."""
        return MockServerConfig(ttft=1, itl=1)

    @pytest.mark.asyncio
    async def test_record_has_timestamps(self, fast_config):
        """Test RequestRecord has proper timestamps."""
        endpoint = create_model_endpoint(streaming=False)
        transport = FakeTransport(model_endpoint=endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(endpoint)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        record = await transport.send_request(request_info, payload)

        assert record.start_perf_ns is not None
        assert record.end_perf_ns is not None
        assert record.timestamp_ns is not None
        assert record.end_perf_ns >= record.start_perf_ns
