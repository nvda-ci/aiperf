# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import ParsedResponse, TextResponseData
from aiperf.workers.worker import Worker


class MockWorker(Worker):
    """Mock implementation of Worker for testing."""

    def __init__(self):
        with (
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector"
            ) as mock_tcp_connector,
            patch(
                "aiperf.common.factories.EndpointFactory.create_instance"
            ) as mock_client_factory,
        ):
            mock_tcp_connector.return_value = Mock()

            mock_client = Mock()
            mock_client.send_request = AsyncMock()
            mock_client_factory.return_value = mock_client

            super().__init__(
                service_config=ServiceConfig(),
                user_config=UserConfig(
                    endpoint=EndpointConfig(model_names=["test-model"]),
                ),
                service_id="mock-service-id",
            )


@pytest.mark.asyncio
class TestWorker:
    @pytest.fixture
    def worker(self):
        """Create a mock Worker for testing."""
        return MockWorker()

    async def test_process_response(self, monkeypatch, worker, sample_request_record):
        """Ensure process_response extracts text correctly from RequestRecord."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text="Hello, world!"),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
        turn = await worker._process_response(sample_request_record)
        assert turn.texts[0].contents == ["Hello, world!"]

    async def test_process_response_empty(
        self, monkeypatch, worker, sample_request_record
    ):
        """Ensure process_response handles empty responses correctly."""
        mock_parsed_response = ParsedResponse(
            perf_ns=0,
            data=TextResponseData(text=""),
        )
        mock_endpoint = Mock()
        mock_endpoint.extract_response_data = Mock(return_value=[mock_parsed_response])
        monkeypatch.setattr(worker.inference_client, "endpoint", mock_endpoint)
        turn = await worker._process_response(sample_request_record)
        assert turn is None
