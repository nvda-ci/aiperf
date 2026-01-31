# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the API service."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from aiohttp.web import Response

from aiperf.api.api_service import APIService
from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.enums import CommandType, MessageType
from aiperf.common.messages import CommandErrorResponse, GetAPIStatusResponse
from aiperf.common.models import ErrorDetails, MetricResult


def get_response_json(response: Response) -> dict[str, Any]:
    """Extract JSON from aiohttp Response."""
    body = response.body
    if hasattr(body, "_value"):
        body = body._value
    return orjson.loads(body)


def get_response_text(response: Response) -> str:
    """Extract text from aiohttp Response."""
    body = response.body
    if hasattr(body, "_value"):
        body = body._value
    if isinstance(body, bytes):
        body = body.decode()
    return body


@pytest.fixture
def service_config() -> ServiceConfig:
    """Create a ServiceConfig instance."""
    return ServiceConfig(api_port=9999, api_host="127.0.0.1")


@pytest.fixture
def user_config() -> UserConfig:
    """Create a UserConfig instance."""
    return UserConfig(
        benchmark_id="test-bench",
        endpoint=EndpointConfig(model_names=["test-model"]),
    )


@pytest.fixture
def api_service(service_config: ServiceConfig, user_config: UserConfig) -> APIService:
    """Create an APIService instance for testing."""
    with patch.object(APIService, "__init__", lambda self, *args, **kwargs: None):
        service = APIService.__new__(APIService)
        service.service_config = service_config
        service.user_config = user_config
        service.service_id = "api-test-1"
        service.api_host = "127.0.0.1"
        service.api_port = 9999
        service._info_labels = None
        service._metrics = []
        service._metrics_lock = asyncio.Lock()
        service._phase_progress_map = {}
        service._phase_progress_lock = asyncio.Lock()
        service._workers_stats = {}
        service._workers_stats_lock = asyncio.Lock()
        service.connection_manager = MagicMock()
        service.client_subscriptions = {}
        service.zmq_subscriptions = set()
        service._handled_message_types = {
            str(MessageType.REALTIME_METRICS),
            str(MessageType.REALTIME_TELEMETRY_METRICS),
            str(MessageType.CREDIT_PHASE_START),
            str(MessageType.CREDIT_PHASE_PROGRESS),
            str(MessageType.CREDIT_PHASE_COMPLETE),
            str(MessageType.WORKER_STATUS_SUMMARY),
            str(MessageType.PROCESSING_STATS),
            str(MessageType.ALL_RECORDS_RECEIVED),
        }
        return service


class TestHTTPEndpoints:
    """Test HTTP API endpoints."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, api_service: APIService) -> None:
        """Test health endpoint returns ok."""
        response = await api_service._handle_health(MagicMock())
        assert response.status == 200
        assert response.text == "ok"

    @pytest.mark.asyncio
    async def test_config_returns_json(self, api_service: APIService) -> None:
        """Test config endpoint returns JSON config."""
        response = await api_service._handle_config(MagicMock())
        assert response.status == 200
        assert response.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_prometheus_empty_metrics(self, api_service: APIService) -> None:
        """Test Prometheus endpoint with no metrics."""
        response = await api_service._handle_prometheus_metrics(MagicMock())
        assert response.status == 200
        assert response.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_prometheus_with_metrics(self, api_service: APIService) -> None:
        """Test Prometheus endpoint with metrics."""
        api_service._metrics = [
            MetricResult(tag="latency", header="Latency", unit="ms", avg=100.0)
        ]
        response = await api_service._handle_prometheus_metrics(MagicMock())
        assert response.status == 200
        assert "aiperf_latency" in get_response_text(response)

    @pytest.mark.asyncio
    async def test_json_metrics_empty(self, api_service: APIService) -> None:
        """Test JSON metrics endpoint with no metrics."""
        response = await api_service._handle_json_metrics(MagicMock())
        assert response.status == 200
        assert response.content_type == "application/json"
        assert get_response_json(response)["metrics"] == {}

    @pytest.mark.asyncio
    async def test_json_metrics_with_data(self, api_service: APIService) -> None:
        """Test JSON metrics endpoint with metrics."""
        api_service._metrics = [
            MetricResult(tag="latency", header="Latency", unit="ms", avg=100.0)
        ]
        response = await api_service._handle_json_metrics(MagicMock())
        data = get_response_json(response)
        assert data["metrics"]["latency"]["avg"] == 100.0

    @pytest.mark.asyncio
    async def test_progress_empty(self, api_service: APIService) -> None:
        """Test progress endpoint with no progress data."""
        response = await api_service._handle_progress(MagicMock())
        assert response.status == 200
        data = get_response_json(response)
        assert data["total"] == 0
        assert data["completed"] == 0
        assert data["percent_complete"] == 0

    @pytest.mark.asyncio
    async def test_workers_empty(self, api_service: APIService) -> None:
        """Test workers endpoint with no workers."""
        response = await api_service._handle_workers(MagicMock())
        assert response.status == 200
        data = get_response_json(response)
        assert data["total_workers"] == 0
        assert data["active_workers"] == 0
        assert data["worker_statuses"] == {}

    @pytest.mark.asyncio
    async def test_status_success(self, api_service: APIService) -> None:
        """Test status endpoint with successful response."""
        api_service.send_command_and_wait_for_response = AsyncMock(
            return_value=GetAPIStatusResponse(
                service_id="test",
                command_id="cmd-1",
                state="running",
                phase="benchmark",
                profile_id="profile-1",
                error=None,
            )
        )
        response = await api_service._handle_status(MagicMock())
        assert response.status == 200
        data = get_response_json(response)
        assert data["state"] == "running"
        assert data["phase"] == "benchmark"
        assert data["profile_id"] == "profile-1"

    @pytest.mark.asyncio
    async def test_status_error(self, api_service: APIService) -> None:
        """Test status endpoint with error response."""
        api_service.send_command_and_wait_for_response = AsyncMock(
            return_value=CommandErrorResponse(
                service_id="test",
                command=CommandType.GET_API_STATUS,
                command_id="cmd-1",
                error=ErrorDetails(message="Something went wrong"),
            )
        )
        response = await api_service._handle_status(MagicMock())
        assert response.status == 500


class TestWebSocketClientMessage:
    """Test WebSocket client message handling."""

    @pytest.mark.asyncio
    async def test_subscribe_message(self, api_service: APIService) -> None:
        """Test handling subscribe message."""
        client_id = "test-client"
        api_service.client_subscriptions[client_id] = set()
        api_service.comms = MagicMock()
        api_service.comms.sub_client = MagicMock()
        api_service.comms.sub_client.subscribe = AsyncMock()
        api_service.info = MagicMock()
        api_service.warning = MagicMock()

        ws = MagicMock()
        ws.send_json = AsyncMock()

        await api_service._handle_client_message(
            client_id, ws, {"type": "subscribe", "message_types": ["realtime_metrics"]}
        )

        ws.send_json.assert_called_once()
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "subscribed"
        assert "realtime_metrics" in call_args["message_types"]

    @pytest.mark.asyncio
    async def test_unsubscribe_message(self, api_service: APIService) -> None:
        """Test handling unsubscribe message."""
        client_id = "test-client"
        api_service.client_subscriptions[client_id] = {"realtime_metrics", "other"}
        api_service.info = MagicMock()

        await api_service._handle_client_message(
            client_id,
            MagicMock(),
            {"type": "unsubscribe", "message_types": ["realtime_metrics"]},
        )

        assert "realtime_metrics" not in api_service.client_subscriptions[client_id]
        assert "other" in api_service.client_subscriptions[client_id]

    @pytest.mark.asyncio
    async def test_ping_message(self, api_service: APIService) -> None:
        """Test handling ping message."""
        ws = MagicMock()
        ws.send_json = AsyncMock()

        await api_service._handle_client_message("test-client", ws, {"type": "ping"})

        ws.send_json.assert_called_once_with({"type": "pong"})


class TestForwardMessage:
    """Test message forwarding to WebSocket clients."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "subscriptions,message_type,expected_clients",
        [
            (  # Subscribed client receives message
                {"client1": {"realtime_metrics"}, "client2": {"other"}},
                MessageType.REALTIME_METRICS,
                ["client1"],
            ),
            (  # Wildcard receives all messages
                {"client1": {"*"}},
                MessageType.CREDIT_PHASE_START,
                ["client1"],
            ),
            (  # Both specific and wildcard receive
                {"client1": {"realtime_metrics"}, "client2": {"*"}},
                MessageType.REALTIME_METRICS,
                ["client1", "client2"],
            ),
        ],
    )  # fmt: skip
    async def test_forward_to_clients(
        self,
        api_service: APIService,
        subscriptions: dict[str, set[str]],
        message_type: MessageType,
        expected_clients: list[str],
    ) -> None:
        """Test forwarding messages to subscribed clients."""
        api_service.client_subscriptions = subscriptions
        api_service.connection_manager.send_to_client = AsyncMock(return_value=True)
        api_service.debug = MagicMock()

        message = MagicMock()
        message.message_type = message_type
        message.model_dump.return_value = {"message_type": message_type.value}

        await api_service._forward_message(message)

        assert api_service.connection_manager.send_to_client.call_count == len(
            expected_clients
        )
        called_clients = [
            call[0][0]
            for call in api_service.connection_manager.send_to_client.call_args_list
        ]
        for client in expected_clients:
            assert client in called_clients
