# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestRecord
from aiperf.workers.inference_client import InferenceClient


class TestInferenceClient:
    """Tests for InferenceClient functionality."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
            ),
        )

    @pytest.fixture
    def inference_client(self, model_endpoint):
        """Create an InferenceClient instance."""
        with (
            patch(
                "aiperf.common.factories.TransportFactory.create_instance"
            ) as mock_transport_factory,
            patch(
                "aiperf.common.factories.EndpointFactory.create_instance"
            ) as mock_endpoint_factory,
        ):
            mock_transport = MagicMock()
            mock_endpoint = MagicMock()
            mock_endpoint.get_endpoint_headers.return_value = {}
            mock_endpoint.get_endpoint_params.return_value = {}
            mock_endpoint.format_payload.return_value = {}
            mock_transport_factory.return_value = mock_transport
            mock_endpoint_factory.return_value = mock_endpoint
            return InferenceClient(
                model_endpoint=model_endpoint, service_id="test-service-id"
            )

    @pytest.mark.asyncio
    async def test_send_request_sets_endpoint_headers(
        self, inference_client, model_endpoint, sample_request_info
    ):
        """Test that send_request sets endpoint_headers on request_info."""
        model_endpoint.endpoint.api_key = "test-key"
        model_endpoint.endpoint.headers = [("X-Custom", "value")]

        request_info = sample_request_info

        expected_headers = {
            "Authorization": "Bearer test-key",
            "X-Custom": "value",
        }
        inference_client.endpoint.get_endpoint_headers.return_value = expected_headers

        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord(request_info=sample_request_info)
        )

        await inference_client.send_request(request_info)

        assert "Authorization" in request_info.endpoint_headers
        assert request_info.endpoint_headers["Authorization"] == "Bearer test-key"
        assert request_info.endpoint_headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_send_request_sets_endpoint_params(
        self, inference_client, model_endpoint, sample_request_info
    ):
        """Test that send_request sets endpoint_params on request_info."""
        model_endpoint.endpoint.url_params = {"api-version": "v1", "timeout": "30"}

        request_info = sample_request_info

        expected_params = {"api-version": "v1", "timeout": "30"}
        inference_client.endpoint.get_endpoint_params.return_value = expected_params

        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord(request_info=sample_request_info)
        )

        await inference_client.send_request(request_info)

        assert request_info.endpoint_params["api-version"] == "v1"
        assert request_info.endpoint_params["timeout"] == "30"

    @pytest.mark.asyncio
    async def test_send_request_calls_transport(
        self,
        inference_client,
        model_endpoint,
        sample_request_info,
        sample_request_record,
    ):
        """Test that send_request delegates to transport."""
        request_info = sample_request_info
        expected_record = sample_request_record

        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        record = await inference_client.send_request(request_info)

        inference_client.transport.send_request.assert_called_once()
        call_args = inference_client.transport.send_request.call_args
        assert call_args[0][0] == request_info
        assert record == expected_record

    @pytest.mark.asyncio
    async def test_send_with_optional_cancel_no_cancellation(
        self, inference_client, sample_request_record
    ):
        """Test _send_with_optional_cancel when cancel_after_ns is None."""

        async def mock_coroutine():
            return sample_request_record

        result = await inference_client._send_with_optional_cancel(
            send_coroutine=mock_coroutine(),
            cancel_after_ns=None,
        )

        assert result == sample_request_record

    @pytest.mark.asyncio
    @patch("asyncio.wait_for")
    async def test_send_with_optional_cancel_success(
        self, mock_wait_for, inference_client, sample_request_record
    ):
        """Test successful request with timeout."""

        async def mock_coroutine():
            return sample_request_record

        # Mock wait_for to consume the coroutine and return the result
        async def mock_wait_for_impl(coro, timeout):
            await coro
            return sample_request_record

        mock_wait_for.side_effect = mock_wait_for_impl

        result = await inference_client._send_with_optional_cancel(
            send_coroutine=mock_coroutine(),
            cancel_after_ns=int(2.0 * 1_000_000_000),
        )

        assert result == sample_request_record
        mock_wait_for.assert_called_once()
        call_args = mock_wait_for.call_args
        assert call_args[1]["timeout"] == 2.0

    @pytest.mark.asyncio
    @patch("asyncio.wait_for", side_effect=asyncio.TimeoutError())
    async def test_send_with_optional_cancel_timeout(
        self, mock_wait_for, inference_client
    ):
        """Test request that times out (returns None)."""

        async def mock_coroutine():
            return Mock()

        result = await inference_client._send_with_optional_cancel(
            send_coroutine=mock_coroutine(),
            cancel_after_ns=int(1.0 * 1_000_000_000),
        )

        assert result is None
        mock_wait_for.assert_called_once()
        call_args = mock_wait_for.call_args
        assert call_args[1]["timeout"] == 1.0

    @pytest.mark.asyncio
    @patch("asyncio.wait_for")
    async def test_timeout_conversion_precision(
        self, mock_wait_for, inference_client, sample_request_record
    ):
        """Test that nanoseconds are correctly converted to seconds."""
        test_cases = [
            (int(0.5 * 1_000_000_000), 0.5),
            (int(1.0 * 1_000_000_000), 1.0),
            (int(2.5 * 1_000_000_000), 2.5),
            (int(10.123456789 * 1_000_000_000), 10.123456789),
        ]

        for cancel_after_ns, expected_timeout in test_cases:
            mock_wait_for.reset_mock()

            async def mock_coroutine():
                return sample_request_record

            async def mock_wait_for_impl(coro, timeout):
                await coro
                return sample_request_record

            mock_wait_for.side_effect = mock_wait_for_impl

            await inference_client._send_with_optional_cancel(
                send_coroutine=mock_coroutine(),
                cancel_after_ns=cancel_after_ns,
            )

            call_args = mock_wait_for.call_args
            actual_timeout = call_args[1]["timeout"]
            assert abs(actual_timeout - expected_timeout) < 1e-9, (
                f"Expected timeout {expected_timeout}, got {actual_timeout}"
            )
