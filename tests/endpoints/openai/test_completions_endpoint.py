# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models import Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.endpoints.openai_completions import CompletionsEndpoint


class TestCompletionsEndpoint:
    """Comprehensive tests for CompletionsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for completions."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="completion-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.COMPLETIONS,
                base_url="http://localhost:8000",
                streaming=False,
            ),
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a CompletionsEndpoint instance."""
        with patch(
            "aiperf.common.factories.TransportFactory.create_instance"
        ) as mock_transport:
            mock_transport.return_value = MagicMock()
            return CompletionsEndpoint(model_endpoint=model_endpoint)

    @pytest.mark.asyncio
    async def test_format_payload_single_prompt(self, endpoint, model_endpoint):
        """Test single prompt formatting."""
        turn = Turn(
            texts=[Text(contents=["Once upon a time"])],
            model="completion-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = await endpoint.format_payload(request_info)

        assert payload["model"] == "completion-model"
        assert payload["stream"] is False
        assert payload["prompt"] == ["Once upon a time"]

    @pytest.mark.asyncio
    async def test_format_payload_multiple_prompts(self, endpoint, model_endpoint):
        """Test multiple prompts are all included."""
        turn = Turn(
            texts=[
                Text(contents=["Prompt 1", "Prompt 2"]),
                Text(contents=["Prompt 3"]),
            ],
            model="completion-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = await endpoint.format_payload(request_info)

        assert len(payload["prompt"]) == 3
        assert payload["prompt"] == ["Prompt 1", "Prompt 2", "Prompt 3"]

    @pytest.mark.asyncio
    async def test_format_payload_filters_empty_prompts(self, endpoint, model_endpoint):
        """Test that empty strings are filtered from prompts."""
        turn = Turn(
            texts=[
                Text(contents=["Valid", "", "Also valid", ""]),
            ],
            model="completion-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = await endpoint.format_payload(request_info)

        assert len(payload["prompt"]) == 2
        assert payload["prompt"] == ["Valid", "Also valid"]

    @pytest.mark.asyncio
    async def test_format_payload_with_max_tokens(self, endpoint, model_endpoint):
        """Test max_tokens is included when specified."""
        turn = Turn(
            texts=[Text(contents=["Generate"])],
            model="completion-model",
            max_tokens=200,
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = await endpoint.format_payload(request_info)

        assert payload["max_tokens"] == 200

    @pytest.mark.asyncio
    async def test_format_payload_no_max_tokens(self, endpoint, model_endpoint):
        """Test max_tokens is not included when None."""
        turn = Turn(
            texts=[Text(contents=["Generate"])],
            model="completion-model",
            max_tokens=None,
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = await endpoint.format_payload(request_info)

        assert "max_tokens" not in payload

    @pytest.mark.asyncio
    async def test_format_payload_streaming_enabled(self, model_endpoint):
        """Test streaming flag from endpoint config."""
        model_endpoint.endpoint.streaming = True

        with patch(
            "aiperf.common.factories.TransportFactory.create_instance"
        ) as mock_transport:
            mock_transport.return_value = MagicMock()
            endpoint = CompletionsEndpoint(model_endpoint=model_endpoint)

            turn = Turn(texts=[Text(contents=["Test"])], model="completion-model")
            request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

            payload = await endpoint.format_payload(request_info)

            assert payload["stream"] is True

    @pytest.mark.asyncio
    async def test_format_payload_extra_parameters(self, model_endpoint):
        """Test extra parameters are merged into payload."""
        model_endpoint.endpoint.extra = [
            ("temperature", 0.8),
            ("top_p", 0.95),
            ("frequency_penalty", 0.5),
        ]

        with patch(
            "aiperf.common.factories.TransportFactory.create_instance"
        ) as mock_transport:
            mock_transport.return_value = MagicMock()
            endpoint = CompletionsEndpoint(model_endpoint=model_endpoint)

            turn = Turn(texts=[Text(contents=["Test"])], model="completion-model")
            request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

            payload = await endpoint.format_payload(request_info)

            assert payload["temperature"] == 0.8
            assert payload["top_p"] == 0.95
            assert payload["frequency_penalty"] == 0.5

    @pytest.mark.asyncio
    async def test_format_payload_empty_texts_list(self, endpoint, model_endpoint):
        """Test behavior with empty texts list."""
        turn = Turn(
            texts=[],  # No texts
            model="completion-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = await endpoint.format_payload(request_info)

        assert payload["prompt"] == []

    @pytest.mark.asyncio
    async def test_format_payload_uses_first_turn_only(self, endpoint, model_endpoint):
        """Test that only request_info.turns[0] is used (hardcoded)."""
        turn1 = Turn(texts=[Text(contents=["First"])], model="model1")
        turn2 = Turn(texts=[Text(contents=["Second"])], model="model2")

        request_info = RequestInfo(
            model_endpoint=model_endpoint, turn_index=0, turns=[turn1, turn2]
        )

        payload = await endpoint.format_payload(request_info)

        # Should only use first turn
        assert payload["prompt"] == ["First"]
        assert payload["model"] == "model1"
