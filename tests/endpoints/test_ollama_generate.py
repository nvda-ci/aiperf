# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models import ParsedResponse
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo, Turn
from aiperf.common.protocols import InferenceServerResponse
from aiperf.endpoints.ollama_generate import OllamaGenerateEndpoint


class TestOllamaGenerateEndpoint:
    """Unit tests for OllamaGenerateEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        endpoint_info = EndpointInfo(
            type=EndpointType.OLLAMA_GENERATE,
            base_url="http://localhost:11434",
            custom_endpoint=None,
        )
        model_list = ModelListInfo(
            models=[ModelInfo(name="llama2")],
            model_selection_strategy=ModelSelectionStrategy.RANDOM,
        )
        return ModelEndpointInfo(models=model_list, endpoint=endpoint_info)

    @pytest.fixture
    def endpoint(self, model_endpoint):
        ep = OllamaGenerateEndpoint(model_endpoint)
        ep.debug = Mock()
        ep.make_text_response_data = Mock(return_value={"text": "parsed"})
        return ep

    def test_metadata_values(self):
        meta = OllamaGenerateEndpoint.metadata()
        assert isinstance(meta, EndpointMetadata)
        assert meta.endpoint_path == "/api/generate"
        assert meta.supports_streaming
        assert meta.produces_tokens
        assert meta.tokenizes_input
        assert meta.metrics_title == "LLM Metrics"

    def test_format_payload_basic(self, endpoint, model_endpoint):
        turn = Turn(texts=[{"contents": ["Hello world"]}])
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["model"] == "llama2"
        assert payload["prompt"] == "Hello world"
        assert payload["stream"] is False

    def test_format_payload_with_streaming(self, endpoint, model_endpoint):
        model_endpoint.endpoint.streaming = True
        turn = Turn(texts=[{"contents": ["Hi there"]}])
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["stream"] is True

    def test_format_payload_with_max_tokens(self, endpoint, model_endpoint):
        turn = Turn(texts=[{"contents": ["test"]}], max_tokens=100)
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["options"]["num_predict"] == 100

    def test_format_payload_with_system_prompt(self, endpoint, model_endpoint):
        model_endpoint.endpoint.extra = {"system": "You are a helpful assistant"}
        turn = Turn(texts=[{"contents": ["test"]}])
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["system"] == "You are a helpful assistant"
        assert "system" not in payload.get("options", {})

    def test_format_payload_with_format(self, endpoint, model_endpoint):
        model_endpoint.endpoint.extra = {"format": "json"}
        turn = Turn(texts=[{"contents": ["test"]}])
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["format"] == "json"
        assert "format" not in payload.get("options", {})

    def test_format_payload_with_options(self, endpoint, model_endpoint):
        model_endpoint.endpoint.extra = {
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "seed": 42,
            }
        }
        turn = Turn(texts=[{"contents": ["test"]}])
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["options"]["temperature"] == 0.7
        assert payload["options"]["top_p"] == 0.9
        assert payload["options"]["top_k"] == 40
        assert payload["options"]["seed"] == 42

    def test_format_payload_with_max_tokens_and_options(self, endpoint, model_endpoint):
        model_endpoint.endpoint.extra = {"options": {"temperature": 0.8}}
        turn = Turn(texts=[{"contents": ["test"]}], max_tokens=100)
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["options"]["num_predict"] == 100
        assert payload["options"]["temperature"] == 0.8

    def test_format_payload_with_raw_flag(self, endpoint, model_endpoint):
        model_endpoint.endpoint.extra = {"raw": True}
        turn = Turn(texts=[{"contents": ["test"]}])
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["raw"] is True
        assert "raw" not in payload.get("options", {})

    def test_format_payload_with_keep_alive(self, endpoint, model_endpoint):
        model_endpoint.endpoint.extra = {"keep_alive": "5m"}
        turn = Turn(texts=[{"contents": ["test"]}])
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["keep_alive"] == "5m"
        assert "keep_alive" not in payload.get("options", {})

    def test_format_payload_with_images(self, endpoint, model_endpoint):
        model_endpoint.endpoint.extra = {"images": ["base64_image_data"]}
        turn = Turn(texts=[{"contents": ["test"]}])
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["images"] == ["base64_image_data"]
        assert "images" not in payload.get("options", {})

    def test_format_payload_multiple_texts(self, endpoint, model_endpoint):
        turn = Turn(texts=[{"contents": ["Hello", "world"]}, {"contents": ["test"]}])
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["prompt"] == "Hello world test"

    def test_format_payload_custom_model(self, endpoint, model_endpoint):
        turn = Turn(texts=[{"contents": ["test"]}], model="mistral")
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["model"] == "mistral"

    def test_format_payload_no_turns_raises(self, endpoint, model_endpoint):
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[])
        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_parse_response_basic(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {
            "model": "llama2",
            "response": "Hello!",
            "done": False,
        }
        response.perf_ns = 123

        result = endpoint.parse_response(response)
        assert isinstance(result, ParsedResponse)
        endpoint.make_text_response_data.assert_called_once_with("Hello!")

    def test_parse_response_with_done_and_usage(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {
            "model": "llama2",
            "response": "Complete response",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        response.perf_ns = 456

        result = endpoint.parse_response(response)
        assert isinstance(result, ParsedResponse)
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    def test_parse_response_done_without_token_counts(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {
            "model": "llama2",
            "response": "Done",
            "done": True,
        }
        response.perf_ns = 789

        result = endpoint.parse_response(response)
        assert isinstance(result, ParsedResponse)
        assert result.usage is None

    def test_parse_response_streaming_chunk(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {
            "model": "llama2",
            "response": "Hi",
            "done": False,
        }
        response.perf_ns = 111

        result = endpoint.parse_response(response)
        assert isinstance(result, ParsedResponse)
        endpoint.make_text_response_data.assert_called_once_with("Hi")
        assert result.usage is None

    def test_parse_response_no_response_field(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {"model": "llama2", "done": False}
        response.perf_ns = 222

        result = endpoint.parse_response(response)
        assert result is None
        endpoint.debug.assert_called()

    def test_parse_response_empty_json(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = None

        result = endpoint.parse_response(response)
        assert result is None

    def test_parse_response_with_partial_usage(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {
            "model": "llama2",
            "response": "test",
            "done": True,
            "prompt_eval_count": 5,
        }
        response.perf_ns = 333

        result = endpoint.parse_response(response)
        assert isinstance(result, ParsedResponse)
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens is None
