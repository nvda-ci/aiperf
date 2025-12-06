# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for the AIPerf Mock Server."""

import asyncio
import logging
import random
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from functools import wraps
from time import perf_counter
from typing import Any, Generic

from aiperf_mock_server.config import server_config
from aiperf_mock_server.metrics import (
    DYNAMO_FRONTEND_INTER_TOKEN_LATENCY_SECONDS,
    DYNAMO_FRONTEND_TIME_TO_FIRST_TOKEN_SECONDS,
    INTER_TOKEN_LATENCY_SECONDS,
    SGLANG_TIME_TO_FIRST_TOKEN_SECONDS,
    TIME_TO_FIRST_TOKEN_SECONDS,
    TOKENS_STREAMED_TOTAL,
    TRTLLM_TIME_PER_OUTPUT_TOKEN_SECONDS,
    TRTLLM_TIME_TO_FIRST_TOKEN_SECONDS,
    VLLM_INTER_TOKEN_LATENCY_SECONDS,
    VLLM_TIME_TO_FIRST_TOKEN_SECONDS,
)
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    ChatDelta,
    ChatStreamChoice,
    ChatStreamCompletionResponse,
    CompletionRequest,
    EmbeddingRequest,
    RankingRequest,
    RequestTypeVarT,
    TextStreamChoice,
    TextStreamCompletionResponse,
)
from aiperf_mock_server.tokens import Tokenizer
from fastapi import HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Decorators
# ============================================================================


def with_error_injection(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to inject errors based on config."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any):
        if (
            server_config.error_rate > 0
            and random.random() * 100 < server_config.error_rate
        ):
            raise HTTPException(status_code=500, detail="Simulated error")
        return await func(*args, **kwargs)

    return wrapper


# ============================================================================
# Timing & Latency Simulation
# ============================================================================


class LatencySimulator:
    """Simulates API latency with TTFT and ITL."""

    __slots__ = (
        "ttft_sec",
        "itl_sec",
        "start_time",
        "token_index",
        "last_token_time",
        "endpoint",
        "model",
        "measured_ttft",
        "measured_decode",
    )

    def __init__(self, endpoint: str = "unknown", model: str = "unknown") -> None:
        self.ttft_sec = server_config.ttft * 0.001
        self.itl_sec = server_config.itl * 0.001
        self.start_time = perf_counter()
        self.token_index = 0
        self.last_token_time: float | None = None
        self.endpoint = endpoint
        self.model = model
        self.measured_ttft: float = 0.0
        self.measured_decode: float = 0.0

    async def wait_for_next_token(self) -> None:
        """Wait for TTFT (first token) or ITL (subsequent tokens)."""
        await self._wait_for_token_at_index(self.token_index)

        now = perf_counter()
        if self.token_index == 0:
            # Record time to first token with real variance
            ttft = now - self.start_time
            self.measured_ttft = ttft  # Store for component metrics
            TIME_TO_FIRST_TOKEN_SECONDS.labels(endpoint=self.endpoint).observe(ttft)
            # Also record vLLM/SGLang/TRT-LLM compatible metrics
            VLLM_TIME_TO_FIRST_TOKEN_SECONDS.observe(ttft)
            SGLANG_TIME_TO_FIRST_TOKEN_SECONDS.observe(ttft)
            TRTLLM_TIME_TO_FIRST_TOKEN_SECONDS.observe(ttft)
            # Also record Dynamo frontend metrics
            DYNAMO_FRONTEND_TIME_TO_FIRST_TOKEN_SECONDS.labels(
                model=self.model
            ).observe(ttft)
        elif self.last_token_time is not None:
            # Record inter-token latency
            itl = now - self.last_token_time
            INTER_TOKEN_LATENCY_SECONDS.labels(endpoint=self.endpoint).observe(itl)
            # Also record vLLM/TRT-LLM compatible metrics
            VLLM_INTER_TOKEN_LATENCY_SECONDS.observe(itl)
            TRTLLM_TIME_PER_OUTPUT_TOKEN_SECONDS.observe(itl)
            # Also record Dynamo frontend metrics
            DYNAMO_FRONTEND_INTER_TOKEN_LATENCY_SECONDS.labels(
                model=self.model
            ).observe(itl)

        self.last_token_time = now
        self.token_index += 1

    async def _wait_for_token_at_index(self, token_index: int) -> None:
        """Wait until the specified token index should be emitted."""
        target_time = self.start_time + self.ttft_sec + (self.itl_sec * token_index)
        remaining = target_time - perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def wait_for_tokens(self, num_tokens: int) -> None:
        """Wait for entire completion (TTFT + ITL * num_tokens).

        Records the measured prefill (TTFT) and decode times for component metrics.
        Note: TTFT/ITL frontend metrics are only recorded for streaming requests
        (via wait_for_next_token) since non-streaming has no "first token" concept.
        """
        # Wait for TTFT first (prefill phase)
        ttft_target = self.start_time + self.ttft_sec
        ttft_remaining = ttft_target - perf_counter()
        if ttft_remaining > 0:
            await asyncio.sleep(ttft_remaining)

        # Record the actual measured prefill time with variance
        self.measured_ttft = perf_counter() - self.start_time

        # Wait for decode phase (ITL * num_tokens)
        decode_target = ttft_target + (self.itl_sec * num_tokens)
        decode_remaining = decode_target - perf_counter()
        if decode_remaining > 0:
            await asyncio.sleep(decode_remaining)

        # Record the actual measured decode time
        self.measured_decode = perf_counter() - self.start_time - self.measured_ttft


class RequestContext(Generic[RequestTypeVarT]):
    """Context object for processing a request."""

    def __init__(self, request: RequestTypeVarT, endpoint: str = "unknown") -> None:
        self.request = request
        self.endpoint = endpoint
        model = getattr(request, "model", "unknown")
        self.latency_sim = LatencySimulator(endpoint=endpoint, model=model)
        self.request_id = create_request_id(request)
        self.tokenized = Tokenizer.tokenize_request(request)

    async def wait_until_completion(self) -> None:
        """Wait until completion is ready."""
        await self.latency_sim.wait_for_tokens(self.tokenized.count)


# ============================================================================
# Completion Handling
# ============================================================================


def create_request_id(request: RequestTypeVarT) -> str:
    """Generate request ID based on request type."""
    match request:
        case ChatCompletionRequest():
            return f"chatcmpl-{uuid.uuid4()}"
        case CompletionRequest():
            return f"cmpl-{uuid.uuid4()}"
        case EmbeddingRequest():
            return f"emb-{uuid.uuid4()}"
        case RankingRequest():
            return f"rank-{uuid.uuid4()}"
        case _:
            raise ValueError(f"Invalid request type: {type(request)}")


# ============================================================================
# Streaming & Response Generation
# ============================================================================


async def stream_chat_completion(
    ctx: RequestContext[ChatCompletionRequest],
) -> AsyncGenerator[str, None]:
    """Stream chat completion tokens as SSE chunks."""
    if ctx.tokenized.reasoning_content_tokens:
        async for chunk in _stream_chat_reasoning_tokens(ctx):
            yield chunk

    async for chunk in _stream_chat_output_tokens(ctx):
        yield chunk

    if ctx.request.include_usage:
        response = ChatStreamCompletionResponse(
            id=ctx.request_id,
            created=int(time.time()),
            model=ctx.request.model,
            choices=[],
            usage=ctx.tokenized.create_usage(),
        )
        yield _format_sse_chunk(response)

    yield "data: [DONE]\n\n"


async def _stream_chat_reasoning_tokens(
    ctx: RequestContext[ChatCompletionRequest],
) -> AsyncGenerator[str, None]:
    """Stream reasoning content tokens for chat completions."""
    for token in ctx.tokenized.reasoning_content_tokens:
        delta = ChatDelta(reasoning_content=token)
        delta.role = "assistant"

        choice = ChatStreamChoice(index=0, finish_reason=None, delta=delta)
        response = ChatStreamCompletionResponse(
            id=ctx.request_id,
            created=int(time.time()),
            model=ctx.request.model,
            choices=[choice],
        )

        await ctx.latency_sim.wait_for_next_token()
        TOKENS_STREAMED_TOTAL.labels(
            endpoint=ctx.endpoint, model=ctx.request.model
        ).inc()
        yield _format_sse_chunk(response)


async def _stream_chat_output_tokens(
    ctx: RequestContext[ChatCompletionRequest],
) -> AsyncGenerator[str, None]:
    """Stream output content tokens for chat completions."""
    has_reasoning = bool(ctx.tokenized.reasoning_content_tokens)

    for i, token in enumerate(ctx.tokenized.tokens):
        delta = ChatDelta(content=token)
        if i == 0 and not has_reasoning:
            delta.role = "assistant"

        choice = ChatStreamChoice(
            index=0,
            finish_reason=ctx.tokenized.finish_reason
            if i == len(ctx.tokenized.tokens) - 1
            else None,
            delta=delta,
        )
        response = ChatStreamCompletionResponse(
            id=ctx.request_id,
            created=int(time.time()),
            model=ctx.request.model,
            choices=[choice],
        )

        await ctx.latency_sim.wait_for_next_token()
        TOKENS_STREAMED_TOTAL.labels(
            endpoint=ctx.endpoint, model=ctx.request.model
        ).inc()
        yield _format_sse_chunk(response)


async def _stream_text_output_tokens(
    ctx: RequestContext[CompletionRequest],
) -> AsyncGenerator[str, None]:
    """Stream output content tokens for text completions."""
    for i, token in enumerate(ctx.tokenized.tokens):
        response = TextStreamCompletionResponse(
            id=ctx.request_id,
            created=int(time.time()),
            model=ctx.request.model,
            choices=[
                TextStreamChoice(
                    index=0,
                    finish_reason=ctx.tokenized.finish_reason
                    if i == len(ctx.tokenized.tokens) - 1
                    else None,
                    text=token,
                )
            ],
        )

        await ctx.latency_sim.wait_for_next_token()
        TOKENS_STREAMED_TOTAL.labels(
            endpoint=ctx.endpoint, model=ctx.request.model
        ).inc()
        yield _format_sse_chunk(response)


async def stream_text_completion(
    ctx: RequestContext[CompletionRequest],
) -> AsyncGenerator[str, None]:
    """Stream text completion tokens as SSE chunks."""
    async for chunk in _stream_text_output_tokens(ctx):
        yield chunk

    if ctx.request.include_usage:
        response = TextStreamCompletionResponse(
            id=ctx.request_id,
            created=int(time.time()),
            model=ctx.request.model,
            choices=[],
            usage=ctx.tokenized.create_usage(),
        )
        yield _format_sse_chunk(response)

    yield "data: [DONE]\n\n"


def _format_sse_chunk(model: BaseModel) -> str:
    """Format data as SSE chunk."""
    json_str = model.model_dump_json(exclude_none=True)
    return f"data: {json_str}\n\n"
