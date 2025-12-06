# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import logging
import random
import time
from contextlib import asynccontextmanager
from time import perf_counter

from aiperf_mock_server.config import server_config
from aiperf_mock_server.dcgm_faker import DCGMFaker
from aiperf_mock_server.metrics import (
    COMPLETION_TOKENS_TOTAL,
    DYNAMO_DECODE_INFLIGHT_REQUESTS,
    DYNAMO_DECODE_REQUEST_DURATION_SECONDS,
    DYNAMO_DECODE_REQUESTS,
    DYNAMO_FRONTEND_INFLIGHT_REQUESTS,
    DYNAMO_FRONTEND_INPUT_SEQUENCE_TOKENS,
    DYNAMO_FRONTEND_OUTPUT_TOKENS,
    DYNAMO_FRONTEND_REQUEST_DURATION_SECONDS,
    DYNAMO_FRONTEND_REQUESTS,
    DYNAMO_PREFILL_INFLIGHT_REQUESTS,
    DYNAMO_PREFILL_REQUEST_DURATION_SECONDS,
    DYNAMO_PREFILL_REQUESTS,
    EMBEDDINGS_GENERATED_TOTAL,
    ERRORS_TOTAL,
    PASSAGES_RANKED_TOTAL,
    PROMPT_TOKENS_TOTAL,
    RANKINGS_GENERATED_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_BY_MODEL,
    REQUESTS_IN_PROGRESS,
    REQUESTS_TOTAL,
    SGLANG_E2E_REQUEST_LATENCY_SECONDS,
    SGLANG_NUM_RUNNING_REQS,
    STREAMING_REQUESTS_TOTAL,
    TOKENS_PER_REQUEST,
    TOKENS_STREAMED_TOTAL,
    TRTLLM_E2E_REQUEST_LATENCY_SECONDS,
    TRTLLM_REQUEST_SUCCESS,
    VLLM_E2E_REQUEST_LATENCY_SECONDS,
    VLLM_GENERATION_TOKENS,
    VLLM_ITERATION_TOKENS_TOTAL,
    VLLM_NUM_REQUESTS_RUNNING,
    VLLM_PROMPT_TOKENS,
    VLLM_REQUEST_SUCCESS,
    generate_aiperf_metrics,
    generate_dynamo_decode_metrics,
    generate_dynamo_frontend_metrics,
    generate_dynamo_prefill_metrics,
    generate_sglang_metrics,
    generate_trtllm_metrics,
    generate_vllm_metrics,
)
from aiperf_mock_server.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    Ranking,
    RankingRequest,
    RankingResponse,
    TextChoice,
    TextCompletionResponse,
)
from aiperf_mock_server.utils import (
    RequestContext,
    stream_chat_completion,
    stream_text_completion,
    with_error_injection,
)
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

dcgm_fakers: list[DCGMFaker] = []
logger = logging.getLogger(__name__)


def _create_dcgm_faker(seed: int | None) -> DCGMFaker:
    """Create a DCGM faker instance with current config."""
    return DCGMFaker(
        gpu_name=server_config.dcgm_gpu_name,
        num_gpus=server_config.dcgm_num_gpus,
        seed=seed,
        hostname=server_config.dcgm_hostname,
        initial_load=server_config.dcgm_initial_load,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize server on startup."""
    logger.info("Server starting: %s", server_config.model_dump())
    if server_config.random_seed is not None:
        random.seed(server_config.random_seed)

    dcgm_fakers.append(_create_dcgm_faker(server_config.dcgm_seed))
    dcgm_fakers.append(
        _create_dcgm_faker(
            None if server_config.dcgm_seed is None else server_config.dcgm_seed + 1
        )
    )

    logger.info(
        "DCGM faker initialized with %d %s GPUs",
        server_config.dcgm_num_gpus,
        server_config.dcgm_gpu_name,
    )
    yield


app = FastAPI(title="AIPerf Mock Server", version="2.0.0", lifespan=lifespan)

# ============================================================================
# Chat Completions
# ============================================================================


@app.post("/v1/chat/completions", response_model=None)
@with_error_injection
async def chat_completions(
    req: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Chat completion endpoint."""
    endpoint = "/v1/chat/completions"
    start_time = perf_counter()
    ctx = RequestContext(req, endpoint=endpoint)

    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
    REQUESTS_BY_MODEL.labels(model=req.model, endpoint=endpoint).inc()

    # Update vLLM/SGLang-style running/waiting gauges
    VLLM_NUM_REQUESTS_RUNNING.inc()
    SGLANG_NUM_RUNNING_REQS.inc()

    # Update Dynamo-style inflight gauges
    DYNAMO_FRONTEND_INFLIGHT_REQUESTS.labels(model=req.model).inc()
    DYNAMO_PREFILL_INFLIGHT_REQUESTS.labels(
        dynamo_endpoint="generate", model=req.model
    ).inc()
    DYNAMO_DECODE_INFLIGHT_REQUESTS.labels(
        dynamo_endpoint="generate", model=req.model
    ).inc()

    try:
        if req.stream:
            STREAMING_REQUESTS_TOTAL.labels(endpoint=endpoint, model=req.model).inc()
            return StreamingResponse(
                _streaming_wrapper(ctx, endpoint, start_time),
                media_type="text/event-stream",
            )

        await ctx.wait_until_completion()

        latency = perf_counter() - start_time

        # Record token metrics
        usage = ctx.tokenized.create_usage()
        PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=req.model).inc(
            usage.prompt_tokens
        )
        COMPLETION_TOKENS_TOTAL.labels(endpoint=endpoint, model=req.model).inc(
            usage.completion_tokens
        )
        TOKENS_PER_REQUEST.labels(endpoint=endpoint, token_type="prompt").observe(
            usage.prompt_tokens
        )
        TOKENS_PER_REQUEST.labels(endpoint=endpoint, token_type="completion").observe(
            usage.completion_tokens
        )

        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(latency)

        # Record vLLM-compatible metrics
        VLLM_E2E_REQUEST_LATENCY_SECONDS.observe(latency)
        VLLM_PROMPT_TOKENS.inc(usage.prompt_tokens)
        VLLM_GENERATION_TOKENS.inc(usage.completion_tokens)
        VLLM_REQUEST_SUCCESS.inc()
        VLLM_ITERATION_TOKENS_TOTAL.observe(usage.total_tokens)

        # Record SGLang-compatible metrics
        SGLANG_E2E_REQUEST_LATENCY_SECONDS.observe(latency)

        # Record TRT-LLM-compatible metrics
        TRTLLM_E2E_REQUEST_LATENCY_SECONDS.observe(latency)
        TRTLLM_REQUEST_SUCCESS.inc()

        # Record Dynamo-compatible metrics
        DYNAMO_FRONTEND_REQUEST_DURATION_SECONDS.labels(model=req.model).observe(
            latency
        )
        DYNAMO_FRONTEND_REQUESTS.labels(model=req.model).inc()
        DYNAMO_FRONTEND_INPUT_SEQUENCE_TOKENS.labels(model=req.model).inc(
            usage.prompt_tokens
        )
        DYNAMO_FRONTEND_OUTPUT_TOKENS.labels(model=req.model).inc(
            usage.completion_tokens
        )
        # Use the measured TTFT/decode durations with real variance
        DYNAMO_PREFILL_REQUEST_DURATION_SECONDS.labels(
            dynamo_endpoint="generate", model=req.model
        ).observe(ctx.latency_sim.measured_ttft)
        DYNAMO_PREFILL_REQUESTS.labels(
            dynamo_endpoint="generate", model=req.model
        ).inc()
        DYNAMO_DECODE_REQUEST_DURATION_SECONDS.labels(
            dynamo_endpoint="generate", model=req.model
        ).observe(ctx.latency_sim.measured_decode)
        DYNAMO_DECODE_REQUESTS.labels(dynamo_endpoint="generate", model=req.model).inc()

        return ChatCompletionResponse(
            id=ctx.request_id,
            created=int(time.time()),
            model=ctx.request.model,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason=ctx.tokenized.finish_reason,
                    message=ChatMessage(
                        role="assistant",
                        content=ctx.tokenized.content,
                        reasoning_content=ctx.tokenized.reasoning_content,
                    ),
                )
            ],
            usage=usage,
        )
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()
        VLLM_NUM_REQUESTS_RUNNING.dec()
        SGLANG_NUM_RUNNING_REQS.dec()
        DYNAMO_FRONTEND_INFLIGHT_REQUESTS.labels(model=req.model).dec()
        DYNAMO_PREFILL_INFLIGHT_REQUESTS.labels(
            dynamo_endpoint="generate", model=req.model
        ).dec()
        DYNAMO_DECODE_INFLIGHT_REQUESTS.labels(
            dynamo_endpoint="generate", model=req.model
        ).dec()


async def _streaming_wrapper(ctx: RequestContext, endpoint: str, start_time: float):
    """Wrapper for streaming that records metrics after completion."""
    try:
        async for chunk in stream_chat_completion(ctx):
            yield chunk
        # Record metrics after streaming completes
        latency = perf_counter() - start_time
        usage = ctx.tokenized.create_usage()
        PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=ctx.request.model).inc(
            usage.prompt_tokens
        )
        COMPLETION_TOKENS_TOTAL.labels(endpoint=endpoint, model=ctx.request.model).inc(
            usage.completion_tokens
        )
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(latency)

        # Record vLLM-compatible metrics
        VLLM_E2E_REQUEST_LATENCY_SECONDS.observe(latency)
        VLLM_PROMPT_TOKENS.inc(usage.prompt_tokens)
        VLLM_GENERATION_TOKENS.inc(usage.completion_tokens)
        VLLM_REQUEST_SUCCESS.inc()

        # Record SGLang-compatible metrics
        SGLANG_E2E_REQUEST_LATENCY_SECONDS.observe(latency)

        # Record TRT-LLM-compatible metrics
        TRTLLM_E2E_REQUEST_LATENCY_SECONDS.observe(latency)
        TRTLLM_REQUEST_SUCCESS.inc()

        # Record Dynamo-compatible metrics
        DYNAMO_FRONTEND_REQUEST_DURATION_SECONDS.labels(
            model=ctx.request.model
        ).observe(latency)
        DYNAMO_FRONTEND_REQUESTS.labels(model=ctx.request.model).inc()
        DYNAMO_FRONTEND_INPUT_SEQUENCE_TOKENS.labels(model=ctx.request.model).inc(
            usage.prompt_tokens
        )
        DYNAMO_FRONTEND_OUTPUT_TOKENS.labels(model=ctx.request.model).inc(
            usage.completion_tokens
        )
        # Use the measured TTFT for prefill, decode is total minus prefill
        prefill_duration = ctx.latency_sim.measured_ttft
        decode_duration = max(0.0, latency - prefill_duration)
        DYNAMO_PREFILL_REQUEST_DURATION_SECONDS.labels(
            dynamo_endpoint="generate", model=ctx.request.model
        ).observe(prefill_duration)
        DYNAMO_PREFILL_REQUESTS.labels(
            dynamo_endpoint="generate", model=ctx.request.model
        ).inc()
        DYNAMO_DECODE_REQUEST_DURATION_SECONDS.labels(
            dynamo_endpoint="generate", model=ctx.request.model
        ).observe(decode_duration)
        DYNAMO_DECODE_REQUESTS.labels(
            dynamo_endpoint="generate", model=ctx.request.model
        ).inc()
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()
        VLLM_NUM_REQUESTS_RUNNING.dec()
        SGLANG_NUM_RUNNING_REQS.dec()
        DYNAMO_FRONTEND_INFLIGHT_REQUESTS.labels(model=ctx.request.model).dec()
        DYNAMO_PREFILL_INFLIGHT_REQUESTS.labels(
            dynamo_endpoint="generate", model=ctx.request.model
        ).dec()
        DYNAMO_DECODE_INFLIGHT_REQUESTS.labels(
            dynamo_endpoint="generate", model=ctx.request.model
        ).dec()


# ============================================================================
# Text Completions
# ============================================================================


@app.post("/v1/completions", response_model=None)
@with_error_injection
async def completions(
    req: CompletionRequest,
) -> TextCompletionResponse | StreamingResponse:
    """Text completion endpoint."""
    endpoint = "/v1/completions"
    start_time = perf_counter()
    ctx = RequestContext(req, endpoint=endpoint)

    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
    REQUESTS_BY_MODEL.labels(model=req.model, endpoint=endpoint).inc()

    try:
        if req.stream:
            STREAMING_REQUESTS_TOTAL.labels(endpoint=endpoint, model=req.model).inc()
            return StreamingResponse(
                _text_streaming_wrapper(ctx, endpoint, start_time),
                media_type="text/event-stream",
            )

        await ctx.wait_until_completion()

        usage = ctx.tokenized.create_usage()
        PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=req.model).inc(
            usage.prompt_tokens
        )
        COMPLETION_TOKENS_TOTAL.labels(endpoint=endpoint, model=req.model).inc(
            usage.completion_tokens
        )
        TOKENS_PER_REQUEST.labels(endpoint=endpoint, token_type="prompt").observe(
            usage.prompt_tokens
        )
        TOKENS_PER_REQUEST.labels(endpoint=endpoint, token_type="completion").observe(
            usage.completion_tokens
        )

        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(
            perf_counter() - start_time
        )

        choice = TextChoice(
            index=0,
            finish_reason=ctx.tokenized.finish_reason,
            text=ctx.tokenized.content,
        )

        return TextCompletionResponse(
            id=ctx.request_id,
            created=int(time.time()),
            model=ctx.request.model,
            choices=[choice],
            usage=usage,
        )
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


async def _text_streaming_wrapper(
    ctx: RequestContext, endpoint: str, start_time: float
):
    """Wrapper for text streaming that records metrics after completion."""
    try:
        async for chunk in stream_text_completion(ctx):
            yield chunk
        usage = ctx.tokenized.create_usage()
        PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=ctx.request.model).inc(
            usage.prompt_tokens
        )
        COMPLETION_TOKENS_TOTAL.labels(endpoint=endpoint, model=ctx.request.model).inc(
            usage.completion_tokens
        )
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(
            perf_counter() - start_time
        )
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


# ============================================================================
# Embeddings
# ============================================================================


@app.post("/v1/embeddings", response_model=None)
@with_error_injection
async def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    """Embedding endpoint."""
    endpoint = "/v1/embeddings"
    start_time = perf_counter()
    ctx = RequestContext(req, endpoint=endpoint)

    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
    REQUESTS_BY_MODEL.labels(model=req.model, endpoint=endpoint).inc()

    try:
        await ctx.wait_until_completion()

        def generate_embedding(text: str) -> list[float]:
            """Generate deterministic embedding from text using stable hash."""
            digest = hashlib.blake2s(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest, byteorder="big")
            rng = random.Random(seed)
            return [rng.random() - 0.5 for _ in range(768)]

        usage = ctx.tokenized.create_usage()
        PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=req.model).inc(
            usage.prompt_tokens
        )
        EMBEDDINGS_GENERATED_TOTAL.labels(model=req.model).inc(len(req.inputs))

        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(
            perf_counter() - start_time
        )

        return EmbeddingResponse(
            id=ctx.request_id,
            created=int(time.time()),
            model=ctx.request.model,
            data=[
                Embedding(
                    index=i,
                    embedding=generate_embedding(text),
                )
                for i, text in enumerate(req.inputs)
            ],
            usage=usage,
        )
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


# ============================================================================
# Rankings
# ============================================================================


def _compute_mock_score(query: str, passage: str) -> float:
    """Compute deterministic mock relevance score for all ranking mocks."""
    combined = f"{query}|{passage}"
    digest = hashlib.blake2s(combined.encode("utf-8")).digest()
    int_digest = int.from_bytes(digest, byteorder="big")
    return (int_digest % 1000) / 1000.0


# ============================================================================
# NIM Rankings Endpoint
# ============================================================================


@app.post("/v1/ranking", response_model=None)
@with_error_injection
async def rankings(req: RankingRequest) -> RankingResponse:
    """Mock NVIDIA NIM /v1/ranking endpoint."""
    endpoint = "/v1/ranking"
    start_time = perf_counter()
    ctx = RequestContext(req, endpoint=endpoint)

    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
    REQUESTS_BY_MODEL.labels(model=req.model, endpoint=endpoint).inc()

    try:
        rankings_list = sorted(
            [
                Ranking(
                    index=i,
                    relevance_score=_compute_mock_score(req.query_text, text),
                )
                for i, text in enumerate(req.passage_texts)
            ],
            key=lambda x: x.relevance_score,
            reverse=True,
        )

        await ctx.wait_until_completion()

        usage = ctx.tokenized.create_usage()
        PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=req.model).inc(
            usage.prompt_tokens
        )
        RANKINGS_GENERATED_TOTAL.labels(endpoint=endpoint).inc()
        PASSAGES_RANKED_TOTAL.labels(endpoint=endpoint).inc(len(req.passage_texts))

        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(
            perf_counter() - start_time
        )

        return RankingResponse(
            id=ctx.request_id,
            model=req.model,
            rankings=rankings_list,
            usage=usage,
        )
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


# ============================================================================
# HuggingFace TEI Rankings Endpoint
# ============================================================================


@app.post("/rerank", response_model=None)
@with_error_injection
async def hf_tei_rerank(req: dict) -> dict:
    """Mock HuggingFace TEI /rerank endpoint."""
    endpoint = "/rerank"
    start_time = perf_counter()

    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()

    try:
        query = req.get("query", "")
        passages = req.get("texts") or req.get("documents") or []

        results = [
            {"index": i, "score": _compute_mock_score(query, p)}
            for i, p in enumerate(passages)
        ]
        results.sort(key=lambda r: r["score"], reverse=True)

        RANKINGS_GENERATED_TOTAL.labels(endpoint=endpoint).inc()
        PASSAGES_RANKED_TOTAL.labels(endpoint=endpoint).inc(len(passages))

        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(
            perf_counter() - start_time
        )

        return {"results": results}
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


# ============================================================================
# Cohere Rankings Endpoint
# ============================================================================


@app.post("/v2/rerank", response_model=None)
@with_error_injection
async def cohere_rerank(req: dict) -> dict:
    """Mock Cohere /v2/rerank endpoint."""
    endpoint = "/v2/rerank"
    start_time = perf_counter()

    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()

    try:
        query = req.get("query", "")
        passages = req.get("documents") or []

        results = [
            {"index": i, "relevance_score": _compute_mock_score(query, p)}
            for i, p in enumerate(passages)
        ]
        results.sort(key=lambda r: r["relevance_score"], reverse=True)

        RANKINGS_GENERATED_TOTAL.labels(endpoint=endpoint).inc()
        PASSAGES_RANKED_TOTAL.labels(endpoint=endpoint).inc(len(passages))

        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(
            perf_counter() - start_time
        )

        return {"results": results}
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


# ============================================================================
# Custom Multimodal Endpoint
# ============================================================================


@app.post("/v1/custom-multimodal", response_model=None)
@with_error_injection
async def custom_multimodal(req: dict) -> dict:
    """Mock endpoint with custom multi-modal format.

    Expected format:
    {
        "modality_bundle": {
            "text_fragments": ["text1", "text2"],
            "visual_assets": {
                "images": ["base64..."],
                "videos": ["base64..."]
            },
            "audio_streams": ["base64..."]
        },
        "inference_params": {
            "model_id": "...",
            "sampling_config": {...}
        }
    }

    Returns format:
    {
        "completion": {
            "generated_text": "...",
            "metadata": {
                "tokens_used": {...}
            }
        }
    }
    """
    endpoint = "/v1/custom-multimodal"
    start_time = perf_counter()

    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()

    try:
        # Extract the multimodal bundle
        bundle = req.get("modality_bundle", {})
        text_fragments = bundle.get("text_fragments", [])
        visual_assets = bundle.get("visual_assets", {})
        images = visual_assets.get("images", [])
        videos = visual_assets.get("videos", [])
        audio_streams = bundle.get("audio_streams", [])

        # Extract inference params
        inference_params = req.get("inference_params", {})
        model_id = inference_params.get("model_id", "default-model")

        REQUESTS_BY_MODEL.labels(model=model_id, endpoint=endpoint).inc()

        # Create a mock request for timing - use a simple valid ChatCompletionRequest
        text_content = " ".join(text_fragments) if text_fragments else "default text"
        mock_req = ChatCompletionRequest(
            model=model_id or "default-model",
            messages=[{"role": "user", "content": text_content}],
        )
        ctx = RequestContext(mock_req, endpoint=endpoint)
        await ctx.wait_until_completion()

        # Build response with custom format
        response_text = f"Processed {len(text_fragments)} text fragments"
        if images:
            response_text += f", {len(images)} images"
        if videos:
            response_text += f", {len(videos)} videos"
        if audio_streams:
            response_text += f", {len(audio_streams)} audio streams"

        usage = ctx.tokenized.create_usage()
        PROMPT_TOKENS_TOTAL.labels(endpoint=endpoint, model=model_id).inc(
            usage.prompt_tokens
        )
        COMPLETION_TOKENS_TOTAL.labels(endpoint=endpoint, model=model_id).inc(
            usage.completion_tokens
        )

        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(
            perf_counter() - start_time
        )

        return {
            "text": response_text,  # Use "text" field for auto-detection
            "completion": {
                "generated_text": response_text,
                "metadata": {
                    "tokens_used": {
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                },
            },
        }
    except Exception as e:
        logger.error(f"Error in custom_multimodal endpoint: {e}", exc_info=True)
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


# ============================================================================
# HuggingFace Generate Endpoint
# ============================================================================


@app.post("/generate", response_model=None)
@with_error_injection
async def huggingface_generate(req: dict):
    """Mock HuggingFace TGI /generate endpoint (non-streaming)."""
    endpoint = "/generate"
    start_time = perf_counter()

    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()

    try:
        prompt = req.get("inputs") or req.get("prompt") or "Hello!"
        max_new_tokens = req.get("parameters", {}).get("max_new_tokens", 50)

        fake_text = f"{prompt} [mocked generation, {max_new_tokens} tokens]"

        COMPLETION_TOKENS_TOTAL.labels(endpoint=endpoint, model="tgi").inc(
            max_new_tokens
        )
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(
            perf_counter() - start_time
        )

        return JSONResponse(content={"generated_text": fake_text})
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
        ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


@app.post("/generate_stream", response_model=None)
@with_error_injection
async def huggingface_generate_stream(req: dict):
    """Mock HuggingFace TGI /generate_stream endpoint (streaming)."""
    endpoint = "/generate_stream"
    start_time = perf_counter()

    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
    STREAMING_REQUESTS_TOTAL.labels(endpoint=endpoint, model="tgi").inc()

    prompt = req.get("inputs") or req.get("prompt") or "Hello!"
    max_new_tokens = req.get("parameters", {}).get("max_new_tokens", 10)

    async def event_stream():
        try:
            for i in range(max_new_tokens):
                yield f'data: {{"token": {{"text": " token_{i}"}}}}\n\n'
                TOKENS_STREAMED_TOTAL.labels(endpoint=endpoint, model="tgi").inc()
                await asyncio.sleep(0.001)
            yield f'data: {{"generated_text": "{prompt} [mocked streaming done]"}}\n\n'
            yield "data: [DONE]\n\n"

            COMPLETION_TOKENS_TOTAL.labels(endpoint=endpoint, model="tgi").inc(
                max_new_tokens
            )
            REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
            REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(
                perf_counter() - start_time
            )
        except Exception as e:
            REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
            ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
            raise
        finally:
            REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================================
# Health & Info
# ============================================================================


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "config": server_config.model_dump()}


@app.get("/")
async def root():
    """Root info."""
    return {
        "message": "AIPerf Mock Server",
        "version": "2.0.0",
        "config": server_config.model_dump(),
    }


# ============================================================================
# DCGM Metrics
# ============================================================================


@app.get("/dcgm{instance_id:int}/metrics")
async def dcgm_metrics(instance_id: int) -> PlainTextResponse:
    """DCGM metrics endpoint (Prometheus format)."""
    index = instance_id - 1
    if index < 0 or index >= len(dcgm_fakers):
        raise HTTPException(status_code=404, detail="Invalid DCGM instance")
    return PlainTextResponse(dcgm_fakers[index].generate(), media_type="text/plain")


# ============================================================================
# Prometheus Metrics Endpoints
# ============================================================================


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """AIPerf mock server Prometheus metrics endpoint.

    Returns AIPerf mock server specific metrics:
    - Request counts by endpoint, method, and status
    - Request latency histograms
    - Token counts (prompt/completion) by endpoint and model
    - Streaming metrics (tokens streamed, TTFT, ITL)
    - In-flight request gauges
    - Error counts by type
    """
    return Response(content=generate_aiperf_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/vllm/metrics")
async def vllm_metrics() -> Response:
    """vLLM-compatible Prometheus metrics endpoint.

    Returns metrics matching vLLM server format:
    - vllm:e2e_request_latency_seconds
    - vllm:time_to_first_token_seconds
    - vllm:inter_token_latency_seconds
    - vllm:prompt_tokens, vllm:generation_tokens
    - vllm:num_requests_running, vllm:num_requests_waiting
    - vllm:request_success, vllm:request_queue_time_seconds
    """
    return Response(content=generate_vllm_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/sglang/metrics")
async def sglang_metrics() -> Response:
    """SGLang-compatible Prometheus metrics endpoint.

    Returns metrics matching SGLang server format:
    - sglang:e2e_request_latency_seconds
    - sglang:time_to_first_token_seconds
    - sglang:queue_time_seconds
    - sglang:num_running_reqs, sglang:num_queue_reqs
    - sglang:gen_throughput, sglang:cache_hit_rate
    """
    return Response(content=generate_sglang_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/trtllm/metrics")
async def trtllm_metrics() -> Response:
    """TensorRT-LLM-compatible Prometheus metrics endpoint.

    Returns metrics matching TensorRT-LLM server format:
    - trtllm:e2e_request_latency_seconds
    - trtllm:time_to_first_token_seconds
    - trtllm:time_per_output_token_seconds
    - trtllm:request_queue_time_seconds
    - trtllm:request_success
    """
    return Response(content=generate_trtllm_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dynamo_frontend/metrics")
async def dynamo_frontend_metrics() -> Response:
    """Dynamo frontend Prometheus metrics endpoint.

    Returns metrics matching Dynamo frontend format:
    - dynamo_frontend_request_duration_seconds
    - dynamo_frontend_time_to_first_token_seconds
    - dynamo_frontend_inter_token_latency_seconds
    - dynamo_frontend_requests
    - dynamo_frontend_input_sequence_tokens, dynamo_frontend_output_tokens
    - dynamo_frontend_queued_requests, dynamo_frontend_inflight_requests
    """
    return Response(
        content=generate_dynamo_frontend_metrics(), media_type=CONTENT_TYPE_LATEST
    )


@app.get("/dynamo_component/prefill/metrics")
async def dynamo_prefill_metrics() -> Response:
    """Dynamo prefill worker Prometheus metrics endpoint.

    Returns metrics matching Dynamo component format for prefill workers:
    - dynamo_component_request_duration_seconds
    - dynamo_component_requests
    - dynamo_component_inflight_requests
    - dynamo_component_kvstats_* (KV cache stats)
    """
    return Response(
        content=generate_dynamo_prefill_metrics(), media_type=CONTENT_TYPE_LATEST
    )


@app.get("/dynamo_component/decode/metrics")
async def dynamo_decode_metrics() -> Response:
    """Dynamo decode worker Prometheus metrics endpoint.

    Returns metrics matching Dynamo component format for decode workers:
    - dynamo_component_request_duration_seconds
    - dynamo_component_requests
    - dynamo_component_inflight_requests
    - dynamo_component_kvstats_* (KV cache stats)
    """
    return Response(
        content=generate_dynamo_decode_metrics(), media_type=CONTENT_TYPE_LATEST
    )
