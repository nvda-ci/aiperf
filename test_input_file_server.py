#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple mock server that records all requests sent to it."""

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from aiohttp import web

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store all received requests
received_requests: list[dict[str, Any]] = []
request_counts: dict[str, int] = defaultdict(int)
output_file: Path | None = None


async def chat_completions(request: web.Request) -> web.Response:
    """Record chat completion requests."""
    body = await request.json()
    request_data = {
        "endpoint": "/v1/chat/completions",
        "method": "POST",
        "headers": dict(request.headers),
        "body": body,
    }
    received_requests.append(request_data)
    request_counts["/v1/chat/completions"] += 1
    logger.info(f"Received request #{request_counts['/v1/chat/completions']}")
    logger.debug(f"Request body: {json.dumps(body, indent=2)}")

    # Return a simple response
    if body.get("stream", False):
        async def stream_response():
            yield b"data: " + json.dumps({
                "id": "test-id",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": body.get("model", "test-model"),
                "choices": [{
                    "index": 0,
                    "delta": {"content": "test"},
                    "finish_reason": None
                }]
            }).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"
        return web.Response(body=stream_response(), content_type="text/event-stream")
    else:
        return web.json_response({
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": body.get("model", "test-model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "test response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        })


async def health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({"status": "healthy"})


async def get_requests(request: web.Request) -> web.Response:
    """Get all recorded requests."""
    return web.json_response({
        "total_requests": len(received_requests),
        "request_counts": dict(request_counts),
        "requests": received_requests
    })


async def reset(request: web.Request) -> web.Response:
    """Reset recorded requests."""
    global received_requests, request_counts
    received_requests.clear()
    request_counts.clear()
    return web.json_response({"status": "reset"})


def save_requests() -> None:
    """Save recorded requests to a JSON file."""
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({
                "total_requests": len(received_requests),
                "request_counts": dict(request_counts),
                "requests": received_requests
            }, f, indent=2)
        logger.info(f"Saved {len(received_requests)} requests to {output_file}")


def create_app() -> web.Application:
    """Create the web application."""
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/health", health)
    app.router.add_get("/requests", get_requests)
    app.router.add_post("/reset", reset)
    return app


async def cleanup(app: web.Application) -> None:
    """Cleanup handler."""
    save_requests()


def main():
    """Main entry point."""
    import sys
    
    global output_file
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("recorded_requests.json")
    
    logger.info(f"Starting mock server on port {port}")
    logger.info(f"Requests will be saved to {output_file}")
    
    app = create_app()
    app.on_cleanup.append(cleanup)
    
    try:
        web.run_app(app, host="127.0.0.1", port=port)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        save_requests()
        logger.info(f"Saved {len(received_requests)} requests to {output_file}")


if __name__ == "__main__":
    main()
