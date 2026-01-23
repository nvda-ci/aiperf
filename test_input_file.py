#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test script to verify that --input-file requests are sent to the server."""

import asyncio
import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def wait_for_server(url: str, timeout: float = 10.0) -> bool:
    """Wait for server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(0.1)
    return False


async def get_recorded_requests(url: str) -> dict:
    """Get recorded requests from the mock server."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{url}/requests") as resp:
            return await resp.json()


async def reset_server(url: str) -> None:
    """Reset the mock server."""
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{url}/reset") as resp:
            assert resp.status == 200


async def main():
    """Main test function."""
    # Find free port
    port = find_free_port()
    server_url = f"http://127.0.0.1:{port}"
    
    # Paths - use converted MultiTurn JSONL format
    input_file = Path("../playground/aiperf_standard_inputs_multiturn.jsonl")
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error("Run convert_inputs_file_to_multiturn.py first to convert InputsFile format")
        sys.exit(1)
    
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    recorded_requests_file = output_dir / "recorded_requests.json"
    
    # Start mock server
    logger.info(f"Starting mock server on port {port}")
    server_process = subprocess.Popen(
        [sys.executable, "test_input_file_server.py", str(port), str(recorded_requests_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    try:
        # Wait for server to be ready
        logger.info("Waiting for server to be ready...")
        if not await wait_for_server(server_url):
            logger.error("Server failed to start")
            sys.exit(1)
        
        logger.info("Server is ready")
        
        # Reset server to clear any initial requests
        await reset_server(server_url)
        
        # Count lines in JSONL file
        with open(input_file) as f:
            line_count = sum(1 for line in f if line.strip())
        logger.info(f"Input file contains {line_count} sessions (JSONL format)")
        
        # Extract model name from input file (read first line of JSONL)
        model_name = "deepseek-r1-nvfp4"  # Default
        try:
            with open(input_file) as f:
                first_line = f.readline()
                if first_line.strip():
                    first_session = json.loads(first_line)
                    if first_session.get("turns") and len(first_session["turns"]) > 0:
                        first_turn = first_session["turns"][0]
                        if "model" in first_turn:
                            model_name = first_turn["model"]
        except Exception as e:
            logger.warning(f"Could not extract model name from input file: {e}")
        
        # Run aiperf with --input-file
        logger.info(f"Running aiperf with --input-file {input_file}")
        logger.info(f"Using model: {model_name}")
        # Test with --custom-dataset-type single_turn (without aiperf modifications)
        cmd = [
            sys.executable, "-m", "aiperf", "profile",
            "--model", model_name,
            "--url", server_url,
            "--endpoint-type", "chat",
            "--input-file", str(input_file),
            "--custom-dataset-type", "multi_turn",  # Use multi_turn for JSONL format
            "--num-requests", "10",  # Limit to 10 requests for testing
            "--artifact-dir", str(output_dir),
            "--max-workers", "1",  # Match user's flag name
            "--concurrency", "1",
            "--tokenizer", "gpt2",  # Use a simple tokenizer that exists
            "--use-server-token-count",  # Use server-reported token counts
            "--ui-type", "none",  # Match user's flag
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        
        logger.info(f"AIPerf exit code: {result.returncode}")
        if result.stdout:
            logger.info(f"AIPerf stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"AIPerf stderr:\n{result.stderr}")
        
        # Wait a bit for any final requests
        await asyncio.sleep(2)
        
        # Get recorded requests
        recorded = await get_recorded_requests(server_url)
        
        logger.info(f"\n{'='*80}")
        logger.info("RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total requests recorded: {recorded['total_requests']}")
        logger.info(f"Request counts: {recorded['request_counts']}")
        
        # Analyze requests
        if recorded['total_requests'] == 0:
            logger.error("❌ NO REQUESTS WERE RECORDED!")
            logger.error("This suggests the input file is not being sent to the server.")
            sys.exit(1)
        
        # Check if we got requests with the expected content
        chat_requests = [r for r in recorded['requests'] if r['endpoint'] == '/v1/chat/completions']
        logger.info(f"Chat completion requests: {len(chat_requests)}")
        
        # Note: For JSONL format, we can't easily count messages without reading the file again
        logger.info("Input file is in MultiTurn JSONL format")
        
        # Compare first few requests
        logger.info("\nFirst few recorded requests:")
        for i, req in enumerate(chat_requests[:3]):
            logger.info(f"\nRequest {i+1}:")
            logger.info(f"  Model: {req['body'].get('model')}")
            logger.info(f"  Messages count: {len(req['body'].get('messages', []))}")
            if req['body'].get('messages'):
                first_msg = req['body']['messages'][0]
                content_preview = str(first_msg.get('content', ''))[:100]
                logger.info(f"  First message preview: {content_preview}...")
        
        logger.info("\n✅ REQUESTS WERE RECORDED!")
        logger.info(f"Saved recorded requests to: {recorded_requests_file}")
        
    finally:
        # Stop server
        logger.info("Stopping mock server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutError:
            server_process.kill()
            server_process.wait()


if __name__ == "__main__":
    asyncio.run(main())
