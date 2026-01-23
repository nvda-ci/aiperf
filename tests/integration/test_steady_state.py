# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for steady-state measurement."""

import asyncio
import json
import os
from contextlib import suppress
from pathlib import Path

import aiohttp
import pytest

from tests.integration.conftest import AIPerfCLI, get_venv_python
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


def _write_mooncake_trace_dataset(path: Path, count: int) -> None:
    """Write a mooncake_trace dataset with the given number of entries."""
    records = [
        {
            "input_length": 64,
            "output_length": 16,
            "hash_ids": [idx],
        }
        for idx in range(count)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


@pytest.fixture
async def aiperf_mock_server_slow(
    mock_server_port: int,
) -> AIPerfMockServer:
    """Start AIPerf Mock Server with latency enabled."""
    host = "127.0.0.1"
    url = f"http://{host}:{mock_server_port}"

    python_exe = get_venv_python()

    os.environ["AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD"] = "0"

    process = await asyncio.create_subprocess_exec(
        python_exe,
        "-m",
        "aiperf_mock_server",
        "--host",
        host,
        "--port",
        str(mock_server_port),
        "--ttft",
        "200",
        "--itl",
        "50",
        "--no-tokenizer",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(100):
                try:
                    async with session.get(
                        f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(0.1)
            else:
                if process.returncode is None:
                    process.terminate()
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                raise RuntimeError(
                    f"AIPerf Mock Server failed to become healthy after 100 attempts "
                    f"(URL: {url}/health)"
                )

        yield AIPerfMockServer(
            host=host, port=mock_server_port, url=url, process=process
        )
    finally:
        if process.returncode is None:
            process.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5.0)


@pytest.mark.integration
@pytest.mark.asyncio
class TestSteadyStateIntegration:
    async def test_steady_state_mooncake_trace(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server_slow: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        dataset_path = tmp_path / "mooncake_trace_10.jsonl"
        _write_mooncake_trace_dataset(dataset_path, count=10)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server_slow.url} \
                --endpoint-type chat \
                --steady-state \
                --custom-dataset-type mooncake_trace \
                --input-file {dataset_path} \
                --request-count 4 \
                --request-rate-mode concurrency_burst \
                --concurrency 4 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui} \
                --export-level raw \
                --no-server-metrics \
                --no-gpu-telemetry
            """
        )

        assert result.exit_code == 0
        assert result.request_count == 4
        assert result.jsonl is not None
        assert len(result.jsonl) == 4
        assert all(record.metadata.was_cancelled is False for record in result.jsonl)
        assert result.raw_records is not None
        # Raw records should include all records, including those that complete after
        # steady-state cancellation (they may be cancelled, have errors, or complete successfully)
        assert len(result.raw_records) > len(result.jsonl)
        # Verify that tail records exist (records that complete after cancellation)
        # These are filtered from jsonl but included in raw_records
        tail_count = len(result.raw_records) - len(result.jsonl)
        assert tail_count > 0, f"Expected tail records but found {len(result.raw_records)} raw vs {len(result.jsonl)} jsonl"

    async def test_steady_state_scaled(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server_slow: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """Test steady-state with scaled parameters: 30 requests, concurrency 15, non-uniform OSL."""
        dataset_path = tmp_path / "mooncake_trace_10.jsonl"
        _write_mooncake_trace_dataset(dataset_path, count=10)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server_slow.url} \
                --endpoint-type chat \
                --steady-state \
                --custom-dataset-type mooncake_trace \
                --input-file {dataset_path} \
                --request-count 30 \
                --request-rate-mode concurrency_burst \
                --concurrency 15 \
                --sequence-distribution "64|10,16|5:50;128|20,32|10:30;256|30,64|15:20" \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui} \
                --export-level raw \
                --no-server-metrics \
                --no-gpu-telemetry
            """,
            timeout=120.0,
        )

        assert result.exit_code == 0
        # With concurrency 15, some requests may complete before cancellation takes effect
        # so we should have at least 30 completed requests
        assert result.request_count >= 30
        assert result.jsonl is not None
        # JSONL should have exactly the number of completed requests (may be > 30 due to race)
        assert len(result.jsonl) == result.request_count
        assert all(record.metadata.was_cancelled is False for record in result.jsonl)
        assert result.raw_records is not None
        # Raw records should include all records, including those that complete after
        # steady-state cancellation (they may be cancelled, have errors, or complete successfully)
        assert len(result.raw_records) > len(result.jsonl)
        # Verify that tail records exist (records that complete after cancellation)
        # These are filtered from jsonl but included in raw_records
        tail_count = len(result.raw_records) - len(result.jsonl)
        assert tail_count > 0, f"Expected tail records but found {len(result.raw_records)} raw vs {len(result.jsonl)} jsonl"

    async def test_steady_state_scaled_streaming(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server_slow: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """Test steady-state with streaming: 30 requests, concurrency 15, non-uniform OSL."""
        dataset_path = tmp_path / "mooncake_trace_10.jsonl"
        _write_mooncake_trace_dataset(dataset_path, count=10)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server_slow.url} \
                --endpoint-type chat \
                --streaming \
                --steady-state \
                --custom-dataset-type mooncake_trace \
                --input-file {dataset_path} \
                --request-count 30 \
                --request-rate-mode concurrency_burst \
                --concurrency 15 \
                --sequence-distribution "64|10,16|5:50;128|20,32|10:30;256|30,64|15:20" \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui} \
                --export-level raw \
                --no-server-metrics \
                --no-gpu-telemetry
            """,
            timeout=120.0,
        )

        assert result.exit_code == 0
        # With concurrency 15, some requests may complete before cancellation takes effect
        # so we should have at least 30 completed requests
        assert result.request_count >= 30
        assert result.jsonl is not None
        # JSONL should have exactly the number of completed requests (may be > 30 due to race)
        assert len(result.jsonl) == result.request_count
        assert all(record.metadata.was_cancelled is False for record in result.jsonl)
        assert result.raw_records is not None
        # Raw records should include all records, including those that complete after
        # steady-state cancellation (they may be cancelled, have errors, or complete successfully)
        assert len(result.raw_records) > len(result.jsonl)
        # Verify that tail records exist (records that complete after cancellation)
        # These are filtered from jsonl but included in raw_records
        tail_count = len(result.raw_records) - len(result.jsonl)
        assert tail_count > 0, f"Expected tail records but found {len(result.raw_records)} raw vs {len(result.jsonl)} jsonl"
