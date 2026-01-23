# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for steady-state measurement with 3-loop warmup/measurement/tail.

The 3-loop steady-state measurement works as follows:
- Loop 1 (a): Warmup loop - requests a1...aN are sent but not measured
- Loop 2 (b): Measurement loop - requests b1...bN define the measurement window
- Loop 3 (c): Tail loop - requests c1...cN may overlap with measurement

The measurement window is [b1_start, bN_end]. Records that overlap with this window
are included in metrics, which includes:
- Late 'a' requests that complete during measurement
- All 'b' requests
- Early 'c' requests that start before measurement ends
"""

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
    async def test_steady_state_3loop_mooncake_trace(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server_slow: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """Test 3-loop steady-state: dataset_size=10, concurrency=4.
        
        With 3-loop steady-state:
        - Loop 1 (a): 10 warmup requests (a1...a10)
        - Loop 2 (b): 10 measurement requests (b1...b10)
        - Loop 3 (c): tail requests until bN completes
        
        The measurement window includes records overlapping [b1_start, b10_end].
        """
        dataset_size = 10
        dataset_path = tmp_path / "mooncake_trace_10.jsonl"
        _write_mooncake_trace_dataset(dataset_path, count=dataset_size)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server_slow.url} \
                --endpoint-type chat \
                --steady-state \
                --custom-dataset-type mooncake_trace \
                --input-file {dataset_path} \
                --request-count {dataset_size} \
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
        # With 3-loop steady-state, we expect records from measurement window (+ overlaps)
        # The exact count depends on timing overlaps
        assert result.request_count >= dataset_size  # At least the measurement loop
        assert result.jsonl is not None
        # JSONL contains records that overlap with measurement window
        assert len(result.jsonl) == result.request_count
        assert result.raw_records is not None
        # Raw records include all records (warmup + measurement + tail)
        # With 3 loops and some overlap, we expect significantly more raw records
        assert len(result.raw_records) >= 2 * dataset_size  # At least warmup + measurement

    async def test_steady_state_3loop_scaled(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server_slow: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """Test 3-loop steady-state with scaled parameters: dataset_size=10, concurrency=10.
        
        With high concurrency (10) and dataset_size (10):
        - Loop 1: 10 warmup requests
        - Loop 2: 10 measurement requests (measurement window)
        - Loop 3: tail requests overlap with measurement
        
        Many warmup requests will complete during measurement, and many tail requests
        will start during measurement, so we expect significant overlap.
        """
        dataset_size = 10
        concurrency = 10
        dataset_path = tmp_path / "mooncake_trace_10.jsonl"
        _write_mooncake_trace_dataset(dataset_path, count=dataset_size)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server_slow.url} \
                --endpoint-type chat \
                --steady-state \
                --custom-dataset-type mooncake_trace \
                --input-file {dataset_path} \
                --request-count {dataset_size} \
                --request-rate-mode concurrency_burst \
                --concurrency {concurrency} \
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
        # With high concurrency and overlap, we expect more than dataset_size records
        assert result.request_count >= dataset_size
        assert result.jsonl is not None
        assert len(result.jsonl) == result.request_count
        assert result.raw_records is not None
        # Raw records should include all records from warmup, measurement, and tail
        assert len(result.raw_records) >= 2 * dataset_size

    async def test_steady_state_3loop_streaming(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server_slow: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """Test 3-loop steady-state with streaming: dataset_size=10, concurrency=10."""
        dataset_size = 10
        concurrency = 10
        dataset_path = tmp_path / "mooncake_trace_10.jsonl"
        _write_mooncake_trace_dataset(dataset_path, count=dataset_size)

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
                --request-count {dataset_size} \
                --request-rate-mode concurrency_burst \
                --concurrency {concurrency} \
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
        assert result.request_count >= dataset_size
        assert result.jsonl is not None
        assert len(result.jsonl) == result.request_count
        assert result.raw_records is not None
        assert len(result.raw_records) >= 2 * dataset_size
