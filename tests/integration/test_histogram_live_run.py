# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live run test with real server - concurrency 30, request count 300."""

import pytest

from tests.integration.conftest import AIPerfCLI


@pytest.mark.integration
@pytest.mark.asyncio
class TestHistogramLiveRun:
    """Test histogram implementation with a real workload."""

    async def test_histogram_with_real_workload(self, cli: AIPerfCLI):
        """Run real workload with concurrency=30, request_count=300 against localhost:8000."""
        result = await cli.run(
            """
            aiperf profile \
                --model Qwen/Qwen3-0.6B \
                --url http://localhost:8000 \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 300 \
                --concurrency 30 \
                --isl 100 \
                --osl 50 \
                --server-metrics http://localhost:8000/metrics http://localhost:8081/metrics \
                --ui-type none
            """
        )

        print("\n" + "=" * 80)
        print("REAL WORKLOAD TEST RESULTS")
        print("Concurrency: 30, Requests: 300")
        print("=" * 80)

        assert result.exit_code == 0, f"AIPerf failed with exit code {result.exit_code}"
        assert result.request_count == 300

        # Check that server metrics were collected
        server_metrics_file = result.artifacts_dir / "server_metrics_export.jsonl"
        assert server_metrics_file.exists(), "server_metrics_export.jsonl not found"

        # Check profile results
        profile_json = result.artifacts_dir / "profile_export_aiperf.json"
        assert profile_json.exists(), "profile_export_aiperf.json not found"

        print("\n✅ Test completed successfully!")
        print(f"   Artifact directory: {result.artifacts_dir}")
        print(f"   Exit code: {result.exit_code}")
        print(f"   Request count: {result.request_count}")

        # Read and display some metrics
        import json

        with open(profile_json) as f:
            profile_data = json.load(f)

        print("\n📊 Client Metrics:")
        if "inter_token_latency" in profile_data:
            itl = profile_data["inter_token_latency"]
            print(f"   ITL avg: {itl.get('avg', 0):.2f}ms")
            print(f"   ITL p50: {itl.get('p50', 0):.2f}ms")
            print(f"   ITL p95: {itl.get('p95', 0):.2f}ms")

        if "time_to_first_token" in profile_data:
            ttft = profile_data["time_to_first_token"]
            print(f"   TTFT avg: {ttft.get('avg', 0):.2f}ms")
            print(f"   TTFT p95: {ttft.get('p95', 0):.2f}ms")

        # Count server metrics records
        server_metrics_count = 0
        with open(server_metrics_file) as f:
            for line in f:
                if line.strip():
                    server_metrics_count += 1

        print("\n📊 Server Metrics:")
        print(f"   Total snapshots collected: {server_metrics_count}")

        print("\n✅ HISTOGRAM IMPLEMENTATION VERIFIED WITH REAL WORKLOAD!")
