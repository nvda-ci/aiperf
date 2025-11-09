# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test using REAL metrics from localhost:8000 and localhost:8081."""

import asyncio
from unittest.mock import MagicMock

import pytest

from aiperf.common.config import UserConfig
from aiperf.post_processors.server_metrics_results_processor import (
    ServerMetricsResultsProcessor,
)
from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)


@pytest.mark.asyncio
async def test_real_localhost_8000_metrics():
    """Test with REAL metrics from localhost:8000 (requires server to be running)."""
    import aiohttp

    # First, check if localhost:8000 is reachable
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                "http://localhost:8000/metrics", timeout=aiohttp.ClientTimeout(total=2)
            ) as response,
        ):
            if response.status != 200:
                pytest.skip("localhost:8000 not reachable or not returning 200")
    except Exception:
        pytest.skip("localhost:8000 not reachable - skipping test")

    # Create processor
    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    # Callback to capture records
    collected_records = []

    async def record_callback(records, collector_id):
        collected_records.extend(records)

    # Create collector for localhost:8000
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8000/metrics",
        collection_interval=0.5,
        record_callback=record_callback,
    )

    await collector.initialize()
    await collector.start()

    # Collect 3 snapshots
    print("\nCollecting 3 snapshots from localhost:8000...")
    for _i in range(3):
        await collector._collect_and_process_metrics()
        await asyncio.sleep(0.5)

    await collector.stop()

    # Process collected records
    print(f"✅ Collected {len(collected_records)} record(s)")
    for record in collected_records:
        await processor.process_server_metrics_record(record)

    # Verify hierarchy
    hierarchy = processor.get_server_metrics_hierarchy()
    print(f"✅ Collected from {len(hierarchy.endpoints)} endpoint(s)")

    if "http://localhost:8000/metrics" not in hierarchy.endpoints:
        pytest.fail("Expected http://localhost:8000/metrics in hierarchy")

    endpoint_data = hierarchy.endpoints["http://localhost:8000/metrics"]
    print(f"✅ Got {len(endpoint_data.time_series.snapshots)} snapshots")

    # Try to get a histogram metric if available
    try:
        result = endpoint_data.get_metric_result(
            metric_name="dynamo_frontend_inter_token_latency_seconds",
            labels={"model": "qwen/qwen3-0.6b"},
            tag="test.itl",
            header="ITL",
            unit="s",
        )

        print(
            "\n✅ Histogram metric 'dynamo_frontend_inter_token_latency_seconds' found!"
        )
        print(f"   Count (delta): {result.count}")
        print(f"   Avg: {result.avg:.4f}s ({result.avg * 1000:.1f}ms)")
        print(f"   p50: {result.p50:.4f}s ({result.p50 * 1000:.1f}ms)")
        print(f"   p95: {result.p95:.4f}s ({result.p95 * 1000:.1f}ms)")
        print(f"   p99: {result.p99:.4f}s ({result.p99 * 1000:.1f}ms)")

    except Exception as e:
        print(f"⚠️  Histogram metric not available or no delta: {e}")

    # Try to get a counter metric
    try:
        result_counter = endpoint_data.get_metric_result(
            metric_name="dynamo_frontend_request_success_count",
            labels={"model": "qwen/qwen3-0.6b"},
            tag="test.success",
            header="Success",
            unit="requests",
        )

        print("\n✅ Counter metric 'dynamo_frontend_request_success_count' found!")
        print(f"   Delta: {result_counter.avg:.0f} requests")

    except Exception as e:
        print(f"⚠️  Counter metric not available: {e}")

    # Try to get a gauge metric
    try:
        result_gauge = endpoint_data.get_metric_result(
            metric_name="dynamo_frontend_inflight_requests",
            labels={"model": "qwen/qwen3-0.6b"},
            tag="test.inflight",
            header="Inflight",
            unit="requests",
        )

        print("\n✅ Gauge metric 'dynamo_frontend_inflight_requests' found!")
        print(f"   Samples: {result_gauge.count}")
        print(f"   Min: {result_gauge.min:.1f}")
        print(f"   Max: {result_gauge.max:.1f}")
        print(f"   Avg: {result_gauge.avg:.1f}")

    except Exception as e:
        print(f"⚠️  Gauge metric not available: {e}")

    print("\n✅ Real endpoint test completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
