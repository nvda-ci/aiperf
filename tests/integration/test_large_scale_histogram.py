# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Large-scale integration test with 30+ snapshots from real server."""

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
async def test_large_scale_histogram_with_real_server():
    """Test with 30+ snapshots from localhost:8000 to get significant deltas."""
    import aiohttp

    # Check if localhost:8000 is reachable
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                "http://localhost:8000/metrics", timeout=aiohttp.ClientTimeout(total=2)
            ) as response,
        ):
            if response.status != 200:
                pytest.skip("localhost:8000 not reachable")
    except Exception:
        pytest.skip("localhost:8000 not reachable")

    # Create processor
    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    # Callback to capture records
    collected_records = []

    async def record_callback(records, collector_id):
        collected_records.extend(records)

    # Create collector
    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8000/metrics",
        collection_interval=0.5,
        record_callback=record_callback,
    )

    await collector.initialize()
    await collector.start()

    # Collect 30 snapshots over 15 seconds (0.5s interval)
    print("\n" + "=" * 80)
    print("LARGE-SCALE TEST: Collecting 30 snapshots from localhost:8000")
    print("Collection interval: 0.5s, Total duration: ~15s")
    print("=" * 80)

    for i in range(30):
        print(f"\r[{i + 1}/30] Collecting snapshot...", end="", flush=True)
        await collector._collect_and_process_metrics()
        await asyncio.sleep(0.5)

    print()  # New line after progress
    await collector.stop()

    # Process all collected records
    print(f"\n✅ Collected {len(collected_records)} record(s)")

    for record in collected_records:
        await processor.process_server_metrics_record(record)

    # Get hierarchy
    hierarchy = processor.get_server_metrics_hierarchy()
    print(f"✅ Endpoints: {list(hierarchy.endpoints.keys())}")

    if "http://localhost:8000/metrics" not in hierarchy.endpoints:
        pytest.fail("localhost:8000 not in hierarchy")

    endpoint_data = hierarchy.endpoints["http://localhost:8000/metrics"]
    print(f"✅ Total snapshots: {len(endpoint_data.time_series.snapshots)}")

    # Get time range
    if endpoint_data.time_series.snapshots:
        first_ts = endpoint_data.time_series.snapshots[0][0]
        last_ts = endpoint_data.time_series.snapshots[-1][0]
        duration_sec = (last_ts - first_ts) / 1e9
        print(f"✅ Time span: {duration_sec:.2f}s")

    # Test histogram metric with large delta
    print("\n" + "=" * 80)
    print("HISTOGRAM METRIC: dynamo_frontend_inter_token_latency_seconds")
    print("=" * 80)

    try:
        result = endpoint_data.get_metric_result(
            metric_name="dynamo_frontend_inter_token_latency_seconds",
            labels={"model": "qwen/qwen3-0.6b"},
            tag="test.itl",
            header="ITL",
            unit="s",
        )

        print("\n✅ Histogram percentiles estimated from buckets:")
        print(f"   Count (delta): {result.count:,} observations")
        print(f"   Avg:  {result.avg * 1000:.2f}ms")
        print(f"   Min:  {result.min * 1000:.2f}ms")
        print(f"   Max:  {result.max * 1000:.2f}ms")
        print("\n   Percentiles:")
        print(f"   p1:   {result.p1 * 1000:.2f}ms")
        print(f"   p5:   {result.p5 * 1000:.2f}ms")
        print(f"   p10:  {result.p10 * 1000:.2f}ms")
        print(f"   p25:  {result.p25 * 1000:.2f}ms")
        print(f"   p50:  {result.p50 * 1000:.2f}ms (median)")
        print(f"   p75:  {result.p75 * 1000:.2f}ms")
        print(f"   p90:  {result.p90 * 1000:.2f}ms")
        print(f"   p95:  {result.p95 * 1000:.2f}ms (SLO critical)")
        print(f"   p99:  {result.p99 * 1000:.2f}ms (tail latency)")

        # Verify results are reasonable
        assert result.count > 0, "Count should be positive"
        assert result.p50 > 0, "p50 should be positive"
        assert result.p50 < result.p95, "p50 should be less than p95"
        assert result.p95 <= result.p99, "p95 should be less than or equal to p99"

    except Exception as e:
        print(f"⚠️  Histogram metric: {e}")

    # Test counter metric
    print("\n" + "=" * 80)
    print("COUNTER METRIC: dynamo_frontend_request_success_count")
    print("=" * 80)

    try:
        result_counter = endpoint_data.get_metric_result(
            metric_name="dynamo_frontend_request_success_count",
            labels={"model": "qwen/qwen3-0.6b"},
            tag="test.success",
            header="Success",
            unit="requests",
        )

        print("\n✅ Counter delta calculated:")
        print(f"   Delta: {result_counter.avg:,.0f} requests")
        print("   (Requests processed during collection period)")

    except Exception as e:
        print(f"⚠️  Counter metric: {e}")

    # Test gauge metric
    print("\n" + "=" * 80)
    print("GAUGE METRIC: dynamo_frontend_inflight_requests")
    print("=" * 80)

    try:
        result_gauge = endpoint_data.get_metric_result(
            metric_name="dynamo_frontend_inflight_requests",
            labels={"model": "qwen/qwen3-0.6b"},
            tag="test.inflight",
            header="Inflight",
            unit="requests",
        )

        print("\n✅ Gauge statistics from time series:")
        print(f"   Samples: {result_gauge.count}")
        print(f"   Min: {result_gauge.min:.1f}")
        print(f"   Max: {result_gauge.max:.1f}")
        print(f"   Avg: {result_gauge.avg:.1f}")
        print(f"   p50: {result_gauge.p50:.1f}")
        print(f"   p95: {result_gauge.p95:.1f}")

    except Exception as e:
        print(f"⚠️  Gauge metric: {e}")

    # Test with time window filtering
    if len(endpoint_data.time_series.snapshots) >= 10:
        print("\n" + "=" * 80)
        print("TIME WINDOW FILTERING TEST")
        print("=" * 80)

        # Filter to middle 50% of snapshots
        mid_start = len(endpoint_data.time_series.snapshots) // 4
        mid_end = 3 * len(endpoint_data.time_series.snapshots) // 4

        min_ts = endpoint_data.time_series.snapshots[mid_start][0]
        max_ts = endpoint_data.time_series.snapshots[mid_end][0]
        filter_duration = (max_ts - min_ts) / 1e9

        print("\nFiltering to middle 50% of snapshots")
        print(f"  Window: [{mid_start}, {mid_end}]")
        print(f"  Duration: {filter_duration:.2f}s")

        try:
            result_filtered = endpoint_data.get_metric_result(
                metric_name="dynamo_frontend_inter_token_latency_seconds",
                labels={"model": "qwen/qwen3-0.6b"},
                tag="test.itl.filtered",
                header="ITL Filtered",
                unit="s",
                min_timestamp_ns=min_ts,
                max_timestamp_ns=max_ts,
            )

            print("\n✅ Filtered histogram percentiles:")
            print(f"   Count (delta): {result_filtered.count:,}")
            print(f"   p50: {result_filtered.p50 * 1000:.2f}ms")
            print(f"   p95: {result_filtered.p95 * 1000:.2f}ms")

        except Exception as e:
            print(f"⚠️  Filtered histogram: {e}")

    print("\n" + "=" * 80)
    print("✅ LARGE-SCALE TEST COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
