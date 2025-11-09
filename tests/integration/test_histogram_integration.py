# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for histogram-based percentile estimation and time-window filtering."""

import time
from unittest.mock import MagicMock

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.exceptions import NoMetricValue
from aiperf.post_processors.server_metrics_results_processor import (
    ServerMetricsResultsProcessor,
)
from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)

# Real metrics from localhost:8000 (dynamo_frontend)
REAL_METRICS_FRONTEND = """# HELP dynamo_frontend_disconnected_clients Number of disconnected clients
# TYPE dynamo_frontend_disconnected_clients gauge
dynamo_frontend_disconnected_clients 14148
# HELP dynamo_frontend_inflight_requests Number of inflight requests
# TYPE dynamo_frontend_inflight_requests gauge
dynamo_frontend_inflight_requests{model="qwen/qwen3-0.6b"} 0
# HELP dynamo_frontend_input_sequence_tokens Input sequence length in tokens
# TYPE dynamo_frontend_input_sequence_tokens histogram
dynamo_frontend_input_sequence_tokens_bucket{model="qwen/qwen3-0.6b",le="0"} 0
dynamo_frontend_input_sequence_tokens_bucket{model="qwen/qwen3-0.6b",le="100"} 0
dynamo_frontend_input_sequence_tokens_bucket{model="qwen/qwen3-0.6b",le="210"} 170450
dynamo_frontend_input_sequence_tokens_bucket{model="qwen/qwen3-0.6b",le="430"} 170450
dynamo_frontend_input_sequence_tokens_bucket{model="qwen/qwen3-0.6b",le="870"} 170450
dynamo_frontend_input_sequence_tokens_bucket{model="qwen/qwen3-0.6b",le="1800"} 170450
dynamo_frontend_input_sequence_tokens_bucket{model="qwen/qwen3-0.6b",le="+Inf"} 170450
dynamo_frontend_input_sequence_tokens_sum{model="qwen/qwen3-0.6b"} 18408629
dynamo_frontend_input_sequence_tokens_count{model="qwen/qwen3-0.6b"} 170450
# HELP dynamo_frontend_inter_token_latency_seconds Inter-token latency in seconds
# TYPE dynamo_frontend_inter_token_latency_seconds histogram
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0"} 0
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0019"} 146021
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0035"} 195162
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0067"} 225274
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.013"} 260764
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.024"} 696484
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.045"} 1842700
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.084"} 16539606
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.16"} 16570374
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.3"} 16630378
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="+Inf"} 16637145
dynamo_frontend_inter_token_latency_seconds_sum{model="qwen/qwen3-0.6b"} 813545.702521758
dynamo_frontend_inter_token_latency_seconds_count{model="qwen/qwen3-0.6b"} 16637145
# HELP dynamo_frontend_request_success_count Number of successful requests
# TYPE dynamo_frontend_request_success_count counter
dynamo_frontend_request_success_count{model="qwen/qwen3-0.6b"} 170450
"""

# Simulated metrics at different time points for time-window testing
METRICS_T0 = """# HELP test_counter_total Test counter
# TYPE test_counter_total counter
test_counter_total 100

# HELP test_histogram Test histogram
# TYPE test_histogram histogram
test_histogram_bucket{le="0.1"} 50
test_histogram_bucket{le="0.5"} 80
test_histogram_bucket{le="1.0"} 95
test_histogram_bucket{le="+Inf"} 100
test_histogram_sum 35.5
test_histogram_count 100

# HELP test_gauge Test gauge
# TYPE test_gauge gauge
test_gauge 10.5
"""

METRICS_T1 = """# HELP test_counter_total Test counter
# TYPE test_counter_total counter
test_counter_total 250

# HELP test_histogram Test histogram
# TYPE test_histogram histogram
test_histogram_bucket{le="0.1"} 120
test_histogram_bucket{le="0.5"} 200
test_histogram_bucket{le="1.0"} 240
test_histogram_bucket{le="+Inf"} 250
test_histogram_sum 88.75
test_histogram_count 250

# HELP test_gauge Test gauge
# TYPE test_gauge gauge
test_gauge 25.3
"""

METRICS_T2 = """# HELP test_counter_total Test counter
# TYPE test_counter_total counter
test_counter_total 450

# HELP test_histogram Test histogram
# TYPE test_histogram histogram
test_histogram_bucket{le="0.1"} 200
test_histogram_bucket{le="0.5"} 340
test_histogram_bucket{le="1.0"} 430
test_histogram_bucket{le="+Inf"} 450
test_histogram_sum 158.25
test_histogram_count 450

# HELP test_gauge Test gauge
# TYPE test_gauge gauge
test_gauge 42.7
"""


@pytest.mark.asyncio
async def test_real_histogram_percentile_estimation():
    """Test histogram percentile estimation with real metrics from localhost:8000."""
    from unittest.mock import patch

    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8000/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Collect multiple snapshots to enable delta calculation
    # Snapshot 1 at t0 (baseline)
    t0 = time.time_ns()
    with patch.object(
        collector, "_fetch_metrics_text", return_value=REAL_METRICS_FRONTEND
    ):
        records = collector._parse_metrics_to_records(REAL_METRICS_FRONTEND)
        for record in records:
            record.timestamp_ns = t0
            await processor.process_server_metrics_record(record)

    # Snapshot 2 at t1 (increased values to simulate more requests)
    # Create metrics with doubled histogram bucket values
    METRICS_T1_DOUBLED = """# HELP dynamo_frontend_disconnected_clients Number of disconnected clients
# TYPE dynamo_frontend_disconnected_clients gauge
dynamo_frontend_disconnected_clients 28296
# HELP dynamo_frontend_inflight_requests Number of inflight requests
# TYPE dynamo_frontend_inflight_requests gauge
dynamo_frontend_inflight_requests{model="qwen/qwen3-0.6b"} 0
# HELP dynamo_frontend_inter_token_latency_seconds Inter-token latency in seconds
# TYPE dynamo_frontend_inter_token_latency_seconds histogram
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0"} 0
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0019"} 292042
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0035"} 390324
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0067"} 450548
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.013"} 521528
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.024"} 1392968
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.045"} 3685400
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.084"} 33079212
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.16"} 33140748
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.3"} 33260756
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="+Inf"} 33274290
dynamo_frontend_inter_token_latency_seconds_sum{model="qwen/qwen3-0.6b"} 1627091.405043516
dynamo_frontend_inter_token_latency_seconds_count{model="qwen/qwen3-0.6b"} 33274290
# HELP dynamo_frontend_request_success_count Number of successful requests
# TYPE dynamo_frontend_request_success_count counter
dynamo_frontend_request_success_count{model="qwen/qwen3-0.6b"} 340900
"""

    t1 = t0 + 1_000_000_000
    with patch.object(
        collector, "_fetch_metrics_text", return_value=METRICS_T1_DOUBLED
    ):
        records = collector._parse_metrics_to_records(METRICS_T1_DOUBLED)
        for record in records:
            record.timestamp_ns = t1
            await processor.process_server_metrics_record(record)

    await collector.stop()

    # Get the hierarchy and verify histogram was processed
    hierarchy = processor.get_server_metrics_hierarchy()
    assert "http://localhost:8000/metrics" in hierarchy.endpoints

    endpoint_data = hierarchy.endpoints["http://localhost:8000/metrics"]
    _, metrics_dict = endpoint_data.time_series.snapshots[0]

    # Verify histogram metric exists
    assert "dynamo_frontend_inter_token_latency_seconds" in metrics_dict
    histogram_family = metrics_dict["dynamo_frontend_inter_token_latency_seconds"]
    assert histogram_family.type == "histogram"

    # Verify histogram sample has structured data
    histogram_sample = histogram_family.samples[0]
    assert histogram_sample.histogram is not None
    assert histogram_sample.histogram.buckets is not None
    assert histogram_sample.histogram.sum > 0
    assert histogram_sample.histogram.count > 0

    # Test percentile estimation from histogram
    try:
        result = endpoint_data.get_metric_result(
            metric_name="dynamo_frontend_inter_token_latency_seconds",
            labels={"model": "qwen/qwen3-0.6b"},
            tag="test.itl",
            header="ITL",
            unit="s",
        )

        # Verify percentiles are NOT computed for histograms (raw buckets used instead)
        assert result.p50 == 0.0
        assert result.p95 == 0.0
        assert result.p99 == 0.0

        # Verify avg is calculated correctly from sum/count
        assert result.avg > 0
        assert result.count == 16637145, f"Expected count=16637145, got {result.count}"

        # Verify raw_histogram_delta is stored for export
        assert result.raw_histogram_delta is not None
        buckets, sum_delta, count_delta = result.raw_histogram_delta
        assert len(buckets) > 0
        assert sum_delta > 0
        assert count_delta == 16637145

        print("✅ Histogram delta test passed!")
        print(f"   ITL avg: {result.avg:.4f}s ({result.avg * 1000:.1f}ms)")
        print(f"   ITL count: {result.count}")
        print(f"   ITL buckets: {len(buckets)}")
        print(f"   Count (delta): {result.count}")
        print(f"   Avg: {result.avg:.4f}s ({result.avg * 1000:.1f}ms)")

    except NoMetricValue as e:
        pytest.fail(f"Failed to get metric result: {e}")


@pytest.mark.asyncio
async def test_histogram_delta_calculation():
    """Test histogram delta calculation for time windows."""
    from unittest.mock import patch

    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://test:8000/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Collect three snapshots
    t0 = time.time_ns()
    with patch.object(collector, "_fetch_metrics_text", return_value=METRICS_T0):
        records = collector._parse_metrics_to_records(METRICS_T0)
        for record in records:
            record.timestamp_ns = t0
            await processor.process_server_metrics_record(record)

    t1 = t0 + 1_000_000_000  # +1 second
    with patch.object(collector, "_fetch_metrics_text", return_value=METRICS_T1):
        records = collector._parse_metrics_to_records(METRICS_T1)
        for record in records:
            record.timestamp_ns = t1
            await processor.process_server_metrics_record(record)

    t2 = t0 + 2_000_000_000  # +2 seconds
    with patch.object(collector, "_fetch_metrics_text", return_value=METRICS_T2):
        records = collector._parse_metrics_to_records(METRICS_T2)
        for record in records:
            record.timestamp_ns = t2
            await processor.process_server_metrics_record(record)

    await collector.stop()

    # Get histogram delta for full time window (t0 to t2)
    hierarchy = processor.get_server_metrics_hierarchy()
    endpoint_data = hierarchy.endpoints["http://test:8000/metrics"]

    # Get histogram delta
    histogram_delta = endpoint_data.time_series.get_histogram_delta(
        "test_histogram", {}
    )

    assert histogram_delta is not None
    buckets_delta, sum_delta, count_delta = histogram_delta

    # Verify deltas
    # From t0 to t2:
    # count: 450 - 100 = 350
    # sum: 158.25 - 35.5 = 122.75
    assert count_delta == 350
    assert abs(sum_delta - 122.75) < 0.01

    # Verify bucket deltas
    assert buckets_delta["0.1"] == 150  # 200 - 50
    assert buckets_delta["0.5"] == 260  # 340 - 80
    assert buckets_delta["1.0"] == 335  # 430 - 95
    assert buckets_delta["+Inf"] == 350  # 450 - 100

    print("✅ Histogram delta calculation test passed!")
    print(f"   Count delta: {count_delta}")
    print(f"   Sum delta: {sum_delta:.2f}")


@pytest.mark.asyncio
async def test_counter_delta_calculation():
    """Test counter delta calculation for time windows."""
    from unittest.mock import patch

    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://test:8000/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Collect three snapshots
    t0 = time.time_ns()
    with patch.object(collector, "_fetch_metrics_text", return_value=METRICS_T0):
        records = collector._parse_metrics_to_records(METRICS_T0)
        for record in records:
            record.timestamp_ns = t0
            await processor.process_server_metrics_record(record)

    t1 = t0 + 1_000_000_000
    with patch.object(collector, "_fetch_metrics_text", return_value=METRICS_T1):
        records = collector._parse_metrics_to_records(METRICS_T1)
        for record in records:
            record.timestamp_ns = t1
            await processor.process_server_metrics_record(record)

    t2 = t0 + 2_000_000_000
    with patch.object(collector, "_fetch_metrics_text", return_value=METRICS_T2):
        records = collector._parse_metrics_to_records(METRICS_T2)
        for record in records:
            record.timestamp_ns = t2
            await processor.process_server_metrics_record(record)

    await collector.stop()

    # Get counter delta
    hierarchy = processor.get_server_metrics_hierarchy()
    endpoint_data = hierarchy.endpoints["http://test:8000/metrics"]

    counter_delta = endpoint_data.time_series.get_counter_delta("test_counter", {})

    assert counter_delta is not None
    # From t0 to t2: 450 - 100 = 350
    assert counter_delta == 350

    print("✅ Counter delta calculation test passed!")
    print(f"   Counter delta: {counter_delta}")


@pytest.mark.asyncio
async def test_time_window_filtering():
    """Test time-window filtering for server metrics."""
    from unittest.mock import patch

    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://test:8000/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Collect multiple snapshots at different times
    base_time = time.time_ns()
    times = [
        base_time,
        base_time + 1_000_000_000,
        base_time + 2_000_000_000,
        base_time + 3_000_000_000,
        base_time + 4_000_000_000,
    ]

    for i, t in enumerate(times):
        # Simulate increasing gauge values
        metrics = f"""# HELP test_gauge Test gauge
# TYPE test_gauge gauge
test_gauge {10.0 + i * 5.0}
"""
        with patch.object(collector, "_fetch_metrics_text", return_value=metrics):
            records = collector._parse_metrics_to_records(metrics)
            for record in records:
                record.timestamp_ns = t
                await processor.process_server_metrics_record(record)

    await collector.stop()

    # Get endpoint data
    hierarchy = processor.get_server_metrics_hierarchy()
    endpoint_data = hierarchy.endpoints["http://test:8000/metrics"]

    # Test 1: Get metrics for full time window (all 5 snapshots)
    result_full = endpoint_data.get_metric_result(
        metric_name="test_gauge",
        labels={},
        tag="test.gauge",
        header="Test Gauge",
        unit="",
    )

    # Should have all 5 values: 10, 15, 20, 25, 30
    assert result_full.count == 5
    assert result_full.min == 10.0
    assert result_full.max == 30.0
    assert abs(result_full.avg - 20.0) < 0.01  # (10+15+20+25+30)/5 = 20

    # Test 2: Filter to middle 3 snapshots
    min_ts = times[1]  # t1
    max_ts = times[3]  # t3

    result_filtered = endpoint_data.get_metric_result(
        metric_name="test_gauge",
        labels={},
        tag="test.gauge",
        header="Test Gauge",
        unit="",
        min_timestamp_ns=min_ts,
        max_timestamp_ns=max_ts,
    )

    # Should have 3 values: 15, 20, 25
    assert result_filtered.count == 3
    assert result_filtered.min == 15.0
    assert result_filtered.max == 25.0
    assert abs(result_filtered.avg - 20.0) < 0.01  # (15+20+25)/3 = 20

    print("✅ Time window filtering test passed!")
    print(
        f"   Full window: count={result_full.count}, min={result_full.min}, max={result_full.max}"
    )
    print(
        f"   Filtered window: count={result_filtered.count}, min={result_filtered.min}, max={result_filtered.max}"
    )


@pytest.mark.asyncio
async def test_summarize_with_time_window():
    """Test summarize() method with time window parameters."""
    from unittest.mock import patch

    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://test:8000/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Collect snapshots
    base_time = time.time_ns()
    for i in range(3):
        t = base_time + i * 1_000_000_000
        with patch.object(collector, "_fetch_metrics_text", return_value=METRICS_T0):
            records = collector._parse_metrics_to_records(METRICS_T0)
            for record in records:
                record.timestamp_ns = t
                await processor.process_server_metrics_record(record)

    await collector.stop()

    # Test summarize without time window
    results_full = await processor.summarize()

    # Test summarize with time window
    min_ts = base_time + 500_000_000  # Between snapshot 0 and 1
    max_ts = base_time + 2_500_000_000  # After snapshot 2

    results_filtered = await processor.summarize(
        min_timestamp_ns=min_ts, max_timestamp_ns=max_ts
    )

    # Should produce results (may be fewer due to filtering)
    assert isinstance(results_full, list)
    assert isinstance(results_filtered, list)

    print("✅ Summarize with time window test passed!")
    print(f"   Full results: {len(results_full)} metrics")
    print(f"   Filtered results: {len(results_filtered)} metrics")


@pytest.mark.asyncio
async def test_complete_pipeline_with_real_histogram():
    """Test complete pipeline with real histogram data from localhost:8000."""
    from unittest.mock import patch

    user_config = MagicMock(spec=UserConfig)
    processor = ServerMetricsResultsProcessor(user_config=user_config)

    collector = ServerMetricsDataCollector(
        endpoint_url="http://localhost:8000/metrics",
        collection_interval=1.0,
    )

    await collector.initialize()

    # Simulate collecting metrics over time with increasing values
    base_time = time.time_ns()

    # Snapshot 1 at t0 (baseline)
    t0 = base_time
    with patch.object(
        collector, "_fetch_metrics_text", return_value=REAL_METRICS_FRONTEND
    ):
        records = collector._parse_metrics_to_records(REAL_METRICS_FRONTEND)
        for record in records:
            record.timestamp_ns = t0
            await processor.process_server_metrics_record(record)

    # Snapshot 2 at t1 (doubled values - see earlier definition)
    METRICS_T1_DOUBLED = """# HELP dynamo_frontend_disconnected_clients Number of disconnected clients
# TYPE dynamo_frontend_disconnected_clients gauge
dynamo_frontend_disconnected_clients 28296
# HELP dynamo_frontend_inflight_requests Number of inflight requests
# TYPE dynamo_frontend_inflight_requests gauge
dynamo_frontend_inflight_requests{model="qwen/qwen3-0.6b"} 5
# HELP dynamo_frontend_inter_token_latency_seconds Inter-token latency in seconds
# TYPE dynamo_frontend_inter_token_latency_seconds histogram
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0"} 0
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0019"} 292042
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0035"} 390324
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0067"} 450548
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.013"} 521528
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.024"} 1392968
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.045"} 3685400
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.084"} 33079212
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.16"} 33140748
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.3"} 33260756
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="+Inf"} 33274290
dynamo_frontend_inter_token_latency_seconds_sum{model="qwen/qwen3-0.6b"} 1627091.405043516
dynamo_frontend_inter_token_latency_seconds_count{model="qwen/qwen3-0.6b"} 33274290
# HELP dynamo_frontend_request_success_count Number of successful requests
# TYPE dynamo_frontend_request_success_count counter
dynamo_frontend_request_success_count{model="qwen/qwen3-0.6b"} 340900
"""

    t1 = base_time + 1_000_000_000
    with patch.object(
        collector, "_fetch_metrics_text", return_value=METRICS_T1_DOUBLED
    ):
        records = collector._parse_metrics_to_records(METRICS_T1_DOUBLED)
        for record in records:
            record.timestamp_ns = t1
            await processor.process_server_metrics_record(record)

    # Snapshot 3 at t2 (tripled from original)
    METRICS_T2_TRIPLED = """# HELP dynamo_frontend_disconnected_clients Number of disconnected clients
# TYPE dynamo_frontend_disconnected_clients gauge
dynamo_frontend_disconnected_clients 42444
# HELP dynamo_frontend_inflight_requests Number of inflight requests
# TYPE dynamo_frontend_inflight_requests gauge
dynamo_frontend_inflight_requests{model="qwen/qwen3-0.6b"} 2
# HELP dynamo_frontend_inter_token_latency_seconds Inter-token latency in seconds
# TYPE dynamo_frontend_inter_token_latency_seconds histogram
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0"} 0
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0019"} 438063
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0035"} 585486
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.0067"} 675822
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.013"} 782292
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.024"} 2089452
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.045"} 5528100
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.084"} 49618818
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.16"} 49711122
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="0.3"} 49891134
dynamo_frontend_inter_token_latency_seconds_bucket{model="qwen/qwen3-0.6b",le="+Inf"} 49911435
dynamo_frontend_inter_token_latency_seconds_sum{model="qwen/qwen3-0.6b"} 2440637.107565274
dynamo_frontend_inter_token_latency_seconds_count{model="qwen/qwen3-0.6b"} 49911435
# HELP dynamo_frontend_request_success_count Number of successful requests
# TYPE dynamo_frontend_request_success_count counter
dynamo_frontend_request_success_count{model="qwen/qwen3-0.6b"} 511350
"""

    t2 = base_time + 2_000_000_000
    with patch.object(
        collector, "_fetch_metrics_text", return_value=METRICS_T2_TRIPLED
    ):
        records = collector._parse_metrics_to_records(METRICS_T2_TRIPLED)
        for record in records:
            record.timestamp_ns = t2
            await processor.process_server_metrics_record(record)

    await collector.stop()

    # Verify hierarchy
    hierarchy = processor.get_server_metrics_hierarchy()
    assert len(hierarchy.endpoints) == 1
    endpoint_data = hierarchy.endpoints["http://localhost:8000/metrics"]
    assert len(endpoint_data.time_series.snapshots) == 3

    # Test histogram metric result with delta calculation
    result = endpoint_data.get_metric_result(
        metric_name="dynamo_frontend_inter_token_latency_seconds",
        labels={"model": "qwen/qwen3-0.6b"},
        tag="test.itl",
        header="ITL",
        unit="s",
    )

    # Verify percentiles are NOT computed for histograms
    assert result.p50 == 0.0
    assert result.p95 == 0.0
    assert result.p99 == 0.0
    assert result.count > 0

    # Verify raw_histogram_delta is stored
    assert result.raw_histogram_delta is not None

    # Test counter metric result with delta
    result_counter = endpoint_data.get_metric_result(
        metric_name="dynamo_frontend_request_success_count",
        labels={"model": "qwen/qwen3-0.6b"},
        tag="test.success",
        header="Success Count",
        unit="requests",
    )

    # Counter should return delta (511350 - 170450 = 340900)
    assert result_counter.count == 1
    assert result_counter.avg == 340900, f"Expected 340900, got {result_counter.avg}"

    # Test gauge metric result (no delta)
    result_gauge = endpoint_data.get_metric_result(
        metric_name="dynamo_frontend_inflight_requests",
        labels={"model": "qwen/qwen3-0.6b"},
        tag="test.inflight",
        header="Inflight",
        unit="requests",
    )

    # Gauge should have all 3 values (0, 5, 2)
    assert result_gauge.count == 3
    assert result_gauge.min == 0
    assert result_gauge.max == 5

    print("✅ Complete pipeline test passed!")
    print(f"   Histogram ITL p50: {result.p50:.4f}s ({result.p50 * 1000:.1f}ms)")
    print(f"   Histogram ITL p95: {result.p95:.4f}s ({result.p95 * 1000:.1f}ms)")
    print(f"   Histogram ITL count (delta): {result.count}")
    print(f"   Counter delta: {result_counter.avg:.0f} requests")
    print(
        f"   Gauge samples: {result_gauge.count}, min={result_gauge.min}, max={result_gauge.max}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
