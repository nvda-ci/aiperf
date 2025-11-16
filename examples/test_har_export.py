#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example demonstrating HAR export functionality.

This script creates a sample HAR file that can be opened in Chrome DevTools
Network panel for inspection.
"""

import asyncio
from pathlib import Path

import orjson

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import CreditPhase, EndpointType, ExportLevel
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import AioHttpTraceData, MetricRecordMetadata
from aiperf.post_processors.har_results_processor import HARResultsProcessor


async def main():
    """Create a sample HAR file with mock trace data."""

    # Setup configuration
    output_dir = Path("./artifacts")
    output_dir.mkdir(exist_ok=True)

    user_config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["gpt-4"],
            type=EndpointType.CHAT,
            url="https://api.openai.com",
            custom_endpoint="/v1/chat/completions",
        ),
        output=OutputConfig(
            artifact_directory=output_dir,
            export_level=ExportLevel.RECORDS,
        ),
    )

    service_config = ServiceConfig()

    # Create HAR processor
    processor = HARResultsProcessor(
        service_id="test-processor",
        service_config=service_config,
        user_config=user_config,
    )

    # Create sample trace data simulating 3 requests
    base_time = 1704067200000000000  # 2024-01-01 00:00:00 UTC
    base_perf = 1_000_000_000  # Base perf_counter

    for i in range(3):
        request_start = base_time + i * 2_000_000_000  # 2 seconds apart
        perf_offset = i * 2_000_000_000  # Same offset for perf_counter

        # Create trace data for this request (using perf_counter internally)
        trace_data = AioHttpTraceData(
            trace_type="aiohttp",
            # Reference timestamps for conversion
            reference_time_ns=base_time,
            reference_perf_ns=base_perf,
            # Request headers
            request_headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-test-key",
                "User-Agent": "AIPerf/0.3.0",
            },
            # Request timing (perf_counter)
            request_send_start_perf_ns=base_perf + perf_offset,
            request_send_end_perf_ns=base_perf
            + perf_offset
            + 15_000_000,  # 15ms to send
            request_write_timestamps_perf_ns=[base_perf + perf_offset],
            request_write_sizes_bytes=[1200],
            # Response headers
            response_status_code=200,
            response_headers={
                "Content-Type": "application/json",
                "X-Request-ID": f"req-{i + 1}",
                "Content-Length": "2048",
            },
            # Response timing (perf_counter)
            response_receive_start_perf_ns=base_perf
            + perf_offset
            + 500_000_000,  # 500ms TTFB
            response_receive_end_perf_ns=base_perf
            + perf_offset
            + 750_000_000,  # +250ms to receive
            response_receive_timestamps_perf_ns=[
                base_perf + perf_offset + 500_000_000,
                base_perf + perf_offset + 750_000_000,
            ],
            response_receive_sizes_bytes=[1024, 1024],
            # Connection phase (only for first request)
            connection_pool_wait_start_perf_ns=base_perf + perf_offset - 100_000_000
            if i == 0
            else None,
            connection_pool_wait_end_perf_ns=base_perf + perf_offset - 90_000_000
            if i == 0
            else None,
            dns_lookup_start_perf_ns=base_perf + perf_offset - 90_000_000
            if i == 0
            else None,
            dns_lookup_end_perf_ns=base_perf + perf_offset - 85_000_000
            if i == 0
            else None,
            tcp_connect_start_perf_ns=base_perf + perf_offset - 85_000_000
            if i == 0
            else None,
            tcp_connect_end_perf_ns=base_perf + perf_offset - 50_000_000
            if i == 0
            else None,
            # Connection reused for subsequent requests
            connection_reused_perf_ns=base_perf + perf_offset - 10_000_000
            if i > 0
            else None,
        )

        # Create metadata
        metadata = MetricRecordMetadata(
            session_num=i + 1,
            x_request_id=f"test-request-{i + 1}",
            x_correlation_id=f"test-correlation-{i + 1}",
            request_start_ns=request_start,
            request_end_ns=request_start + 750_000_000,
            worker_id="worker-1",
            record_processor_id="processor-1",
            benchmark_phase=CreditPhase.PROFILING,
        )

        # Create record data
        record_data = MetricRecordsData(
            metadata=metadata,
            metrics={"request_latency_ns": 750_000_000},
            trace_data=trace_data,
            error=None,
        )

        # Process the record
        await processor.process_result(record_data)

    # Finalize HAR file
    await processor.summarize()

    print(f"✓ HAR file created: {processor.output_file}")
    print(f"✓ Total entries: {len(processor._entries)}")
    print("\nTo view in Chrome DevTools:")
    print("1. Open Chrome DevTools (F12)")
    print("2. Go to Network tab")
    print("3. Right-click and select 'Import HAR file...'")
    print(f"4. Select: {processor.output_file.absolute()}")

    # Pretty print first entry for inspection
    har_bytes = processor.output_file.read_bytes()
    har_dict = orjson.loads(har_bytes)

    print(f"\n{'=' * 60}")
    print("Sample HAR Entry:")
    print(f"{'=' * 60}")

    entry = har_dict["log"]["entries"][0]
    print(f"Method: {entry['request']['method']}")
    print(f"URL: {entry['request']['url']}")
    print(f"Status: {entry['response']['status']} {entry['response']['statusText']}")
    print("\nTimings (ms):")
    for key, value in entry["timings"].items():
        if value is not None:
            print(f"  {key:12s}: {value:>8.2f}")
    print(f"  {'total':12s}: {entry['time']:>8.2f}")


if __name__ == "__main__":
    asyncio.run(main())
