# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example demonstrating aiohttp trace timing functionality."""

import asyncio

from aiperf.transports.aiohttp_client import AioHttpClient


async def example_with_trace_timing():
    """Example showing how trace timestamps are automatically captured."""
    client = AioHttpClient(timeout=30.0)

    try:
        # Make a request - trace timestamps are automatically captured
        record = await client.get_request(
            "https://httpbin.org/get",
            headers={"User-Agent": "AIPerf-TraceExample/1.0"},
        )

        if record.trace_data:
            timestamps = record.trace_data

            print("=" * 80)
            print("HTTP Trace Timing Report")
            print("=" * 80)
            print(f"URL: {timestamps.request_url}")
            print(f"Method: {timestamps.request_method}")
            print()

            # Connection Information
            print("Connection:")
            if timestamps.connection_was_reused:
                print("  ✓ Connection reused from pool")
                print(
                    f"    Reused at: {timestamps.connection_reuseconn_ns / 1_000_000:.2f} ms"
                )
            else:
                print("  ✗ New connection created")
                if timestamps.connection_create_duration_ns:
                    print(
                        f"    Creation time: {timestamps.connection_create_duration_ns / 1_000_000:.2f} ms"
                    )
            print()

            # DNS Information
            print("DNS Resolution:")
            if timestamps.dns_was_cached:
                print("  ✓ DNS cached")
            elif timestamps.dns_resolution_duration_ns:
                print("  ✗ DNS lookup performed")
                print(
                    f"    Resolution time: {timestamps.dns_resolution_duration_ns / 1_000_000:.2f} ms"
                )
            if timestamps.dns_host:
                print(f"    Host: {timestamps.dns_host}")
            print()

            # Request Timing
            print("Request Timing:")
            if timestamps.request_send_duration_ns:
                print(
                    f"  Send duration: {timestamps.request_send_duration_ns / 1_000_000:.2f} ms"
                )
            if timestamps.request_headers_duration_ns:
                print(
                    f"  Headers sent in: {timestamps.request_headers_duration_ns / 1_000_000:.2f} ms"
                )
            if timestamps.total_request_bytes > 0:
                print(f"  Total bytes sent: {timestamps.total_request_bytes}")
            print()

            # Response Timing (Critical Metrics)
            print("Response Timing:")
            if timestamps.time_to_first_byte_ns:
                ttfb_ms = timestamps.time_to_first_byte_ns / 1_000_000
                print(f"  Time to First Byte (TTFB): {ttfb_ms:.2f} ms")
            if timestamps.time_to_last_byte_ns:
                ttlb_ms = timestamps.time_to_last_byte_ns / 1_000_000
                print(f"  Time to Last Byte (TTLB): {ttlb_ms:.2f} ms")
            if timestamps.server_processing_time_ns:
                server_ms = timestamps.server_processing_time_ns / 1_000_000
                print(f"  Estimated server time: {server_ms:.2f} ms")
            if timestamps.network_transfer_time_ns:
                network_ms = timestamps.network_transfer_time_ns / 1_000_000
                print(f"  Network transfer time: {network_ms:.2f} ms")
            print()

            # Response Details
            print("Response Details:")
            print(f"  Status: {timestamps.response_status}")
            print(f"  Total bytes received: {timestamps.total_response_bytes}")
            print(f"  Response chunks: {timestamps.total_response_chunks}")
            if timestamps.compression_type:
                print(f"  Compression: {timestamps.compression_type}")
            print()

            # Bandwidth
            if timestamps.download_rate_bytes_per_sec:
                download_mbps = timestamps.download_rate_bytes_per_sec / 1_000_000
                print("Download Rate:")
                print(f"  {download_mbps:.2f} MB/s")
                print()

            # Chunk Analysis (if streaming/multiple chunks)
            if timestamps.is_streaming_response:
                print("Chunk Analysis:")
                print(
                    f"  Chunks received: {len(timestamps.response_chunk_received_ns)}"
                )
                if timestamps.response_chunk_latency_avg_ns:
                    avg_ms = timestamps.response_chunk_latency_avg_ns / 1_000_000
                    print(f"  Avg inter-chunk latency: {avg_ms:.2f} ms")
                if timestamps.response_chunk_latency_jitter_ns:
                    jitter_ms = timestamps.response_chunk_latency_jitter_ns / 1_000_000
                    print(f"  Latency jitter (std dev): {jitter_ms:.2f} ms")
                print()

            # Overall Efficiency Score
            if timestamps.request_efficiency_score:
                score = timestamps.request_efficiency_score
                print(f"Overall Efficiency Score: {score:.1f}/100")
                if score >= 80:
                    print("  ✓ Excellent!")
                elif score >= 60:
                    print("  ⚠ Good")
                else:
                    print("  ⚠ Could be improved")

            print("=" * 80)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(example_with_trace_timing())
