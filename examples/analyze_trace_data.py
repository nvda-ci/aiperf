#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Analyze aiohttp trace data from AIPerf JSONL export.

Usage:
    python analyze_trace_data.py <path_to_profile_export.jsonl>

Example:
    python analyze_trace_data.py results/profile_export.jsonl
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def ns_to_ms(nanoseconds: int | None) -> float | None:
    """Convert nanoseconds to milliseconds."""
    return nanoseconds / 1_000_000 if nanoseconds is not None else None


def format_ms(milliseconds: float | None) -> str:
    """Format milliseconds for display."""
    if milliseconds is None:
        return "N/A"
    if milliseconds < 1:
        return f"{milliseconds * 1000:.2f}μs"
    return f"{milliseconds:.2f}ms"


def format_bytes(bytes_value: int | None) -> str:
    """Format bytes for display."""
    if bytes_value is None:
        return "N/A"
    if bytes_value < 1024:
        return f"{bytes_value}B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.2f}KB"
    else:
        return f"{bytes_value / (1024 * 1024):.2f}MB"


def compute_percentiles(
    values: list[float], percentiles: list[int]
) -> dict[int, float]:
    """Compute percentiles for a list of values."""
    if not values:
        return {p: 0.0 for p in percentiles}
    sorted_values = sorted(values)
    result = {}
    for p in percentiles:
        idx = int(len(sorted_values) * p / 100)
        idx = min(idx, len(sorted_values) - 1)
        result[p] = sorted_values[idx]
    return result


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_stats_table(title: str, stats: dict[str, Any], formatter=None) -> None:
    """Print statistics in a formatted table."""
    if not stats or all(v is None or v == 0 for v in stats.values()):
        print(f"\n{title}: No data available")
        return

    print(f"\n{title}:")
    print(f"  {'Metric':<30} {'Value':>15}")
    print(f"  {'-' * 30} {'-' * 15}")

    for key, value in stats.items():
        if formatter:
            formatted = formatter(value)
        else:
            formatted = f"{value:.2f}" if isinstance(value, float) else str(value)
        print(f"  {key:<30} {formatted:>15}")


def analyze_trace_data(jsonl_path: Path) -> None:
    """Analyze trace data from JSONL file and print extensive statistics."""

    # Data collectors
    records = []
    connection_metrics = []
    dns_metrics = []
    request_metrics = []
    response_metrics = []
    ttfb_metrics = []
    ttlb_metrics = []
    chunk_counts = []
    chunk_sizes = []
    request_bytes = []
    response_bytes = []
    inter_chunk_latencies = []

    # k6-compatible metric collectors
    k6_http_req_blocked = []
    k6_http_req_looking_up = []
    k6_http_req_connecting = []
    k6_http_req_tls_handshaking = []
    k6_http_req_sending = []
    k6_http_req_waiting = []
    k6_http_req_receiving = []
    k6_http_req_duration = []
    k6_http_req_failed_count = 0

    # Token analytics collectors
    input_tokens = []
    output_tokens = []
    total_tokens = []
    tokens_per_second = []
    time_per_output_token = []
    bytes_per_token = []
    chunks_per_token = []

    # Validation: Compare trace data vs AIPerf metrics
    validation_comparisons = {
        "request_latency_vs_trace": [],  # AIPerf request_latency vs trace total duration
        "ttft_vs_trace": [],  # AIPerf TTFT vs trace TTFB
        "request_latency_delta_ms": [],  # Difference between metrics
        "ttft_delta_ms": [],  # Difference between TTFT calculations
    }

    # Counters
    total_records = 0
    records_with_trace = 0
    connection_reused_count = 0
    connection_created_count = 0
    dns_cached_count = 0
    dns_resolved_count = 0
    streaming_requests = 0
    non_streaming_requests = 0

    # Read and parse JSONL file
    print(f"Reading trace data from: {jsonl_path}")
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                total_records += 1
                records.append(record)

                trace_data = record.get("trace_data")
                if not trace_data:
                    continue

                records_with_trace += 1

                # Connection metrics
                conn_queue_start = trace_data.get("connection_queued_start_ns")
                conn_queue_end = trace_data.get("connection_queued_end_ns")
                if conn_queue_start and conn_queue_end:
                    queue_wait = ns_to_ms(conn_queue_end - conn_queue_start)
                    if queue_wait:
                        connection_metrics.append(queue_wait)

                conn_create_start = trace_data.get("connection_create_start_ns")
                conn_create_end = trace_data.get("connection_create_end_ns")
                if conn_create_start and conn_create_end:
                    create_duration = ns_to_ms(conn_create_end - conn_create_start)
                    if create_duration:
                        connection_metrics.append(create_duration)
                    connection_created_count += 1

                if trace_data.get("connection_reuseconn_ns"):
                    connection_reused_count += 1

                # DNS metrics
                dns_start = trace_data.get("dns_resolvehost_start_ns")
                dns_end = trace_data.get("dns_resolvehost_end_ns")
                if dns_start and dns_end:
                    dns_duration = ns_to_ms(dns_end - dns_start)
                    if dns_duration:
                        dns_metrics.append(dns_duration)
                    dns_resolved_count += 1

                if trace_data.get("dns_cache_hit_ns"):
                    dns_cached_count += 1

                # Request timing
                req_start = trace_data.get("request_start_ns")
                req_end = trace_data.get("request_end_ns")
                if req_start and req_end:
                    req_duration = ns_to_ms(req_end - req_start)
                    if req_duration:
                        request_metrics.append(req_duration)

                # Response timing and chunks
                resp_chunks = trace_data.get("response_receive_timestamps_ns", [])
                resp_chunk_sizes = trace_data.get("response_receive_sizes_bytes", [])

                if resp_chunks:
                    # TTFB (Time to First Byte)
                    if req_end and resp_chunks:
                        ttfb = ns_to_ms(resp_chunks[0] - req_end)
                        if ttfb and ttfb > 0:
                            ttfb_metrics.append(ttfb)

                    # TTLB (Time to Last Byte)
                    if req_end and resp_chunks:
                        ttlb = ns_to_ms(resp_chunks[-1] - req_end)
                        if ttlb and ttlb > 0:
                            ttlb_metrics.append(ttlb)

                    # Chunk statistics
                    chunk_counts.append(len(resp_chunks))
                    if len(resp_chunks) > 1:
                        streaming_requests += 1
                        # Inter-chunk latencies
                        for i in range(1, len(resp_chunks)):
                            icl = ns_to_ms(resp_chunks[i] - resp_chunks[i - 1])
                            if icl:
                                inter_chunk_latencies.append(icl)
                    else:
                        non_streaming_requests += 1

                # Bytes transferred
                req_chunk_sizes = trace_data.get("request_chunk_sizes", [])
                if req_chunk_sizes:
                    total_req_bytes = sum(req_chunk_sizes)
                    request_bytes.append(total_req_bytes)

                if resp_chunk_sizes:
                    total_resp_bytes = sum(resp_chunk_sizes)
                    response_bytes.append(total_resp_bytes)
                    chunk_sizes.extend(resp_chunk_sizes)

                # Collect k6-compatible metrics
                # These use derived properties that match k6's metric names
                blocked = trace_data.get("connection_queued_start_ns")
                blocked_end = trace_data.get("connection_queued_end_ns")
                if blocked and blocked_end:
                    k6_http_req_blocked.append(ns_to_ms(blocked_end - blocked))

                dns_start = trace_data.get("dns_resolvehost_start_ns")
                dns_end = trace_data.get("dns_resolvehost_end_ns")
                if dns_start and dns_end:
                    k6_http_req_looking_up.append(ns_to_ms(dns_end - dns_start))

                conn_start = trace_data.get("connection_create_start_ns")
                conn_end = trace_data.get("connection_create_end_ns")
                if conn_start and conn_end:
                    k6_http_req_connecting.append(ns_to_ms(conn_end - conn_start))

                tls_start = trace_data.get("tls_handshake_start_ns")
                tls_end = trace_data.get("tls_handshake_end_ns")
                if tls_start and tls_end:
                    k6_http_req_tls_handshaking.append(ns_to_ms(tls_end - tls_start))

                if req_start and req_end:
                    k6_http_req_sending.append(ns_to_ms(req_end - req_start))

                # http_req_waiting is TTFB (server processing time)
                if req_end and resp_chunks:
                    ttfb_ns = resp_chunks[0] - req_end
                    k6_http_req_waiting.append(ns_to_ms(ttfb_ns))

                # http_req_receiving is network transfer time (TTLB - TTFB)
                if resp_chunks and len(resp_chunks) > 1:
                    receiving_ns = resp_chunks[-1] - resp_chunks[0]
                    k6_http_req_receiving.append(ns_to_ms(receiving_ns))
                elif resp_chunks:
                    # Single chunk - no transfer time
                    k6_http_req_receiving.append(0.0)

                # http_req_duration = sending + waiting + receiving
                if req_start and req_end and resp_chunks:
                    sending_ns = req_end - req_start
                    waiting_ns = resp_chunks[0] - req_end
                    receiving_ns = (
                        resp_chunks[-1] - resp_chunks[0] if len(resp_chunks) > 1 else 0
                    )
                    total_duration = ns_to_ms(sending_ns + waiting_ns + receiving_ns)
                    k6_http_req_duration.append(total_duration)

                # Track failures
                if trace_data.get("request_exception_ns"):
                    k6_http_req_failed_count += 1
                elif trace_data.get("response_status"):
                    status = trace_data["response_status"]
                    if status < 200 or status >= 400:
                        k6_http_req_failed_count += 1

                # Token analytics and validation (from metrics field)
                metrics = record.get("metrics", {})

                # Helper function to extract value from nested or flat format
                def get_metric_value(metric_dict, *keys):
                    """Extract value from metric, handling both flat and nested {value, unit} formats."""
                    for key in keys:
                        val = metric_dict.get(key)
                        if val is not None:
                            # Handle nested format {"value": X, "unit": "Y"}
                            if isinstance(val, dict) and "value" in val:
                                return val["value"]
                            # Handle flat format
                            return val
                    return None

                # Validation: Compare trace data with AIPerf metrics
                # Compare request_latency (AIPerf computed) vs trace-based duration
                aiperf_latency = get_metric_value(metrics, "request_latency")
                if aiperf_latency and req_start and req_end and resp_chunks:
                    # Calculate trace-based total duration
                    sending_ns = req_end - req_start
                    waiting_ns = resp_chunks[0] - req_end
                    receiving_ns = (
                        resp_chunks[-1] - resp_chunks[0] if len(resp_chunks) > 1 else 0
                    )
                    trace_duration = ns_to_ms(sending_ns + waiting_ns + receiving_ns)

                    validation_comparisons["request_latency_vs_trace"].append(
                        (aiperf_latency, trace_duration)
                    )
                    delta = abs(aiperf_latency - trace_duration)
                    validation_comparisons["request_latency_delta_ms"].append(delta)

                # Compare TTFT (AIPerf computed) vs trace TTFB
                aiperf_ttft = get_metric_value(
                    metrics, "time_to_first_token", "ttft", "time_to_first_output_token"
                )
                if aiperf_ttft and req_end and resp_chunks:
                    trace_ttfb = ns_to_ms(resp_chunks[0] - req_end)
                    validation_comparisons["ttft_vs_trace"].append(
                        (aiperf_ttft, trace_ttfb)
                    )
                    delta = abs(aiperf_ttft - trace_ttfb)
                    validation_comparisons["ttft_delta_ms"].append(delta)

                # Try multiple field names for input/output tokens
                input_tok = get_metric_value(
                    metrics,
                    "input_tokens",
                    "num_input_tokens",
                    "input_sequence_length",
                    "usage_prompt_tokens",
                )
                output_tok = get_metric_value(
                    metrics,
                    "output_tokens",
                    "num_output_tokens",
                    "output_sequence_length",
                    "output_token_count",
                    "usage_completion_tokens",
                )
                total_tok = get_metric_value(metrics, "usage_total_tokens")

                if input_tok is not None:
                    input_tokens.append(input_tok)
                if output_tok is not None:
                    output_tokens.append(output_tok)
                if total_tok is not None:
                    total_tokens.append(total_tok)
                elif input_tok is not None and output_tok is not None:
                    total_tokens.append(input_tok + output_tok)

                # Calculate token throughput and efficiency
                if output_tok and output_tok > 0:
                    # Time per output token (from TTFB to TTLB)
                    if req_end and resp_chunks and len(resp_chunks) > 0:
                        generation_time_ms = ns_to_ms(resp_chunks[-1] - req_end)
                        if generation_time_ms and generation_time_ms > 0:
                            tps = (
                                output_tok / generation_time_ms
                            ) * 1000  # tokens/second
                            tokens_per_second.append(tps)
                            time_per_token_ms = generation_time_ms / output_tok
                            time_per_output_token.append(time_per_token_ms)

                    # Bytes per token
                    if resp_chunk_sizes and sum(resp_chunk_sizes) > 0:
                        bpt = sum(resp_chunk_sizes) / output_tok
                        bytes_per_token.append(bpt)

                    # Chunks per token
                    if resp_chunks and len(resp_chunks) > 0:
                        cpt = len(resp_chunks) / output_tok
                        chunks_per_token.append(cpt)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")

    # Print comprehensive statistics
    print_section("OVERVIEW")
    print(f"  Total records:              {total_records:>10,}")
    print(
        f"  Records with trace data:    {records_with_trace:>10,} ({records_with_trace / total_records * 100:.1f}%)"
    )
    print(f"  Streaming requests:         {streaming_requests:>10,}")
    print(f"  Non-streaming requests:     {non_streaming_requests:>10,}")
    if records_with_trace > 0:
        failure_rate = (k6_http_req_failed_count / records_with_trace) * 100
        print(
            f"  Failed requests:            {k6_http_req_failed_count:>10,} ({failure_rate:.2f}%)"
        )

    # k6-compatible metrics summary
    print_section("K6-COMPATIBLE HTTP METRICS")

    if k6_http_req_duration:
        percentiles = compute_percentiles(k6_http_req_duration, [50, 90, 95, 99, 99.9])
        print_stats_table(
            "http_req_duration (k6: sending + waiting + receiving)",
            {
                "Count": len(k6_http_req_duration),
                "Min": min(k6_http_req_duration),
                "Max": max(k6_http_req_duration),
                "Avg": statistics.mean(k6_http_req_duration),
                "Med": percentiles[50],
                "p(90)": percentiles[90],
                "p(95)": percentiles[95],
                "p(99)": percentiles[99],
                "p(99.9)": percentiles[99.9],
            },
            formatter=format_ms,
        )

    if k6_http_req_blocked:
        percentiles = compute_percentiles(k6_http_req_blocked, [50, 90, 95, 99])
        print_stats_table(
            "http_req_blocked (k6: connection pool queue wait)",
            {
                "Count": len(k6_http_req_blocked),
                "Avg": statistics.mean(k6_http_req_blocked),
                "Med": percentiles[50],
                "Max": max(k6_http_req_blocked),
            },
            formatter=format_ms,
        )

    if k6_http_req_looking_up:
        percentiles = compute_percentiles(k6_http_req_looking_up, [50, 90, 95, 99])
        print_stats_table(
            "http_req_looking_up (k6: DNS resolution time)",
            {
                "Count": len(k6_http_req_looking_up),
                "Avg": statistics.mean(k6_http_req_looking_up),
                "Med": percentiles[50],
                "Max": max(k6_http_req_looking_up),
            },
            formatter=format_ms,
        )

    if k6_http_req_connecting:
        percentiles = compute_percentiles(k6_http_req_connecting, [50, 90, 95, 99])
        print_stats_table(
            "http_req_connecting (k6: TCP connection establishment)",
            {
                "Count": len(k6_http_req_connecting),
                "Avg": statistics.mean(k6_http_req_connecting),
                "Med": percentiles[50],
                "Max": max(k6_http_req_connecting),
            },
            formatter=format_ms,
        )

    if k6_http_req_tls_handshaking:
        percentiles = compute_percentiles(k6_http_req_tls_handshaking, [50, 90, 95, 99])
        print_stats_table(
            "http_req_tls_handshaking (k6: TLS/SSL handshake)",
            {
                "Count": len(k6_http_req_tls_handshaking),
                "Avg": statistics.mean(k6_http_req_tls_handshaking),
                "Med": percentiles[50],
                "Max": max(k6_http_req_tls_handshaking),
            },
            formatter=format_ms,
        )

    if k6_http_req_sending:
        percentiles = compute_percentiles(k6_http_req_sending, [50, 90, 95, 99])
        print_stats_table(
            "http_req_sending (k6: request send time)",
            {
                "Count": len(k6_http_req_sending),
                "Avg": statistics.mean(k6_http_req_sending),
                "Med": percentiles[50],
                "Max": max(k6_http_req_sending),
            },
            formatter=format_ms,
        )

    if k6_http_req_waiting:
        percentiles = compute_percentiles(k6_http_req_waiting, [50, 90, 95, 99, 99.9])
        print_stats_table(
            "http_req_waiting (k6: TTFB / server processing)",
            {
                "Count": len(k6_http_req_waiting),
                "Min": min(k6_http_req_waiting),
                "Max": max(k6_http_req_waiting),
                "Avg": statistics.mean(k6_http_req_waiting),
                "Med": percentiles[50],
                "p(90)": percentiles[90],
                "p(95)": percentiles[95],
                "p(99)": percentiles[99],
                "p(99.9)": percentiles[99.9],
            },
            formatter=format_ms,
        )

    if k6_http_req_receiving:
        percentiles = compute_percentiles(k6_http_req_receiving, [50, 90, 95, 99])
        print_stats_table(
            "http_req_receiving (k6: response transfer time)",
            {
                "Count": len(k6_http_req_receiving),
                "Avg": statistics.mean(k6_http_req_receiving),
                "Med": percentiles[50],
                "Max": max(k6_http_req_receiving),
            },
            formatter=format_ms,
        )

    # Connection statistics
    print_section("CONNECTION STATISTICS")
    print(f"  Connections created:        {connection_created_count:>10,}")
    print(f"  Connections reused:         {connection_reused_count:>10,}")
    if connection_created_count + connection_reused_count > 0:
        reuse_rate = (
            connection_reused_count
            / (connection_created_count + connection_reused_count)
            * 100
        )
        print(f"  Connection reuse rate:      {reuse_rate:>10.1f}%")

    if connection_metrics:
        percentiles = compute_percentiles(connection_metrics, [50, 90, 95, 99])
        print_stats_table(
            "Connection Establishment Time",
            {
                "Min": min(connection_metrics),
                "Max": max(connection_metrics),
                "Mean": statistics.mean(connection_metrics),
                "Median (p50)": percentiles[50],
                "p90": percentiles[90],
                "p95": percentiles[95],
                "p99": percentiles[99],
            },
            formatter=format_ms,
        )

    # DNS statistics
    print_section("DNS STATISTICS")
    print(f"  DNS resolutions performed:  {dns_resolved_count:>10,}")
    print(f"  DNS cache hits:             {dns_cached_count:>10,}")
    if dns_resolved_count + dns_cached_count > 0:
        cache_rate = dns_cached_count / (dns_resolved_count + dns_cached_count) * 100
        print(f"  DNS cache hit rate:         {cache_rate:>10.1f}%")

    if dns_metrics:
        percentiles = compute_percentiles(dns_metrics, [50, 90, 95, 99])
        print_stats_table(
            "DNS Resolution Time",
            {
                "Min": min(dns_metrics),
                "Max": max(dns_metrics),
                "Mean": statistics.mean(dns_metrics),
                "Median (p50)": percentiles[50],
                "p90": percentiles[90],
                "p95": percentiles[95],
                "p99": percentiles[99],
            },
            formatter=format_ms,
        )

    # Request timing
    if request_metrics:
        print_section("REQUEST SEND TIME")
        percentiles = compute_percentiles(request_metrics, [50, 90, 95, 99])
        print_stats_table(
            "Request Send Duration",
            {
                "Min": min(request_metrics),
                "Max": max(request_metrics),
                "Mean": statistics.mean(request_metrics),
                "Median (p50)": percentiles[50],
                "p90": percentiles[90],
                "p95": percentiles[95],
                "p99": percentiles[99],
            },
            formatter=format_ms,
        )

    # Response timing
    if ttfb_metrics:
        print_section("RESPONSE TIMING - TIME TO FIRST BYTE (TTFB)")
        percentiles = compute_percentiles(ttfb_metrics, [50, 90, 95, 99])
        print_stats_table(
            "TTFB (includes network + server processing)",
            {
                "Min": min(ttfb_metrics),
                "Max": max(ttfb_metrics),
                "Mean": statistics.mean(ttfb_metrics),
                "Median (p50)": percentiles[50],
                "p90": percentiles[90],
                "p95": percentiles[95],
                "p99": percentiles[99],
            },
            formatter=format_ms,
        )

    if ttlb_metrics:
        print_section("RESPONSE TIMING - TIME TO LAST BYTE (TTLB)")
        percentiles = compute_percentiles(ttlb_metrics, [50, 90, 95, 99])
        print_stats_table(
            "TTLB (total response time)",
            {
                "Min": min(ttlb_metrics),
                "Max": max(ttlb_metrics),
                "Mean": statistics.mean(ttlb_metrics),
                "Median (p50)": percentiles[50],
                "p90": percentiles[90],
                "p95": percentiles[95],
                "p99": percentiles[99],
            },
            formatter=format_ms,
        )

    # Streaming statistics
    if streaming_requests > 0:
        print_section("STREAMING STATISTICS")
        if chunk_counts:
            print_stats_table(
                "Chunks per Request",
                {
                    "Min": min(chunk_counts),
                    "Max": max(chunk_counts),
                    "Mean": statistics.mean(chunk_counts),
                    "Median": statistics.median(chunk_counts),
                },
            )

        if inter_chunk_latencies:
            percentiles = compute_percentiles(inter_chunk_latencies, [50, 90, 95, 99])
            print_stats_table(
                "Inter-Chunk Latency (ICL)",
                {
                    "Min": min(inter_chunk_latencies),
                    "Max": max(inter_chunk_latencies),
                    "Mean": statistics.mean(inter_chunk_latencies),
                    "Median (p50)": percentiles[50],
                    "p90": percentiles[90],
                    "p95": percentiles[95],
                    "p99": percentiles[99],
                },
                formatter=format_ms,
            )

        if chunk_sizes:
            print_stats_table(
                "Response Chunk Sizes",
                {
                    "Min": min(chunk_sizes),
                    "Max": max(chunk_sizes),
                    "Mean": statistics.mean(chunk_sizes),
                    "Median": statistics.median(chunk_sizes),
                    "Total": sum(chunk_sizes),
                },
                formatter=format_bytes,
            )

    # Bandwidth statistics
    print_section("BANDWIDTH STATISTICS")
    if request_bytes:
        print_stats_table(
            "Request Size",
            {
                "Min": min(request_bytes),
                "Max": max(request_bytes),
                "Mean": statistics.mean(request_bytes),
                "Median": statistics.median(request_bytes),
                "Total": sum(request_bytes),
            },
            formatter=format_bytes,
        )

    if response_bytes:
        print_stats_table(
            "Response Size",
            {
                "Min": min(response_bytes),
                "Max": max(response_bytes),
                "Mean": statistics.mean(response_bytes),
                "Median": statistics.median(response_bytes),
                "Total": sum(response_bytes),
            },
            formatter=format_bytes,
        )

    # Token analytics
    if input_tokens or output_tokens:
        print_section("TOKEN ANALYTICS")

        if input_tokens:
            print_stats_table(
                "Input Tokens",
                {
                    "Min": min(input_tokens),
                    "Max": max(input_tokens),
                    "Mean": statistics.mean(input_tokens),
                    "Median": statistics.median(input_tokens),
                    "Total": sum(input_tokens),
                },
            )

        if output_tokens:
            print_stats_table(
                "Output Tokens",
                {
                    "Min": min(output_tokens),
                    "Max": max(output_tokens),
                    "Mean": statistics.mean(output_tokens),
                    "Median": statistics.median(output_tokens),
                    "Total": sum(output_tokens),
                },
            )

        if tokens_per_second:
            percentiles = compute_percentiles(tokens_per_second, [50, 90, 95, 99])
            print_stats_table(
                "Token Throughput (tokens/second)",
                {
                    "Min": min(tokens_per_second),
                    "Max": max(tokens_per_second),
                    "Mean": statistics.mean(tokens_per_second),
                    "Median (p50)": percentiles[50],
                    "p90": percentiles[90],
                    "p95": percentiles[95],
                    "p99": percentiles[99],
                },
            )

        if time_per_output_token:
            percentiles = compute_percentiles(time_per_output_token, [50, 90, 95, 99])
            print_stats_table(
                "Time per Output Token",
                {
                    "Min": min(time_per_output_token),
                    "Max": max(time_per_output_token),
                    "Mean": statistics.mean(time_per_output_token),
                    "Median (p50)": percentiles[50],
                    "p90": percentiles[90],
                    "p95": percentiles[95],
                    "p99": percentiles[99],
                },
                formatter=format_ms,
            )

        if bytes_per_token:
            print_stats_table(
                "Bytes per Token (network efficiency)",
                {
                    "Min": min(bytes_per_token),
                    "Max": max(bytes_per_token),
                    "Mean": statistics.mean(bytes_per_token),
                    "Median": statistics.median(bytes_per_token),
                },
                formatter=format_bytes,
            )

        if chunks_per_token:
            print_stats_table(
                "Chunks per Token (streaming granularity)",
                {
                    "Min": min(chunks_per_token),
                    "Max": max(chunks_per_token),
                    "Mean": statistics.mean(chunks_per_token),
                    "Median": statistics.median(chunks_per_token),
                },
            )

    # Token vs Response correlation insights
    if output_tokens and ttfb_metrics and inter_chunk_latencies:
        print_section("TOKEN-TO-RESPONSE CORRELATION")

        # Average tokens vs timing
        if tokens_per_second:
            print("\n  Token Generation Performance:")
            print(
                f"    Avg tokens/second:         {statistics.mean(tokens_per_second):>15.2f}"
            )
            print(
                f"    Avg time/token:            {format_ms(statistics.mean(time_per_output_token)):>15}"
            )

        # Correlation with response metrics
        if inter_chunk_latencies and output_tokens:
            avg_icl = statistics.mean(inter_chunk_latencies)
            avg_output = statistics.mean(output_tokens)
            print("\n  Streaming Characteristics:")
            print(f"    Avg output tokens:         {avg_output:>15.2f}")
            print(f"    Avg inter-chunk latency:   {format_ms(avg_icl):>15}")
            if chunks_per_token:
                avg_cpt = statistics.mean(chunks_per_token)
                print(f"    Avg chunks per token:      {avg_cpt:>15.3f}")
                if avg_cpt > 0:
                    tokens_per_chunk = 1.0 / avg_cpt
                    print(f"    Avg tokens per chunk:      {tokens_per_chunk:>15.2f}")

        # Network efficiency per token
        if bytes_per_token and response_bytes:
            avg_bpt = statistics.mean(bytes_per_token)
            total_resp = sum(response_bytes)
            total_out = sum(output_tokens)
            print("\n  Network Efficiency:")
            print(f"    Avg bytes per token:       {format_bytes(avg_bpt):>15}")
            print(f"    Total response bytes:      {format_bytes(total_resp):>15}")
            print(f"    Total output tokens:       {total_out:>15,}")
            print(
                f"    Overall bytes/token:       {format_bytes(total_resp / total_out if total_out > 0 else 0):>15}"
            )

    # Network efficiency insights
    if ttfb_metrics and ttlb_metrics:
        print_section("NETWORK EFFICIENCY INSIGHTS")
        # Calculate network transfer time (TTLB - TTFB)
        transfer_times = [
            ttlb - ttfb for ttfb, ttlb in zip(ttfb_metrics, ttlb_metrics, strict=False)
        ]
        if transfer_times:
            avg_ttfb = statistics.mean(ttfb_metrics)
            avg_transfer = statistics.mean(transfer_times)
            print("\n  Average breakdown:")
            print(f"    Server processing + TTFB:  {format_ms(avg_ttfb):>15}")
            print(f"    Network transfer time:     {format_ms(avg_transfer):>15}")
            if avg_ttfb > 0:
                ratio = avg_transfer / avg_ttfb
                print(f"    Transfer/Processing ratio: {ratio:>15.2f}x")

    # Validation: Compare AIPerf metrics vs trace data
    if validation_comparisons["request_latency_vs_trace"]:
        print_section("METRIC VALIDATION: AIPerf vs Trace Data")

        # Request latency comparison
        latency_deltas = validation_comparisons["request_latency_delta_ms"]
        latency_pairs = validation_comparisons["request_latency_vs_trace"]

        if latency_deltas:
            avg_delta = statistics.mean(latency_deltas)
            max_delta = max(latency_deltas)
            avg_aiperf = statistics.mean([p[0] for p in latency_pairs])
            avg_trace = statistics.mean([p[1] for p in latency_pairs])
            delta_pct = (avg_delta / avg_aiperf * 100) if avg_aiperf > 0 else 0

            print("\n  Request Latency Comparison:")
            print(f"    AIPerf request_latency (avg):  {format_ms(avg_aiperf):>15}")
            print(f"    Trace http_req_duration (avg): {format_ms(avg_trace):>15}")
            print(
                f"    Average delta:                  {format_ms(avg_delta):>15} ({delta_pct:.2f}%)"
            )
            print(f"    Max delta:                      {format_ms(max_delta):>15}")

            if delta_pct < 1.0:
                print("    ✅ Excellent correlation (< 1% difference)")
            elif delta_pct < 5.0:
                print("    ✓  Good correlation (< 5% difference)")
            else:
                print(
                    "    ⚠️  Significant difference (> 5%) - investigate timing source discrepancy"
                )

        # TTFT comparison
        ttft_deltas = validation_comparisons["ttft_delta_ms"]
        ttft_pairs = validation_comparisons["ttft_vs_trace"]

        if ttft_deltas:
            avg_delta = statistics.mean(ttft_deltas)
            max_delta = max(ttft_deltas)
            avg_aiperf = statistics.mean([p[0] for p in ttft_pairs])
            avg_trace = statistics.mean([p[1] for p in ttft_pairs])
            delta_pct = (avg_delta / avg_aiperf * 100) if avg_aiperf > 0 else 0

            print("\n  TTFT/TTFB Comparison:")
            print(f"    AIPerf TTFT (avg):              {format_ms(avg_aiperf):>15}")
            print(f"    Trace http_req_waiting (avg):  {format_ms(avg_trace):>15}")
            print(
                f"    Average delta:                  {format_ms(avg_delta):>15} ({delta_pct:.2f}%)"
            )
            print(f"    Max delta:                      {format_ms(max_delta):>15}")

            if delta_pct < 1.0:
                print("    ✅ Excellent correlation (< 1% difference)")
            elif delta_pct < 5.0:
                print("    ✓  Good correlation (< 5% difference)")
            else:
                print(
                    "    ⚠️  Significant difference (> 5%) - investigate timing methodology"
                )

        print("\n  Timing Source Information:")
        print("    AIPerf metrics:  Computed from RequestRecord timestamps")
        print("    Trace data:      Captured via aiohttp trace callbacks")
        print("    Expected delta:  < 1ms (sub-millisecond precision)")

    print_section("ANALYSIS COMPLETE")
    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze aiohttp trace data from AIPerf JSONL export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to the profile_export.jsonl file",
    )

    args = parser.parse_args()

    if not args.jsonl_file.exists():
        print(f"Error: File not found: {args.jsonl_file}")
        return

    if not args.jsonl_file.suffix == ".jsonl":
        print(f"Warning: File does not have .jsonl extension: {args.jsonl_file}")

    analyze_trace_data(args.jsonl_file)


if __name__ == "__main__":
    main()
