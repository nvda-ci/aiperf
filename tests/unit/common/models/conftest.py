# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for model tests."""

import pytest


@pytest.fixture
def base_message_data():
    """Base message data template."""
    return {
        "service_id": "test-service",
        "request_ns": 1234567890,
    }


@pytest.fixture
def error_details():
    """Standard error details for testing."""
    return {
        "message": "Test error",
        "type": "TestError",
    }


@pytest.fixture
def process_records_result():
    """Minimal valid ProcessRecordsResult data."""
    return {
        "results": {
            "records": [],
            "completed": True,
            "start_ns": 0,
            "end_ns": 1000,
            "profile_results": {},
        },
        "errors": [],
    }


# Trace data fixtures
@pytest.fixture
def base_trace_timestamps():
    """Baseline timestamps for trace data (perf_counter_ns)."""
    base = 1000000000  # 1 second in ns
    return {
        "reference_perf": base,
        "reference_time": base * 1732000000,  # Realistic wall-clock time
        "request_start": base + 10_000_000,  # +10ms
        "request_headers_sent": base + 15_000_000,  # +15ms
        "request_end": base + 20_000_000,  # +20ms
        "response_start": base + 50_000_000,  # +50ms
        "response_headers": base + 55_000_000,  # +55ms
        "response_end": base + 100_000_000,  # +100ms
        "error": base + 75_000_000,  # +75ms
    }


@pytest.fixture
def aiohttp_trace_timestamps(base_trace_timestamps):
    """Extended timestamps for aiohttp-specific trace data."""
    base = base_trace_timestamps
    return {
        **base,
        "pool_wait_start": base["reference_perf"] + 1_000_000,  # +1ms
        "pool_wait_end": base["reference_perf"] + 2_000_000,  # +2ms
        "dns_start": base["reference_perf"] + 2_500_000,  # +2.5ms
        "dns_end": base["reference_perf"] + 4_000_000,  # +4ms
        "tcp_start": base["reference_perf"] + 4_500_000,  # +4.5ms
        "tcp_end": base["reference_perf"] + 7_000_000,  # +7ms
        "tls_start": base["reference_perf"] + 7_500_000,  # +7.5ms
        "tls_end": base["reference_perf"] + 9_000_000,  # +9ms
        "connection_reused": base["reference_perf"] + 2_000_000,  # +2ms
    }


@pytest.fixture
def sample_headers():
    """Sample HTTP headers for testing."""
    return {
        "content-type": "application/json",
        "user-agent": "aiperf/1.0",
        "accept": "*/*",
    }


def create_base_trace_data(**overrides):
    """Helper to create BaseTraceData with default values."""
    from aiperf.common.models import BaseTraceData

    defaults = {"trace_type": "base"}
    return BaseTraceData(**{**defaults, **overrides})


def create_aiohttp_trace_data(**overrides):
    """Helper to create AioHttpTraceData with default values."""
    from aiperf.common.models import AioHttpTraceData

    return AioHttpTraceData(**overrides)
