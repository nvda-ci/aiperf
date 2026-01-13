# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test rate timing WITHOUT looptime to verify basic functionality."""

import asyncio

import pytest

from aiperf.common import random_generator as rng

# NO pytestmark = pytest.mark.looptime - run in real time


@pytest.mark.asyncio
async def test_rate_without_looptime(create_orchestrator_harness, time_traveler):
    """Test issuing 10 credits without looptime (real time)."""
    rng.reset()
    rng.init(42)

    loop = asyncio.get_running_loop()
    print(f"\nLoop type: {type(loop).__name__}")
    print(f"Looptime on: {getattr(loop, 'looptime_on', 'N/A')}")

    # Use only 10 credits with high rate
    harness = create_orchestrator_harness(
        conversations=[(f"conv{i}", 1) for i in range(10)],
        request_count=10,
        request_rate=1000.0,  # Very high rate for fast completion
        random_seed=42,
    )

    # Use auto_return
    harness.router.auto_return = True
    await harness.orchestrator.initialize()

    # Run with timeout
    try:
        await asyncio.wait_for(harness.orchestrator.start(), timeout=5.0)
    except asyncio.TimeoutError:
        print(f"TIMEOUT! Credits sent: {len(harness.sent_credits)}")
        raise

    print(f"Credits sent: {len(harness.sent_credits)}")
    assert len(harness.sent_credits) == 10, (
        f"Expected 10 credits, got {len(harness.sent_credits)}"
    )
