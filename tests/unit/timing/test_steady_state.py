# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid

import pytest

from aiperf.common.enums import CreditPhase, RequestRateMode, TimingMode
from aiperf.common.messages import CreditReturnMessage, ProfileCancelCommand
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.request_rate_strategy import RequestRateStrategy
from tests.unit.timing.conftest import profiling_phase_stats_from_config

# Use a short grace period for testing
TEST_GRACE_PERIOD = 0.1


@pytest.mark.asyncio
async def test_steady_state_3loop_triggers_cancel_after_measurement_loop(mock_credit_manager):
    """Test that steady-state with 3-loop triggers cancel when measurement loop (second loop) completes.
    
    For dataset_size=2:
    - Loop 1 (a): credits 0, 1 (warmup)
    - Loop 2 (b): credits 2, 3 (measurement)
    - Loop 3 (c): credits 4+ (tail)
    
    Cancel should trigger after grace period when credit 3 (last of measurement loop) returns.
    """
    dataset_size = 2
    config = TimingManagerConfig(
        timing_mode=TimingMode.REQUEST_RATE,
        request_count=dataset_size,  # Will be used for validation
        request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
        concurrency=1,
        steady_state=True,
        steady_state_grace_period=TEST_GRACE_PERIOD,  # Short grace period for testing
        dataset_size=dataset_size,  # Required for 3-loop tracking
    )
    phase_stats = profiling_phase_stats_from_config(config)

    strategy = RequestRateStrategy(config, mock_credit_manager)
    strategy.phase_stats[CreditPhase.PROFILING] = phase_stats

    # Verify measurement loop boundaries are set correctly
    assert strategy._measurement_loop_start_credit == dataset_size  # 2
    assert strategy._measurement_loop_end_credit == 2 * dataset_size - 1  # 3

    # Simulate credit issuance for warmup loop (a) - credits 0, 1
    for credit_num in range(dataset_size):
        strategy._record_measurement_start_if_needed(phase_stats, credit_num)
    
    # measurement_start_ns should not be set during warmup
    assert strategy._measurement_start_ns is None

    # Return credits from warmup loop (a) - should NOT trigger cancel
    for credit_num in range(dataset_size):  # 0, 1
        await strategy._on_credit_return(
            CreditReturnMessage(
                service_id="worker-1",
                phase=CreditPhase.PROFILING,
                credit_drop_id=str(uuid.uuid4()),
                credit_num=credit_num,
                requests_sent=1,
            )
        )
    
    assert not strategy._steady_state_stop_event.is_set()
    assert not any(
        isinstance(message, ProfileCancelCommand)
        for message in mock_credit_manager.publish_calls
    )

    # Simulate credit issuance for measurement loop (b) - credits 2, 3
    # When credit 2 (b1) is issued, measurement_start_ns should be set
    strategy._record_measurement_start_if_needed(phase_stats, dataset_size)  # credit 2 (b1)
    assert strategy._measurement_start_ns is not None
    measurement_start = strategy._measurement_start_ns
    
    strategy._record_measurement_start_if_needed(phase_stats, dataset_size + 1)  # credit 3 (b2)
    # measurement_start_ns should not change
    assert strategy._measurement_start_ns == measurement_start

    # Return first credit from measurement loop (b1) - should NOT trigger cancel yet
    await strategy._on_credit_return(
        CreditReturnMessage(
            service_id="worker-1",
            phase=CreditPhase.PROFILING,
            credit_drop_id=str(uuid.uuid4()),
            credit_num=dataset_size,  # credit 2 (b1)
            requests_sent=1,
        )
    )
    
    assert not strategy._steady_state_stop_event.is_set()

    # Return last credit from measurement loop (bN) - should record measurement_end and start grace period
    await strategy._on_credit_return(
        CreditReturnMessage(
            service_id="worker-1",
            phase=CreditPhase.PROFILING,
            credit_drop_id=str(uuid.uuid4()),
            credit_num=2 * dataset_size - 1,  # credit 3 (b2/bN)
            requests_sent=1,
        )
    )

    # Measurement should be complete but cancel not yet sent (in grace period)
    assert strategy._measurement_complete is True
    assert strategy._measurement_start_ns is not None
    assert strategy._measurement_end_ns is not None
    assert not strategy._steady_state_stop_event.is_set()
    
    # Wait for all background tasks (including grace period timer) to complete
    if strategy.tasks:
        await asyncio.gather(*strategy.tasks, return_exceptions=True)
    
    # Now cancel should have been triggered
    assert strategy._steady_state_stop_event.is_set()
    assert any(
        isinstance(message, ProfileCancelCommand)
        and message.reason == "steady_state"
        for message in mock_credit_manager.publish_calls
    )


def test_steady_state_measurement_start_recorded_on_first_measurement_credit(mock_credit_manager):
    """Test that measurement_start_ns is recorded when the first credit of measurement loop is issued."""
    dataset_size = 3
    config = TimingManagerConfig(
        timing_mode=TimingMode.REQUEST_RATE,
        request_count=dataset_size,
        request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
        concurrency=1,
        steady_state=True,
        steady_state_grace_period=TEST_GRACE_PERIOD,
        dataset_size=dataset_size,
    )
    phase_stats = profiling_phase_stats_from_config(config)

    strategy = RequestRateStrategy(config, mock_credit_manager)
    strategy.phase_stats[CreditPhase.PROFILING] = phase_stats

    # Initially, measurement_start_ns should be None
    assert strategy._measurement_start_ns is None

    # Issue credits for warmup loop - measurement should not start
    for credit_num in range(dataset_size):  # 0, 1, 2
        strategy._record_measurement_start_if_needed(phase_stats, credit_num)
    
    assert strategy._measurement_start_ns is None

    # Issue first credit of measurement loop - measurement should start
    strategy._record_measurement_start_if_needed(phase_stats, dataset_size)  # credit 3 (b1)
    
    assert strategy._measurement_start_ns is not None

    # Further credits should not change measurement_start_ns
    original_start = strategy._measurement_start_ns
    strategy._record_measurement_start_if_needed(phase_stats, dataset_size + 1)  # credit 4 (b2)
    
    assert strategy._measurement_start_ns == original_start


@pytest.mark.asyncio
async def test_steady_state_fallback_without_dataset_size(mock_credit_manager):
    """Test that steady-state falls back to old behavior when dataset_size is not set."""
    config = TimingManagerConfig(
        timing_mode=TimingMode.REQUEST_RATE,
        request_count=2,
        request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
        concurrency=1,
        steady_state=True,
        steady_state_grace_period=TEST_GRACE_PERIOD,
        # dataset_size not set - should use fallback behavior
    )
    phase_stats = profiling_phase_stats_from_config(config)

    strategy = RequestRateStrategy(config, mock_credit_manager)
    strategy.phase_stats[CreditPhase.PROFILING] = phase_stats

    # Without dataset_size, measurement loop boundaries should not be set
    assert strategy._measurement_loop_start_credit is None
    assert strategy._measurement_loop_end_credit is None

    # Return credits without credit_num - should use old behavior (completed >= request_count)
    await strategy._on_credit_return(
        CreditReturnMessage(
            service_id="worker-1",
            phase=CreditPhase.PROFILING,
            credit_drop_id=str(uuid.uuid4()),
            requests_sent=1,
        )
    )
    
    assert not strategy._steady_state_stop_event.is_set()

    await strategy._on_credit_return(
        CreditReturnMessage(
            service_id="worker-1",
            phase=CreditPhase.PROFILING,
            credit_drop_id=str(uuid.uuid4()),
            requests_sent=1,
        )
    )

    # Measurement is complete but we need to wait for grace period
    assert strategy._measurement_complete is True
    assert not strategy._steady_state_stop_event.is_set()
    
    # Wait for all background tasks (including grace period timer) to complete
    if strategy.tasks:
        await asyncio.gather(*strategy.tasks, return_exceptions=True)
    
    # With 2 completions and request_count=2, old behavior should trigger cancel after grace
    assert strategy._steady_state_stop_event.is_set()
    assert any(
        isinstance(message, ProfileCancelCommand)
        and message.reason == "steady_state"
        for message in mock_credit_manager.publish_calls
    )
