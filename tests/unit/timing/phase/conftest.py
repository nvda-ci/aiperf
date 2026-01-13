# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for timing phases tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.models import CreditPhaseStats
from aiperf.timing.config import CreditPhaseConfig


@pytest.fixture
def mock_pub_client() -> MagicMock:
    """Create a MagicMock pub client with async publish method."""
    mock = MagicMock()
    mock.publish = AsyncMock()
    return mock


@pytest.fixture
def mock_phase_publisher() -> MagicMock:
    """Create a MagicMock phase publisher."""
    mock = MagicMock()
    mock.publish_phase_start = AsyncMock()
    mock.publish_phase_sending_complete = AsyncMock()
    mock.publish_phase_complete = AsyncMock()
    mock.publish_progress = AsyncMock()
    mock.publish_credits_complete = AsyncMock()
    return mock


@pytest.fixture
def mock_credit_router() -> MagicMock:
    """Create a MagicMock credit router."""
    mock = MagicMock()
    mock.send_credit = AsyncMock()
    mock.cancel_all_credits = AsyncMock()
    mock.mark_credits_complete = MagicMock()
    mock.reset = MagicMock()
    return mock


@pytest.fixture
def mock_credit_manager_simple() -> MagicMock:
    """Create a simple MagicMock credit manager."""
    mock = MagicMock()
    mock.configure_for_phase = MagicMock()
    # Return tuple (session_released, prefill_released) for release_stuck_slots
    mock.release_stuck_slots = MagicMock(return_value=(0, 0))
    return mock


@pytest.fixture
def mock_credit_manager() -> MagicMock:
    """Create a MagicMock credit manager (alias for mock_credit_manager_simple)."""
    mock = MagicMock()
    mock.configure_for_phase = MagicMock()
    # Return tuple (session_released, prefill_released) for release_stuck_slots
    mock.release_stuck_slots = MagicMock(return_value=(0, 0))
    return mock


@pytest.fixture
def sample_phase_config() -> CreditPhaseConfig:
    """Sample phase configuration for tests."""
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=100,
    )


@pytest.fixture
def sample_phase_stats() -> CreditPhaseStats:
    """Sample phase stats for tests."""
    return CreditPhaseStats(
        phase=CreditPhase.PROFILING,
        requests_sent=50,
        requests_completed=30,
        requests_cancelled=2,
        final_requests_sent=50,
        start_ns=1000000,
    )
