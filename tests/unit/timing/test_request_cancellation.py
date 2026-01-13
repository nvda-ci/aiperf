# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RequestCancellationPolicy.

Tests:
- Enable/disable behavior
- Rate bounds (0%, 50%, 100%)
- Phase-specific behavior (warmup vs profiling)
- Deterministic RNG behavior
- Delay calculation
"""

import pytest

from aiperf.common import random_generator as rng
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.timing.config import RequestCancellationConfig
from aiperf.timing.request_cancellation import RequestCancellationSimulator

# =============================================================================
# Helper Functions
# =============================================================================


def make_cancellation_config(
    cancellation_rate: float | None = None,
    cancellation_delay: float = 0.0,
    random_seed: int | None = None,
) -> RequestCancellationConfig:
    """Create RequestCancellationConfig for testing."""
    # random_seed is not used by RequestCancellationConfig but kept for API compatibility
    # (the global RNG is seeded separately in tests)
    _ = random_seed
    return RequestCancellationConfig(
        rate=cancellation_rate,
        delay=cancellation_delay,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def policy_disabled() -> RequestCancellationSimulator:
    """Policy with cancellation disabled."""
    return RequestCancellationSimulator(make_cancellation_config())


@pytest.fixture
def policy_100_percent() -> RequestCancellationSimulator:
    """Policy with 100% cancellation rate."""
    return RequestCancellationSimulator(
        make_cancellation_config(cancellation_rate=100.0, cancellation_delay=1.0)
    )


@pytest.fixture
def policy_0_percent() -> RequestCancellationSimulator:
    """Policy with 0% cancellation rate."""
    return RequestCancellationSimulator(
        make_cancellation_config(cancellation_rate=0.0, cancellation_delay=1.0)
    )


# =============================================================================
# Enable/Disable Tests
# =============================================================================


class TestCancellationEnabled:
    """Tests for cancellation enable/disable behavior."""

    def test_disabled_when_rate_not_configured(self, policy_disabled):
        """Cancellation disabled when rate is not configured."""
        assert not policy_disabled.is_cancellation_enabled

    def test_disabled_returns_none(self, policy_disabled):
        """Disabled policy always returns None."""
        assert policy_disabled.next_cancellation_delay_ns() is None

    def test_enabled_when_rate_is_positive(self, policy_100_percent):
        """Cancellation enabled when rate is positive."""
        assert policy_100_percent.is_cancellation_enabled

    def test_disabled_when_rate_is_zero(self, policy_0_percent):
        """Cancellation disabled when rate is 0.0."""
        assert not policy_0_percent.is_cancellation_enabled

    @pytest.mark.parametrize(
        "rate,expected_enabled",
        [
            (None, False),
            (0.0, False),
            (0.1, True),
            (50.0, True),
            (100.0, True),
        ],
    )  # fmt: skip
    def test_enabled_status_by_rate(self, rate: float | None, expected_enabled: bool):
        """is_cancellation_enabled reflects rate configuration."""
        config = make_cancellation_config(cancellation_rate=rate)
        policy = RequestCancellationSimulator(config)

        assert policy.is_cancellation_enabled == expected_enabled


# =============================================================================
# Rate Bounds Tests
# =============================================================================


class TestCancellationRate:
    """Tests for cancellation rate behavior."""

    def test_zero_rate_never_cancels(self, policy_0_percent):
        """0% rate never triggers cancellation."""
        for _ in range(100):
            assert policy_0_percent.next_cancellation_delay_ns() is None

    def test_full_rate_always_cancels(self, policy_100_percent):
        """100% rate always triggers cancellation."""
        for _ in range(100):
            delay = policy_100_percent.next_cancellation_delay_ns()
            assert delay == int(1.0 * NANOS_PER_SECOND)

    def test_50_percent_rate_cancels_approximately_half(self):
        """50% rate cancels approximately half of requests."""
        rng.reset()
        rng.init(42)
        config = make_cancellation_config(
            cancellation_rate=50.0, cancellation_delay=1.0, random_seed=42
        )
        policy = RequestCancellationSimulator(config)

        decisions = [policy.next_cancellation_delay_ns() for _ in range(100)]
        cancellation_count = sum(1 for d in decisions if d is not None)

        assert 30 <= cancellation_count <= 70  # Allow variance


# =============================================================================
# Delay Calculation Tests
# =============================================================================


class TestCancellationDelay:
    """Tests for cancellation delay calculation."""

    @pytest.mark.parametrize(
        "delay_seconds,expected_ns",
        [
            (0.0, 0),
            (0.5, int(0.5 * NANOS_PER_SECOND)),
            (1.0, int(1.0 * NANOS_PER_SECOND)),
            (2.5, int(2.5 * NANOS_PER_SECOND)),
            (10.0, int(10.0 * NANOS_PER_SECOND)),
        ],
    )  # fmt: skip
    def test_delay_converted_to_nanoseconds(
        self, delay_seconds: float, expected_ns: int
    ):
        """Delay is correctly converted from seconds to nanoseconds."""
        config = make_cancellation_config(
            cancellation_rate=100.0, cancellation_delay=delay_seconds
        )
        policy = RequestCancellationSimulator(config)

        assert policy.next_cancellation_delay_ns() == expected_ns


# =============================================================================
# Phase-Specific Behavior Tests
# =============================================================================


class TestPhaseSpecificBehavior:
    """Tests for phase-specific cancellation behavior."""

    def test_warmup_phase_skips_cancellation(self):
        """Warmup phase always returns None regardless of rate."""
        config = make_cancellation_config(
            cancellation_rate=100.0, cancellation_delay=1.0
        )
        policy = RequestCancellationSimulator(config)

        for _ in range(10):
            result = policy.next_cancellation_delay_ns(phase=CreditPhase.WARMUP)
            assert result is None

    def test_profiling_phase_applies_cancellation(self):
        """Profiling phase applies cancellation policy."""
        config = make_cancellation_config(
            cancellation_rate=100.0, cancellation_delay=1.0
        )
        policy = RequestCancellationSimulator(config)

        result = policy.next_cancellation_delay_ns(phase=CreditPhase.PROFILING)
        assert result == int(1.0 * NANOS_PER_SECOND)

    def test_no_phase_applies_cancellation(self):
        """No phase specified applies cancellation policy."""
        config = make_cancellation_config(
            cancellation_rate=100.0, cancellation_delay=1.0
        )
        policy = RequestCancellationSimulator(config)

        result = policy.next_cancellation_delay_ns(phase=None)
        assert result == int(1.0 * NANOS_PER_SECOND)

    def test_warmup_phase_does_not_consume_rng(self):
        """Warmup phase does not consume RNG state (no RNG call made)."""
        rng.reset()
        rng.init(42)
        config = make_cancellation_config(
            cancellation_rate=50.0, cancellation_delay=1.0, random_seed=42
        )
        policy = RequestCancellationSimulator(config)

        # Call with warmup phase multiple times
        for _ in range(10):
            policy.next_cancellation_delay_ns(phase=CreditPhase.WARMUP)

        # Now call with profiling - should get same result as if warmup calls never happened
        rng.reset()
        rng.init(42)
        fresh_policy = RequestCancellationSimulator(config)

        # First profiling call should match
        result1 = policy.next_cancellation_delay_ns(phase=CreditPhase.PROFILING)
        result2 = fresh_policy.next_cancellation_delay_ns(phase=CreditPhase.PROFILING)
        assert result1 == result2


# =============================================================================
# Deterministic Behavior Tests
# =============================================================================


class TestDeterministicBehavior:
    """Tests for deterministic RNG behavior."""

    def test_same_seed_produces_same_sequence(self):
        """Same seed produces identical cancellation sequences."""
        rng.reset()
        rng.init(42)
        policy1 = RequestCancellationSimulator(
            make_cancellation_config(
                cancellation_rate=50.0, cancellation_delay=1.0, random_seed=42
            )
        )
        decisions1 = [policy1.next_cancellation_delay_ns() for _ in range(50)]

        rng.reset()
        rng.init(42)
        policy2 = RequestCancellationSimulator(
            make_cancellation_config(
                cancellation_rate=50.0, cancellation_delay=1.0, random_seed=42
            )
        )
        decisions2 = [policy2.next_cancellation_delay_ns() for _ in range(50)]

        assert decisions1 == decisions2

    def test_different_seeds_produce_different_sequences(self):
        """Different seeds produce different cancellation sequences."""
        rng.reset()
        rng.init(42)
        policy1 = RequestCancellationSimulator(
            make_cancellation_config(
                cancellation_rate=50.0, cancellation_delay=1.0, random_seed=42
            )
        )
        decisions1 = [policy1.next_cancellation_delay_ns() for _ in range(50)]

        rng.reset()
        rng.init(123)
        policy2 = RequestCancellationSimulator(
            make_cancellation_config(
                cancellation_rate=50.0, cancellation_delay=1.0, random_seed=123
            )
        )
        decisions2 = [policy2.next_cancellation_delay_ns() for _ in range(50)]

        assert decisions1 != decisions2
