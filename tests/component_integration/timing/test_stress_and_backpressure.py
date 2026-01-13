# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stress and backpressure tests for timing strategies.

These tests stress-test the timing strategies under high load conditions:

- High concurrency with low rate (concurrency contention)
- Low concurrency with high rate (backpressure)
- Rapid credit issuance and return (tight timing)
- Many sessions competing for limited resources
- Multi-turn with rapid turn transitions
- Mixed workload patterns

Tests verify:
- No deadlocks or hangs
- All credits eventually returned
- No data corruption
- Consistent behavior under stress

NOTE: These are NOT true race condition tests. Asyncio is single-threaded,
so true concurrent access to shared state is not tested here. These tests
validate system behavior under stress and backpressure conditions.
"""

import pytest

from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.timing.conftest import (
    TimingTestConfig,
    build_timing_command,
    defaults,
)
from tests.harness.analyzers import CreditFlowAnalyzer
from tests.harness.utils import AIPerfCLI


def build_burst_command(config: TimingTestConfig) -> str:
    """Build burst mode command."""
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {config.num_sessions} \
            --concurrency {config.concurrency} \
            --osl {config.osl} \
            --ui {defaults.ui}
    """
    if config.turns_per_session > 1:
        cmd += (
            f" --session-turns-mean {config.turns_per_session} --session-turns-stddev 0"
        )
    return cmd


@pytest.mark.component_integration
class TestConcurrencyContention:
    """Tests for high concurrency contention scenarios."""

    def test_high_concurrency_burst(self, cli: AIPerfCLI):
        """Test burst mode with very high concurrency."""
        config = TimingTestConfig(
            num_sessions=100,
            qps=0,
            concurrency=50,
            timeout=90.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    def test_concurrency_equals_sessions(self, cli: AIPerfCLI):
        """Test when all sessions can run concurrently."""
        config = TimingTestConfig(
            num_sessions=30,
            qps=0,
            concurrency=30,
            timeout=60.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_severe_concurrency_limit(self, cli: AIPerfCLI):
        """Test with severe concurrency limit (1)."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=0,
            concurrency=1,
            timeout=60.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_concurrency_with_rate_limit(self, cli: AIPerfCLI):
        """Test concurrency combined with rate limiting."""
        config = TimingTestConfig(
            num_sessions=30,
            qps=100.0,
            concurrency=5,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions


@pytest.mark.component_integration
class TestBackpressureScenarios:
    """Tests for backpressure conditions (rate faster than processing)."""

    def test_high_rate_low_concurrency(self, cli: AIPerfCLI):
        """Test high request rate with low concurrency (backpressure)."""
        config = TimingTestConfig(
            num_sessions=50,
            qps=500.0,
            concurrency=3,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    def test_poisson_with_low_concurrency(self, cli: AIPerfCLI):
        """Test Poisson rate with low concurrency (bursty backpressure)."""
        config = TimingTestConfig(
            num_sessions=40,
            qps=200.0,
            concurrency=4,
        )
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions


@pytest.mark.component_integration
class TestRapidCreditTurnover:
    """Tests for rapid credit issuance and return."""

    def test_rapid_burst_small_sessions(self, cli: AIPerfCLI):
        """Test rapid burst with many small sessions."""
        config = TimingTestConfig(
            num_sessions=100,
            qps=0,
            concurrency=20,
            timeout=90.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_rapid_multi_turn(self, cli: AIPerfCLI):
        """Test rapid multi-turn transitions."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=0,
            turns_per_session=5,
            concurrency=10,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.turn_indices_sequential()

    def test_high_rate_multi_turn(self, cli: AIPerfCLI):
        """Test high rate with multi-turn."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=200.0,
            turns_per_session=4,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests


@pytest.mark.component_integration
class TestResourceCompetition:
    """Tests for many sessions competing for limited resources."""

    @pytest.mark.slow
    def test_many_sessions_limited_concurrency(self, cli: AIPerfCLI):
        """Test many sessions with very limited concurrency."""
        config = TimingTestConfig(
            num_sessions=50,
            qps=0,
            concurrency=2,
            timeout=90.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    @pytest.mark.slow
    def test_many_multi_turn_limited_concurrency(self, cli: AIPerfCLI):
        """Test many multi-turn sessions with limited concurrency."""
        config = TimingTestConfig(
            num_sessions=25,
            qps=0,
            turns_per_session=4,
            concurrency=3,
            timeout=120.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests


@pytest.mark.component_integration
class TestUserCentricStress:
    """Stress tests specific to user-centric mode."""

    def test_user_centric_high_session_count(self, cli: AIPerfCLI):
        """Test user-centric with many users competing.

        User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. We verify at least the initial user turns
        are issued.
        """
        config = TimingTestConfig(
            num_sessions=80,
            qps=200.0,
            timeout=90.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Duration-based: at least num_sessions requests (initial user turns)
        assert result.request_count >= config.num_sessions

    def test_user_centric_rapid_turns(self, cli: AIPerfCLI):
        """Test user-centric with rapid turn transitions.

        User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. We verify at least the initial user turns
        are issued and credits are balanced.
        """
        config = TimingTestConfig(
            num_sessions=20,
            qps=150.0,
            turns_per_session=6,
            timeout=90.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Duration-based: at least num_sessions requests (initial user turns)
        assert result.request_count >= config.num_sessions

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    @pytest.mark.slow
    def test_user_centric_high_gap(self, cli: AIPerfCLI):
        """Test user-centric with high gap (many users, low rate).

        User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. With 30 users at 30 QPS in 1 second,
        we may not get all initial turns if timing is tight.
        """
        config = TimingTestConfig(
            num_sessions=30,
            qps=30.0,  # gap = 1 second per user
            turns_per_session=2,
            timeout=120.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Duration-based: verify at least some requests were issued
        # With high gap (1s per user) in 1s duration, we may not get all users
        assert result.request_count >= 1


@pytest.mark.component_integration
@pytest.mark.stress
class TestMixedWorkloadStress:
    """Mixed workload stress tests combining different patterns."""

    def test_constant_high_volume(self, cli: AIPerfCLI):
        """High volume constant rate stress test."""
        config = TimingTestConfig(
            num_sessions=150,
            qps=400.0,
            timeout=120.0,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_poisson_high_volume(self, cli: AIPerfCLI):
        """High volume Poisson rate stress test."""
        config = TimingTestConfig(
            num_sessions=150,
            qps=400.0,
            timeout=120.0,
        )
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_burst_high_volume(self, cli: AIPerfCLI):
        """High volume burst mode stress test."""
        config = TimingTestConfig(
            num_sessions=150,
            qps=0,
            concurrency=30,
            timeout=120.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_user_centric_high_volume(self, cli: AIPerfCLI):
        """High volume user-centric mode stress test.

        User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. We verify at least the initial user turns
        are issued.
        """
        config = TimingTestConfig(
            num_sessions=150,
            qps=400.0,
            timeout=120.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Duration-based: at least num_sessions requests (initial user turns)
        assert result.request_count >= config.num_sessions


@pytest.mark.component_integration
@pytest.mark.stress
class TestCreditBalanceUnderStress:
    """Verify credit balance is maintained under stress conditions."""

    @pytest.mark.parametrize(
        "rate_mode",
        ["constant", "poisson"],
    )
    def test_credit_balance_with_rate(self, cli: AIPerfCLI, rate_mode: str):
        """Verify credits balanced with rate limiting under stress."""
        config = TimingTestConfig(
            num_sessions=50,
            qps=200.0,
            turns_per_session=3,
            timeout=90.0,
        )
        cmd = build_timing_command(config, arrival_pattern=rate_mode)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced(), (
            f"Credits unbalanced ({rate_mode}): "
            f"{analyzer.total_credits} sent, {analyzer.total_returns} returned"
        )

    def test_credit_balance_burst(self, cli: AIPerfCLI):
        """Verify credits balanced in burst mode under stress."""
        config = TimingTestConfig(
            num_sessions=50,
            qps=0,
            turns_per_session=3,
            concurrency=15,
            timeout=90.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced()

    def test_credit_balance_user_centric(self, cli: AIPerfCLI):
        """Verify credits balanced in user-centric mode under stress."""
        config = TimingTestConfig(
            num_sessions=50,
            qps=200.0,
            turns_per_session=3,
            timeout=90.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced()
