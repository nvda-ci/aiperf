# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge case and boundary tests for timing strategies.

These tests verify correct behavior at edge cases and boundary conditions:

- Minimum values (1 session, 1 turn, etc.)
- Maximum practical values (high counts, high rates)
- Boundary values (concurrency = sessions, etc.)
- Zero and near-zero values
- Unusual combinations
- Configuration validation

Tests are organized by the type of edge case being tested.
"""

import pytest

from tests.component_integration.timing.conftest import (
    TimingTestConfig,
    build_timing_command,
    defaults,
)
from tests.harness.analyzers import ConcurrencyAnalyzer
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
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """
    if config.turns_per_session > 1:
        cmd += (
            f" --session-turns-mean {config.turns_per_session} --session-turns-stddev 0"
        )
    if config.prefill_concurrency is not None:
        cmd += f" --prefill-concurrency {config.prefill_concurrency}"
    return cmd


@pytest.mark.component_integration
class TestMinimumValues:
    """Tests with minimum practical values."""

    def test_single_session_constant_rate(self, cli: AIPerfCLI):
        """Test with single session (minimum)."""
        config = TimingTestConfig(num_sessions=1, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 1

    def test_single_session_poisson_rate(self, cli: AIPerfCLI):
        """Test single session with Poisson."""
        config = TimingTestConfig(num_sessions=1, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 1

    def test_single_session_burst(self, cli: AIPerfCLI):
        """Test single session burst mode."""
        config = TimingTestConfig(num_sessions=1, qps=0, concurrency=1)
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 1

    def test_single_session_user_centric(self, cli: AIPerfCLI):
        """Test single session user-centric mode.

        User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. We verify at least one request is issued.
        """
        config = TimingTestConfig(num_sessions=1, qps=100.0)
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=30.0)

        # Duration-based: at least 1 request
        assert result.request_count >= 1

    def test_single_turn_multi_session(self, cli: AIPerfCLI):
        """Test multiple sessions with single turn each."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=100.0,
            turns_per_session=1,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestMaximumPracticalValues:
    """Tests with high practical values."""

    def test_many_sessions_constant(self, cli: AIPerfCLI):
        """Test with many sessions constant rate."""
        config = TimingTestConfig(
            num_sessions=200,
            qps=500.0,
            timeout=120.0,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_many_sessions_poisson(self, cli: AIPerfCLI):
        """Test with many sessions Poisson rate."""
        config = TimingTestConfig(
            num_sessions=200,
            qps=500.0,
            timeout=120.0,
        )
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_many_turns_per_session(self, cli: AIPerfCLI):
        """Test with many turns per session."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=100.0,
            turns_per_session=10,
            timeout=90.0,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests

    def test_high_qps(self, cli: AIPerfCLI):
        """Test with very high QPS."""
        config = TimingTestConfig(
            num_sessions=100,
            qps=1000.0,
            timeout=90.0,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_high_concurrency_burst(self, cli: AIPerfCLI):
        """Test burst with high concurrency."""
        config = TimingTestConfig(
            num_sessions=100,
            qps=0,
            concurrency=100,
            timeout=90.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions


@pytest.mark.component_integration
class TestBoundaryEquality:
    """Tests where values are equal (boundary conditions)."""

    def test_concurrency_equals_sessions(self, cli: AIPerfCLI):
        """Test when concurrency equals number of sessions."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=0,
            concurrency=20,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_sessions_equals_turns_total(self, cli: AIPerfCLI):
        """Test when total requests equals a power of 2."""
        config = TimingTestConfig(
            num_sessions=16,
            qps=100.0,
            turns_per_session=4,  # 64 total
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == 64


@pytest.mark.component_integration
class TestLowRates:
    """Tests with low QPS values."""

    def test_very_low_constant_rate(self, cli: AIPerfCLI):
        """Test very low constant rate (long gaps)."""
        config = TimingTestConfig(
            num_sessions=3,
            qps=2.0,  # 500ms gap
            timeout=30.0,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_very_low_poisson_rate(self, cli: AIPerfCLI):
        """Test very low Poisson rate."""
        config = TimingTestConfig(
            num_sessions=3,
            qps=2.0,
            timeout=30.0,
        )
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_very_low_user_centric_rate(self, cli: AIPerfCLI):
        """Test very low user-centric rate.

        User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. We verify at least some requests are issued.
        """
        config = TimingTestConfig(
            num_sessions=3,
            qps=3.0,
            timeout=30.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Duration-based: at least num_sessions requests (initial user turns)
        assert result.request_count >= config.num_sessions


@pytest.mark.component_integration
class TestConcurrencyMinimum:
    """Tests with minimum concurrency (1)."""

    def test_concurrency_one_burst(self, cli: AIPerfCLI):
        """Test burst with concurrency=1 (sequential execution)."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=0,
            concurrency=1,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_concurrency_one_with_rate(self, cli: AIPerfCLI):
        """Test concurrency=1 with rate limiting."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=100.0,
            concurrency=1,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_concurrency_one_multi_turn(self, cli: AIPerfCLI):
        """Test concurrency=1 with multi-turn."""
        config = TimingTestConfig(
            num_sessions=5,
            qps=0,
            turns_per_session=4,
            concurrency=1,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests


@pytest.mark.component_integration
class TestPrefillConcurrencyEdgeCases:
    """Tests for prefill concurrency edge cases.

    Note: These tests use burst mode to ensure prefill limits are actually hit.
    For rate-limited modes to hit prefill limits, QPS × TTFT >= prefill_concurrency,
    which requires very high QPS (e.g., prefill_concurrency=1 needs QPS >= 200).
    """

    def test_prefill_equals_concurrency(self, cli: AIPerfCLI):
        """Test when prefill concurrency equals regular concurrency (burst mode)."""
        concurrency = 5
        config = TimingTestConfig(
            num_sessions=30,  # >> concurrency to ensure limit is hit
            qps=0,  # Burst mode
            concurrency=concurrency,
            prefill_concurrency=concurrency,
        )

        # Burst mode should hit both limits
        assert config.will_hit_concurrency_limit()
        assert config.will_hit_prefill_limit()

        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        # Verify both limits were respected and reached
        conc_analyzer = ConcurrencyAnalyzer(result)
        max_concurrent = conc_analyzer.get_max_concurrent()
        max_prefill = conc_analyzer.get_max_prefill_concurrent()

        assert max_concurrent <= concurrency, (
            f"Max concurrent {max_concurrent} exceeded limit {concurrency}"
        )
        assert max_concurrent == concurrency, (
            f"Concurrency limit not reached: max={max_concurrent}, limit={concurrency}"
        )
        assert max_prefill <= concurrency, (
            f"Max prefill {max_prefill} exceeded limit {concurrency}"
        )
        assert max_prefill == concurrency, (
            f"Prefill limit not reached: max={max_prefill}, limit={concurrency}"
        )

    def test_prefill_one(self, cli: AIPerfCLI):
        """Test with prefill concurrency=1 (strict serialization, burst mode)."""
        prefill_concurrency = 1
        config = TimingTestConfig(
            num_sessions=15,
            qps=0,  # Burst mode
            concurrency=10,
            prefill_concurrency=prefill_concurrency,
        )

        # Burst mode should hit the prefill limit
        assert config.will_hit_prefill_limit()

        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        # Verify strict serialization of prefill phase
        conc_analyzer = ConcurrencyAnalyzer(result)
        max_prefill = conc_analyzer.get_max_prefill_concurrent()
        assert max_prefill <= prefill_concurrency, (
            f"Max prefill {max_prefill} exceeded limit {prefill_concurrency}"
        )
        assert max_prefill == prefill_concurrency, (
            f"Prefill limit not reached: max={max_prefill}, limit={prefill_concurrency}"
        )


@pytest.mark.component_integration
class TestUnusualCombinations:
    """Tests with unusual parameter combinations."""

    def test_two_sessions_high_rate(self, cli: AIPerfCLI):
        """Test just 2 sessions with very high rate."""
        config = TimingTestConfig(
            num_sessions=2,
            qps=1000.0,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 2

    def test_many_sessions_low_rate(self, cli: AIPerfCLI):
        """Test many sessions with very low rate."""
        config = TimingTestConfig(
            num_sessions=50,
            qps=50.0,  # 20ms gap, 1 second for 50 requests
            timeout=30.0,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_single_session_many_turns(self, cli: AIPerfCLI):
        """Test single session with many turns."""
        config = TimingTestConfig(
            num_sessions=1,
            qps=50.0,
            turns_per_session=10,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests


@pytest.mark.component_integration
class TestAllModesCompleteness:
    """Verify all timing modes complete successfully with various configs."""

    @pytest.mark.parametrize(
        "num_sessions,turns",
        [
            (5, 1),
            (10, 2),
            (15, 3),
            (20, 4),
        ],
    )
    def test_constant_rate_matrix(self, cli: AIPerfCLI, num_sessions: int, turns: int):
        """Test constant rate across session/turn combinations."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=100.0,
            turns_per_session=turns,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests

    @pytest.mark.parametrize(
        "num_sessions,turns",
        [
            (5, 1),
            (10, 2),
            (15, 3),
            (20, 4),
        ],
    )
    def test_poisson_rate_matrix(self, cli: AIPerfCLI, num_sessions: int, turns: int):
        """Test Poisson rate across session/turn combinations."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=100.0,
            turns_per_session=turns,
        )
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests

    @pytest.mark.parametrize(
        "num_sessions,turns",
        [
            (5, 1),
            (10, 2),
            (15, 3),
            (20, 4),
        ],
    )
    def test_user_centric_rate_matrix(
        self, cli: AIPerfCLI, num_sessions: int, turns: int
    ):
        """Test user-centric rate across session/turn combinations.

        User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. We verify at least the initial user turns
        are issued.
        """
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=100.0,
            turns_per_session=turns,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Duration-based: at least num_sessions requests (initial user turns)
        assert result.request_count >= config.num_sessions

    @pytest.mark.parametrize(
        "num_sessions,turns,concurrency",
        [
            (10, 1, 5),
            (15, 2, 5),
            (20, 3, 10),
            (25, 4, 10),
        ],
    )
    def test_burst_mode_matrix(
        self, cli: AIPerfCLI, num_sessions: int, turns: int, concurrency: int
    ):
        """Test burst mode across various combinations."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,
            turns_per_session=turns,
            concurrency=concurrency,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests
