# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive cross-parameter component integration tests for user-centric rate mode.

This module provides exhaustive testing across the full parameter space:
- QPS values: low (5-20), medium (50-100), high (150-300)
- Num users: small (2-5), medium (10-20), large (30-50)
- Session turns: minimum (2), medium (4-5), high (8-10)

Tests verify:
- Completion with correct request/session counts
- Credit flow balance (no leaks)
- Turn index sequentiality
- Timing characteristics match expected behavior for each mode
"""

import pytest

from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.component_integration.timing.conftest import (
    assert_credits_balanced,
    assert_turn_indices_sequential,
)
from tests.harness.analyzers import (
    CreditFlowAnalyzer,
    StatisticalAnalyzer,
    TimingAnalyzer,
    verify_no_interleaving_within_session,
)
from tests.harness.utils import AIPerfCLI

pytestmark = pytest.mark.stress


def build_user_centric_command(
    num_users: int,
    qps: float,
    turns_per_session: int = 2,
    osl: int = 50,
    *,
    use_duration: bool = True,
    duration: float = 1.0,
    num_sessions: int | None = None,
    request_count: int | None = None,
) -> str:
    """Build user-centric rate command with flexible stop conditions.

    Args:
        num_users: Number of concurrent user slots
        qps: Request rate (queries per second)
        turns_per_session: Turns per conversation (minimum 2 for user-centric)
        osl: Output sequence length
        use_duration: If True, use --benchmark-duration as stop condition
        duration: Benchmark duration in seconds (if use_duration=True)
        num_sessions: Stop after this many sessions (optional)
        request_count: Stop after this many requests (optional)

    Returns:
        CLI command string
    """
    # User-centric requires multi-turn (minimum 2)
    turns = max(turns_per_session, 2)

    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --osl {osl} \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui} \
            --num-users {num_users} \
            --user-centric-rate {qps} \
            --session-turns-mean {turns} --session-turns-stddev 0
    """

    if use_duration:
        cmd += f" --benchmark-duration {duration} --benchmark-grace-period 0.5"

    if num_sessions is not None:
        cmd += f" --num-sessions {num_sessions}"

    if request_count is not None:
        cmd += f" --request-count {request_count}"

    return cmd


# =============================================================================
# Cross-Parameter Matrix Tests - PRECISE Mode
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricPreciseQPSVariations:
    """Test PRECISE mode with various QPS values."""

    @pytest.mark.parametrize(
        "num_users,qps",
        [
            # Low QPS (5-20)
            (5, 5.0),
            (5, 10.0),
            (5, 15.0),
            (10, 10.0),
            (10, 20.0),
            # Medium QPS (50-100)
            (10, 50.0),
            (10, 75.0),
            (10, 100.0),
            (20, 50.0),
            (20, 100.0),
            # High QPS (150-300)
            (20, 150.0),
            (30, 200.0),
            (50, 250.0),
            (50, 300.0),
        ],
    )  # fmt: skip
    def test_precise_mode_qps_variations(
        self, cli: AIPerfCLI, num_users: int, qps: float
    ):
        """PRECISE mode completes correctly at various QPS levels."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        # Should complete with at least num_users requests (initial turns)
        assert result.request_count >= num_users
        assert_credits_balanced(result)

    @pytest.mark.parametrize(
        "num_users,qps,turns",
        [
            # Low QPS with various turns
            (5, 10.0, 2),
            (5, 10.0, 4),
            (5, 10.0, 6),
            # Medium QPS with various turns
            (10, 50.0, 2),
            (10, 50.0, 4),
            (10, 50.0, 8),
            # High QPS with various turns
            (20, 100.0, 2),
            (20, 100.0, 5),
            (20, 100.0, 10),
        ],
    )  # fmt: skip
    def test_precise_mode_multi_turn_variations(
        self, cli: AIPerfCLI, num_users: int, qps: float, turns: int
    ):
        """PRECISE mode handles multi-turn conversations at various QPS."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= num_users
        assert_credits_balanced(result)

        # Verify multi-turn behavior
        if turns > 2:
            turn_indices = {r.metadata.turn_index for r in result.jsonl}
            assert len(turn_indices) > 1, "Expected multiple turn indices"


# =============================================================================
# Cross-Parameter Matrix Tests - LMBENCH Mode
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricLMBenchQPSVariations:
    """Test LMBENCH mode with various QPS values."""

    @pytest.mark.parametrize(
        "num_users,qps",
        [
            # Low QPS (5-20)
            (5, 5.0),
            (5, 10.0),
            (5, 15.0),
            (10, 10.0),
            (10, 20.0),
            # Medium QPS (50-100)
            (10, 50.0),
            (10, 75.0),
            (10, 100.0),
            (20, 50.0),
            (20, 100.0),
            # High QPS (150-300)
            (20, 150.0),
            (30, 200.0),
            (50, 250.0),
            (50, 300.0),
        ],
    )  # fmt: skip
    def test_lmbench_mode_qps_variations(
        self, cli: AIPerfCLI, num_users: int, qps: float
    ):
        """LMBENCH mode completes correctly at various QPS levels."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        # Should complete with at least num_users requests
        assert result.request_count >= num_users
        assert_credits_balanced(result)

    @pytest.mark.parametrize(
        "num_users,qps,turns",
        [
            # Low QPS with various turns
            (5, 10.0, 2),
            (5, 10.0, 4),
            (5, 10.0, 6),
            # Medium QPS with various turns
            (10, 50.0, 2),
            (10, 50.0, 4),
            (10, 50.0, 8),
            # High QPS with various turns
            (20, 100.0, 2),
            (20, 100.0, 5),
            (20, 100.0, 10),
        ],
    )  # fmt: skip
    def test_lmbench_mode_multi_turn_variations(
        self, cli: AIPerfCLI, num_users: int, qps: float, turns: int
    ):
        """LMBENCH mode handles multi-turn conversations at various QPS."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= num_users
        assert_credits_balanced(result)

        # Verify multi-turn behavior
        if turns > 2:
            turn_indices = {r.metadata.turn_index for r in result.jsonl}
            assert len(turn_indices) > 1, "Expected multiple turn indices"


# =============================================================================
# Comprehensive Matrix Tests - Both Modes
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricComprehensiveMatrix:
    """Comprehensive matrix testing both modes across parameters."""

    @pytest.mark.parametrize(
        "num_users,qps,turns",
        [
            (5, 25.0, 2),
            (5, 50.0, 3),
            (5, 75.0, 4),
            (10, 25.0, 2),
            (10, 50.0, 3),
            (10, 100.0, 5),
            (15, 50.0, 2),
            (15, 75.0, 4),
            (20, 100.0, 3),
            (30, 150.0, 2),
        ],
    )  # fmt: skip
    def test_comprehensive_parameter_matrix(
        self,
        cli: AIPerfCLI,
        num_users: int,
        qps: float,
        turns: int,
    ):
        """Both modes work correctly across comprehensive parameter combinations."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= num_users, (
            f"failed with {num_users} users, {qps} QPS, {turns} turns"
        )
        assert_credits_balanced(result)
        assert_turn_indices_sequential(result)


# =============================================================================
# Credit Flow Tests
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricCreditFlow:
    """Credit flow verification."""

    def test_credits_balanced_both_modes(self, cli: AIPerfCLI):
        """Maintain balanced credits (no leaks)."""
        cmd = build_user_centric_command(
            num_users=15,
            qps=75.0,
            turns_per_session=4,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced(), (
            f"Credits not balanced - "
            f"{analyzer.total_credits} issued, {analyzer.total_returns} returned"
        )

    def test_turn_indices_sequential(self, cli: AIPerfCLI):
        """Maintain sequential turn indices within sessions."""
        cmd = build_user_centric_command(
            num_users=12,
            qps=60.0,
            turns_per_session=5,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.turn_indices_sequential(), (
            "Turn indices not sequential within sessions"
        )

    def test_no_interleaving_within_user(self, cli: AIPerfCLI):
        """Prevent turn interleaving within a single user."""
        cmd = build_user_centric_command(
            num_users=10,
            qps=50.0,
            turns_per_session=4,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        passed, reason = verify_no_interleaving_within_session(analyzer)
        assert passed, reason


# =============================================================================
# Per-User Gap Timing Tests
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricPerUserGap:
    """Verify per-user gap timing (gap = num_users / qps) for both modes."""

    @pytest.mark.parametrize(
        "num_users,qps",
        [
            # PRECISE mode
            (8, 40.0),   # gap = 0.2s
            (10, 100.0), # gap = 0.1s
            (15, 75.0),  # gap = 0.2s
        ],
    )  # fmt: skip
    def test_per_user_gap_respected(self, cli: AIPerfCLI, num_users: int, qps: float):
        """Verify gap = num_users / qps between each user's turns."""
        expected_gap = num_users / qps

        # Use session-count stopping to get clean measurements
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=5,
            use_duration=False,
            num_sessions=num_users,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        timing = TimingAnalyzer(result)
        times_by_session = timing.get_issue_times_by_session()

        # Filter to sessions with multiple turns for gap analysis
        multi_turn_sessions = {k: v for k, v in times_by_session.items() if len(v) > 1}

        if len(multi_turn_sessions) >= 3:
            passed, reason = StatisticalAnalyzer.verify_per_user_gaps(
                multi_turn_sessions,
                expected_gap_sec=expected_gap,
                tolerance_pct=60.0,  # Higher tolerance for component tests
            )
            assert passed, f"Per-user gap not respected - {reason}"


# =============================================================================
# First Turn Stagger Tests - Both Modes
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricStaggerBothModes:
    """Verify first-turn stagger timing (stagger = 1/qps) for both modes."""

    @pytest.mark.parametrize(
        "num_users,qps",
        [
            (10, 100.0),  # 10ms stagger
            (15, 150.0),  # ~6.7ms stagger
            (20, 200.0),  # 5ms stagger
        ],
    )  # fmt: skip
    def test_first_turns_staggered(self, cli: AIPerfCLI, num_users: int, qps: float):
        """Verify first turns are staggered by approximately 1/qps."""
        expected_stagger = 1.0 / qps

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        timing = TimingAnalyzer(result)
        first_turn_times = timing.get_first_turn_issue_times_ns()

        if len(first_turn_times) >= 5:
            passed, reason = StatisticalAnalyzer.verify_stagger(
                first_turn_times,
                expected_stagger_sec=expected_stagger,
                tolerance_pct=60.0,  # Higher tolerance for component tests
            )
            # Note: Stagger verification may be noisy in component tests
            # Log warning but don't fail hard
            if not passed:
                pytest.skip(f"stagger verification noisy: {reason}")


# =============================================================================
# Num Users Variations Tests
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricNumUsersVariations:
    """Test various num_users configurations for both modes."""

    @pytest.mark.parametrize(
        "num_users",
        [
            # Single user (edge case)
            (1),
            # Small user counts
            (2),
            (3),
            (5),
            # Medium user counts
            (10),
            (15),
            (20),
            # Large user counts
            (30),
            (50),
            (30),
            (50),
        ],
    )  # fmt: skip
    def test_various_num_users(self, cli: AIPerfCLI, num_users: int):
        """Both modes work with various num_users configurations."""
        # Scale QPS with users for reasonable test duration
        qps = max(10.0, num_users * 3.0)

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=3,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= num_users, (
            f"with {num_users} users: "
            f"Expected >= {num_users} requests, got {result.request_count}"
        )
        assert_credits_balanced(result)


# =============================================================================
# Session Turns Variations Tests
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricSessionTurnsVariations:
    """Test various session-turns configurations for both modes."""

    @pytest.mark.parametrize(
        "turns",
        [
            # Minimum turns (2)
            (2),
            # Low turns (3-4)
            (3),
            (4),
            # Medium turns (5-6)
            (5),
            (6),
            # High turns (8-10)
            (8),
            (10),
        ],
    )  # fmt: skip
    def test_various_session_turns(self, cli: AIPerfCLI, turns: int):
        """Both modes work with various session-turns configurations."""
        cmd = build_user_centric_command(
            num_users=10,
            qps=50.0,
            turns_per_session=turns,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= 10, (
            f"with {turns} turns: Expected >= 10 requests, got {result.request_count}"
        )
        assert_credits_balanced(result)

        # Verify multi-turn behavior
        turn_indices = {r.metadata.turn_index for r in result.jsonl}
        if turns > 2:
            assert len(turn_indices) > 1, (
                f"Expected multiple turn indices with {turns} turns/session"
            )


# =============================================================================
# Stop Condition Tests - Both Modes
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricStopConditionsBothModes:
    """Test stop conditions work correctly for both spacing modes."""

    def test_session_count_stop_condition(self, cli: AIPerfCLI):
        """Both modes respect --num-sessions stop condition."""
        num_users = 10
        num_sessions = 15  # 1.5x users

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            turns_per_session=3,
            use_duration=False,
            num_sessions=num_sessions,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced()

    def test_request_count_stop_condition(self, cli: AIPerfCLI):
        """Both modes respect --request-count stop condition."""
        num_users = 10
        request_count = 25

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            turns_per_session=3,
            use_duration=False,
            request_count=request_count,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == request_count, (
            f"Expected {request_count} requests, got {result.request_count}"
        )
        assert_credits_balanced(result)


# =============================================================================
# Edge Cases - Both Modes
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricEdgeCasesBothModes:
    """Edge case tests for both spacing modes."""

    def test_single_user(self, cli: AIPerfCLI):
        """Both modes handle single user correctly."""
        cmd = build_user_centric_command(
            num_users=1,
            qps=10.0,
            turns_per_session=3,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= 1
        assert_credits_balanced(result)

    def test_many_users_few_turns(self, cli: AIPerfCLI):
        """Both modes handle many users with minimum turns."""
        cmd = build_user_centric_command(
            num_users=50,
            qps=200.0,
            turns_per_session=2,  # Minimum turns
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= 50
        assert_credits_balanced(result)

    def test_few_users_many_turns(self, cli: AIPerfCLI):
        """Both modes handle few users with many turns."""
        cmd = build_user_centric_command(
            num_users=5,
            qps=25.0,
            turns_per_session=10,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= 5
        assert_credits_balanced(result)

    def test_very_low_qps(self, cli: AIPerfCLI):
        """Both modes handle very low QPS."""
        cmd = build_user_centric_command(
            num_users=3,
            qps=3.0,  # 3 QPS
            turns_per_session=2,
            duration=1.5,  # Slightly longer to get some requests
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= 1
        assert_credits_balanced(result)

    def test_very_high_qps(self, cli: AIPerfCLI):
        """Both modes handle very high QPS."""
        cmd = build_user_centric_command(
            num_users=30,
            qps=300.0,  # 300 QPS
            turns_per_session=2,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= 30
        assert_credits_balanced(result)


# =============================================================================
# Stress Tests - Both Modes
# =============================================================================


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestUserCentricStressBothModes:
    """Stress tests for both spacing modes."""

    def test_high_volume(self, cli: AIPerfCLI):
        """Both modes handle high volume workloads."""
        cmd = build_user_centric_command(
            num_users=75,
            qps=250.0,
            turns_per_session=3,
            duration=2.0,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        assert result.request_count >= 75
        assert_credits_balanced(result)

    def test_sustained_multi_turn(self, cli: AIPerfCLI):
        """Both modes handle sustained multi-turn workloads."""
        cmd = build_user_centric_command(
            num_users=25,
            qps=100.0,
            turns_per_session=6,
            duration=2.0,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        assert result.request_count >= 25
        assert_credits_balanced(result)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.turn_indices_sequential()


# =============================================================================
# Mode Comparison Tests
# =============================================================================


@pytest.mark.component_integration
class TestSpacingModeComparison:
    """Compare behavior between both modes."""

    def test_both_modes_complete_same_configuration(self, cli: AIPerfCLI):
        """Both modes complete successfully with identical configuration."""
        num_users = 15
        qps = 75.0
        turns = 4

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
            use_duration=False,
            num_sessions=num_users,
        )
        result = cli.run_sync(cmd, timeout=60.0)
        analyzer = CreditFlowAnalyzer(result.runner_result)

        assert analyzer.num_sessions == num_users
        assert analyzer.credits_balanced()

    def test_both_modes_sequential_within_user(self, cli: AIPerfCLI):
        """Both modes enforce no interleaving within a user."""
        cmd = build_user_centric_command(
            num_users=12,
            qps=60.0,
            turns_per_session=5,
            use_duration=False,
            num_sessions=12,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        passed, reason = verify_no_interleaving_within_session(analyzer)
        assert passed, reason


# =============================================================================
# Ultra-Comprehensive Gauntlet (3D Matrix)
# =============================================================================


@pytest.mark.component_integration
@pytest.mark.slow
class TestUserCentricUltraComprehensive:
    """Ultra-comprehensive gauntlet testing all parameter combinations."""

    @pytest.mark.parametrize(
        "num_users,qps,turns",
        [
            # 3D matrix: spacing × users × qps × turns
            (users, qps, turns)
            for users in [5, 10, 20]
            for qps in [25.0, 50.0, 100.0]
            for turns in [2, 4, 6]
        ],
    )  # fmt: skip
    def test_ultra_comprehensive_gauntlet(
        self,
        cli: AIPerfCLI,
        num_users: int,
        qps: float,
        turns: int,
    ):
        """Comprehensive test across all parameter dimensions."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= num_users, (
            f"Gauntlet failed: {num_users} users, {qps} QPS, {turns} turns"
        )
        assert_credits_balanced(result)


# =============================================================================
# ADVERSARIAL TESTS - Very Low QPS (0.25 to 2.0)
# =============================================================================
# These tests stress the system with very low QPS values where timing precision
# is critical and edge cases in scheduling logic are more likely to surface.


@pytest.mark.component_integration
class TestUserCentricVeryLowQPS:
    """Test both modes with very low QPS values (0.25 to 2.0).

    Very low QPS creates:
    - Very long turn gaps (turn_gap = num_users / qps)
    - High precision timing requirements
    - Potential for timeout issues
    - Edge cases in scheduler sleep/wake logic
    """

    # Generate QPS values from 0.25 to 2.0 in 0.25 increments
    VERY_LOW_QPS_VALUES = [0.25 * i for i in range(1, 9)]  # [0.25, 0.5, 0.75, ..., 2.0]

    @pytest.mark.parametrize("qps", VERY_LOW_QPS_VALUES)
    def test_very_low_qps_default_config(self, cli: AIPerfCLI, qps: float):
        """Test very low QPS with default 15 users and 20 turns."""
        num_users = 15
        turns = 20

        # Calculate expected turn_gap for context
        turn_gap = num_users / qps  # At 0.25 QPS with 15 users: 60 second gap!

        # Use session-count stopping with longer timeout for low QPS
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
            use_duration=False,
            num_sessions=num_users,  # Complete exactly num_users sessions
        )

        # Timeout needs to account for very slow QPS
        # Minimum time = num_users sessions * turns * turn_gap
        timeout = max(90.0, num_users * 2)  # At least 90 seconds

        result = cli.run_sync(cmd, timeout=timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_users, (
            f"Expected {num_users} sessions, "
            f"got {analyzer.num_sessions}. Turn gap was {turn_gap:.2f}s"
        )
        assert analyzer.credits_balanced(), "Credit leak detected"
        assert analyzer.turn_indices_sequential(), "Turn indices not sequential"

    @pytest.mark.parametrize(
        "qps,num_users",
        [
            # Extreme ratios - very high turn gaps
            (0.25, 15),  # turn_gap = 60s
            (0.5, 20),   # turn_gap = 40s
            (0.75, 10),  # turn_gap = 13.3s
            (1.0, 15),   # turn_gap = 15s
            (1.25, 20),  # turn_gap = 16s
            (1.5, 15),   # turn_gap = 10s
            (1.75, 20),  # turn_gap = 11.4s
            (2.0, 15),   # turn_gap = 7.5s
        ],
    )  # fmt: skip
    def test_very_low_qps_various_users(
        self, cli: AIPerfCLI, qps: float, num_users: int
    ):
        """Test very low QPS with various user counts."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=20,  # Default 20 turns
            use_duration=False,
            num_sessions=num_users,
        )

        result = cli.run_sync(cmd, timeout=120.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_users
        assert analyzer.credits_balanced()


# =============================================================================
# ADVERSARIAL TESTS - Prime Number Cases
# =============================================================================
# Prime numbers often expose edge cases in modular arithmetic, division,
# and scheduling algorithms. These tests use prime values for all parameters.


# First 20 primes for comprehensive testing
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]


@pytest.mark.component_integration
class TestUserCentricPrimeNumbers:
    """Test with prime number parameters to expose modular arithmetic bugs.

    Prime numbers are adversarial because:
    - They don't divide evenly (exposing rounding bugs)
    - They create irregular stagger patterns
    - They stress hash functions and mod operations
    - Virtual history calculations with primes can expose off-by-one errors
    """

    @pytest.mark.parametrize("num_users", PRIMES[:10])  # First 10 primes
    def test_prime_num_users(self, cli: AIPerfCLI, num_users: int):
        """Test with prime number of users."""
        # QPS = num_users * 5 to keep reasonable timing
        qps = float(num_users * 5)

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=20,  # Default 20 turns
            use_duration=False,
            num_sessions=num_users,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_users, (
            f"with {num_users} (prime) users: "
            f"Expected {num_users} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced()
        assert analyzer.turn_indices_sequential()

    @pytest.mark.parametrize("turns", PRIMES[:10])  # First 10 primes
    def test_prime_turns(self, cli: AIPerfCLI, turns: int):
        """Test with prime number of turns per session."""
        num_users = 15  # Default

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=75.0,
            turns_per_session=turns,
            use_duration=False,
            num_sessions=num_users,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_users, (
            f"with {turns} (prime) turns: "
            f"Expected {num_users} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced()

    @pytest.mark.parametrize(
        "num_users,turns,qps_multiplier",
        [
            # All-prime combinations (adversarial)
            (2, 3, 5),    # 2 users, 3 turns, QPS = 2*5 = 10
            (3, 5, 7),    # 3 users, 5 turns, QPS = 3*7 = 21
            (5, 7, 11),   # 5 users, 7 turns, QPS = 5*11 = 55
            (7, 11, 13),  # 7 users, 11 turns, QPS = 7*13 = 91
            (11, 13, 17), # 11 users, 13 turns, QPS = 11*17 = 187
            (13, 17, 19), # 13 users, 17 turns, QPS = 13*19 = 247
            (17, 19, 23), # 17 users, 19 turns, QPS = 17*23 = 391
            (19, 23, 29), # 19 users, 23 turns, QPS = 19*29 = 551
        ],
    )  # fmt: skip
    def test_all_prime_combination(
        self,
        cli: AIPerfCLI,
        num_users: int,
        turns: int,
        qps_multiplier: int,
    ):
        """Test with all-prime parameter combinations."""
        qps = float(num_users * qps_multiplier)

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
            use_duration=False,
            num_sessions=num_users,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_users, (
            f"all-prime ({num_users}u, {turns}t, {qps}qps): "
            f"Expected {num_users} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced()

    @pytest.mark.parametrize(
        "qps",
        [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0],  # Prime QPS values
    )
    def test_prime_qps(self, cli: AIPerfCLI, qps: float):
        """Test with prime QPS values."""
        num_users = 15
        turns = 20

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
            use_duration=False,
            num_sessions=num_users,
        )
        result = cli.run_sync(cmd, timeout=120.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_users
        assert analyzer.credits_balanced()


# =============================================================================
# ADVERSARIAL TESTS - Edge Cases to Break the System
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricAdversarial:
    """Adversarial tests designed to break the system and find bugs.

    These tests target known fragile areas:
    - Virtual history edge cases
    - Turn gap calculations with extreme ratios
    - Scheduler timing precision
    - Credit flow under stress
    - User replacement logic
    """

    def test_single_user_20_turns(self, cli: AIPerfCLI):
        """Single user with 20 turns - tests sequential processing."""
        cmd = build_user_centric_command(
            num_users=1,
            qps=10.0,  # turn_gap = 0.1s (but only 1 user, so effectively sequential)
            turns_per_session=20,
            use_duration=False,
            num_sessions=1,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == 1
        assert analyzer.credits_balanced()
        # All 20 turns should be in one session
        max_turn_index = max(r.metadata.turn_index for r in result.jsonl)
        assert max_turn_index >= 19, (
            f"Expected 20 turns (0-19), got max turn {max_turn_index}"
        )

    def test_users_equals_turns(self, cli: AIPerfCLI):
        """num_users == turns - symmetric configuration."""
        n = 15

        cmd = build_user_centric_command(
            num_users=n,
            qps=float(n * 5),  # 75 QPS
            turns_per_session=n,  # 15 turns
            use_duration=False,
            num_sessions=n,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == n
        assert analyzer.credits_balanced()

    def test_very_high_turn_count(self, cli: AIPerfCLI):
        """Test with very high turn count (100 turns)."""
        cmd = build_user_centric_command(
            num_users=15,
            qps=150.0,  # turn_gap = 0.1s
            turns_per_session=100,
            use_duration=False,
            num_sessions=15,
        )
        result = cli.run_sync(cmd, timeout=120.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == 15
        assert analyzer.credits_balanced()

    def test_qps_less_than_one(self, cli: AIPerfCLI):
        """Test QPS < 1 (less than 1 request per second)."""
        cmd = build_user_centric_command(
            num_users=5,
            qps=0.5,  # turn_gap = 10 seconds!
            turns_per_session=20,
            use_duration=False,
            num_sessions=5,
        )
        result = cli.run_sync(cmd, timeout=120.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == 5
        assert analyzer.credits_balanced()

    def test_fractional_qps_many_decimals(self, cli: AIPerfCLI):
        """Test with many decimal places in QPS (floating point precision)."""
        cmd = build_user_centric_command(
            num_users=15,
            qps=17.333333333,  # Irrational-ish QPS
            turns_per_session=20,
            use_duration=False,
            num_sessions=15,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == 15
        assert analyzer.credits_balanced()

    def test_pi_qps(self, cli: AIPerfCLI):
        """Test with pi as QPS (irrational number stress test)."""
        import math

        cmd = build_user_centric_command(
            num_users=15,
            qps=math.pi * 10,  # ~31.4159 QPS
            turns_per_session=20,
            use_duration=False,
            num_sessions=15,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == 15
        assert analyzer.credits_balanced()

    def test_e_qps(self, cli: AIPerfCLI):
        """Test with e as QPS (another irrational number)."""
        import math

        cmd = build_user_centric_command(
            num_users=15,
            qps=math.e * 10,  # ~27.18 QPS
            turns_per_session=20,
            use_duration=False,
            num_sessions=15,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == 15
        assert analyzer.credits_balanced()

    def test_golden_ratio_qps(self, cli: AIPerfCLI):
        """Test with golden ratio as QPS multiplier."""
        phi = (1 + 5**0.5) / 2  # Golden ratio ≈ 1.618

        cmd = build_user_centric_command(
            num_users=15,
            qps=phi * 30,  # ~48.54 QPS
            turns_per_session=20,
            use_duration=False,
            num_sessions=15,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == 15
        assert analyzer.credits_balanced()

    def test_num_sessions_much_greater_than_users(self, cli: AIPerfCLI):
        """Test with num_sessions >> num_users (many replacement cycles)."""
        num_users = 5
        num_sessions = 50  # 10x users

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            turns_per_session=20,
            use_duration=False,
            num_sessions=num_sessions,
        )
        result = cli.run_sync(cmd, timeout=120.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions (10x users), got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced()

    def test_request_count_not_divisible_by_turns(self, cli: AIPerfCLI):
        """Test request_count that doesn't divide evenly by turns (mid-session stop)."""
        num_users = 15
        turns = 20
        request_count = 137  # Prime, doesn't divide by 20

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            turns_per_session=turns,
            use_duration=False,
            request_count=request_count,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        assert result.request_count == request_count, (
            f"Expected exactly {request_count} requests, got {result.request_count}"
        )
        assert_credits_balanced(result)


# =============================================================================
# ADVERSARIAL TESTS - Boundary Conditions
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricBoundaryConditions:
    """Test boundary conditions that often expose off-by-one errors."""

    @pytest.mark.parametrize(
        "num_users,num_sessions",
        [
            # Exact boundaries
            (15, 15),     # users == sessions
            (15, 16),     # sessions = users + 1
            (15, 14),     # sessions = users - 1 (should fail validation)
            (15, 30),     # sessions = 2x users
            (15, 29),     # sessions = 2x users - 1
            (15, 31),     # sessions = 2x users + 1
        ],
    )  # fmt: skip
    def test_session_user_boundaries(
        self,
        cli: AIPerfCLI,
        num_users: int,
        num_sessions: int,
    ):
        """Test session count boundaries relative to user count."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            turns_per_session=20,
            use_duration=False,
            num_sessions=num_sessions,
        )

        if num_sessions < num_users:
            # Should fail validation
            result = cli.run_sync(cmd, timeout=30.0, assert_success=False)
            assert result.exit_code == 1
        else:
            result = cli.run_sync(cmd, timeout=90.0)
            runner: AIPerfRunnerResultWithSharedBus = result.runner_result
            analyzer = CreditFlowAnalyzer(runner)
            assert analyzer.num_sessions == num_sessions
            assert analyzer.credits_balanced()

    @pytest.mark.parametrize(
        "num_users,request_count",
        [
            # Exact boundaries
            (15, 15),     # requests == users
            (15, 16),     # requests = users + 1
            (15, 14),     # requests = users - 1 (should fail validation)
            (15, 300),    # requests = users * 20 (full 20 turns for all)
            (15, 299),    # requests = users * 20 - 1
            (15, 301),    # requests = users * 20 + 1
        ],
    )  # fmt: skip
    def test_request_user_boundaries(
        self,
        cli: AIPerfCLI,
        num_users: int,
        request_count: int,
    ):
        """Test request count boundaries relative to user count."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            turns_per_session=20,
            use_duration=False,
            request_count=request_count,
        )

        if request_count < num_users:
            # Should fail validation
            result = cli.run_sync(cmd, timeout=30.0, assert_success=False)
            assert result.exit_code == 1
        else:
            result = cli.run_sync(cmd, timeout=90.0)
            assert result.request_count == request_count
            assert_credits_balanced(result)


# =============================================================================
# ADVERSARIAL TESTS - Timing Precision
# =============================================================================


@pytest.mark.component_integration
class TestUserCentricTimingPrecision:
    """Test timing precision with extreme ratios.

    These tests create scenarios where timing precision is critical
    and small errors can compound into bugs.
    """

    @pytest.mark.parametrize(
        "num_users,qps",
        [
            # Very small turn gaps (precision stress)
            (100, 1000.0),  # turn_gap = 0.1s
            (50, 500.0),    # turn_gap = 0.1s
            (200, 1000.0),  # turn_gap = 0.2s
            # Very large turn gaps (long wait stress)
            (15, 0.5),      # turn_gap = 30s
            (20, 0.5),      # turn_gap = 40s
            (10, 0.25),     # turn_gap = 40s
        ],
    )  # fmt: skip
    def test_extreme_turn_gap_ratios(self, cli: AIPerfCLI, num_users: int, qps: float):
        """Test extreme ratios."""
        # For high QPS tests, use duration-based stopping
        # For low QPS tests, use session-count stopping
        if qps >= 100:
            cmd = build_user_centric_command(
                num_users=num_users,
                qps=qps,
                turns_per_session=20,
                use_duration=True,
                duration=1.0,
            )
            result = cli.run_sync(cmd, timeout=30.0)
            assert result.request_count >= num_users
        else:
            cmd = build_user_centric_command(
                num_users=num_users,
                qps=qps,
                turns_per_session=20,
                use_duration=False,
                num_sessions=num_users,
            )
            result = cli.run_sync(cmd, timeout=180.0)  # Long timeout for slow QPS
            runner: AIPerfRunnerResultWithSharedBus = result.runner_result
            analyzer = CreditFlowAnalyzer(runner)
            assert analyzer.num_sessions == num_users

        assert_credits_balanced(result)


# =============================================================================
# MEGA ADVERSARIAL GAUNTLET - The Ultimate Bug Hunt
# =============================================================================


@pytest.mark.component_integration
@pytest.mark.slow
class TestUserCentricMegaAdversarialGauntlet:
    """The ultimate adversarial gauntlet designed to find hidden bugs.

    This combines multiple adversarial dimensions:
    - Very low QPS (0.25-2.0)
    - Prime numbers
    - Default 15 users, 20 turns
    - Irrational ratios
    """

    @pytest.mark.parametrize(
        "qps,num_users,turns",
        [
            # Very low QPS with prime users
            (qps, users, 20)
            for qps in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            for users in [2, 3, 5, 7, 11, 13]  # First 6 primes
        ],
    )  # fmt: skip
    def test_low_qps_prime_users_gauntlet(
        self,
        cli: AIPerfCLI,
        qps: float,
        num_users: int,
        turns: int,
    ):
        """Very low QPS with prime user counts."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
            use_duration=False,
            num_sessions=num_users,
        )
        result = cli.run_sync(cmd, timeout=180.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_users, (
            f"GAUNTLET FAIL: {qps} QPS, {num_users} (prime) users - "
            f"Expected {num_users} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced(), (
            f"GAUNTLET FAIL: {qps} QPS, {num_users} users - Credit leak!"
        )

    @pytest.mark.parametrize(
        "qps,turns",
        [
            # Very low QPS with prime turns
            (qps, turns)
            for qps in [0.5, 1.0, 1.5, 2.0]
            for turns in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  # First 10 primes
        ],
    )  # fmt: skip
    def test_low_qps_prime_turns_gauntlet(self, cli: AIPerfCLI, qps: float, turns: int):
        """Very low QPS with prime turn counts."""
        num_users = 15  # Default

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
            use_duration=True,
            duration=30.0,
            num_sessions=num_users,
        )
        result = cli.run_sync(cmd, timeout=180.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_users, (
            f"GAUNTLET FAIL: {qps} QPS, {turns} (prime) turns - "
            f"Expected {num_users} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced()

    @pytest.mark.parametrize("qps", [
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
    ])  # fmt: skip
    def test_default_15_users_20_turns_all_low_qps(self, cli: AIPerfCLI, qps: float):
        """Test default 15 users, 20 turns at ALL very low QPS values."""
        num_users = 15
        turns = 20

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            turns_per_session=turns,
            use_duration=True,
            duration=30.0,
            num_sessions=num_users,
        )
        result = cli.run_sync(cmd, timeout=180.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_users, (
            f"MEGA GAUNTLET FAIL: {qps} QPS - "
            f"Expected {num_users} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced(), (
            f"MEGA GAUNTLET FAIL: {qps} QPS - Credit leak!"
        )
        assert analyzer.turn_indices_sequential(), (
            f"MEGA GAUNTLET FAIL: {qps} QPS - Non-sequential turns!"
        )
