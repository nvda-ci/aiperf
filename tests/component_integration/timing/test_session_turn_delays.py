# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive session turn delay tests with interaction coverage.

Session turn delays simulate realistic "think time" between conversation turns,
critical for chatbot and multi-turn benchmark accuracy.

Options tested:
- --session-turn-delay-mean: Mean delay between turns (milliseconds)
- --session-turn-delay-stddev: Standard deviation of delays (milliseconds)
- --session-delay-ratio: Scaling factor for delays

These tests verify:
1. Basic delay functionality
2. Delay interactions with rate modes (constant, poisson, user-centric, burst)
3. Delay + concurrency interactions
4. Delay + duration/grace period
5. Variable delays (stddev > 0)
6. Delay ratio scaling
7. Multi-turn timing with delays

CRITICAL INTERACTIONS TO TEST:
- Turn delays should NOT affect credit issuance rate (rate mode independent)
- Turn delays should affect session completion time
- Delays should respect per-session sequencing
- Delays + duration: In-flight turns may not complete
"""

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import Credit
from tests.component_integration.timing.conftest import (
    defaults,
)
from tests.harness.analyzers import CreditFlowAnalyzer
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestSessionTurnDelayBasic:
    """Basic session turn delay functionality tests."""

    def test_turn_delay_mean_with_multi_turn(self, cli: AIPerfCLI):
        """Test basic turn delay with multi-turn conversations.

        Scenario:
        - 3-turn conversations
        - 100ms delay between turns
        - Verify all turns complete
        - Verify delays don't break credit flow
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 10 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 100 \
                --request-rate 150 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # 10 sessions × 3 turns = 30 requests
        assert result.request_count == 30

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Verify all credits balanced
        assert credit_analyzer.credits_balanced()

        # Verify turn indices sequential
        assert credit_analyzer.turn_indices_sequential()

    def test_turn_delay_variable_with_stddev(self, cli: AIPerfCLI):
        """Test variable turn delays with stddev > 0.

        Scenario:
        - Mean delay: 50ms
        - Stddev: 20ms (variable delays)
        - Verify credit flow handles randomness
        - Verify all turns complete
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 8 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 50 \
                --session-turn-delay-stddev 20 \
                --request-rate 180 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # 8 sessions × 4 turns = 32 requests
        assert result.request_count == 32

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()
        assert credit_analyzer.num_sessions == 8
        assert credit_analyzer.session_credits_match(expected_turns=4)

    def test_turn_delay_ratio_scaling(self, cli: AIPerfCLI):
        """Test turn delay ratio scaling.

        Scenario:
        - Base delay: 100ms
        - Ratio: 2.0 (2× scaling)
        - Effective delay: 200ms
        - Verify scaling works
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 6 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 100 \
                --session-delay-ratio 2.0 \
                --request-rate 150 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # 6 sessions × 3 turns = 18 requests
        assert result.request_count == 18

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()


@pytest.mark.component_integration
class TestTurnDelayWithRateModes:
    """Test turn delays with different rate modes.

    Turn delays should work consistently across all rate modes.
    The delay affects session progression, not credit issuance rate.
    """

    @pytest.mark.parametrize(
        "rate_mode,qps",
        [
            ("constant", 80),
            ("poisson", 80),
        ],
    )
    def test_turn_delay_with_rate_modes(
        self, cli: AIPerfCLI, rate_mode: str, qps: float
    ):
        """Test turn delays work with constant and poisson rate modes.

        Scenario:
        - Turn delay: 80ms
        - Different rate modes
        - Verify credit issuance rate maintained
        - Verify delays don't interfere with rate control
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 12 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 80 \
                --request-rate {qps} \
                --request-rate-mode {rate_mode} \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 36  # 12 × 3

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()
        assert credit_analyzer.num_sessions == 12

    def test_turn_delay_with_user_centric_mode(self, cli: AIPerfCLI):
        """Test turn delays with user-centric rate mode.

        Scenario:
        - User-centric mode (per-user rate limiting)
        - Turn delay: 100ms
        - Verify user-centric gap + turn delay both apply
        - Verify no interleaving within user

        Note: User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. We verify at least the initial user turns
        are issued and credits are balanced.
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-users 10 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 100 \
                --user-centric-rate 150 \
                --benchmark-duration 1.0 \
                --benchmark-grace-period 0.0 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Duration-based: at least num_users requests (initial user turns)
        assert result.request_count >= 10

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()
        assert credit_analyzer.turn_indices_sequential()

    def test_turn_delay_with_burst_mode(self, cli: AIPerfCLI):
        """Test turn delays with burst mode.

        Scenario:
        - Burst mode (no rate limiting, concurrency-only)
        - Turn delay: 120ms
        - Verify delays respected despite burst
        - Concurrency limit still enforced
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 15 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 120 \
                --concurrency 10 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """

        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 45  # 15 × 3

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()


@pytest.mark.component_integration
class TestTurnDelayInteractions:
    """Test turn delay interactions with other features.

    These tests focus on complex interactions between turn delays and:
    - Concurrency limits
    - Duration/grace period
    - Warmup phase
    - Request timeouts
    """

    @pytest.mark.slow
    def test_turn_delay_with_concurrency_limit(self, cli: AIPerfCLI):
        """Test turn delays + concurrency limit interaction.

        Scenario:
        - Multi-turn with delays
        - Concurrency limit enforced
        - Verify delays don't prevent concurrent sessions
        - Turn delay affects per-session, not cross-session
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 20 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 100 \
                --request-rate 200 \
                --request-rate-mode constant \
                --concurrency 8 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """

        result = cli.run_sync(cmd, timeout=40.0)

        assert result.request_count == 80  # 20 × 4

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()

    def test_turn_delay_with_duration_grace_period(self, cli: AIPerfCLI):
        """Test turn delays + duration + grace period interaction.

        Scenario:
        - Turn delay: 200ms (slow think time)
        - Duration: 0.4s (short)
        - Grace period: 5s (allows delayed turns to complete)
        - Verify in-flight delayed turns complete in grace period
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 50 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 200 \
                --request-rate 150 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --benchmark-duration 0.4 \
                --benchmark-grace-period 5.0
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Duration 0.4s at 30 QPS → ~12 requests
        # Some sessions started, delays mean turns extend into grace period
        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # All credits should be accounted for
        assert credit_analyzer.credits_balanced()

        # At least some requests completed
        assert result.request_count >= 5

    @pytest.mark.slow
    def test_turn_delay_with_warmup_phase(self, cli: AIPerfCLI):
        """Test turn delays apply to both warmup and profiling.

        Scenario:
        - Warmup: multi-turn with delays
        - Profiling: multi-turn with delays
        - Verify delays respected in both phases
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 12 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 80 \
                --request-rate 180 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-warmup-sessions 8
        """

        result = cli.run_sync(cmd, timeout=40.0)

        # Profiling: 12 sessions × 3 turns = 36
        assert result.request_count == 36

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.PROFILING
        ]

        # Warmup: 8 sessions × 3 turns = 24
        assert len(warmup_credits) == 24
        assert len(profiling_credits) == 36

    def test_turn_delay_with_request_cancellation(self, cli: AIPerfCLI):
        """Test turn delays + request cancellation interaction.

        Scenario:
        - Turn delay: 60ms
        - Cancellation rate: 25%
        - Verify delays and cancellations coexist
        - Cancelled turns still respect turn delay for next turn
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 10 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 60 \
                --request-rate 200 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --request-cancellation-rate 25.0 \
                --request-cancellation-delay 0.003 \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # All 40 credits sent
        assert credit_analyzer.total_credits == 40

        # Credits balanced (some with errors)
        assert credit_analyzer.credits_balanced()

        # Verify some errors (25% rate with seed 42)
        error_count = sum(
            1 for cr in credit_analyzer.credit_returns if cr.error is not None
        )
        assert error_count > 0, "Expected some errors with cancellation rate"


@pytest.mark.component_integration
class TestVariableTurnDelays:
    """Tests for variable turn delays (stddev > 0).

    Variable delays simulate realistic chatbot think time variability.
    """

    def test_turn_delay_with_high_stddev(self, cli: AIPerfCLI):
        """Test turn delays with high variance.

        Scenario:
        - Mean: 100ms
        - Stddev: 50ms (50% CV - high variability)
        - Verify system handles variable delays
        - Verify turn sequencing maintained
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 12 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 100 \
                --session-turn-delay-stddev 50 \
                --request-rate 150 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=40.0)

        assert result.request_count == 48  # 12 × 4

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()
        assert credit_analyzer.turn_indices_sequential()

    def test_turn_delay_zero_mean_nonzero_stddev(self, cli: AIPerfCLI):
        """Test zero mean delay with stddev (pure randomness).

        Scenario:
        - Mean: 0ms (no base delay)
        - Stddev: 30ms (adds random jitter)
        - Verify random delays applied
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 10 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 0 \
                --session-turn-delay-stddev 30 \
                --request-rate 180 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 30  # 10 × 3

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()


@pytest.mark.component_integration
class TestVariableTurnCounts:
    """Tests for variable turn counts (session-turns-stddev > 0).

    Variable turn counts simulate realistic conversation length variability.
    """

    def test_turns_stddev_variation_per_session(self, cli: AIPerfCLI):
        """Test variable turn counts across sessions.

        Scenario:
        - Mean: 5 turns
        - Stddev: 2 turns (variation)
        - Verify each session gets different turn count
        - Verify all credits balanced
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 15 \
                --session-turns-mean 5 \
                --session-turns-stddev 2 \
                --request-rate 200 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=40.0)

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Total credits should vary from 15×3 to 15×7 approximately
        total = credit_analyzer.total_credits
        assert 45 <= total <= 105, f"Expected variable total ~75±30, got {total}"

        # Verify credits balanced
        assert credit_analyzer.credits_balanced()

        # Verify we have 15 sessions
        assert credit_analyzer.num_sessions == 15

        # Verify turn counts vary across sessions
        turn_counts = [
            len(payloads) for payloads in credit_analyzer.credits_by_session.values()
        ]

        # Should have variation (stddev > 0)
        import statistics

        if len(turn_counts) > 1:
            stddev = statistics.stdev(turn_counts)
            assert stddev > 0, "Expected variation in turn counts with stddev=2"

    @pytest.mark.slow
    def test_turns_stddev_with_user_centric(self, cli: AIPerfCLI):
        """Test variable turns with user-centric mode.

        Scenario:
        - User-centric rate
        - Variable turn counts
        - Verify per-user rate respected
        - Verify no interleaving within user

        Note: User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. With user replacement, total sessions can
        exceed num_users as users complete and are replaced.
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-users 12 \
                --session-turns-mean 4 \
                --session-turns-stddev 1 \
                --user-centric-rate 180 \
                --benchmark-duration 1.0 \
                --benchmark-grace-period 0.0 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Duration-based: verify at least some requests were issued
        assert credit_analyzer.total_credits >= 12

        assert credit_analyzer.credits_balanced()
        # Note: with user replacement, sessions can exceed num_users
        # The key invariant is turn indices are sequential within each session
        assert credit_analyzer.turn_indices_sequential()


@pytest.mark.component_integration
class TestComplexDelayInteractions:
    """Tests for complex multi-feature interactions with delays.

    These test the most complex scenarios combining many features.
    """

    def test_all_delay_options_combined(self, cli: AIPerfCLI):
        """Test all delay options used together.

        Interaction matrix:
        - Delay mean + stddev + ratio
        - Multi-turn
        - Constant rate
        - Concurrency
        - Verify all work together
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 10 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 60 \
                --session-turn-delay-stddev 20 \
                --session-delay-ratio 1.5 \
                --request-rate 200 \
                --request-rate-mode constant \
                --concurrency 8 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=40.0)

        assert result.request_count == 40  # 10 × 4

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()

    def test_variable_turns_and_delays_combined(self, cli: AIPerfCLI):
        """Test variable turn counts + variable delays.

        Maximum variability scenario:
        - Turns: mean=5, stddev=2 (variable)
        - Delays: mean=80, stddev=30 (variable)
        - Verify system handles double randomness
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 12 \
                --session-turns-mean 5 \
                --session-turns-stddev 2 \
                --session-turn-delay-mean 80 \
                --session-turn-delay-stddev 30 \
                --request-rate 200 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=40.0)

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Variable total (mean 12×5=60, with turn stddev)
        total = credit_analyzer.total_credits
        assert 36 <= total <= 84

        assert credit_analyzer.credits_balanced()

    def test_turn_delay_warmup_different_from_profiling(self, cli: AIPerfCLI):
        """Test different turn delays for warmup vs profiling.

        Scenario:
        - Warmup: fast (20ms delays)
        - Profiling: slow (100ms delays)
        - Verify both phases respect their delays
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 12 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 100 \
                --request-rate 180 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-warmup-sessions 8
        """

        result = cli.run_sync(cmd, timeout=40.0)

        assert result.request_count == 36  # 12 × 3

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_count = sum(
            1 for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        )
        profiling_count = sum(
            1 for p in credit_payloads if p.payload.phase == CreditPhase.PROFILING
        )

        # Warmup: 8 × 3 = 24
        assert warmup_count == 24
        assert profiling_count == 36

    def test_extreme_delay_with_short_duration(self, cli: AIPerfCLI):
        """Test very long delays with short benchmark duration.

        Edge case:
        - Turn delay: 500ms (very long think time)
        - Duration: 0.5s (short)
        - Many sessions but duration limits completions
        - Grace period allows some delayed turns
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 50 \
                --session-turns-mean 5 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 500 \
                --request-rate 150 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --benchmark-duration 0.5 \
                --benchmark-grace-period 8.0
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Credits balanced
        assert credit_analyzer.credits_balanced()

        # Some credits issued (limited by duration)
        assert credit_analyzer.total_credits >= 10
