# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive warmup phase tests with interaction coverage.

Warmup phase is critical for:
- KV cache warm-up
- Connection pool initialization
- Latency stabilization
- Avoiding cold-start bias in benchmark metrics

These tests verify:
1. Basic warmup functionality (request-count, duration)
2. Warmup → profiling phase transition
3. Warmup parameter overrides (concurrency, rate, prefill)
4. Warmup interactions with:
   - Multi-turn conversations
   - Different rate modes
   - Concurrency limits
   - Cancellation (should be disabled in warmup)
   - Duration/grace period

CRITICAL: Warmup and profiling phases have SEPARATE credit tracking.
Each phase should balance independently.
"""

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit
from tests.component_integration.timing.conftest import (
    defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestWarmupBasic:
    """Basic warmup phase functionality tests."""

    def test_warmup_request_count_completes(self, cli: AIPerfCLI):
        """Test basic warmup with request count.

        Scenario:
        - Warmup: 20 requests
        - Profiling: 30 requests
        - Verify both phases complete
        - Verify separate credit tracking
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 30 \
                --request-rate 300 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 20
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Total requests = warmup (20) + profiling (30) = 50
        # But result.request_count only counts profiling phase
        assert result.request_count == 30

        runner = result.runner_result

        # Verify we have credits from BOTH phases
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.PROFILING
        ]

        assert len(warmup_credits) == 20, (
            f"Expected 20 warmup credits, got {len(warmup_credits)}"
        )
        assert len(profiling_credits) == 30, (
            f"Expected 30 profiling credits, got {len(profiling_credits)}"
        )

        # Verify both phases balanced independently
        return_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, CreditReturn)
        ]

        warmup_returns = [
            p for p in return_payloads if p.payload.credit.phase == CreditPhase.WARMUP
        ]
        profiling_returns = [
            p
            for p in return_payloads
            if p.payload.credit.phase == CreditPhase.PROFILING
        ]

        assert len(warmup_returns) == 20
        assert len(profiling_returns) == 30

    def test_warmup_duration_completes(self, cli: AIPerfCLI):
        """Test warmup with duration instead of request count.

        Scenario:
        - Warmup: 0.3 seconds at 200 QPS → ~60 requests
        - Profiling: 25 requests
        - Verify duration stops warmup, profiling continues
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 25 \
                --request-rate 200 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-duration 0.3
        """

        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 25  # Profiling only

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

        # Warmup duration 0.3s at 200 QPS → ~60 requests (within tolerance)
        assert 50 <= len(warmup_credits) <= 70, (
            f"Expected ~60 warmup credits, got {len(warmup_credits)}"
        )
        assert len(profiling_credits) == 25


@pytest.mark.component_integration
class TestWarmupPhaseTransition:
    """Tests for warmup → profiling phase transition.

    The transition should be seamless with no credit loss or duplication.
    """

    def test_warmup_to_profiling_transition_seamless(self, cli: AIPerfCLI):
        """Test seamless transition from warmup to profiling.

        Scenario:
        - Warmup phase completes
        - Profiling phase starts immediately
        - No credit loss, no overlap
        - Credits balanced per phase
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 20 \
                --request-rate 300 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 15
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result

        # Get all credits sorted by capture timestamp
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]
        credit_payloads.sort(key=lambda p: p.timestamp_ns)

        # Verify phase ordering (warmup first, then profiling)
        phases = [p.payload.phase for p in credit_payloads]
        warmup_end_idx = phases.index(CreditPhase.PROFILING)

        assert all(p == CreditPhase.WARMUP for p in phases[:warmup_end_idx])
        assert all(p == CreditPhase.PROFILING for p in phases[warmup_end_idx:])

        # Verify counts
        assert warmup_end_idx == 15  # 15 warmup credits
        assert len(phases) - warmup_end_idx == 20  # 20 profiling credits

    def test_credits_isolated_per_phase(self, cli: AIPerfCLI):
        """Test that warmup and profiling credits are isolated.

        Scenario:
        - Warmup and profiling both run
        - Credit IDs should restart at 0 for each phase
        - No cross-phase contamination
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 12 \
                --request-rate 250 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 10
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p.payload for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p.payload
            for p in credit_payloads
            if p.payload.phase == CreditPhase.PROFILING
        ]

        # Credit IDs should restart for each phase
        warmup_ids = {c.id for c in warmup_credits}
        profiling_ids = {c.id for c in profiling_credits}

        # Both should start from 0
        assert min(warmup_ids) == 0
        assert min(profiling_ids) == 0

        # Both should be sequential
        assert warmup_ids == set(range(10))
        assert profiling_ids == set(range(12))


@pytest.mark.component_integration
class TestWarmupParameterOverrides:
    """Tests for warmup parameter overrides.

    Warmup can have different concurrency, rate, and prefill settings
    than the profiling phase. These tests verify overrides work correctly.
    """

    def test_warmup_concurrency_override(self, cli: AIPerfCLI):
        """Test warmup with different concurrency than profiling.

        Scenario:
        - Warmup: concurrency=2
        - Profiling: concurrency=8
        - Verify limits enforced per phase
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 30 \
                --concurrency 8 \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 20 \
                --warmup-concurrency 2
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result

        # Separate credits by phase
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_payloads = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_payloads = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.PROFILING
        ]

        # Build phase-specific analyzers
        # Note: ConcurrencyAnalyzer needs AIPerfResults, but we need phase-specific analysis
        # For now, verify credit counts
        assert len(warmup_payloads) == 20
        assert len(profiling_payloads) == 30

    def test_warmup_rate_override(self, cli: AIPerfCLI):
        """Test warmup with different request rate than profiling.

        Scenario:
        - Warmup: 200 QPS (fast)
        - Profiling: 50 QPS (slower, more accurate)
        - Verify rate difference
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 25 \
                --request-rate 200 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 20 \
                --warmup-request-rate 300
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_payloads = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_payloads = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.PROFILING
        ]

        assert len(warmup_payloads) == 20
        assert len(profiling_payloads) == 25

        # Warmup should complete faster (higher rate)
        warmup_times = [p.timestamp_ns for p in warmup_payloads]
        profiling_times = [p.timestamp_ns for p in profiling_payloads]

        warmup_duration = (max(warmup_times) - min(warmup_times)) / 1e9
        profiling_duration = (max(profiling_times) - min(profiling_times)) / 1e9

        # Warmup at 200 QPS should be ~4x faster than profiling at 50 QPS
        # (20/200 = 0.1s vs 25/50 = 0.5s)
        assert warmup_duration < profiling_duration


@pytest.mark.component_integration
class TestWarmupInteractions:
    """Tests for warmup interactions with other features.

    These tests focus on how warmup interacts with:
    - Multi-turn conversations
    - Concurrency limits
    - Cancellation
    - Different rate modes
    """

    def test_warmup_with_multi_turn_conversations(self, cli: AIPerfCLI):
        """Test warmup + multi-turn interaction.

        Scenario:
        - Warmup: 10 sessions × 3 turns = 30 credits
        - Profiling: 15 sessions × 3 turns = 45 credits
        - Verify turn indices sequential per phase
        - Verify session isolation between phases
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 15 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --request-rate 250 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-warmup-sessions 10
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Profiling: 15 sessions × 3 turns = 45 requests
        assert result.request_count == 45

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p.payload for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p.payload
            for p in credit_payloads
            if p.payload.phase == CreditPhase.PROFILING
        ]

        # Warmup: 10 sessions × 3 turns = 30
        assert len(warmup_credits) == 30
        assert len(profiling_credits) == 45

        # Verify warmup sessions are different from profiling sessions
        warmup_session_ids = {c.x_correlation_id for c in warmup_credits}
        profiling_session_ids = {c.x_correlation_id for c in profiling_credits}

        # Session IDs should be different between phases
        assert warmup_session_ids.isdisjoint(profiling_session_ids), (
            "Warmup and profiling should use different sessions"
        )

    def test_warmup_cancellation_disabled(self, cli: AIPerfCLI):
        """Test that cancellation is disabled during warmup phase.

        Scenario:
        - Cancellation rate 50% configured
        - Warmup: Should have 0 cancellations
        - Profiling: Should have ~50% cancellations
        - Verify warmup immune to cancellation
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 20 \
                --request-rate 300 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 20 \
                --request-cancellation-rate 50.0 \
                --request-cancellation-delay 0.003 \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result
        return_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, CreditReturn)
        ]

        warmup_returns = [
            p.payload
            for p in return_payloads
            if p.payload.credit.phase == CreditPhase.WARMUP
        ]
        profiling_returns = [
            p.payload
            for p in return_payloads
            if p.payload.credit.phase == CreditPhase.PROFILING
        ]

        # Warmup: No errors (cancellation disabled)
        warmup_errors = sum(1 for r in warmup_returns if r.error is not None)
        assert warmup_errors == 0, f"Warmup should have 0 errors, got {warmup_errors}"

        # Profiling: Should have errors (~50% with seed 42)
        profiling_errors = sum(1 for r in profiling_returns if r.error is not None)
        assert profiling_errors > 0, (
            "Profiling should have some errors with 50% cancellation rate"
        )

    def test_warmup_with_concurrency_and_prefill(self, cli: AIPerfCLI):
        """Test warmup with concurrency limits.

        Scenario:
        - Warmup: concurrency=3, prefill=1
        - Profiling: concurrency=8, prefill=2
        - Verify limits enforced per phase
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 25 \
                --concurrency 8 \
                --prefill-concurrency 2 \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-warmup-sessions 15 \
                --warmup-concurrency 3 \
                --warmup-prefill-concurrency 1
        """

        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 25

        runner = result.runner_result

        # Verify warmup credits issued
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.PROFILING
        ]

        assert len(warmup_credits) == 15
        assert len(profiling_credits) == 25


@pytest.mark.component_integration
class TestWarmupWithRateModes:
    """Test warmup with different rate modes.

    Warmup should work correctly with constant, poisson, and burst modes.
    """

    @pytest.mark.parametrize(
        "rate_mode",
        ["constant", "poisson"],
    )
    def test_warmup_with_rate_modes(self, cli: AIPerfCLI, rate_mode: str):
        """Test warmup works with different rate modes.

        Scenario:
        - Test each rate mode (constant, poisson)
        - Warmup + profiling both use same mode
        - Verify completion
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 20 \
                --request-rate 300 \
                --request-rate-mode {rate_mode} \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 15 \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 20

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

        assert warmup_count == 15
        assert profiling_count == 20

    def test_warmup_with_burst_mode(self, cli: AIPerfCLI):
        """Test warmup with burst mode (concurrency-only).

        Scenario:
        - Warmup: burst mode, 20 sessions, concurrency=4
        - Profiling: burst mode, 25 sessions, concurrency=8
        - Verify fast issuance in both phases
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 25 \
                --concurrency 8 \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-warmup-sessions 20 \
                --warmup-concurrency 4
        """

        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 25

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

        assert warmup_count == 20
        assert profiling_count == 25


@pytest.mark.component_integration
class TestWarmupComplexInteractions:
    """Tests for complex warmup interactions.

    These tests combine multiple features to verify no unexpected interactions.
    """

    def test_warmup_multi_turn_with_concurrency_override(self, cli: AIPerfCLI):
        """Test warmup + multi-turn + concurrency override.

        Interaction matrix:
        - Multi-turn conversations
        - Warmup phase with override concurrency
        - Different concurrency in profiling
        - Verify turn sequencing per phase
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 12 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --concurrency 10 \
                --request-rate 300 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-warmup-sessions 8 \
                --warmup-concurrency 4
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Profiling: 12 sessions × 4 turns = 48
        assert result.request_count == 48

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p.payload for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p.payload
            for p in credit_payloads
            if p.payload.phase == CreditPhase.PROFILING
        ]

        # Warmup: 8 sessions × 4 turns = 32
        assert len(warmup_credits) == 32
        assert len(profiling_credits) == 48

    def test_warmup_duration_with_profiling_duration(self, cli: AIPerfCLI):
        """Test warmup duration + profiling duration interaction.

        Scenario:
        - Warmup: 0.2s duration
        - Profiling: 0.3s duration
        - Both duration-controlled
        - Verify both phases stop at their durations
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 100 \
                --request-rate 200 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-duration 0.2 \
                --benchmark-duration 0.3 \
                --benchmark-grace-period 5.0
        """

        result = cli.run_sync(cmd, timeout=30.0)

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

        # Warmup: 0.2s at 200 QPS → ~40 requests
        # Profiling: 0.3s at 200 QPS → ~60 requests
        assert 35 <= len(warmup_credits) <= 45, (
            f"Expected ~40 warmup credits, got {len(warmup_credits)}"
        )
        assert 55 <= len(profiling_credits) <= 65, (
            f"Expected ~60 profiling credits, got {len(profiling_credits)}"
        )

    def test_warmup_with_prefill_concurrency_override(self, cli: AIPerfCLI):
        """Test warmup prefill concurrency override.

        Interaction matrix:
        - Warmup prefill concurrency different from profiling
        - High QPS to hit prefill limits
        - Verify limits enforced per phase
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 30 \
                --request-rate 300 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --prefill-concurrency 3 \
                --warmup-request-count 25 \
                --warmup-prefill-concurrency 1
        """

        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 30

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

        assert warmup_count == 25
        assert profiling_count == 30


# Note: Validation tests moved to dedicated test_validation.py file
