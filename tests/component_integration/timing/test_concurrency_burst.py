# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for concurrency burst timing mode.

Concurrency burst mode issues credits with zero delay - throughput is
controlled entirely by the concurrency semaphore.

Key characteristics:
- No rate limiting - credits issued as fast as concurrency allows
- Effective rate = concurrency / avg_response_time
- Requires concurrency to be set (no request-rate)
- Maximum throughput mode

Tests cover:
- Basic functionality at various concurrency levels
- Credit flow verification
- Timing behavior (minimal gaps expected)
- Multi-turn conversations
- Concurrency limit enforcement
- Edge cases
"""

import pytest

from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.timing.conftest import (
    BaseConcurrencyTests,
    TimingTestConfig,
    assert_concurrency_limit_hit,
    assert_concurrency_limit_respected,
    assert_request_count,
    defaults,
)
from tests.harness.analyzers import (
    CreditFlowAnalyzer,
    TimingAnalyzer,
)
from tests.harness.utils import AIPerfCLI

# Fast OSL for concurrency burst tests (10 for slightly longer decode phase)
TEST_OSL_BURST = 10


def build_burst_command(
    config: TimingTestConfig,
    *,
    extra_args: str = "",
    osl: int | None = None,
) -> str:
    """Build a CLI command for concurrency burst tests.

    Burst mode requires concurrency but no request-rate.
    """
    osl_value = osl if osl is not None else TEST_OSL_BURST
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {config.num_sessions} \
            --concurrency {config.concurrency} \
            --osl {osl_value} \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """

    if config.turns_per_session > 1:
        cmd += (
            f" --session-turns-mean {config.turns_per_session} --session-turns-stddev 0"
        )

    if config.prefill_concurrency is not None:
        cmd += f" --prefill-concurrency {config.prefill_concurrency}"

    if extra_args:
        cmd += f" {extra_args}"

    return cmd


@pytest.mark.component_integration
class TestConcurrencyBurstBasic:
    """Basic functionality tests for concurrency burst timing."""

    @pytest.mark.parametrize(
        "num_sessions,concurrency",
        [
            (15, 3),
            (25, 5),
            (40, 10),
            (60, 15),
        ],
    )
    def test_burst_mode_completes(
        self, cli: AIPerfCLI, num_sessions: int, concurrency: int
    ):
        """Test burst mode completes at various concurrency levels."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,  # No rate for burst mode
            concurrency=concurrency,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions
        assert result.has_streaming_metrics

    def test_burst_mode_multi_turn(self, cli: AIPerfCLI):
        """Test burst mode with multi-turn conversations."""
        config = TimingTestConfig(
            num_sessions=12,
            qps=0,
            turns_per_session=4,
            concurrency=6,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests
        assert result.has_streaming_metrics


@pytest.mark.component_integration
class TestConcurrencyBurstCreditFlow:
    """Credit flow verification for concurrency burst timing."""

    def test_credits_balanced(self, cli: AIPerfCLI):
        """Verify all credits sent are returned."""
        config = TimingTestConfig(
            num_sessions=30,
            qps=0,
            concurrency=8,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced(), (
            f"Credits not balanced: {analyzer.total_credits} sent, "
            f"{analyzer.total_returns} returned"
        )

    def test_credits_per_session(self, cli: AIPerfCLI):
        """Verify each session gets expected credits."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=0,
            turns_per_session=3,
            concurrency=5,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == config.num_sessions
        assert analyzer.session_credits_match(config.turns_per_session)

    @pytest.mark.slow
    def test_turn_indices_sequential(self, cli: AIPerfCLI):
        """Verify turn indices are sequential per session."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=0,
            turns_per_session=5,
            concurrency=4,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.turn_indices_sequential()


@pytest.mark.component_integration
class TestConcurrencyBurstTiming:
    """Timing behavior tests for concurrency burst mode."""

    def test_minimal_gaps(self, cli: AIPerfCLI):
        """Verify gaps are minimal in burst mode (no rate limiting)."""
        config = TimingTestConfig(
            num_sessions=25,
            qps=0,
            concurrency=10,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 5, (
            f"Insufficient data for gap analysis: got {len(gaps)} gaps, need >= 5. "
            f"25 sessions with concurrency=10 should produce enough data."
        )
        # Burst mode should have very small gaps (nearly 0)
        mean_gap = timing.calculate_mean(gaps)
        # Most gaps should be < 10ms in burst mode
        assert mean_gap < 0.1, f"Mean gap {mean_gap:.4f}s too high for burst mode"

    def test_fast_initial_burst(self, cli: AIPerfCLI):
        """Verify initial burst of credits happens quickly."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=0,
            concurrency=10,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()

        assert len(issue_times) >= 10, (
            f"Insufficient data for initial burst analysis: got {len(issue_times)} issue times, "
            f"need >= 10. 20 sessions with concurrency=10 should produce enough data."
        )
        # First 10 credits should be issued very quickly
        first_batch_duration = (issue_times[9] - issue_times[0]) / 1e9
        # Should be < 100ms for 10 credits in burst mode
        assert first_batch_duration < 0.2, (
            f"First 10 credits took {first_batch_duration:.3f}s, expected < 0.2s"
        )


@pytest.mark.component_integration
class TestConcurrencyBurstLimits(BaseConcurrencyTests):
    """Tests for concurrency limit enforcement in burst mode.

    Inherits common concurrency tests from BaseConcurrencyTests, with customization
    for burst mode (qps=0) behavior. Uses parametrized test for single values instead
    of (concurrency, qps) pairs since burst mode has qps=0.

    Tests: test_with_concurrency_limit (customized), test_with_prefill_concurrency,
           test_multi_turn_with_concurrency
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build burst mode command."""
        return build_burst_command(config)

    @pytest.mark.parametrize(
        "concurrency",
        [2, 4, 8, 12],
    )
    def test_with_concurrency_limit(self, cli: AIPerfCLI, concurrency: int):
        """Test burst mode respects and reaches concurrency limit.

        Override base class to use concurrency-only parameters (no QPS).
        Burst mode (qps=0) issues credits as fast as possible.
        """
        # Ensure we have enough sessions to hit the limit
        num_sessions = max(30, concurrency * 3)
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,  # Burst mode
            concurrency=concurrency,
        )

        # Validate test parameters will actually hit the limit
        assert config.will_hit_concurrency_limit(), (
            f"Test config won't hit concurrency limit: "
            f"num_sessions={num_sessions}, concurrency={concurrency}"
        )

        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.num_sessions)
        assert_concurrency_limit_respected(result, concurrency)
        assert_concurrency_limit_hit(result, concurrency)

    def test_with_prefill_concurrency(self, cli: AIPerfCLI):
        """Test burst mode with prefill concurrency limit.

        Override base class to ensure enough sessions for burst mode.
        """
        prefill_concurrency = 3
        # Ensure we have enough sessions to hit the prefill limit
        num_sessions = max(25, prefill_concurrency * 5)
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,  # Burst mode
            concurrency=10,
            prefill_concurrency=prefill_concurrency,
        )

        # Validate test parameters will hit the prefill limit
        assert config.will_hit_prefill_limit(), (
            f"Test config won't hit prefill limit: "
            f"num_sessions={num_sessions}, prefill_concurrency={prefill_concurrency}"
        )

        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.num_sessions)
        assert_concurrency_limit_respected(result, prefill_concurrency, prefill=True)
        assert_concurrency_limit_hit(result, prefill_concurrency, prefill=True)

    def test_multi_turn_with_concurrency(self, cli: AIPerfCLI):
        """Test multi-turn burst with concurrency.

        Override base class to use burst mode (qps=0).
        """
        config = TimingTestConfig(
            num_sessions=10,
            qps=0,  # Burst mode
            turns_per_session=4,
            concurrency=4,
        )

        assert config.will_hit_concurrency_limit()

        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.expected_requests)
        assert_concurrency_limit_hit(result, config.concurrency)

    def test_low_concurrency_high_sessions(self, cli: AIPerfCLI):
        """Test low concurrency with many sessions (queuing behavior)."""
        config = TimingTestConfig(
            num_sessions=40,
            qps=0,
            concurrency=2,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.num_sessions)


@pytest.mark.component_integration
class TestConcurrencyBurstEdgeCases:
    """Edge case tests for concurrency burst timing."""

    def test_single_concurrency(self, cli: AIPerfCLI):
        """Test with concurrency of 1 (sequential execution)."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=0,
            concurrency=1,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_concurrency_equals_sessions(self, cli: AIPerfCLI):
        """Test when concurrency equals number of sessions."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=0,
            concurrency=15,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_single_session_burst(self, cli: AIPerfCLI):
        """Test single session in burst mode (concurrency = 1)."""
        config = TimingTestConfig(
            num_sessions=1,
            qps=0,
            concurrency=1,  # Concurrency must be <= num_sessions
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 1

    def test_many_turns_burst(self, cli: AIPerfCLI):
        """Test many turns per session in burst mode."""
        config = TimingTestConfig(
            num_sessions=5,
            qps=0,
            turns_per_session=8,
            concurrency=3,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests


@pytest.mark.component_integration
@pytest.mark.stress
class TestConcurrencyBurstStress:
    """Stress tests for concurrency burst timing."""

    def test_high_volume_burst(self, cli: AIPerfCLI):
        """Test high volume in burst mode."""
        config = TimingTestConfig(
            num_sessions=100,
            qps=0,
            concurrency=20,
            timeout=90.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_sustained_burst_multi_turn(self, cli: AIPerfCLI):
        """Test sustained multi-turn burst workload."""
        config = TimingTestConfig(
            num_sessions=25,
            qps=0,
            turns_per_session=4,
            concurrency=10,
            timeout=90.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    def test_high_concurrency_burst(self, cli: AIPerfCLI):
        """Test with very high concurrency."""
        config = TimingTestConfig(
            num_sessions=50,
            qps=0,
            concurrency=50,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions
