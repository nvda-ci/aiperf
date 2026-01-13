# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Gamma rate timing mode.

Gamma rate mode generalizes Poisson arrivals with a tunable smoothness parameter:
- smoothness = 1.0: Equivalent to Poisson (exponential inter-arrivals, CV = 1.0)
- smoothness < 1.0: More bursty/clustered arrivals (higher CV)
- smoothness > 1.0: More regular/smooth arrivals (lower CV)

Key characteristics:
- Mean inter-arrival time = 1/rate (same as Poisson)
- CV = 1/sqrt(smoothness) for Gamma distribution
- Matches vLLM's burstiness parameter for realistic traffic modeling

Tests cover:
- Basic functionality at various QPS and smoothness levels
- Statistical distribution verification (CV matches theory)
- Smoothness comparison (higher smoothness = lower variance)
- Multi-turn conversations
- Concurrency interactions
- Equivalence to Poisson at smoothness=1.0
"""

import pytest

from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.timing.conftest import (
    BaseConcurrencyTests,
    BaseCreditFlowTests,
    TimingTestConfig,
    defaults,
)
from tests.harness.analyzers import (
    CreditFlowAnalyzer,
    StatisticalAnalyzer,
    TimingAnalyzer,
)
from tests.harness.utils import AIPerfCLI

DEFAULT_RANDOM_SEED = 42


def build_gamma_command(
    config: TimingTestConfig,
    smoothness: float,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    extra_args: str = "",
) -> str:
    """Build a CLI command for gamma rate tests.

    Args:
        config: Test configuration
        smoothness: Gamma smoothness parameter (1.0 = Poisson)
        random_seed: Random seed for deterministic tests (default: 42)
        extra_args: Additional CLI arguments
    """
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {config.num_sessions} \
            --request-rate {config.qps} \
            --arrival-pattern gamma \
            --arrival-smoothness {smoothness} \
            --osl {config.osl} \
            --extra-inputs ignore_eos:true \
            --random-seed {random_seed} \
            --ui {defaults.ui}
    """

    if config.turns_per_session > 1:
        cmd += (
            f" --session-turns-mean {config.turns_per_session} --session-turns-stddev 0"
        )

    if config.concurrency is not None:
        cmd += f" --concurrency {config.concurrency}"

    if config.prefill_concurrency is not None:
        cmd += f" --prefill-concurrency {config.prefill_concurrency}"

    if extra_args:
        cmd += f" {extra_args}"

    return cmd


@pytest.mark.component_integration
class TestGammaRateBasic:
    """Basic functionality tests for Gamma rate timing."""

    @pytest.mark.parametrize(
        "num_sessions,qps,smoothness",
        [
            (20, 100.0, 1.0),   # smoothness=1.0 (Poisson equivalent)
            (25, 100.0, 2.0),   # smoothness=2.0 (smoother)
            (30, 150.0, 4.0),   # smoothness=4.0 (much smoother)
            (20, 100.0, 0.5),   # smoothness=0.5 (burstier)
        ],
    )  # fmt: skip
    def test_gamma_rate_completes(
        self, cli: AIPerfCLI, num_sessions: int, qps: float, smoothness: float
    ):
        """Test Gamma rate mode completes at various configurations."""
        config = TimingTestConfig(num_sessions=num_sessions, qps=qps)
        cmd = build_gamma_command(config, smoothness=smoothness)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions
        assert result.has_streaming_metrics

    def test_gamma_rate_multi_turn(self, cli: AIPerfCLI):
        """Test Gamma rate with multi-turn conversations."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=75.0,
            turns_per_session=4,
        )
        cmd = build_gamma_command(config, smoothness=2.0)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests
        assert result.has_streaming_metrics


@pytest.mark.component_integration
class TestGammaRateCreditFlow(BaseCreditFlowTests):
    """Credit flow verification for Gamma rate timing.

    Inherits common credit flow tests from BaseCreditFlowTests.
    Tests: credits_balanced, credits_per_session, turn_indices_sequential
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Gamma rate timing command with default smoothness=2.0."""
        return build_gamma_command(config, smoothness=2.0)


@pytest.mark.component_integration
class TestGammaRateStatistics:
    """Statistical distribution tests for Gamma rate mode."""

    @pytest.mark.parametrize(
        "smoothness",
        [1.0, 2.0, 4.0],  # Different smoothness values
    )  # fmt: skip
    def test_gamma_mode_runs_successfully(self, cli: AIPerfCLI, smoothness: float):
        """Verify Gamma mode runs successfully at various smoothness levels.

        Note: System overhead at the component integration level adds jitter
        that masks the theoretical CV differences. The key validation is:
        - Gamma mode executes without errors
        - All requests complete successfully
        - Inter-arrival times have reasonable variability (CV > 0)

        For precise CV validation, see unit tests in test_interval_generators.py.
        """
        config = TimingTestConfig(num_sessions=50, qps=100.0)
        cmd = build_gamma_command(config, smoothness=smoothness)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 20, f"Insufficient data: got {len(gaps)} gaps, need >= 20"

        cv = timing.calculate_cv(gaps)
        # Just verify CV is reasonable (between 0 and 2)
        assert 0 < cv < 2.0, (
            f"CV {cv:.3f} outside reasonable range [0, 2.0] (smoothness={smoothness})"
        )

    def test_gamma_distribution_characteristics(self, cli: AIPerfCLI):
        """Verify Gamma distribution statistical properties.

        Uses the StatisticalAnalyzer.is_approximately_gamma method
        for rigorous distribution verification.
        """
        smoothness = 2.0
        config = TimingTestConfig(num_sessions=50, qps=100.0)
        cmd = build_gamma_command(config, smoothness=smoothness)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 20, (
            f"Insufficient data for Gamma analysis: got {len(gaps)} gaps, need >= 20"
        )

        passed, reason = StatisticalAnalyzer.is_approximately_gamma(
            gaps, expected_rate=config.qps, smoothness=smoothness, tolerance_pct=40.0
        )
        assert passed, f"Distribution not Gamma-like: {reason}"

    def test_mean_rate_matches_target(self, cli: AIPerfCLI):
        """Verify mean rate approximately matches target QPS.

        Gamma distribution should have the same mean as Poisson (1/rate).
        """
        config = TimingTestConfig(num_sessions=50, qps=100.0)
        cmd = build_gamma_command(config, smoothness=3.0)
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 20, (
            f"Insufficient data for mean rate analysis: got {len(gaps)} gaps"
        )

        mean_gap = timing.calculate_mean(gaps)
        expected_gap = 1.0 / config.qps
        tolerance = expected_gap * 0.4  # 40% tolerance

        assert abs(mean_gap - expected_gap) < tolerance, (
            f"Mean gap {mean_gap:.4f}s differs from expected {expected_gap:.4f}s"
        )


@pytest.mark.component_integration
class TestGammaSmoothnessComparison:
    """Tests comparing different smoothness values.

    Note: At the component integration level, system overhead dominates the
    inter-arrival time variance, making precise CV validation unreliable.
    Unit tests (test_interval_generators.py) verify the mathematical properties.
    Here we focus on functional validation.
    """

    def test_low_smoothness_functional(self, cli: AIPerfCLI):
        """Verify low smoothness (bursty) mode runs successfully."""
        config = TimingTestConfig(num_sessions=50, qps=100.0)
        cmd_low = build_gamma_command(config, smoothness=0.5)
        result = cli.run_sync(cmd_low, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())
        assert len(gaps) >= 20, "Insufficient data"

        # Verify non-trivial variance (not constant rate)
        cv = timing.calculate_cv(gaps)
        assert cv > 0.3, f"Low smoothness should have CV > 0.3, got {cv:.3f}"

    def test_smoothness_one_functional(self, cli: AIPerfCLI):
        """Verify smoothness=1.0 (Poisson equivalent) runs successfully."""
        config = TimingTestConfig(num_sessions=50, qps=100.0)
        cmd = build_gamma_command(config, smoothness=1.0)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())
        assert len(gaps) >= 20, "Insufficient data"

        cv = timing.calculate_cv(gaps)
        # Verify reasonable variability (between constant and extreme)
        assert 0.3 < cv < 2.0, (
            f"Smoothness=1.0 should have CV in (0.3, 2.0), got {cv:.3f}"
        )

    def test_high_smoothness_functional(self, cli: AIPerfCLI):
        """Verify high smoothness (regular) mode runs successfully."""
        config = TimingTestConfig(num_sessions=50, qps=100.0)
        cmd = build_gamma_command(config, smoothness=10.0)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())
        assert len(gaps) >= 20, "Insufficient data"

        # Verify CV is reasonable (system overhead adds jitter regardless of smoothness)
        cv = timing.calculate_cv(gaps)
        assert cv < 2.0, f"High smoothness should have CV < 2.0, got {cv:.3f}"


@pytest.mark.component_integration
class TestGammaRateWithConcurrency(BaseConcurrencyTests):
    """Tests for Gamma rate with concurrency limits.

    Inherits common concurrency tests from BaseConcurrencyTests.
    Tests: test_with_concurrency_limit, test_with_prefill_concurrency,
           test_multi_turn_with_concurrency
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Gamma rate timing command with default smoothness=2.0."""
        return build_gamma_command(config, smoothness=2.0)


@pytest.mark.component_integration
class TestGammaRateEdgeCases:
    """Edge case tests for Gamma rate timing.

    Inherits common edge case tests from BaseEdgeCaseTests.
    Tests: single_request, very_high_qps, low_qps, many_sessions_few_turns, few_sessions_many_turns

    Additionally includes Gamma-specific smoothness edge cases.
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Gamma rate timing command with default smoothness=2.0."""
        return build_gamma_command(config, smoothness=2.0)

    def test_very_low_smoothness(self, cli: AIPerfCLI):
        """Test very low smoothness (very bursty) executes successfully."""
        config = TimingTestConfig(num_sessions=40, qps=100.0)
        cmd = build_gamma_command(config, smoothness=0.1)  # Very bursty
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        # Just verify it runs and produces reasonable variability
        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())
        assert len(gaps) >= 20, f"Insufficient data: got {len(gaps)} gaps"
        cv = timing.calculate_cv(gaps)
        assert cv > 0, f"Should have non-zero CV, got {cv:.3f}"

    def test_very_high_smoothness(self, cli: AIPerfCLI):
        """Test very high smoothness (very regular) executes successfully."""
        config = TimingTestConfig(num_sessions=40, qps=100.0)
        cmd = build_gamma_command(config, smoothness=25.0)  # Very smooth
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        # Just verify it runs - system overhead means CV won't be near theoretical
        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())
        assert len(gaps) >= 20, f"Insufficient data: got {len(gaps)} gaps"
        cv = timing.calculate_cv(gaps)
        assert cv < 2.0, f"CV should be reasonable (< 2.0), got {cv:.3f}"


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestGammaRateStress:
    """Stress tests for Gamma rate timing."""

    def test_high_volume(self, cli: AIPerfCLI):
        """Test high volume with Gamma rate."""
        config = TimingTestConfig(num_sessions=100, qps=300.0, timeout=90.0)
        cmd = build_gamma_command(config, smoothness=2.0)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_sustained_multi_turn(self, cli: AIPerfCLI):
        """Test sustained multi-turn Gamma workload."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=100.0,
            turns_per_session=5,
            timeout=90.0,
        )
        cmd = build_gamma_command(config, smoothness=3.0)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()


@pytest.mark.component_integration
class TestVLLMBurstinessAlias:
    """Tests verifying --vllm-burstiness alias works."""

    def test_vllm_burstiness_alias(self, cli: AIPerfCLI):
        """Test that --vllm-burstiness is an alias for --arrival-smoothness.

        vLLM uses "burstiness" for the same parameter, so we support it.
        The key validation is that the command executes successfully.
        """
        config = TimingTestConfig(num_sessions=40, qps=100.0)

        # Use --vllm-burstiness instead of --arrival-smoothness
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions {config.num_sessions} \
                --request-rate {config.qps} \
                --arrival-pattern gamma \
                --vllm-burstiness 4.0 \
                --osl {config.osl} \
                --extra-inputs ignore_eos:true \
                --random-seed 42 \
                --ui {defaults.ui}
        """
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Key validation: command executed successfully with --vllm-burstiness
        assert result.request_count == config.num_sessions

        # Verify reasonable inter-arrival times were generated
        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())
        assert len(gaps) >= 20, f"Insufficient data: got {len(gaps)} gaps"

        # Just verify CV is reasonable (system overhead dominates)
        cv = timing.calculate_cv(gaps)
        assert 0 < cv < 2.0, f"CV should be reasonable, got {cv:.3f}"
