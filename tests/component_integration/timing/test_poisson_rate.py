# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Poisson rate timing mode.

Poisson rate mode issues credits with exponentially distributed inter-arrival
times, simulating realistic traffic patterns with natural variability.

Key characteristics:
- Mean inter-arrival time = 1/rate
- Coefficient of variation (CV) ~ 1.0 for exponential distribution
- Natural variability around the target rate

Tests cover:
- Basic functionality at various QPS levels
- Statistical distribution verification
- Multi-turn conversations
- Concurrency interactions
- Comparison with constant rate (higher variability expected)
"""

import pytest

from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.timing.conftest import (
    BaseConcurrencyTests,
    BaseCreditFlowTests,
    TimingTestConfig,
    build_timing_command,
)
from tests.harness.analyzers import (
    CreditFlowAnalyzer,
    StatisticalAnalyzer,
    TimingAnalyzer,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestPoissonRateBasic:
    """Basic functionality tests for Poisson rate timing."""

    @pytest.mark.parametrize(
        "num_sessions,qps",
        [
            (15, 50.0),
            (25, 100.0),
            (35, 150.0),
            (50, 200.0),
        ],
    )
    def test_poisson_rate_completes(
        self, cli: AIPerfCLI, num_sessions: int, qps: float
    ):
        """Test Poisson rate mode completes at various QPS levels."""
        config = TimingTestConfig(num_sessions=num_sessions, qps=qps)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions
        assert result.has_streaming_metrics

    def test_poisson_rate_multi_turn(self, cli: AIPerfCLI):
        """Test Poisson rate with multi-turn conversations."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=75.0,
            turns_per_session=4,
        )
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests
        assert result.has_streaming_metrics


@pytest.mark.component_integration
class TestPoissonRateCreditFlow(BaseCreditFlowTests):
    """Credit flow verification for Poisson rate timing.

    Inherits common credit flow tests from BaseCreditFlowTests.
    Tests: credits_balanced, credits_per_session, turn_indices_sequential
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Poisson rate timing command."""
        return build_timing_command(config, arrival_pattern="poisson")


@pytest.mark.component_integration
class TestPoissonRateStatistics:
    """Statistical distribution tests for Poisson rate mode.

    Tests verify that the timing system produces inter-arrival times following
    an exponential distribution (Poisson process). Key properties tested:

    1. Mean ≈ 1/rate (correct average spacing)
    2. Std ≈ Mean (exponential property: σ = μ)
    3. CV ≈ 1.0 (coefficient of variation for exponential)
    4. CDF property: ~63.2% of values below mean
    5. Independence: consecutive intervals uncorrelated
    6. Index of dispersion ≈ 1.0 (variance/mean of event counts)
    """

    def test_poisson_distribution_comprehensive(self, cli: AIPerfCLI):
        """Comprehensive Poisson validation using multiple statistical tests.

        Runs 4 independent statistical tests and passes if at least 3 pass.
        This is more robust than single-test validation.
        """
        config = TimingTestConfig(num_sessions=60, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 30, (
            f"Insufficient data for comprehensive Poisson analysis: got {len(gaps)} gaps, "
            f"need >= 30. 60 sessions should produce enough data."
        )

        passed, summary, details = StatisticalAnalyzer.comprehensive_poisson_check(
            gaps, expected_rate=config.qps, tolerance_pct=30.0
        )

        assert passed, f"Comprehensive Poisson check failed: {summary}"

    def test_exponential_mean_std_cv(self, cli: AIPerfCLI):
        """Verify exponential distribution: Mean ≈ Std ≈ 1/rate, CV ≈ 1.0.

        For exponential distribution with rate λ:
        - Mean = 1/λ (expected inter-arrival time)
        - Standard deviation = 1/λ (same as mean!)
        - CV = σ/μ = 1.0

        This is the defining characteristic of exponential distribution.
        """
        config = TimingTestConfig(num_sessions=50, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())

        assert len(gaps) >= 20, (
            f"Insufficient data for mean/std/cv analysis: got {len(gaps)} gaps, need >= 20."
        )

        passed, reason = StatisticalAnalyzer.verify_exponential_mean_std_cv(
            gaps, expected_rate=config.qps, tolerance_pct=25.0
        )
        assert passed, f"Exponential mean/std/cv check failed: {reason}"

    def test_exponential_cdf_property(self, cli: AIPerfCLI):
        """Verify exponential CDF: ~63.2% of values are below the mean.

        This is a fundamental property of exponential distribution:
        P(X < μ) = 1 - e^(-1) ≈ 0.6321

        This test is rate-independent and only checks the distribution shape.
        """
        config = TimingTestConfig(num_sessions=60, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())

        assert len(gaps) >= 30, (
            f"Insufficient data for CDF property test: got {len(gaps)} gaps, need >= 30."
        )

        passed, reason = StatisticalAnalyzer.verify_exponential_cdf_property(gaps)
        assert passed, f"CDF property check failed: {reason}"

    def test_memoryless_property(self, cli: AIPerfCLI):
        """Verify memoryless property: consecutive intervals are independent.

        The Poisson process is memoryless - knowing one inter-arrival time tells
        you nothing about the next. This means consecutive intervals should have
        near-zero correlation.
        """
        config = TimingTestConfig(num_sessions=60, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())

        assert len(gaps) >= 30, (
            f"Insufficient data for independence test: got {len(gaps)} gaps, need >= 30."
        )

        passed, reason = StatisticalAnalyzer.verify_independence(gaps)
        assert passed, f"Independence (memoryless) check failed: {reason}"

    def test_higher_cv_than_constant(self, cli: AIPerfCLI):
        """Verify CV is higher than constant rate (more variability).

        Poisson process should have CV ~ 1.0 (exponential distribution).
        Constant rate should have CV < 0.3.
        """
        config = TimingTestConfig(num_sessions=40, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())

        assert len(gaps) >= 20, (
            f"Insufficient data for CV analysis: got {len(gaps)} gaps, need >= 20."
        )

        cv = timing.calculate_cv(gaps)
        # Poisson should have CV in range [0.5, 1.5]
        # Constant rate should have CV < 0.3
        assert 0.5 < cv < 1.5, (
            f"CV {cv:.4f} outside expected range [0.5, 1.5] for Poisson process"
        )

    def test_mean_rate_matches_target(self, cli: AIPerfCLI):
        """Verify mean rate approximately matches target QPS."""
        config = TimingTestConfig(num_sessions=50, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())

        assert len(gaps) >= 20, (
            f"Insufficient data for mean rate analysis: got {len(gaps)} gaps, need >= 20."
        )

        mean_gap = timing.calculate_mean(gaps)
        expected_gap = 1.0 / config.qps
        tolerance = expected_gap * 0.4  # 40% tolerance

        assert abs(mean_gap - expected_gap) < tolerance, (
            f"Mean gap {mean_gap:.4f}s differs from expected {expected_gap:.4f}s"
        )


@pytest.mark.component_integration
class TestPoissonRateWithConcurrency(BaseConcurrencyTests):
    """Tests for Poisson rate with concurrency limits.

    Inherits common concurrency tests from BaseConcurrencyTests.
    Tests: test_with_concurrency_limit, test_with_prefill_concurrency,
           test_multi_turn_with_concurrency
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Poisson rate timing command."""
        return build_timing_command(config, arrival_pattern="poisson")


@pytest.mark.component_integration
class TestPoissonRateEdgeCases:
    """Edge case tests for Poisson rate timing.

    Inherits common edge case tests from BaseEdgeCaseTests.
    Tests: single_request, very_high_qps, low_qps, many_sessions_few_turns, few_sessions_many_turns
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Poisson rate timing command."""
        return build_timing_command(config, arrival_pattern="poisson")


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestPoissonRateStress:
    """Stress tests for Poisson rate timing."""

    def test_high_volume(self, cli: AIPerfCLI):
        """Test high volume with Poisson rate."""
        config = TimingTestConfig(num_sessions=100, qps=300.0, timeout=90.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_sustained_multi_turn(self, cli: AIPerfCLI):
        """Test sustained multi-turn Poisson workload."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=100.0,
            turns_per_session=5,
            timeout=90.0,
        )
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    def test_bursty_pattern(self, cli: AIPerfCLI):
        """Test that Poisson produces bursty patterns (some clustering).

        Exponential distribution has high variability, so we should see some
        very short gaps (bursts) where requests cluster together.
        """
        config = TimingTestConfig(num_sessions=50, qps=150.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 10, (
            f"Insufficient data for bursty pattern analysis: got {len(gaps)} gaps, need >= 10."
        )
        # Poisson should have some very short intervals (bursts)
        min_gap = min(gaps)
        mean_gap = timing.calculate_mean(gaps)
        # Some gaps should be much shorter than mean
        assert min_gap < mean_gap * 0.5, (
            f"No bursty behavior detected: min_gap={min_gap:.4f}s, mean_gap={mean_gap:.4f}s. "
            f"Expected min_gap < {mean_gap * 0.5:.4f}s for exponential distribution."
        )
