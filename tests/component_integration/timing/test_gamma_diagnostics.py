# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Gamma distribution arrival patterns.

Verifies that the Gamma interval generator produces correct coefficient of
variation (CV) values across different smoothness parameters.

For Gamma distribution: CV = 1 / sqrt(smoothness)
- smoothness=1.0 → CV=1.0 (equivalent to Poisson/exponential)
- smoothness=4.0 → CV=0.5 (more regular arrivals)
- smoothness=0.25 → CV=2.0 (more bursty arrivals)

Run with -s flag to see detailed statistics:
    pytest tests/component_integration/timing/test_gamma_diagnostics.py -v -s
"""

import statistics

import pytest

from tests.component_integration.timing.conftest import (
    defaults,
)
from tests.harness.analyzers import TimingAnalyzer
from tests.harness.utils import AIPerfCLI

# CV tolerance for assertions. 15% accounts for:
# - Statistical variance with small sample sizes (~80-100 samples)
# - Minor scheduling overhead
# - Event loop timing variations
CV_TOLERANCE_PCT = 25.0

# Minimum sample size required for reliable CV measurement
MIN_SAMPLE_SIZE = 30


def assert_cv_within_tolerance(
    actual_cv: float,
    expected_cv: float,
    tolerance_pct: float = CV_TOLERANCE_PCT,
    context: str = "",
) -> None:
    """Assert that actual CV is within tolerance of expected CV.

    Args:
        actual_cv: Measured coefficient of variation
        expected_cv: Theoretical CV (1/sqrt(smoothness))
        tolerance_pct: Allowed percentage deviation (default 15%)
        context: Optional context string for error message
    """
    error_pct = abs(actual_cv - expected_cv) / expected_cv * 100
    assert error_pct <= tolerance_pct, (
        f"CV error {error_pct:.1f}% exceeds {tolerance_pct}% tolerance. "
        f"Expected CV={expected_cv:.4f}, got {actual_cv:.4f}. {context}"
    )


def build_gamma_cmd(num_sessions: int, qps: float, smoothness: float) -> str:
    """Build Gamma rate command."""
    return f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {num_sessions} \
            --request-rate {qps} \
            --arrival-pattern gamma \
            --arrival-smoothness {smoothness} \
            --osl 50 \
            --extra-inputs ignore_eos:true \
            --random-seed 42 \
            --ui {defaults.ui}
    """


def print_stats(gaps: list[float], smoothness: float, qps: float) -> dict:
    """Print detailed statistics and return them."""
    if len(gaps) < 2:
        print(f"  ERROR: Only {len(gaps)} gaps, need at least 2")
        return {}

    actual_mean = statistics.mean(gaps)
    actual_std = statistics.stdev(gaps)
    actual_cv = actual_std / actual_mean if actual_mean > 0 else 0

    # Actual QPS = 1 / mean_gap (inverse of mean inter-arrival time)
    actual_qps = 1.0 / actual_mean if actual_mean > 0 else 0
    qps_error_pct = ((actual_qps - qps) / qps * 100) if qps > 0 else 0

    expected_mean = 1.0 / qps
    expected_cv = 1.0 / (smoothness**0.5)
    expected_std = expected_mean * expected_cv

    # Calculate how much the scheduler jitter affects things
    # If we assume scheduler adds ~1ms jitter
    scheduler_jitter_ms = 1.0
    scheduler_jitter_sec = scheduler_jitter_ms / 1000
    jitter_as_pct_of_std = (
        (scheduler_jitter_sec / expected_std) * 100 if expected_std > 0 else 0
    )

    print(f"\n{'=' * 70}")
    print(f"  SMOOTHNESS = {smoothness}, TARGET QPS = {qps}")
    print(f"{'=' * 70}")
    print(f"  Sample size: {len(gaps)} gaps")
    print()
    print(f"  {'Metric':<25} {'Actual':>12} {'Expected':>12} {'Error %':>10}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10}")
    print(
        f"  {'QPS (req/sec)':<25} {actual_qps:>12.2f} {qps:>12.2f} {qps_error_pct:>+10.1f}%"
    )
    print(
        f"  {'Mean (sec)':<25} {actual_mean:>12.6f} {expected_mean:>12.6f} {((actual_mean - expected_mean) / expected_mean * 100):>+10.1f}%"
    )
    print(
        f"  {'Std (sec)':<25} {actual_std:>12.6f} {expected_std:>12.6f} {((actual_std - expected_std) / expected_std * 100):>+10.1f}%"
    )
    print(
        f"  {'CV':<25} {actual_cv:>12.4f} {expected_cv:>12.4f} {((actual_cv - expected_cv) / expected_cv * 100):>+10.1f}%"
    )
    print()
    print("  Scheduler Impact Analysis:")
    print(f"    Expected std: {expected_std * 1000:.2f} ms")
    print(f"    ~1ms jitter is {jitter_as_pct_of_std:.1f}% of expected std")
    print(f"{'=' * 70}\n")

    return {
        "smoothness": smoothness,
        "qps": qps,
        "actual_qps": actual_qps,
        "qps_error_pct": qps_error_pct,
        "sample_size": len(gaps),
        "actual_mean": actual_mean,
        "actual_std": actual_std,
        "actual_cv": actual_cv,
        "expected_mean": expected_mean,
        "expected_std": expected_std,
        "expected_cv": expected_cv,
        "cv_error_pct": ((actual_cv - expected_cv) / expected_cv * 100),
        "jitter_impact_pct": jitter_as_pct_of_std,
    }


@pytest.mark.component_integration
class TestGammaDiagnostics:
    """Tests for Gamma arrival pattern CV accuracy.

    Verifies that the coefficient of variation (CV) of inter-arrival times
    matches theoretical values: CV = 1/sqrt(smoothness).

    All tests assert CV is within 15% of expected value, accounting for
    statistical variance with small sample sizes.

    Run with -s for detailed output:
        pytest tests/component_integration/timing/test_gamma_diagnostics.py -v -s
    """

    @pytest.mark.parametrize(
        "smoothness",
        [0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0],
    )
    def test_gamma_cv_by_smoothness(self, cli: AIPerfCLI, smoothness: float):
        """Verify Gamma CV matches theoretical value for given smoothness.

        CV = 1/sqrt(smoothness), so:
        - smoothness=0.25 → CV=2.0 (bursty)
        - smoothness=4.0 → CV=0.5 (regular)
        - smoothness=25.0 → CV=0.2 (very regular)
        """
        qps = 100.0
        num_sessions = 80

        cmd = build_gamma_cmd(num_sessions, qps, smoothness)
        result = cli.run_sync(cmd, timeout=60.0)

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())

        stats = print_stats(gaps, smoothness, qps)

        # Assertions
        assert len(gaps) >= MIN_SAMPLE_SIZE, (
            f"Insufficient samples: {len(gaps)} < {MIN_SAMPLE_SIZE}"
        )
        assert_cv_within_tolerance(
            stats["actual_cv"],
            stats["expected_cv"],
            context=f"smoothness={smoothness}",
        )

    @pytest.mark.slow
    def test_gamma_cv_at_lower_qps(self, cli: AIPerfCLI):
        """Verify Gamma CV at lower QPS where scheduler jitter matters less.

        At QPS=20, intervals are 50ms, so 1ms jitter is only 2% noise.
        This should produce even more accurate CV values.
        """
        print("\n" + "=" * 70)
        print("  GAMMA CV DIAGNOSTIC TEST - LOWER QPS (20)")
        print("  Lower QPS = longer intervals = less scheduler jitter impact")
        print("=" * 70)

        qps = 20.0  # 50ms intervals
        num_sessions = 60
        smoothness_values = [0.5, 1.0, 4.0, 16.0]

        results = []
        for smoothness in smoothness_values:
            cmd = build_gamma_cmd(num_sessions, qps, smoothness)
            result = cli.run_sync(cmd, timeout=60.0)

            timing = TimingAnalyzer(result)
            gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())

            stats = print_stats(gaps, smoothness, qps)
            if stats:
                results.append(stats)

        # Print summary table
        print("\n" + "=" * 70)
        print("  SUMMARY TABLE (QPS=20)")
        print("=" * 70)
        print(
            f"  {'Smooth':>8} {'Tgt QPS':>9} {'Act QPS':>9} {'QPS Err':>9} {'Exp CV':>9} {'Act CV':>9} {'CV Err':>9}"
        )
        print(
            f"  {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}"
        )
        for r in results:
            print(
                f"  {r['smoothness']:>8.2f} "
                f"{r['qps']:>9.1f} "
                f"{r['actual_qps']:>9.1f} "
                f"{r['qps_error_pct']:>+8.1f}% "
                f"{r['expected_cv']:>9.4f} "
                f"{r['actual_cv']:>9.4f} "
                f"{r['cv_error_pct']:>+8.1f}%"
            )
        print("=" * 70 + "\n")

        # Assertions - all smoothness values should have correct CV
        for r in results:
            assert_cv_within_tolerance(
                r["actual_cv"],
                r["expected_cv"],
                context=f"smoothness={r['smoothness']}, qps={qps}",
            )

    def test_single_smoothness_detailed(self, cli: AIPerfCLI):
        """Verify smoothness=4.0 (CV=0.5) with detailed percentile output.

        Good for quick iteration when testing code changes.
        """
        print("\n" + "=" * 70)
        print("  SINGLE SMOOTHNESS DETAILED TEST")
        print("=" * 70)

        smoothness = 4.0  # Expected CV = 0.5
        qps = 100.0
        num_sessions = 100

        cmd = build_gamma_cmd(num_sessions, qps, smoothness)
        result = cli.run_sync(cmd, timeout=60.0)

        timing = TimingAnalyzer(result)
        gaps = timing.calculate_gaps_sec(timing.get_credit_issue_times_ns())

        stats = print_stats(gaps, smoothness, qps)

        # Print percentile distribution
        if len(gaps) >= 10:
            sorted_gaps = sorted(gaps)
            print("  Percentile Distribution (ms):")
            print(f"    Min:  {sorted_gaps[0] * 1000:>8.3f}")
            print(f"    P10:  {sorted_gaps[len(sorted_gaps) // 10] * 1000:>8.3f}")
            print(f"    P25:  {sorted_gaps[len(sorted_gaps) // 4] * 1000:>8.3f}")
            print(f"    P50:  {sorted_gaps[len(sorted_gaps) // 2] * 1000:>8.3f}")
            print(f"    P75:  {sorted_gaps[3 * len(sorted_gaps) // 4] * 1000:>8.3f}")
            print(f"    P90:  {sorted_gaps[9 * len(sorted_gaps) // 10] * 1000:>8.3f}")
            print(f"    Max:  {sorted_gaps[-1] * 1000:>8.3f}")
            print()

            # Count gaps below 1ms (scheduler floor)
            below_1ms = sum(1 for g in gaps if g < 0.001)
            print(f"  Gaps below 1ms: {below_1ms} ({below_1ms / len(gaps) * 100:.1f}%)")
            print("=" * 70 + "\n")

        # Assertions
        assert len(gaps) >= MIN_SAMPLE_SIZE
        assert_cv_within_tolerance(
            stats["actual_cv"],
            stats["expected_cv"],
            context=f"smoothness={smoothness}",
        )

    def test_poisson_arrival_pattern(self, cli: AIPerfCLI):
        """Verify Poisson arrival pattern produces CV≈1.0.

        Poisson arrivals have exponential inter-arrival times with CV=1.0.
        """
        qps = 100.0
        num_sessions = 80

        cmd_poisson = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions {num_sessions} \
                --request-rate {qps} \
                --arrival-pattern poisson \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --random-seed 42 \
                --ui {defaults.ui}
        """
        result_poisson = cli.run_sync(cmd_poisson, timeout=60.0)
        timing_poisson = TimingAnalyzer(result_poisson)
        gaps_poisson = timing_poisson.calculate_gaps_sec(
            timing_poisson.get_credit_issue_times_ns()
        )

        print("\n  POISSON (--arrival-pattern poisson):")
        stats_poisson = print_stats(gaps_poisson, smoothness=1.0, qps=qps)

        # Assertions - Poisson should have CV ≈ 1.0
        assert len(gaps_poisson) >= MIN_SAMPLE_SIZE
        assert_cv_within_tolerance(
            stats_poisson["actual_cv"],
            1.0,
            context="Poisson arrival pattern",
        )
