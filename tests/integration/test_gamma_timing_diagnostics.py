# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full integration diagnostic tests for Gamma rate timing.

These tests run aiperf as a SUBPROCESS with the mock server as a SEPARATE PROCESS,
eliminating event loop contention that exists in component_integration tests.

Run with:
    pytest tests/integration/test_gamma_timing_diagnostics.py -v -s -n0

Compare results before/after scheduler changes to see if CV values improve.
"""

import statistics

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults

defaults = IntegrationTestDefaults


def calculate_cv(values: list[float]) -> float:
    """Calculate coefficient of variation."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def print_gamma_stats(
    gaps_sec: list[float], smoothness: float, qps: float, label: str = ""
) -> dict:
    """Print detailed statistics for Gamma distribution analysis."""
    if len(gaps_sec) < 2:
        print(f"  ERROR: Only {len(gaps_sec)} gaps, need at least 2")
        return {}

    actual_mean = statistics.mean(gaps_sec)
    actual_std = statistics.stdev(gaps_sec)
    actual_cv = actual_std / actual_mean if actual_mean > 0 else 0

    # Actual QPS = 1 / mean_gap (inverse of mean inter-arrival time)
    actual_qps = 1.0 / actual_mean if actual_mean > 0 else 0
    qps_error_pct = ((actual_qps - qps) / qps * 100) if qps > 0 else 0

    expected_mean = 1.0 / qps
    expected_cv = 1.0 / (smoothness**0.5)
    expected_std = expected_mean * expected_cv

    # Scheduler jitter impact analysis
    scheduler_jitter_ms = 1.0
    scheduler_jitter_sec = scheduler_jitter_ms / 1000
    jitter_impact_pct = (
        (scheduler_jitter_sec / expected_std) * 100 if expected_std > 0 else 0
    )

    header = f"  SMOOTHNESS = {smoothness}, TARGET QPS = {qps}"
    if label:
        header += f" [{label}]"

    print(f"\n{'=' * 70}")
    print(header)
    print(f"{'=' * 70}")
    print(f"  Sample size: {len(gaps_sec)} gaps")
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
    print(f"  ~1ms jitter is {jitter_impact_pct:.1f}% of expected std")
    print(f"{'=' * 70}")

    return {
        "smoothness": smoothness,
        "qps": qps,
        "actual_qps": actual_qps,
        "qps_error_pct": qps_error_pct,
        "sample_size": len(gaps_sec),
        "actual_mean": actual_mean,
        "actual_std": actual_std,
        "actual_cv": actual_cv,
        "expected_mean": expected_mean,
        "expected_std": expected_std,
        "expected_cv": expected_cv,
        "cv_error_pct": ((actual_cv - expected_cv) / expected_cv * 100),
        "jitter_impact_pct": jitter_impact_pct,
    }


@pytest.mark.integration
@pytest.mark.asyncio
class TestGammaTimingDiagnostics:
    """Full integration diagnostic tests for Gamma rate timing.

    These tests run with separate processes for mock server and aiperf,
    eliminating event loop contention.
    """

    async def test_gamma_smoothness_comparison(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test multiple smoothness values and compare CV.

        This is the key diagnostic test - run it before and after scheduler
        changes to see if CV values improve.
        """
        print("\n" + "=" * 70)
        print("  GAMMA CV DIAGNOSTIC - FULL INTEGRATION TEST")
        print("  (Separate processes for mock server and aiperf)")
        print("=" * 70)

        qps = 100.0
        num_sessions = 100  # More samples for better statistics
        smoothness_values = [0.5, 1.0, 2.0, 4.0, 9.0, 16.0]

        results = []

        for smoothness in smoothness_values:
            cmd = f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {aiperf_mock_server.url} \
                    --streaming \
                    --num-sessions {num_sessions} \
                    --request-rate {qps} \
                    --arrival-pattern gamma \
                    --arrival-smoothness {smoothness} \
                    --osl 50 \
                    --random-seed 42 \
                    --workers-max 1 \
                    --ui simple
            """

            result = await cli.run(cmd, timeout=120.0)

            assert result.exit_code == 0, (
                f"Command failed with exit code {result.exit_code}"
            )
            assert result.jsonl is not None, "No JSONL records found"
            assert len(result.jsonl) >= num_sessions * 0.9, (
                f"Expected ~{num_sessions} records, got {len(result.jsonl)}"
            )

            # Extract request_start_ns from metadata and calculate gaps
            start_times_ns = sorted(
                record.metadata.request_start_ns
                for record in result.jsonl
                if record.metadata.request_start_ns is not None
            )

            gaps_ns = [
                start_times_ns[i] - start_times_ns[i - 1]
                for i in range(1, len(start_times_ns))
            ]
            gaps_sec = [g / 1e9 for g in gaps_ns]

            stats = print_gamma_stats(gaps_sec, smoothness, qps, "FULL INTEGRATION")
            if stats:
                results.append(stats)

        # Print summary table
        print("\n" + "=" * 70)
        print("  SUMMARY TABLE - FULL INTEGRATION")
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

    async def test_gamma_single_smoothness_detailed(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Single smoothness test with detailed output.

        Good for quick iteration when testing code changes.
        """
        print("\n" + "=" * 70)
        print("  SINGLE SMOOTHNESS DETAILED - FULL INTEGRATION")
        print("=" * 70)

        smoothness = 4.0  # Expected CV = 0.5
        qps = 100.0
        num_sessions = 100

        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --streaming \
                --num-sessions {num_sessions} \
                --request-rate {qps} \
                --arrival-pattern gamma \
                --arrival-smoothness {smoothness} \
                --osl 50 \
                --random-seed 42 \
                --workers-max 1 \
                --ui simple
        """

        result = await cli.run(cmd, timeout=120.0)

        assert result.exit_code == 0
        assert result.jsonl is not None

        # Extract and analyze timing
        start_times_ns = sorted(
            record.metadata.request_start_ns
            for record in result.jsonl
            if record.metadata.request_start_ns is not None
        )

        gaps_ns = [
            start_times_ns[i] - start_times_ns[i - 1]
            for i in range(1, len(start_times_ns))
        ]
        gaps_sec = [g / 1e9 for g in gaps_ns]
        gaps_ms = [g * 1000 for g in gaps_sec]

        print_gamma_stats(gaps_sec, smoothness, qps, "DETAILED")

        # Print percentile distribution
        if len(gaps_ms) >= 10:
            sorted_gaps = sorted(gaps_ms)
            print("\n  Percentile Distribution (ms):")
            print(f"    Min:  {sorted_gaps[0]:>8.3f}")
            print(f"    P10:  {sorted_gaps[len(sorted_gaps) // 10]:>8.3f}")
            print(f"    P25:  {sorted_gaps[len(sorted_gaps) // 4]:>8.3f}")
            print(f"    P50:  {sorted_gaps[len(sorted_gaps) // 2]:>8.3f}")
            print(f"    P75:  {sorted_gaps[3 * len(sorted_gaps) // 4]:>8.3f}")
            print(f"    P90:  {sorted_gaps[9 * len(sorted_gaps) // 10]:>8.3f}")
            print(f"    Max:  {sorted_gaps[-1]:>8.3f}")

            # Count gaps below 1ms
            below_1ms = sum(1 for g in gaps_ms if g < 1.0)
            print(
                f"\n  Gaps below 1ms: {below_1ms} ({below_1ms / len(gaps_ms) * 100:.1f}%)"
            )
            print("=" * 70 + "\n")

    async def test_gamma_vs_poisson_comparison(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Compare Gamma(smoothness=1) with Poisson - should be equivalent."""
        print("\n" + "=" * 70)
        print("  GAMMA vs POISSON COMPARISON - FULL INTEGRATION")
        print("=" * 70)

        qps = 100.0
        num_sessions = 100

        # Run Gamma with smoothness=1.0
        cmd_gamma = f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --streaming \
                --num-sessions {num_sessions} \
                --request-rate {qps} \
                --arrival-pattern gamma \
                --arrival-smoothness 1.0 \
                --osl 50 \
                --random-seed 42 \
                --workers-max 1 \
                --ui simple
        """

        result_gamma = await cli.run(cmd_gamma, timeout=120.0)
        assert result_gamma.exit_code == 0
        assert result_gamma.jsonl is not None

        start_times_gamma = sorted(
            r.metadata.request_start_ns
            for r in result_gamma.jsonl
            if r.metadata.request_start_ns
        )
        gaps_gamma = [
            (start_times_gamma[i] - start_times_gamma[i - 1]) / 1e9
            for i in range(1, len(start_times_gamma))
        ]

        print("\n  GAMMA (smoothness=1.0):")
        stats_gamma = print_gamma_stats(gaps_gamma, 1.0, qps, "GAMMA")

        # Run Poisson
        cmd_poisson = f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --streaming \
                --num-sessions {num_sessions} \
                --request-rate {qps} \
                --arrival-pattern poisson \
                --osl 50 \
                --random-seed 42 \
                --workers-max 1 \
                --ui simple
        """

        result_poisson = await cli.run(cmd_poisson, timeout=120.0)
        assert result_poisson.exit_code == 0
        assert result_poisson.jsonl is not None

        start_times_poisson = sorted(
            r.metadata.request_start_ns
            for r in result_poisson.jsonl
            if r.metadata.request_start_ns
        )
        gaps_poisson = [
            (start_times_poisson[i] - start_times_poisson[i - 1]) / 1e9
            for i in range(1, len(start_times_poisson))
        ]

        print("\n  POISSON (--arrival-pattern poisson):")
        stats_poisson = print_gamma_stats(gaps_poisson, 1.0, qps, "POISSON")

        # Compare
        if stats_gamma and stats_poisson:
            print("\n  COMPARISON:")
            print(f"    Gamma CV:   {stats_gamma['actual_cv']:.4f}")
            print(f"    Poisson CV: {stats_poisson['actual_cv']:.4f}")
            print(
                f"    Difference: {abs(stats_gamma['actual_cv'] - stats_poisson['actual_cv']):.4f}"
            )
            print("=" * 70 + "\n")
