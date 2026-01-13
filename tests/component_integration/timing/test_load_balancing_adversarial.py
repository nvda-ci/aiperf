# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for sticky credit router load balancing.

These tests attempt to break fair load balancing with:
- Pathological request patterns
- Edge case configurations
- Race condition triggers
- Statistical edge cases

If these tests fail, the load balancer needs improvement.
"""

import pytest

from tests.component_integration.timing.conftest import (
    defaults,
)
from tests.harness.analyzers import LoadBalancingAnalyzer
from tests.harness.utils import AIPerfCLI


def build_adversarial_command(
    *,
    num_sessions: int,
    workers_max: int,
    turns_per_session: int = 1,
    turns_stddev: int = 0,
    qps: float = 0,
    concurrency: int | None = None,
    osl: int = 5,  # Fast OSL for behavior tests (5-10x speedup)
    random_seed: int = 42,
) -> str:
    """Build command for adversarial testing."""
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {num_sessions} \
            --osl {osl} \
            --extra-inputs ignore_eos:true \
            --workers-max {workers_max} \
            --random-seed {random_seed} \
            --ui {defaults.ui}
    """

    if turns_per_session > 1 or turns_stddev > 0:
        cmd += f" --session-turns-mean {turns_per_session} --session-turns-stddev {turns_stddev}"

    if concurrency is not None:
        cmd += f" --concurrency {concurrency}"

    if qps > 0:
        cmd += f" --request-rate {qps} --request-rate-mode constant"

    return cmd


@pytest.mark.component_integration
class TestSmallSampleAdversarial:
    """Adversarial tests with small sample sizes where variance is naturally high."""

    @pytest.mark.parametrize(
        "num_sessions,workers_max",
        [
            (5, 5),     # 1 session per worker - any deviation is 100%
            (6, 5),     # 1.2 sessions per worker - guaranteed imbalance
            (7, 5),     # 1.4 sessions per worker
            (10, 7),    # Prime workers, not divisible
            (11, 7),    # Prime both
            (13, 11),   # Both prime, very close
        ],
    )  # fmt: skip
    def test_small_samples_still_reasonable(
        self,
        cli: AIPerfCLI,
        num_sessions: int,
        workers_max: int,
    ):
        """Small samples have high variance but should still be somewhat fair."""
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            concurrency=workers_max,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        # With small samples, we can't expect perfect fairness
        # But max/min ratio shouldn't be extreme
        ratio = analyzer.max_min_ratio()

        # Calculate theoretical worst case: if sessions aren't evenly divisible
        # worst case is ceil(n/w) vs floor(n/w)
        floor_val = num_sessions // workers_max
        ceil_val = floor_val + (1 if num_sessions % workers_max else 0)
        theoretical_worst_ratio = (
            ceil_val / floor_val if floor_val > 0 else float("inf")
        )

        # Should be at or near theoretical best possible
        assert ratio <= theoretical_worst_ratio * 1.1, (
            f"Ratio {ratio:.3f} worse than theoretical worst {theoretical_worst_ratio:.3f}. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )

    def test_single_session_single_worker(self, cli: AIPerfCLI):
        """Degenerate case: 1 session, 1 worker."""
        cmd = build_adversarial_command(
            num_sessions=1,
            workers_max=1,
            concurrency=1,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 1

        analyzer = LoadBalancingAnalyzer(result)
        assert analyzer.jains_fairness_index() == 1.0  # Trivially fair

    def test_more_workers_than_sessions(self, cli: AIPerfCLI):
        """Edge case: more workers than sessions - some workers get nothing."""
        cmd = build_adversarial_command(
            num_sessions=3,
            workers_max=10,
            concurrency=3,  # Can't exceed num_sessions
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 3

        analyzer = LoadBalancingAnalyzer(result)

        # Only 3 workers should have received work
        assert analyzer.num_workers == 3, (
            f"Expected 3 workers to receive work, got {analyzer.num_workers}"
        )

        # Those 3 should each have exactly 1
        for worker_id, count in analyzer.credits_per_worker().items():
            assert count == 1, f"Worker {worker_id} got {count} credits, expected 1"


@pytest.mark.component_integration
@pytest.mark.stress
class TestHighConcurrencyAdversarial:
    """Adversarial tests with extreme concurrency to trigger race conditions."""

    def test_concurrency_equals_sessions(self, cli: AIPerfCLI):
        """All sessions start simultaneously - tests tie-breaking."""
        num_sessions = 100
        workers_max = 5
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            concurrency=num_sessions,  # All at once
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        # Even with all sessions starting at once, distribution should be fair
        # because the router assigns one at a time and tracks load
        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.95, (
            f"JFI {jfi:.4f} too low for simultaneous start. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )

    def test_concurrency_much_higher_than_workers(self, cli: AIPerfCLI):
        """Concurrency >> workers - heavy contention."""
        num_sessions = 200
        workers_max = 3
        concurrency = 100  # 33x the workers
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            concurrency=concurrency,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.95, (
            f"JFI {jfi:.4f} too low under heavy contention. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )

    def test_very_high_qps_burst(self, cli: AIPerfCLI):
        """Extremely high QPS - rapid-fire requests."""
        num_sessions = 500
        workers_max = 5
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            qps=2000.0,  # 2000 requests/sec
        )
        result = cli.run_sync(cmd, timeout=90.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.97, (
            f"JFI {jfi:.4f} degraded under high QPS. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )


@pytest.mark.component_integration
class TestPrimeNumberAdversarial:
    """Adversarial tests with prime numbers that don't divide evenly."""

    @pytest.mark.parametrize(
        "num_sessions,workers_max",
        [
            (97, 7),    # Prime sessions, prime workers
            (101, 11),  # Both prime
            (103, 13),  # Both prime, close values
            (127, 17),  # Larger primes
            (199, 23),  # Even larger primes
        ],
    )  # fmt: skip
    def test_prime_combinations(
        self,
        cli: AIPerfCLI,
        num_sessions: int,
        workers_max: int,
    ):
        """Prime numbers can't divide evenly but should still be fair."""
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            qps=500.0,
        )
        result = cli.run_sync(cmd, timeout=120.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        # Even with primes, JFI should be high
        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.90, (
            f"JFI {jfi:.4f} too low for prime combination {num_sessions}/{workers_max}. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )

        # Max-min difference follows binomial distribution variance
        # For n sessions into k workers with random tie-breaking:
        # StdDev ≈ sqrt(n * (1/k) * (1 - 1/k))
        # Allow 3 stddevs for rare events
        import math

        expected_std = math.sqrt(
            num_sessions * (1 / workers_max) * (1 - 1 / workers_max)
        )
        max_allowed_diff = max(2, int(3 * expected_std))  # At least 2, or 3 stddevs

        credits = list(analyzer.credits_per_worker().values())
        diff = max(credits) - min(credits)
        assert diff <= max_allowed_diff, (
            f"Max-min diff {diff} > {max_allowed_diff} for {num_sessions}/{workers_max}. "
            f"(expected_std={expected_std:.2f}) Distribution: {analyzer.credits_per_worker()}"
        )


@pytest.mark.component_integration
class TestMultiTurnAdversarial:
    """Adversarial tests with multi-turn sessions that stress sticky routing."""

    @pytest.mark.stress
    @pytest.mark.slow
    def test_very_long_sessions(self, cli: AIPerfCLI):
        """Sessions with many turns - worker gets "stuck" with long session."""
        num_sessions = 20
        turns_per_session = 20  # 400 total credits
        workers_max = 4
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            turns_per_session=turns_per_session,
            qps=400.0,
        )
        result = cli.run_sync(cmd, timeout=120.0)

        expected = num_sessions * turns_per_session
        assert result.request_count == expected

        analyzer = LoadBalancingAnalyzer(result)

        # Sticky routing should still maintain fairness
        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.85, (
            f"JFI {jfi:.4f} too low for long sessions. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )

        # Sticky routing must be maintained
        passed, reason = analyzer.verify_sticky_routing()
        assert passed, reason

    @pytest.mark.slow
    def test_single_turn_vs_multi_turn_mix_simulated(self, cli: AIPerfCLI):
        """Mix of session lengths using stddev to create variance.

        With mean=5 and stddev=4, we get sessions ranging from 1 to ~13 turns.
        This creates uneven load per session, stressing the balancer.
        """
        num_sessions = 50
        workers_max = 5
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            turns_per_session=5,
            turns_stddev=4,  # High variance in session length
            qps=400.0,
            random_seed=42,
        )
        result = cli.run_sync(cmd, timeout=120.0)

        # Can't predict exact count due to stddev
        assert result.request_count > 0

        analyzer = LoadBalancingAnalyzer(result)

        # With variable session lengths, total credit distribution may vary
        # but sessions should still be evenly distributed
        sessions_per_worker = analyzer.sessions_per_worker()
        expected = num_sessions / workers_max

        for worker_id, count in sessions_per_worker.items():
            deviation_pct = abs(count - expected) / expected * 100
            assert deviation_pct <= 40, (
                f"Worker {worker_id} has {count} sessions, "
                f"expected ~{expected:.1f} (deviation {deviation_pct:.1f}%)"
            )

    @pytest.mark.slow
    def test_many_short_sessions_few_workers(self, cli: AIPerfCLI):
        """Many 2-turn sessions with few workers - rapid session turnover."""
        num_sessions = 200
        turns_per_session = 2
        workers_max = 2
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            turns_per_session=turns_per_session,
            concurrency=20,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        expected = num_sessions * turns_per_session
        assert result.request_count == expected

        analyzer = LoadBalancingAnalyzer(result)

        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.95, (
            f"JFI {jfi:.4f} too low for rapid session turnover. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )


@pytest.mark.component_integration
class TestRandomSeedAdversarial:
    """Test that different random seeds don't break fairness."""

    @pytest.mark.parametrize("seed", [1, 13, 42, 99, 12345, 999999])
    def test_various_seeds_fair(self, cli: AIPerfCLI, seed: int):
        """Different seeds should all produce fair distributions."""
        num_sessions = 100
        workers_max = 5
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            concurrency=20,
            random_seed=seed,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.90, (
            f"JFI {jfi:.4f} too low for seed {seed}. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )


@pytest.mark.component_integration
class TestWorkerCountExtremes:
    """Adversarial tests with extreme worker counts."""

    def test_many_workers_few_sessions(self, cli: AIPerfCLI):
        """10 workers for 20 sessions - 2 each, tests even spreading."""
        cmd = build_adversarial_command(
            num_sessions=20,
            workers_max=10,
            concurrency=20,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == 20

        analyzer = LoadBalancingAnalyzer(result)

        # Should be exactly 2 per worker
        for worker_id, count in analyzer.credits_per_worker().items():
            assert count == 2, (
                f"Worker {worker_id} got {count}, expected exactly 2. "
                f"Distribution: {analyzer.credits_per_worker()}"
            )

    def test_single_worker_high_load(self, cli: AIPerfCLI):
        """Single worker handles everything - baseline for comparison."""
        num_sessions = 100
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=1,
            concurrency=10,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        assert analyzer.num_workers == 1
        assert analyzer.jains_fairness_index() == 1.0

    def test_workers_equal_sessions_multi_turn(self, cli: AIPerfCLI):
        """N workers, N sessions, M turns each - perfect balance expected."""
        num_sessions = 8
        workers_max = 8
        turns_per_session = 5
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            turns_per_session=turns_per_session,
            concurrency=num_sessions,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        expected = num_sessions * turns_per_session
        assert result.request_count == expected

        analyzer = LoadBalancingAnalyzer(result)

        # Each worker should have exactly 1 session with 5 turns
        for worker_id, count in analyzer.credits_per_worker().items():
            assert count == turns_per_session, (
                f"Worker {worker_id} got {count}, expected {turns_per_session}. "
                f"Distribution: {analyzer.credits_per_worker()}"
            )

        assert analyzer.jains_fairness_index() == 1.0


@pytest.mark.component_integration
@pytest.mark.stress
class TestTimingPatternAdversarial:
    """Adversarial tests with specific timing patterns."""

    @pytest.mark.slow
    def test_original_problematic_scenario(self, cli: AIPerfCLI):
        """Original problematic scenario: 10 sessions, 5 workers, slow QPS.

        Before the fix, this produced distribution like {3, 4, 1, 1, 1} with JFI=0.714.
        With the new algorithm using total_sent_credits as tie-breaker,
        this should now produce perfect {2, 2, 2, 2, 2} distribution.
        """
        num_sessions = 10
        workers_max = 5
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            qps=500.0,  # Fast issuance
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)
        jfi = analyzer.jains_fairness_index()
        distribution = sorted(analyzer.credits_per_worker().values())

        # With the new algorithm, should achieve perfect fairness
        assert jfi == 1.0, (
            f"JFI {jfi:.4f} != 1.0. Distribution: {distribution}. "
            f"Expected [2, 2, 2, 2, 2] with new tie-breaking algorithm."
        )
        assert distribution == [2, 2, 2, 2, 2], (
            f"Distribution {distribution} not perfectly balanced. "
            f"Expected [2, 2, 2, 2, 2]."
        )

    @pytest.mark.slow
    def test_very_slow_rate(self, cli: AIPerfCLI):
        """Very slow request rate - tests deterministic tie-breaking.

        With slow QPS, each request completes before the next arrives, so all
        workers are at load=0 when selection happens. The new algorithm uses
        total_sent_credits as tie-breaker, ensuring perfect distribution.
        """
        num_sessions = 50
        workers_max = 5
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            qps=300.0,  # High rate
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        # With deterministic tie-breaking, should achieve perfect fairness
        jfi = analyzer.jains_fairness_index()
        assert jfi == 1.0, (
            f"JFI {jfi:.4f} != 1.0. Distribution: {analyzer.credits_per_worker()}"
        )

    @pytest.mark.slow
    def test_qps_lower_than_workers(self, cli: AIPerfCLI):
        """QPS < workers - requests trickle in slower than worker count."""
        num_sessions = 50
        workers_max = 10
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            qps=500.0,  # High rate
        )
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.85, (
            f"JFI {jfi:.4f} too low for low QPS. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestStressTest:
    """High-volume stress tests."""

    def test_thousand_sessions(self, cli: AIPerfCLI):
        """1000 single-turn sessions - statistical fairness should be excellent."""
        num_sessions = 1000
        workers_max = 7  # Prime, doesn't divide evenly
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            qps=500.0,
        )
        result = cli.run_sync(cmd, timeout=120.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        # High volume should produce excellent fairness
        jfi = analyzer.jains_fairness_index()
        gini = analyzer.gini_coefficient()
        ratio = analyzer.max_min_ratio()

        assert jfi >= 0.999, f"JFI {jfi:.6f} should be near-perfect at high volume"
        assert gini <= 0.01, f"Gini {gini:.6f} should be very low at high volume"
        assert ratio <= 1.05, f"Ratio {ratio:.4f} should be very close to 1.0"

    def test_thousand_multi_turn_sessions(self, cli: AIPerfCLI):
        """1000 multi-turn sessions - 5000 total credits."""
        num_sessions = 1000
        turns_per_session = 5
        workers_max = 8
        cmd = build_adversarial_command(
            num_sessions=num_sessions,
            workers_max=workers_max,
            turns_per_session=turns_per_session,
            qps=500.0,
        )
        result = cli.run_sync(cmd, timeout=180.0)

        expected = num_sessions * turns_per_session
        assert result.request_count == expected

        analyzer = LoadBalancingAnalyzer(result)

        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.998, (
            f"JFI {jfi:.6f} should be near-perfect for high-volume multi-turn. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )

        # Sticky routing must still work
        passed, reason = analyzer.verify_sticky_routing()
        assert passed, reason
