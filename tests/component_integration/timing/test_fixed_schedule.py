# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for fixed schedule timing mode.

Fixed schedule mode replays conversation traces at precise timestamps from dataset
metadata. First turns are sent at absolute timestamps, subsequent turns are
dispatched based on delay_ms or calculated from timestamp_ms.

Tests cover:
- Basic functionality with timestamp-based scheduling
- Credit flow verification (balanced, per-session)
- Timing accuracy (requests at correct timestamps)
- Multi-turn conversations with delays
- Concurrency interactions
- Load balancing across workers
- Edge cases (single request, sparse/dense schedules)
"""

from dataclasses import dataclass
from pathlib import Path

import orjson
import pytest

from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit
from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.component_integration.timing.conftest import (
    defaults,
)
from tests.harness.analyzers import (
    ConcurrencyAnalyzer,
    CreditFlowAnalyzer,
    LoadBalancingAnalyzer,
)
from tests.harness.utils import AIPerfCLI, AIPerfResults


def get_request_count(result: AIPerfResults) -> int:
    """Get request count from results, falling back to JSONL if JSON export fails.

    Fixed schedule mode has a validation conflict with the default dataset_sampling_strategy,
    which causes JSON export to fail. Fall back to JSONL record count.
    """
    # Try JSON export first (uses result.json.request_count.avg)
    if result.json and result.json.request_count:
        return int(result.json.request_count.avg)
    # Fall back to JSONL record count
    if result.jsonl:
        return len(result.jsonl)
    return 0


@dataclass
class FixedScheduleTestConfig:
    """Configuration for a fixed schedule test scenario."""

    num_sessions: int
    turns_per_session: int = 1
    schedule_duration_ms: int = 400  # Total schedule duration in ms (keep fast!)
    delay_ms: int = 5  # Delay between turns (for multi-turn)
    workers_max: int = 3
    concurrency: int | None = None
    prefill_concurrency: int | None = None
    osl: int = 50
    timeout: float = 60.0

    @property
    def expected_requests(self) -> int:
        """Calculate expected total requests."""
        return self.num_sessions * self.turns_per_session


def generate_trace_file(
    path: Path,
    config: FixedScheduleTestConfig,
    *,
    stagger_ms: int | None = None,
) -> Path:
    """Generate a mooncake trace file for fixed schedule testing.

    Args:
        path: Directory to create trace file in
        config: Test configuration
        stagger_ms: If set, stagger first turns by this interval.
                    If None, distributes evenly across schedule_duration_ms.

    Returns:
        Path to created trace file
    """
    trace_file = path / "trace.jsonl"

    if stagger_ms is None:
        # Distribute first turns evenly across schedule duration
        if config.num_sessions > 1:
            stagger_ms = config.schedule_duration_ms // (config.num_sessions - 1)
        else:
            stagger_ms = 0

    with open(trace_file, "w") as f:
        for session_idx in range(config.num_sessions):
            for turn_idx in range(config.turns_per_session):
                if turn_idx == 0:
                    # First turn - needs timestamp
                    line = {
                        "session_id": f"session_{session_idx}",
                        "timestamp": session_idx * stagger_ms,
                        "input_length": 100,
                    }
                else:
                    # Subsequent turns - use delay
                    line = {
                        "session_id": f"session_{session_idx}",
                        "delay": config.delay_ms,
                        "input_length": 100,
                    }
                f.write(orjson.dumps(line).decode() + "\n")

    return trace_file


def build_fixed_schedule_command(
    config: FixedScheduleTestConfig,
    trace_file: Path,
    *,
    auto_offset: bool = True,
    extra_args: str = "",
) -> str:
    """Build a CLI command for fixed schedule tests.

    Args:
        config: Test configuration
        trace_file: Path to trace file
        auto_offset: Whether to auto-offset timestamps
        extra_args: Additional CLI arguments

    Returns:
        CLI command string
    """
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --fixed-schedule \
            --custom-dataset-type mooncake_trace \
            --input-file {trace_file} \
            --workers-max {config.workers_max} \
            --osl {config.osl} \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """

    if auto_offset:
        cmd += " --fixed-schedule-auto-offset"

    if config.concurrency is not None:
        cmd += f" --concurrency {config.concurrency}"

    if config.prefill_concurrency is not None:
        cmd += f" --prefill-concurrency {config.prefill_concurrency}"

    if extra_args:
        cmd += f" {extra_args}"

    return cmd


# =============================================================================
# Basic Functionality Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleBasic:
    """Basic functionality tests for fixed schedule timing."""

    @pytest.mark.parametrize(
        "num_sessions",
        [5, 10, 20, 50],
    )
    def test_fixed_schedule_completes(
        self, cli: AIPerfCLI, tmp_path: Path, num_sessions: int
    ):
        """Test fixed schedule mode completes with various session counts."""
        config = FixedScheduleTestConfig(num_sessions=num_sessions)
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == num_sessions

    def test_fixed_schedule_multi_turn(self, cli: AIPerfCLI, tmp_path: Path):
        """Test fixed schedule with multi-turn conversations."""
        config = FixedScheduleTestConfig(
            num_sessions=15,
            turns_per_session=4,
            delay_ms=5,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

    def test_fixed_schedule_single_session(self, cli: AIPerfCLI, tmp_path: Path):
        """Test fixed schedule with single session."""
        config = FixedScheduleTestConfig(num_sessions=1, turns_per_session=5)
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == 5


# =============================================================================
# Credit Flow Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleCreditFlow:
    """Credit flow verification for fixed schedule timing."""

    def test_credits_balanced(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify all credits sent are returned."""
        config = FixedScheduleTestConfig(num_sessions=20)
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced(), (
            f"Credits not balanced: {analyzer.total_credits} sent, "
            f"{analyzer.total_returns} returned"
        )

    def test_credits_per_session(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify each session gets expected credits."""
        config = FixedScheduleTestConfig(
            num_sessions=12,
            turns_per_session=3,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # Build credits per session count
        credits_per_session = {
            sid: len(payloads) for sid, payloads in analyzer.credits_by_session.items()
        }
        assert len(credits_per_session) == config.num_sessions

        for session_id, count in credits_per_session.items():
            assert count == config.turns_per_session, (
                f"Session {session_id} has {count} credits, "
                f"expected {config.turns_per_session}"
            )

    def test_turn_indices_sequential(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify turn indices are sequential within each session."""
        config = FixedScheduleTestConfig(
            num_sessions=10,
            turns_per_session=5,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # turn_indices_sequential() returns bool only
        assert analyzer.turn_indices_sequential()


# =============================================================================
# Timing Accuracy Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleTiming:
    """Timing accuracy tests for fixed schedule mode."""

    def test_first_turns_staggered(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify first turns are sent at staggered timestamps."""
        stagger_ms = 50  # 50ms between first turns
        config = FixedScheduleTestConfig(
            num_sessions=10,
            schedule_duration_ms=450,  # 9 gaps of 50ms
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=stagger_ms)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # Get first turn issue times (returns list[int], not dict)
        first_turn_times = analyzer.get_first_turn_issue_times_ns()
        assert len(first_turn_times) == config.num_sessions

        # Verify stagger pattern (with tolerance)
        # times are already sorted from the method
        gaps = [
            (first_turn_times[i] - first_turn_times[i - 1])
            for i in range(1, len(first_turn_times))
        ]

        expected_gap_ns = stagger_ms * 1_000_000
        tolerance_ms = 20  # ±20ms tolerance

        for gap in gaps:
            error_ms = abs(gap - expected_gap_ns) / 1_000_000
            assert error_ms < tolerance_ms, (
                f"Gap {gap / 1e6:.2f}ms differs from expected {stagger_ms}ms by {error_ms:.1f}ms"
            )

    def test_concurrent_first_turns(self, cli: AIPerfCLI, tmp_path: Path):
        """Test multiple sessions with same timestamp (concurrent start)."""
        config = FixedScheduleTestConfig(num_sessions=10)
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)  # All at t=0
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions

        # All first turns should be issued within a small window
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # get_first_turn_issue_times_ns() returns list[int]
        first_turn_times = analyzer.get_first_turn_issue_times_ns()
        assert len(first_turn_times) == config.num_sessions

        # Max spread should be less than 100ms for concurrent starts
        max_spread_ns = max(first_turn_times) - min(first_turn_times)
        assert max_spread_ns < 100_000_000, (
            f"First turn spread {max_spread_ns / 1e6:.2f}ms too large for concurrent start"
        )


# =============================================================================
# Multi-turn Conversation Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleMultiTurn:
    """Multi-turn conversation tests for fixed schedule mode."""

    def test_delays_between_turns(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify delays between turns are respected with timing check."""
        config = FixedScheduleTestConfig(
            num_sessions=3,
            turns_per_session=3,
            delay_ms=40,  # 40ms between turns
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

        # Verify delay timing accuracy
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        credit_payloads = runner.payloads_by_type(Credit, sent=True)
        return_payloads = runner.payloads_by_type(CreditReturn, sent=True)

        for session_id in {p.payload.x_correlation_id for p in credit_payloads}:
            session_credits = sorted(
                [
                    p
                    for p in credit_payloads
                    if p.payload.x_correlation_id == session_id
                ],
                key=lambda p: p.payload.turn_index,
            )
            session_returns = sorted(
                [
                    p
                    for p in return_payloads
                    if p.payload.credit.x_correlation_id == session_id
                ],
                key=lambda p: p.payload.credit.turn_index,
            )

            for i in range(len(session_returns) - 1):
                actual_delay_ms = (
                    session_credits[i + 1].timestamp_ns
                    - session_returns[i].timestamp_ns
                ) / 1_000_000
                error_ms = abs(actual_delay_ms - config.delay_ms)
                assert error_ms < 25, f"Delay error {error_ms:.1f}ms exceeds tolerance"

    def test_zero_delay_turns(self, cli: AIPerfCLI, tmp_path: Path):
        """Test multi-turn with zero delay (immediate subsequent turns)."""
        config = FixedScheduleTestConfig(
            num_sessions=10,
            turns_per_session=5,
            delay_ms=0,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

        # Verify all sessions completed all turns
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # Use session_credits_match helper
        assert analyzer.session_credits_match(expected_turns=config.turns_per_session)

    def test_many_turns_per_session(self, cli: AIPerfCLI, tmp_path: Path):
        """Test sessions with many turns."""
        config = FixedScheduleTestConfig(
            num_sessions=5,
            turns_per_session=10,
            delay_ms=2,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests


# =============================================================================
# Concurrency Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleConcurrency:
    """Concurrency limit tests for fixed schedule mode."""

    def test_respects_concurrency_limit(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify concurrency limit is respected."""
        concurrency_limit = 5
        config = FixedScheduleTestConfig(
            num_sessions=20,
            concurrency=concurrency_limit,
        )
        # All sessions start at t=0 to stress concurrency
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions

        analyzer = ConcurrencyAnalyzer(result)

        # concurrency_within_limit() returns bool only
        assert analyzer.concurrency_within_limit(concurrency_limit)

    def test_respects_prefill_concurrency_limit(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify prefill concurrency limit is respected."""
        prefill_limit = 3
        config = FixedScheduleTestConfig(
            num_sessions=15,
            prefill_concurrency=prefill_limit,
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions

        analyzer = ConcurrencyAnalyzer(result)

        # prefill_concurrency_within_limit() returns bool only
        assert analyzer.prefill_concurrency_within_limit(prefill_limit)

    def test_concurrency_with_multi_turn(self, cli: AIPerfCLI, tmp_path: Path):
        """Test concurrency limits with multi-turn conversations."""
        concurrency_limit = 8
        config = FixedScheduleTestConfig(
            num_sessions=10,
            turns_per_session=4,
            delay_ms=0,  # Zero delay to maximize concurrent requests
            concurrency=concurrency_limit,
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests


# =============================================================================
# Load Balancing Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleLoadBalancing:
    """Load balancing tests for fixed schedule mode."""

    def test_fair_distribution_across_workers(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify credits are distributed fairly across workers."""
        config = FixedScheduleTestConfig(
            num_sessions=30,
            workers_max=5,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        analyzer = LoadBalancingAnalyzer(result)  # Needs AIPerfResults, not runner

        passed, reason = analyzer.verify_fair_distribution(tolerance_pct=30.0)
        assert passed, reason

    def test_sticky_routing_maintained(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify all turns of a session go to the same worker."""
        config = FixedScheduleTestConfig(
            num_sessions=15,
            turns_per_session=5,
            workers_max=4,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        analyzer = LoadBalancingAnalyzer(result)

        passed, reason = analyzer.verify_sticky_routing()
        assert passed, reason

    def test_first_turn_distribution(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify first turns are distributed fairly (determines session assignment)."""
        config = FixedScheduleTestConfig(
            num_sessions=40,
            turns_per_session=3,
            workers_max=5,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        analyzer = LoadBalancingAnalyzer(result)

        passed, reason = analyzer.verify_first_turn_distribution(tolerance_pct=30.0)
        assert passed, reason

    def test_jains_fairness_index(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify Jain's Fairness Index is acceptable."""
        config = FixedScheduleTestConfig(
            num_sessions=100,
            workers_max=5,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        analyzer = LoadBalancingAnalyzer(result)

        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.9, f"Jain's Fairness Index {jfi:.4f} below threshold 0.9"


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleEdgeCases:
    """Edge case tests for fixed schedule mode."""

    def test_single_session_single_turn(self, cli: AIPerfCLI, tmp_path: Path):
        """Test minimal case: single session, single turn."""
        config = FixedScheduleTestConfig(num_sessions=1, turns_per_session=1)
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == 1

    def test_sparse_schedule(self, cli: AIPerfCLI, tmp_path: Path):
        """Test sparse schedule with large gaps between sessions."""
        config = FixedScheduleTestConfig(
            num_sessions=5,
            schedule_duration_ms=400,  # 100ms gaps (still "sparse")
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions

    def test_dense_schedule(self, cli: AIPerfCLI, tmp_path: Path):
        """Test dense schedule with many sessions at similar timestamps."""
        config = FixedScheduleTestConfig(
            num_sessions=50,
            schedule_duration_ms=100,  # 2ms average gap
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions

    def test_single_worker(self, cli: AIPerfCLI, tmp_path: Path):
        """Test with single worker (all requests routed to same worker)."""
        config = FixedScheduleTestConfig(
            num_sessions=15,
            turns_per_session=3,
            workers_max=1,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

    def test_more_workers_than_sessions(self, cli: AIPerfCLI, tmp_path: Path):
        """Test with more workers than sessions."""
        config = FixedScheduleTestConfig(
            num_sessions=3,
            workers_max=10,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions


# =============================================================================
# Stress Tests
# =============================================================================


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestFixedScheduleStress:
    """Stress tests for fixed schedule mode."""

    def test_high_session_count(self, cli: AIPerfCLI, tmp_path: Path):
        """Test with high number of sessions."""
        config = FixedScheduleTestConfig(
            num_sessions=200,
            schedule_duration_ms=400,  # 2ms gaps, keep fast
            workers_max=5,
            timeout=60.0,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions

    def test_high_turn_count(self, cli: AIPerfCLI, tmp_path: Path):
        """Test with high number of turns per session."""
        config = FixedScheduleTestConfig(
            num_sessions=20,
            turns_per_session=25,
            delay_ms=1,
            workers_max=5,
            timeout=120.0,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

    def test_high_concurrency_burst(self, cli: AIPerfCLI, tmp_path: Path):
        """Test burst of requests with high concurrency."""
        config = FixedScheduleTestConfig(
            num_sessions=50,
            turns_per_session=2,
            delay_ms=0,
            concurrency=20,
            workers_max=5,
            timeout=120.0,
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests


# =============================================================================
# Schedule Adherence Verification Tests (100% Precision)
# =============================================================================


@pytest.mark.component_integration
class TestScheduleTimingPrecision:
    """Tests for absolute timestamp accuracy and schedule adherence.

    These tests measure ACTUAL vs EXPECTED timing to verify the system
    sends credits at the precise scheduled timestamps (within tolerance).
    """

    def test_absolute_timestamp_accuracy_first_turns(
        self, cli: AIPerfCLI, tmp_path: Path
    ):
        """Measure first turn timestamp accuracy vs schedule.

        Scenario:
        - 5 sessions at precise timestamps: 0, 100, 200, 300, 400 ms
        - Measure when credits actually sent
        - Verify within ±20ms of schedule (system scheduling tolerance)
        """
        stagger_ms = 100
        config = FixedScheduleTestConfig(
            num_sessions=5,
            schedule_duration_ms=400,  # 0, 100, 200, 300, 400
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=stagger_ms)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # Get first turn credits sorted by timestamp
        first_turn_payloads = [
            p for p in analyzer.credit_payloads if p.payload.turn_index == 0
        ]
        first_turn_payloads.sort(key=lambda p: p.timestamp_ns)

        # Get baseline (first credit sent time)
        baseline_ns = first_turn_payloads[0].timestamp_ns

        # Verify each subsequent first turn is at expected offset
        for idx, payload in enumerate(first_turn_payloads):
            expected_offset_ms = idx * stagger_ms
            actual_offset_ms = (payload.timestamp_ns - baseline_ns) / 1_000_000

            # Within ±20ms tolerance
            error_ms = abs(actual_offset_ms - expected_offset_ms)
            assert error_ms < 20, (
                f"Session {idx}: expected offset {expected_offset_ms}ms, "
                f"actual {actual_offset_ms:.2f}ms, error {error_ms:.2f}ms"
            )

    def test_delay_application_accuracy(self, cli: AIPerfCLI, tmp_path: Path):
        """Measure inter-turn delay accuracy.

        Scenario:
        - Multi-turn with 100ms delay between turns
        - Measure actual time from credit return to next credit issue
        - Verify within ±20ms tolerance
        """
        config = FixedScheduleTestConfig(
            num_sessions=5,
            turns_per_session=4,
            delay_ms=30,  # Short delay, still testable
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result

        # Get Credit and CreditReturn payloads by session

        credit_payloads = runner.payloads_by_type(Credit, sent=True)
        return_payloads = runner.payloads_by_type(CreditReturn, sent=True)

        # Build timeline per session
        for session_id in {p.payload.x_correlation_id for p in credit_payloads}:
            session_credits = [
                p for p in credit_payloads if p.payload.x_correlation_id == session_id
            ]
            session_returns = [
                p
                for p in return_payloads
                if p.payload.credit.x_correlation_id == session_id
            ]

            # Sort by turn index
            session_credits.sort(key=lambda p: p.payload.turn_index)
            session_returns.sort(key=lambda p: p.payload.credit.turn_index)

            # Measure delay between turn N return and turn N+1 credit
            for i in range(len(session_returns) - 1):
                return_time_ns = session_returns[i].timestamp_ns
                next_credit_time_ns = session_credits[i + 1].timestamp_ns

                actual_delay_ms = (next_credit_time_ns - return_time_ns) / 1_000_000

                # Should be approximately delay_ms (30ms)
                # Tolerance: ±20ms for system scheduling + processing
                error_ms = abs(actual_delay_ms - config.delay_ms)
                assert error_ms < 20, (
                    f"Session {session_id[:8]}, turn {i}→{i + 1}: "
                    f"expected {config.delay_ms}ms delay, actual {actual_delay_ms:.2f}ms, "
                    f"error {error_ms:.2f}ms"
                )


@pytest.mark.component_integration
class TestScheduleVariants:
    """Tests for different schedule patterns and formats.

    These tests verify the system handles various trace formats correctly.
    """

    def test_subsequent_turns_with_absolute_timestamps(
        self, cli: AIPerfCLI, tmp_path: Path
    ):
        """Test multi-turn where ALL turns have absolute timestamps (no delays).

        Trace format:
        {"session_id": "A", "timestamp": 0, "input_length": 100}
        {"session_id": "A", "timestamp": 150, "input_length": 100}
        {"session_id": "A", "timestamp": 400, "input_length": 100}
        """
        trace_file = tmp_path / "trace.jsonl"

        # Create trace with all timestamps (no delays)
        with open(trace_file, "w") as f:
            for session_idx in range(3):
                # 3 turns per session, all with absolute timestamps
                timestamps = [0, 150, 400]  # ms
                for _turn_idx, ts in enumerate(timestamps):
                    line = {
                        "session_id": f"session_{session_idx}",
                        "timestamp": session_idx * 100 + ts,  # Stagger sessions
                        "input_length": 100,
                    }
                    f.write(orjson.dumps(line).decode() + "\n")

        config = FixedScheduleTestConfig(num_sessions=3, turns_per_session=3)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 9  # 3 sessions × 3 turns

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # Verify all credits balanced
        assert analyzer.credits_balanced()

        # Verify 3 sessions
        assert analyzer.num_sessions == 3

        # Verify each session has 3 turns
        assert analyzer.session_credits_match(expected_turns=3)

    def test_mixed_timestamps_and_delays(self, cli: AIPerfCLI, tmp_path: Path):
        """Test conversation with mixed timestamp/delay pattern.

        Critical test: T1 timestamp, T2 delay, T3 timestamp, T4 delay
        """
        trace_file = tmp_path / "trace.jsonl"

        with open(trace_file, "w") as f:

            def write(d):
                f.write(orjson.dumps(d).decode() + "\n")

            # Session A: Mixed pattern
            write({"session_id": "A", "timestamp": 0, "input_length": 100})
            write({"session_id": "A", "delay": 50, "input_length": 100})
            write({"session_id": "A", "timestamp": 300, "input_length": 100})
            write({"session_id": "A", "delay": 80, "input_length": 100})
            # Session B: Another mixed pattern
            write({"session_id": "B", "timestamp": 100, "input_length": 100})
            write({"session_id": "B", "delay": 100, "input_length": 100})
            write({"session_id": "B", "timestamp": 500, "input_length": 100})

        config = FixedScheduleTestConfig(num_sessions=2)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=30.0)

        # Session A: 4 turns, Session B: 3 turns = 7 total
        assert get_request_count(result) == 7

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced()
        assert analyzer.num_sessions == 2

    def test_delay_only_subsequent_turn(self, cli: AIPerfCLI, tmp_path: Path):
        """Test subsequent turn with only delay field (no timestamp)."""
        trace_file = tmp_path / "trace.jsonl"
        expected_delay_ms = 30

        with open(trace_file, "w") as f:

            def write(d):
                f.write(orjson.dumps(d).decode() + "\n")

            write({"session_id": "X", "timestamp": 0, "input_length": 100})
            write({"session_id": "X", "delay": expected_delay_ms, "input_length": 100})

        config = FixedScheduleTestConfig(num_sessions=1, turns_per_session=2)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 2

        # Verify delay timing
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        credit_payloads = runner.payloads_by_type(Credit, sent=True)
        return_payloads = runner.payloads_by_type(CreditReturn, sent=True)

        credits = sorted(credit_payloads, key=lambda p: p.payload.turn_index)
        returns = sorted(return_payloads, key=lambda p: p.payload.credit.turn_index)

        actual_delay_ms = (
            credits[1].timestamp_ns - returns[0].timestamp_ns
        ) / 1_000_000
        assert abs(actual_delay_ms - expected_delay_ms) < 25, (
            f"Delay error: expected {expected_delay_ms}ms, got {actual_delay_ms:.1f}ms"
        )


@pytest.mark.component_integration
class TestScheduleOffsets:
    """Tests for schedule offset configurations.

    Schedule offsets control how trace timestamps are interpreted.
    """

    def test_auto_offset_uses_minimum_timestamp(self, cli: AIPerfCLI, tmp_path: Path):
        """Test auto-offset sets schedule zero to minimum timestamp.

        Scenario:
        - Timestamps: 500, 600, 700 ms
        - Auto-offset: schedule_zero = 500
        - Effective schedule: 0, 100, 200 ms (fast execution!)
        """
        trace_file = tmp_path / "trace.jsonl"

        with open(trace_file, "w") as f:

            def write(d):
                f.write(orjson.dumps(d).decode() + "\n")

            # All timestamps offset by 500ms (keep test fast!)
            for i in range(3):
                write(
                    {
                        "session_id": f"session_{i}",
                        "timestamp": 500 + i * 100,
                        "input_length": 100,
                    }
                )

        config = FixedScheduleTestConfig(num_sessions=3)
        cmd = build_fixed_schedule_command(config, trace_file, auto_offset=True)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 3

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # First credit should issue at relative t=0 (schedule_zero=5000)
        first_turn_times = analyzer.get_first_turn_issue_times_ns()
        baseline = min(first_turn_times)

        # Verify 100ms gaps
        for i, time_ns in enumerate(sorted(first_turn_times)):
            expected_offset_ms = i * 100
            actual_offset_ms = (time_ns - baseline) / 1_000_000

            error_ms = abs(actual_offset_ms - expected_offset_ms)
            assert error_ms < 20, f"Timing error {error_ms:.2f}ms for session {i}"

    def test_no_auto_offset_uses_absolute_timestamps(
        self, cli: AIPerfCLI, tmp_path: Path
    ):
        """Test without auto-offset, timestamps used as absolute values.

        Scenario:
        - Timestamps: 0, 50, 100 ms
        - No auto-offset
        - Schedule zero = 0, no adjustment
        """
        trace_file = tmp_path / "trace.jsonl"

        with open(trace_file, "w") as f:

            def write(d):
                f.write(orjson.dumps(d).decode() + "\n")

            for i in range(3):
                # 0, 50, 100 ms
                write(
                    {
                        "session_id": f"session_{i}",
                        "timestamp": i * 50,
                        "input_length": 100,
                    }
                )

        config = FixedScheduleTestConfig(num_sessions=3)
        cmd = build_fixed_schedule_command(config, trace_file, auto_offset=False)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 3

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        first_turn_times = analyzer.get_first_turn_issue_times_ns()
        min(first_turn_times)

        # Verify 50ms gaps
        sorted_times = sorted(first_turn_times)
        for i in range(1, len(sorted_times)):
            gap_ms = (sorted_times[i] - sorted_times[i - 1]) / 1_000_000
            assert abs(gap_ms - 50) < 15, (
                f"Gap {gap_ms:.2f}ms differs from expected 50ms"
            )


@pytest.mark.component_integration
class TestScheduleEdgeCases:
    """Advanced edge cases for schedule adherence."""

    def test_sub_millisecond_timestamps(self, cli: AIPerfCLI, tmp_path: Path):
        """Test float timestamps with sub-millisecond precision.

        Scenario:
        - Timestamps: 0.0, 100.5, 200.25, 300.75 ms
        - Verify ~100ms gaps between sessions
        """
        trace_file = tmp_path / "trace.jsonl"

        with open(trace_file, "w") as f:

            def write(d):
                f.write(orjson.dumps(d).decode() + "\n")

            timestamps = [0.0, 100.5, 200.25, 300.75]
            for i, ts in enumerate(timestamps):
                write(
                    {"session_id": f"session_{i}", "timestamp": ts, "input_length": 100}
                )

        config = FixedScheduleTestConfig(num_sessions=4)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 4

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced()

        # Verify ~100ms gaps between first turns
        first_turn_times = sorted(analyzer.get_first_turn_issue_times_ns())
        for i in range(1, len(first_turn_times)):
            gap_ms = (first_turn_times[i] - first_turn_times[i - 1]) / 1_000_000
            assert 80 < gap_ms < 120, f"Gap {i} was {gap_ms:.1f}ms, expected ~100ms"

    def test_zero_timestamp_start(self, cli: AIPerfCLI, tmp_path: Path):
        """Test schedule starting at timestamp=0 with 100ms gaps."""
        trace_file = tmp_path / "trace.jsonl"

        with open(trace_file, "w") as f:

            def write(d):
                f.write(orjson.dumps(d).decode() + "\n")

            for i in range(5):
                # 0, 100, 200, 300, 400 ms
                write(
                    {
                        "session_id": f"session_{i}",
                        "timestamp": i * 100,
                        "input_length": 100,
                    }
                )

        config = FixedScheduleTestConfig(num_sessions=5)
        cmd = build_fixed_schedule_command(config, trace_file, auto_offset=False)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 5

        # Verify 100ms gaps
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        first_turn_times = sorted(analyzer.get_first_turn_issue_times_ns())

        for i in range(1, len(first_turn_times)):
            gap_ms = (first_turn_times[i] - first_turn_times[i - 1]) / 1_000_000
            assert abs(gap_ms - 100) < 20, f"Gap {i} was {gap_ms:.1f}ms, expected 100ms"

    def test_concurrent_timestamps_across_workers(self, cli: AIPerfCLI, tmp_path: Path):
        """Test multiple sessions scheduled at same timestamp across workers.

        Scenario:
        - 10 sessions all at t=0
        - 3 workers
        - All should issue concurrently (within small window)
        - Each worker handles its assigned sessions
        """
        config = FixedScheduleTestConfig(
            num_sessions=10,
            workers_max=3,
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)  # All at t=0
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 10

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # All first turns should issue within 50ms window
        first_turn_times = analyzer.get_first_turn_issue_times_ns()
        max_spread_ns = max(first_turn_times) - min(first_turn_times)

        assert max_spread_ns < 50_000_000, (
            f"Concurrent start spread {max_spread_ns / 1e6:.2f}ms too large"
        )

    def test_variable_delays_per_turn(self, cli: AIPerfCLI, tmp_path: Path):
        """Test different delays for each subsequent turn.

        Scenario:
        - T0→T1: 20ms delay
        - T1→T2: 60ms delay
        - T2→T3: 40ms delay
        """
        trace_file = tmp_path / "trace.jsonl"
        expected_delays = [20, 60, 40]  # ms

        with open(trace_file, "w") as f:

            def write(d):
                f.write(orjson.dumps(d).decode() + "\n")

            write({"session_id": "VAR", "timestamp": 0, "input_length": 100})
            for delay in expected_delays:
                write({"session_id": "VAR", "delay": delay, "input_length": 100})

        config = FixedScheduleTestConfig(num_sessions=1, turns_per_session=4)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 4

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        credit_payloads = runner.payloads_by_type(Credit, sent=True)
        return_payloads = runner.payloads_by_type(CreditReturn, sent=True)

        credits = sorted(credit_payloads, key=lambda p: p.payload.turn_index)
        returns = sorted(return_payloads, key=lambda p: p.payload.credit.turn_index)

        # Verify each delay
        for i, expected_delay in enumerate(expected_delays):
            actual_delay_ms = (
                credits[i + 1].timestamp_ns - returns[i].timestamp_ns
            ) / 1_000_000
            error_ms = abs(actual_delay_ms - expected_delay)
            assert error_ms < 25, (
                f"Turn {i}→{i + 1}: expected {expected_delay}ms delay, got {actual_delay_ms:.1f}ms"
            )


@pytest.mark.component_integration
class TestScheduleWithConcurrencyBackpressure:
    """Tests for schedule adherence under concurrency pressure.

    When schedule is faster than concurrency allows, verify queueing and timing.
    """

    def test_schedule_faster_than_concurrency_allows(
        self, cli: AIPerfCLI, tmp_path: Path
    ):
        """Test schedule with rate exceeding concurrency capacity.

        Scenario:
        - 20 sessions at t=0 (all concurrent)
        - Concurrency=4
        - First 4 issue immediately
        - Remaining 16 queue
        - Verify adherence to concurrency limit
        """
        config = FixedScheduleTestConfig(
            num_sessions=20,
            concurrency=4,
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)  # All at t=0
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 20

        analyzer = ConcurrencyAnalyzer(result)

        max_concurrent = analyzer.get_max_concurrent()
        assert max_concurrent <= 4, f"Concurrency {max_concurrent} exceeded limit 4"
        assert max_concurrent == 4, (
            f"Should hit concurrency limit, got {max_concurrent}"
        )

    def test_staggered_schedule_respects_concurrency(
        self, cli: AIPerfCLI, tmp_path: Path
    ):
        """Test staggered schedule with tight concurrency.

        Scenario:
        - 30 sessions @ 10ms intervals = 300ms total schedule
        - Concurrency=5
        - Request duration ~55ms
        - Verify both schedule AND concurrency respected
        """
        config = FixedScheduleTestConfig(
            num_sessions=30,
            schedule_duration_ms=290,  # 10ms gaps
            concurrency=5,
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=10)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 30

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = ConcurrencyAnalyzer(result)

        # Concurrency limit respected
        assert analyzer.concurrency_within_limit(5)

        # Verify timing still reasonably accurate
        credit_analyzer = CreditFlowAnalyzer(runner)
        first_turn_times = credit_analyzer.get_first_turn_issue_times_ns()

        # Check gaps are approximately 10ms (with tolerance for concurrency queueing)
        sorted_times = sorted(first_turn_times)
        gaps_ms = [
            (sorted_times[i] - sorted_times[i - 1]) / 1_000_000
            for i in range(1, len(sorted_times))
        ]

        mean_gap_ms = sum(gaps_ms) / len(gaps_ms)
        # Mean should be close to 10ms, but concurrency may cause variation
        assert 8 <= mean_gap_ms <= 25, (
            f"Mean gap {mean_gap_ms:.2f}ms out of expected range"
        )
