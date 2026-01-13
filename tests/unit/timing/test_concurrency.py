# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for concurrency limiting components.

Tests DynamicConcurrencyLimit, GlobalPhaseConcurrencyLimiter, and ConcurrencyManager.
"""

import asyncio
import contextlib

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.timing.concurrency import (
    ConcurrencyManager,
    ConcurrencyStats,
    DynamicConcurrencyLimit,
    GlobalPhaseConcurrencyLimiter,
)
from aiperf.timing.config import CreditPhaseConfig


class TestDynamicConcurrencyLimitBasic:
    """Basic operations tests."""

    def test_initial_state(self) -> None:
        """Initial state has zero permits and zero debt."""
        limit = DynamicConcurrencyLimit()
        assert limit.current_limit == 0
        assert limit.debt == 0
        assert limit.effective_slots == 0

    def test_set_limit_from_zero(self) -> None:
        """Setting limit from zero adds permits."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(10)
        assert limit.current_limit == 10
        assert limit.effective_slots == 10
        assert limit.debt == 0

    def test_set_limit_negative_raises(self) -> None:
        """Negative limit raises ValueError."""
        limit = DynamicConcurrencyLimit()
        with pytest.raises(ValueError, match="non-negative"):
            limit.set_limit(-1)

    @pytest.mark.asyncio
    async def test_acquire_succeeds_with_permits(self) -> None:
        """Acquire succeeds immediately when permits available."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(1)

        await asyncio.wait_for(limit.acquire(), timeout=0.1)
        assert limit.effective_slots == 0

    @pytest.mark.asyncio
    async def test_acquire_blocks_without_permits(self) -> None:
        """Acquire blocks when no permits available."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(0)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limit.acquire(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_release_frees_permit(self) -> None:
        """Release adds permit back to semaphore."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(1)

        await limit.acquire()
        assert limit.effective_slots == 0

        limit.release()
        assert limit.effective_slots == 1

    @pytest.mark.asyncio
    async def test_acquire_release_cycle(self) -> None:
        """Multiple acquire/release cycles work correctly."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(2)

        await limit.acquire()
        await limit.acquire()
        assert limit.effective_slots == 0

        limit.release()
        limit.release()
        assert limit.effective_slots == 2


class TestDynamicConcurrencyLimitIncrease:
    """Tests for increasing the limit."""

    def test_increase_adds_permits(self) -> None:
        """Increasing limit adds permits."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(10)
        limit.set_limit(25)

        assert limit.current_limit == 25
        assert limit.effective_slots == 25
        assert limit.debt == 0

    @pytest.mark.asyncio
    async def test_increase_cancels_debt_fully(self) -> None:
        """Increase that exceeds debt cancels all debt and adds remaining."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)

        # Acquire all to make decrease create debt (not drain)
        for _ in range(50):
            await limit.acquire()

        limit.set_limit(25)  # debt = 25 (no available slots to drain)
        assert limit.debt == 25

        limit.set_limit(60)  # +35, cancels 25 debt, adds 10 permits
        assert limit.current_limit == 60
        assert limit.debt == 0
        assert limit.effective_slots == 10  # 10 new slots available

    @pytest.mark.asyncio
    async def test_increase_cancels_debt_partially(self) -> None:
        """Increase smaller than debt only cancels partial debt."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)

        # Acquire all to make decrease create debt
        for _ in range(50):
            await limit.acquire()

        limit.set_limit(25)  # debt = 25
        assert limit.debt == 25

        limit.set_limit(35)  # +10, cancels 10 debt
        assert limit.current_limit == 35
        assert limit.debt == 15

    @pytest.mark.asyncio
    async def test_increase_exactly_cancels_debt(self) -> None:
        """Increase equal to debt cancels exactly, adds no permits."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)

        # Acquire all to make decrease create debt
        for _ in range(50):
            await limit.acquire()

        limit.set_limit(25)  # debt = 25
        assert limit.debt == 25

        limit.set_limit(50)  # +25, cancels 25 debt exactly
        assert limit.current_limit == 50
        assert limit.debt == 0
        assert limit.effective_slots == 0  # still all acquired

    @pytest.mark.asyncio
    async def test_increase_wakes_waiters(self) -> None:
        """Increasing limit wakes blocked acquirers."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(0)

        acquired = []

        async def waiter(id: int) -> None:
            await limit.acquire()
            acquired.append(id)

        # Start waiters
        tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
        await asyncio.sleep(0.01)  # Let them block

        assert len(acquired) == 0

        # Increase limit - should wake waiters
        limit.set_limit(3)
        await asyncio.sleep(0.01)

        assert len(acquired) == 3
        for t in tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t


class TestDynamicConcurrencyLimitDecrease:
    """Tests for decreasing the limit."""

    def test_decrease_drains_available_slots(self) -> None:
        """Decreasing limit drains available slots immediately (no debt)."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)
        limit.set_limit(25)

        assert limit.current_limit == 25
        # Available slots are drained immediately - no debt!
        assert limit.debt == 0
        assert limit.effective_slots == 25

    def test_decrease_to_zero_drains_all(self) -> None:
        """Decreasing to zero drains all available slots."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(10)
        limit.set_limit(0)

        assert limit.current_limit == 0
        # All slots drained immediately - no debt
        assert limit.debt == 0
        assert limit.effective_slots == 0

    def test_multiple_decreases_drain_each_time(self) -> None:
        """Multiple decreases drain available slots each time."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(100)
        limit.set_limit(75)  # drains 25, no debt
        assert limit.debt == 0
        assert limit.effective_slots == 75

        limit.set_limit(50)  # drains 25 more, no debt
        assert limit.debt == 0
        assert limit.effective_slots == 50

    @pytest.mark.asyncio
    async def test_decrease_with_in_flight_creates_debt(self) -> None:
        """Decreasing with in-flight requests creates debt for excess."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)

        # Acquire all 50 slots (all in-flight)
        for _ in range(50):
            await limit.acquire()

        # Now decrease - can't drain (locked), so creates debt
        limit.set_limit(25)
        assert limit.debt == 25
        assert limit.effective_slots == 0

    @pytest.mark.asyncio
    async def test_decrease_partial_drain_partial_debt(self) -> None:
        """Decrease drains available first, remainder becomes debt."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)

        # Acquire 40, leaving 10 available
        for _ in range(40):
            await limit.acquire()
        assert limit.effective_slots == 10

        # Decrease by 25: drain 10 available, 15 becomes debt
        limit.set_limit(25)
        assert limit.debt == 15
        assert limit.effective_slots == 0

    @pytest.mark.asyncio
    async def test_release_absorbs_debt(self) -> None:
        """Releases are absorbed when debt exists (don't increase semaphore)."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)

        # Acquire all to create locked state
        for _ in range(50):
            await limit.acquire()

        # Decrease creates debt (no available slots to drain)
        limit.set_limit(25)
        assert limit.debt == 25

        # This release is absorbed (decreases debt, doesn't free slot)
        limit.release()
        assert limit.debt == 24

    @pytest.mark.asyncio
    async def test_releases_drain_debt_then_free(self) -> None:
        """After debt is drained, releases free permits normally."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(5)

        # Acquire all 5
        for _ in range(5):
            await limit.acquire()

        # Decrease creates debt (no available slots)
        limit.set_limit(3)
        assert limit.debt == 2
        assert limit.effective_slots == 0

        # First 2 releases absorbed (reduce debt)
        limit.release()
        limit.release()
        assert limit.debt == 0
        assert limit.effective_slots == 0

        # 3rd release actually frees a slot
        limit.release()
        assert limit.effective_slots == 1


class TestDynamicConcurrencyLimitTransitions:
    """Tests for realistic phase transition scenarios."""

    @pytest.mark.asyncio
    async def test_concurrency_ramp_up(self) -> None:
        """Simulates concurrency ramp: 10 → 25 → 50 → 100."""
        limit = DynamicConcurrencyLimit()

        # Phase 1: c=10
        limit.set_limit(10)
        assert limit.effective_slots == 10

        # Acquire all 10
        for _ in range(10):
            await limit.acquire()
        assert limit.effective_slots == 0

        # Phase 2: c=25 (while 10 in-flight)
        limit.set_limit(25)
        assert limit.effective_slots == 15  # 15 new permits

        # Phase 3: c=50
        limit.set_limit(50)
        assert limit.effective_slots == 40  # 25 more permits

        # Phase 4: c=100
        limit.set_limit(100)
        assert limit.effective_slots == 90  # 50 more permits

    @pytest.mark.asyncio
    async def test_seamless_transition_with_drain(self) -> None:
        """Simulates seamless phase transition where old phase drains."""
        limit = DynamicConcurrencyLimit()

        # Phase 1 at capacity
        limit.set_limit(10)
        for _ in range(10):
            await limit.acquire()

        # Transition to Phase 2
        limit.set_limit(25)
        assert limit.effective_slots == 15

        # Acquire new permits (Phase 2 starting)
        for _ in range(15):
            await limit.acquire()
        assert limit.effective_slots == 0

        # Phase 1 requests complete (release 10)
        for _ in range(10):
            limit.release()
        assert limit.effective_slots == 10

    @pytest.mark.asyncio
    async def test_decrease_during_in_flight(self) -> None:
        """Decreasing limit while requests in-flight drains gracefully."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)

        # 50 in-flight
        for _ in range(50):
            await limit.acquire()

        # Decrease to 25 (25 debt)
        limit.set_limit(25)
        assert limit.debt == 25

        # First 25 releases absorbed
        for _ in range(25):
            limit.release()
        assert limit.debt == 0
        assert limit.effective_slots == 0

        # Remaining 25 releases free permits
        for _ in range(25):
            limit.release()
        assert limit.effective_slots == 25

    def test_oscillating_limits_with_immediate_drain(self) -> None:
        """Handles oscillating limit changes with immediate drain."""
        limit = DynamicConcurrencyLimit()

        limit.set_limit(100)
        assert limit.effective_slots == 100

        # Decrease drains immediately (no in-flight)
        limit.set_limit(50)
        assert limit.debt == 0  # drained, not debt
        assert limit.effective_slots == 50

        # Increase adds slots
        limit.set_limit(75)
        assert limit.debt == 0
        assert limit.effective_slots == 75

        # Decrease drains again
        limit.set_limit(25)
        assert limit.debt == 0
        assert limit.effective_slots == 25

        # Increase adds slots
        limit.set_limit(100)
        assert limit.debt == 0
        assert limit.effective_slots == 100

    @pytest.mark.asyncio
    async def test_oscillating_limits_with_in_flight(self) -> None:
        """Handles oscillating limits with in-flight requests (debt scenario)."""
        limit = DynamicConcurrencyLimit()

        limit.set_limit(100)
        # Acquire all 100
        for _ in range(100):
            await limit.acquire()
        assert limit.effective_slots == 0

        # Decrease with all in-flight creates debt
        limit.set_limit(50)
        assert limit.debt == 50

        # Increase cancels debt
        limit.set_limit(75)
        assert limit.debt == 25  # 50 - 25 = 25 remaining

        # Decrease adds more debt
        limit.set_limit(25)
        assert limit.debt == 75  # 25 + 50 = 75

        # Increase cancels all debt
        limit.set_limit(100)
        assert limit.debt == 0


class TestDynamicConcurrencyLimitConcurrency:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_multiple_waiters_single_release(self) -> None:
        """Multiple waiters, single release wakes exactly one."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(0)

        acquired_count = 0

        async def waiter() -> None:
            nonlocal acquired_count
            await limit.acquire()
            acquired_count += 1

        # Start 5 waiters
        tasks = [asyncio.create_task(waiter()) for _ in range(5)]
        await asyncio.sleep(0.01)
        assert acquired_count == 0

        # Release 1 - should wake exactly 1
        limit.set_limit(1)
        await asyncio.sleep(0.01)
        assert acquired_count == 1

        # Release 2 more
        limit.set_limit(3)
        await asyncio.sleep(0.01)
        assert acquired_count == 3

        # Cleanup
        for t in tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

    @pytest.mark.asyncio
    async def test_rapid_acquire_release(self) -> None:
        """Rapid acquire/release cycles don't lose permits."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(10)

        async def worker(iterations: int) -> int:
            count = 0
            for _ in range(iterations):
                await limit.acquire()
                count += 1
                await asyncio.sleep(0)  # Yield
                limit.release()
            return count

        # Run 5 workers, 100 iterations each
        results = await asyncio.gather(*[worker(100) for _ in range(5)])

        assert sum(results) == 500
        assert limit.effective_slots == 10  # All permits returned

    @pytest.mark.asyncio
    async def test_limit_change_during_acquire(self) -> None:
        """Limit change while acquires are pending works correctly."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(2)

        # Acquire both permits
        await limit.acquire()
        await limit.acquire()

        acquired_after_increase = False

        async def waiter() -> None:
            nonlocal acquired_after_increase
            await limit.acquire()
            acquired_after_increase = True

        # Start waiter (will block)
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        assert not acquired_after_increase

        # Increase limit while waiter is pending
        limit.set_limit(3)
        await asyncio.sleep(0.01)
        assert acquired_after_increase

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestDynamicConcurrencyLimitEdgeCases:
    """Edge case tests."""

    def test_set_same_limit(self) -> None:
        """Setting same limit is a no-op."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(10)

        limit.set_limit(10)

        assert limit.current_limit == 10
        assert limit.debt == 0
        assert limit.effective_slots == 10

    @pytest.mark.asyncio
    async def test_set_same_limit_with_debt(self) -> None:
        """Setting same limit with existing debt is a no-op."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)
        # Acquire all slots to create debt scenario on decrease
        for _ in range(50):
            await limit.acquire()

        limit.set_limit(25)  # Creates debt since all slots in-flight
        assert limit.debt == 25

        limit.set_limit(25)  # Same limit - no-op
        assert limit.debt == 25  # Unchanged

    def test_release_without_acquire(self) -> None:
        """Release without acquire adds permit (semaphore behavior)."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(10)

        limit.release()

        assert limit.effective_slots == 11

    @pytest.mark.asyncio
    async def test_large_debt_small_increase(self) -> None:
        """Small increase with large debt only reduces debt."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(1000)
        # Acquire all slots to create debt scenario
        for _ in range(1000):
            await limit.acquire()

        limit.set_limit(0)  # debt = 1000 (all in-flight)
        assert limit.debt == 1000

        limit.set_limit(1)  # Only cancels 1 debt
        assert limit.debt == 999
        assert limit.effective_slots == 0  # all still in-flight

    @pytest.mark.asyncio
    async def test_zero_limit_blocks_all(self) -> None:
        """Zero limit blocks all acquires."""
        limit = DynamicConcurrencyLimit(0)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limit.acquire(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_debt_exactly_equals_releases(self) -> None:
        """When releases exactly equal debt, debt is cleared."""
        limit = DynamicConcurrencyLimit(10)
        # Acquire all 10 slots to simulate in-flight
        for _ in range(10):
            await limit.acquire()

        limit.set_limit(5)  # debt = 5 (all in-flight, none to drain)
        assert limit.debt == 5
        assert limit.effective_slots == 0  # all acquired

        # 5 releases absorbed by debt
        for _ in range(5):
            limit.release()

        assert limit.debt == 0
        assert limit.effective_slots == 0  # still 5 in-flight


class TestNoOvershootGuarantee:
    """Tests for the hybrid decrease mechanism's immediate enforcement.

    When decreasing limits with available slots, the hybrid approach drains
    those slots immediately rather than using debt. This ensures the new
    limit is enforced instantly.
    """

    @pytest.mark.asyncio
    async def test_decrease_immediately_enforces_limit(self) -> None:
        """Decreasing with available slots enforces limit immediately."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)
        assert limit.effective_slots == 50

        # Decrease to 25 - enforced immediately
        limit.set_limit(25)

        # Only 25 slots available
        assert limit.effective_slots == 25
        assert limit.debt == 0  # Slots drained, no debt needed

        # Verify we can only acquire exactly 25 slots
        for _i in range(25):
            await limit.acquire()
        assert limit.effective_slots == 0

        # 26th acquire should block
        task = asyncio.create_task(limit.acquire())
        await asyncio.sleep(0.01)
        assert not task.done(), "26th acquire should block"

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_partial_decrease_drains_then_debts(self) -> None:
        """Partial drain + debt: some slots available, some in-flight."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(100)

        # Acquire 60 slots (40 available)
        for _ in range(60):
            await limit.acquire()
        assert limit.effective_slots == 40

        # Decrease to 25: need to reduce by 75
        # - Drain 40 available slots immediately
        # - Remaining 35 becomes debt (for the 60 in-flight, only want 25)
        limit.set_limit(25)

        assert limit.effective_slots == 0  # All available drained
        assert limit.debt == 35  # 75 - 40 = 35

        # No more acquires possible until releases
        task = asyncio.create_task(limit.acquire())
        await asyncio.sleep(0.01)
        assert not task.done(), "Should block - no available slots"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_rapid_decrease_enforces_each_step(self) -> None:
        """Each decrease step enforces immediately, no cumulative overshoot."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(100)
        assert limit.effective_slots == 100

        # Rapid decreases - each should enforce immediately
        limit.set_limit(75)
        assert limit.effective_slots == 75
        assert limit.debt == 0

        limit.set_limit(50)
        assert limit.effective_slots == 50
        assert limit.debt == 0

        limit.set_limit(25)
        assert limit.effective_slots == 25
        assert limit.debt == 0

        limit.set_limit(10)
        assert limit.effective_slots == 10
        assert limit.debt == 0

        # Only 10 acquires possible
        for _ in range(10):
            await limit.acquire()

        task = asyncio.create_task(limit.acquire())
        await asyncio.sleep(0.01)
        assert not task.done(), "11th acquire should block"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def test_semaphore_value_drained_on_decrease(self) -> None:
        """Semaphore._value is drained directly when decreasing with available slots."""
        limit = DynamicConcurrencyLimit()
        limit.set_limit(50)

        limit.set_limit(25)

        # Semaphore value should be drained to 25, not left at 50 with debt
        assert limit._semaphore._value == 25
        assert limit.debt == 0
        assert limit.effective_slots == 25


# =============================================================================
# GlobalPhaseConcurrencyLimiter Tests
# =============================================================================


class TestGlobalPhaseConcurrencyLimiterBasic:
    """Basic operations tests for GlobalPhaseConcurrencyLimiter."""

    def test_initial_state_disabled(self) -> None:
        """Initial state should be disabled."""
        limiter = GlobalPhaseConcurrencyLimiter()
        assert not limiter.enabled

    def test_configure_for_phase_enables_limiter(self) -> None:
        """Configuring with a limit should enable the limiter."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.PROFILING, 10)
        assert limiter.enabled

    def test_configure_with_none_disables_limiter(self) -> None:
        """Configuring with None should disable the limiter."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.PROFILING, 10)
        assert limiter.enabled

        limiter.configure_for_phase(CreditPhase.WARMUP, None)
        assert not limiter.enabled

    @pytest.mark.asyncio
    async def test_acquire_requires_configured_phase(self) -> None:
        """Acquire should raise for unconfigured phase."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.PROFILING, 10)

        with pytest.raises(ValueError, match="not configured"):
            await limiter.acquire(CreditPhase.WARMUP, lambda: True)

    @pytest.mark.asyncio
    async def test_acquire_succeeds_with_slots(self) -> None:
        """Acquire should succeed when slots available."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.PROFILING, 5)

        result = await limiter.acquire(CreditPhase.PROFILING, lambda: True)

        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_returns_false_when_can_proceed_false(self) -> None:
        """Acquire should return False when can_proceed_fn returns False."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.PROFILING, 5)

        result = await limiter.acquire(CreditPhase.PROFILING, lambda: False)

        assert result is False

    def test_release_requires_configured_phase(self) -> None:
        """Release should raise for unconfigured phase."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.PROFILING, 10)

        with pytest.raises(ValueError, match="not configured"):
            limiter.release(CreditPhase.WARMUP)


class TestGlobalPhaseConcurrencyLimiterMultiPhase:
    """Tests for multi-phase scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_phases_independent(self) -> None:
        """Each phase should have independent limits."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.WARMUP, 5)
        limiter.configure_for_phase(CreditPhase.PROFILING, 10)

        # Acquire all WARMUP slots
        for _ in range(5):
            await limiter.acquire(CreditPhase.WARMUP, lambda: True)

        # PROFILING should still have slots available
        result = await limiter.acquire(CreditPhase.PROFILING, lambda: True)
        assert result is True

    @pytest.mark.asyncio
    async def test_held_slots_tracking(self) -> None:
        """get_held_slots should track acquired slots per phase."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.PROFILING, 10)

        assert limiter.get_held_slots(CreditPhase.PROFILING) == 0

        await limiter.acquire(CreditPhase.PROFILING, lambda: True)
        await limiter.acquire(CreditPhase.PROFILING, lambda: True)

        assert limiter.get_held_slots(CreditPhase.PROFILING) == 2

        limiter.release(CreditPhase.PROFILING)

        assert limiter.get_held_slots(CreditPhase.PROFILING) == 1

    def test_unconfigured_phase_held_slots_zero(self) -> None:
        """get_held_slots should return 0 for unconfigured phase."""
        limiter = GlobalPhaseConcurrencyLimiter()
        assert limiter.get_held_slots(CreditPhase.PROFILING) == 0


class TestGlobalPhaseConcurrencyLimiterStats:
    """Tests for stats tracking."""

    @pytest.mark.asyncio
    async def test_global_stats_track_operations(self) -> None:
        """Global stats should track acquire/release counts."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.PROFILING, 10)

        await limiter.acquire(CreditPhase.PROFILING, lambda: True)
        await limiter.acquire(CreditPhase.PROFILING, lambda: True)
        limiter.release(CreditPhase.PROFILING)

        stats = limiter.global_stats
        assert stats.acquire_count == 2
        assert stats.release_count == 1

    @pytest.mark.asyncio
    async def test_phase_stats_track_operations(self) -> None:
        """Phase stats should track operations per phase."""
        limiter = GlobalPhaseConcurrencyLimiter()
        limiter.configure_for_phase(CreditPhase.PROFILING, 10)

        await limiter.acquire(CreditPhase.PROFILING, lambda: True)
        limiter.release(CreditPhase.PROFILING)

        stats = limiter.get_phase_stats(CreditPhase.PROFILING)
        assert stats is not None
        assert stats.acquire_count == 1
        assert stats.release_count == 1

    def test_unconfigured_phase_stats_none(self) -> None:
        """get_phase_stats should return None for unconfigured phase."""
        limiter = GlobalPhaseConcurrencyLimiter()
        assert limiter.get_phase_stats(CreditPhase.PROFILING) is None


# =============================================================================
# ConcurrencyManager Tests
# =============================================================================


def make_phase_config(
    phase: CreditPhase = CreditPhase.PROFILING,
    concurrency: int | None = None,
    prefill_concurrency: int | None = None,
) -> CreditPhaseConfig:
    """Create CreditPhaseConfig for testing."""
    return CreditPhaseConfig(
        phase=phase,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
        concurrency=concurrency,
        prefill_concurrency=prefill_concurrency,
    )


class TestConcurrencyManagerBasic:
    """Basic operations tests for ConcurrencyManager."""

    def test_initial_state_disabled(self) -> None:
        """Both limiters should be disabled initially."""
        manager = ConcurrencyManager()
        assert manager._session_limiter.enabled is False
        assert manager._prefill_limiter.enabled is False

    def test_configure_enables_session_limiter(self) -> None:
        """Configuring with concurrency should enable session limiter."""
        manager = ConcurrencyManager()
        config = make_phase_config(concurrency=10)

        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        assert manager._session_limiter.enabled is True
        assert manager._prefill_limiter.enabled is False

    def test_configure_enables_prefill_limiter(self) -> None:
        """Configuring with prefill_concurrency should enable prefill limiter."""
        manager = ConcurrencyManager()
        config = make_phase_config(prefill_concurrency=5)

        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        assert manager._session_limiter.enabled is False
        assert manager._prefill_limiter.enabled is True

    def test_configure_enables_both_limiters(self) -> None:
        """Configuring with both should enable both limiters."""
        manager = ConcurrencyManager()
        config = make_phase_config(concurrency=10, prefill_concurrency=5)

        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        assert manager._session_limiter.enabled is True
        assert manager._prefill_limiter.enabled is True


class TestConcurrencyManagerSessionSlots:
    """Tests for session slot management."""

    @pytest.mark.asyncio
    async def test_acquire_session_slot_disabled_calls_check(self) -> None:
        """When disabled, should still call can_proceed_fn."""
        manager = ConcurrencyManager()
        config = make_phase_config()  # No concurrency
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        check_called = False

        def check() -> bool:
            nonlocal check_called
            check_called = True
            return True

        result = await manager.acquire_session_slot(CreditPhase.PROFILING, check)

        assert result is True
        assert check_called is True

    @pytest.mark.asyncio
    async def test_acquire_session_slot_enabled(self) -> None:
        """When enabled, should acquire slot."""
        manager = ConcurrencyManager()
        config = make_phase_config(concurrency=5)
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        result = await manager.acquire_session_slot(CreditPhase.PROFILING, lambda: True)

        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_session_slot_returns_false_on_check_fail(self) -> None:
        """Should return False when can_proceed_fn returns False."""
        manager = ConcurrencyManager()
        config = make_phase_config(concurrency=5)
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        result = await manager.acquire_session_slot(
            CreditPhase.PROFILING, lambda: False
        )

        assert result is False

    def test_release_session_slot_disabled_noop(self) -> None:
        """Release should be no-op when disabled."""
        manager = ConcurrencyManager()
        config = make_phase_config()  # No concurrency
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        # Should not raise
        manager.release_session_slot(CreditPhase.PROFILING)

    @pytest.mark.asyncio
    async def test_release_session_slot_enabled(self) -> None:
        """Release should free slot when enabled."""
        manager = ConcurrencyManager()
        config = make_phase_config(concurrency=1)
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        await manager.acquire_session_slot(CreditPhase.PROFILING, lambda: True)

        # Second acquire should block
        task = asyncio.create_task(
            manager.acquire_session_slot(CreditPhase.PROFILING, lambda: True)
        )
        await asyncio.sleep(0.01)
        assert not task.done()

        # Release should unblock
        manager.release_session_slot(CreditPhase.PROFILING)
        await asyncio.sleep(0.01)
        assert task.done()

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestConcurrencyManagerPrefillSlots:
    """Tests for prefill slot management."""

    @pytest.mark.asyncio
    async def test_acquire_prefill_slot_disabled_calls_check(self) -> None:
        """When disabled, should still call can_proceed_fn."""
        manager = ConcurrencyManager()
        config = make_phase_config()  # No prefill_concurrency
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        check_called = False

        def check() -> bool:
            nonlocal check_called
            check_called = True
            return True

        result = await manager.acquire_prefill_slot(CreditPhase.PROFILING, check)

        assert result is True
        assert check_called is True

    @pytest.mark.asyncio
    async def test_acquire_prefill_slot_enabled(self) -> None:
        """When enabled, should acquire slot."""
        manager = ConcurrencyManager()
        config = make_phase_config(prefill_concurrency=5)
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        result = await manager.acquire_prefill_slot(CreditPhase.PROFILING, lambda: True)

        assert result is True

    def test_release_prefill_slot_disabled_noop(self) -> None:
        """Release should be no-op when disabled."""
        manager = ConcurrencyManager()
        config = make_phase_config()  # No prefill_concurrency
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        # Should not raise
        manager.release_prefill_slot(CreditPhase.PROFILING)


class TestConcurrencyManagerStuckSlots:
    """Tests for releasing stuck slots."""

    @pytest.mark.asyncio
    async def test_release_stuck_slots_returns_counts(self) -> None:
        """release_stuck_slots should return correct counts."""
        manager = ConcurrencyManager()
        config = make_phase_config(concurrency=10, prefill_concurrency=5)
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        # Acquire some slots
        for _ in range(3):
            await manager.acquire_session_slot(CreditPhase.PROFILING, lambda: True)
        for _ in range(2):
            await manager.acquire_prefill_slot(CreditPhase.PROFILING, lambda: True)

        session_released, prefill_released = manager.release_stuck_slots(
            CreditPhase.PROFILING
        )

        assert session_released == 3
        assert prefill_released == 2

    @pytest.mark.asyncio
    async def test_release_stuck_slots_disabled_returns_zero(self) -> None:
        """release_stuck_slots should return (0, 0) when disabled."""
        manager = ConcurrencyManager()
        config = make_phase_config()  # No concurrency
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        session_released, prefill_released = manager.release_stuck_slots(
            CreditPhase.PROFILING
        )

        assert session_released == 0
        assert prefill_released == 0


class TestConcurrencyManagerStats:
    """Tests for stats retrieval."""

    def test_get_session_stats_disabled_returns_none(self) -> None:
        """Should return None when session limiter disabled."""
        manager = ConcurrencyManager()
        config = make_phase_config()  # No concurrency
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        assert manager.get_session_stats() is None
        assert manager.get_session_stats(CreditPhase.PROFILING) is None

    @pytest.mark.asyncio
    async def test_get_session_stats_enabled(self) -> None:
        """Should return stats when session limiter enabled."""
        manager = ConcurrencyManager()
        config = make_phase_config(concurrency=10)
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        await manager.acquire_session_slot(CreditPhase.PROFILING, lambda: True)

        global_stats = manager.get_session_stats()
        phase_stats = manager.get_session_stats(CreditPhase.PROFILING)

        assert global_stats is not None
        assert global_stats.acquire_count == 1
        assert phase_stats is not None
        assert phase_stats.acquire_count == 1


class TestConcurrencyManagerDynamicLimits:
    """Tests for dynamic limit adjustment."""

    @pytest.mark.asyncio
    async def test_set_session_limit_updates_limits(self) -> None:
        """set_session_limit should update both global and phase limits."""
        manager = ConcurrencyManager()
        config = make_phase_config(concurrency=5)
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        # Acquire all 5
        for _ in range(5):
            await manager.acquire_session_slot(CreditPhase.PROFILING, lambda: True)

        # Increase limit
        manager.set_session_limit(CreditPhase.PROFILING, 10)

        # Should now be able to acquire more
        result = await manager.acquire_session_slot(CreditPhase.PROFILING, lambda: True)
        assert result is True

    def test_set_session_limit_disabled_noop(self) -> None:
        """set_session_limit should be no-op when disabled."""
        manager = ConcurrencyManager()
        config = make_phase_config()  # No concurrency
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        # Should not raise
        manager.set_session_limit(CreditPhase.PROFILING, 10)

    @pytest.mark.asyncio
    async def test_set_prefill_limit_updates_limits(self) -> None:
        """set_prefill_limit should update both global and phase limits."""
        manager = ConcurrencyManager()
        config = make_phase_config(prefill_concurrency=3)
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        # Acquire all 3
        for _ in range(3):
            await manager.acquire_prefill_slot(CreditPhase.PROFILING, lambda: True)

        # Increase limit
        manager.set_prefill_limit(CreditPhase.PROFILING, 5)

        # Should now be able to acquire more
        result = await manager.acquire_prefill_slot(CreditPhase.PROFILING, lambda: True)
        assert result is True

    def test_set_prefill_limit_disabled_noop(self) -> None:
        """set_prefill_limit should be no-op when disabled."""
        manager = ConcurrencyManager()
        config = make_phase_config()  # No prefill_concurrency
        manager.configure_for_phase(
            config.phase, config.concurrency, config.prefill_concurrency
        )

        # Should not raise
        manager.set_prefill_limit(CreditPhase.PROFILING, 10)


class TestConcurrencyStatsDataclass:
    """Tests for ConcurrencyStats dataclass."""

    def test_default_values(self) -> None:
        """Default values should be zero."""
        stats = ConcurrencyStats()
        assert stats.acquire_count == 0
        assert stats.release_count == 0
        assert stats.wait_count == 0

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        stats = ConcurrencyStats(acquire_count=10, release_count=5, wait_count=2)
        assert stats.acquire_count == 10
        assert stats.release_count == 5
        assert stats.wait_count == 2
