# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CreditCallbackHandler.

Tests credit lifecycle callbacks from CreditRouter.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_concurrency():
    """Mock concurrency manager."""
    mock = MagicMock()
    mock.release_session_slot = MagicMock()
    mock.release_prefill_slot = MagicMock()
    return mock


@pytest.fixture
def mock_progress():
    """Mock progress tracker."""
    mock = MagicMock()
    mock.increment_returned = MagicMock(return_value=False)  # Not final return
    mock.increment_prefill_released = MagicMock()
    mock.all_credits_returned_event = asyncio.Event()
    mock.in_flight_sessions = 0
    return mock


@pytest.fixture
def mock_lifecycle():
    """Mock phase lifecycle."""
    mock = MagicMock()
    mock.is_complete = False
    return mock


@pytest.fixture
def mock_stop_checker():
    """Mock stop condition checker."""
    mock = MagicMock()
    mock.can_send_any_turn = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_strategy():
    """Mock timing strategy."""
    mock = MagicMock()
    mock.handle_credit_return = AsyncMock()
    return mock


@pytest.fixture
def callback_handler(mock_concurrency):
    """Create CreditCallbackHandler."""
    return CreditCallbackHandler(mock_concurrency)


@pytest.fixture
def registered_handler(
    callback_handler,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
):
    """Create CreditCallbackHandler with phase registered."""
    callback_handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    return callback_handler


def make_credit(
    credit_id: int = 1,
    conversation_id: str = "conv1",
    turn_index: int = 0,
    num_turns: int = 1,
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    """Create a Credit for testing."""
    return Credit(
        id=credit_id,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=f"corr-{conversation_id}",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
    )


def make_credit_return(
    credit: Credit,
    cancelled: bool = False,
    first_token_sent: bool = True,
) -> CreditReturn:
    """Create a CreditReturn for testing."""
    return CreditReturn(
        credit=credit,
        cancelled=cancelled,
        first_token_sent=first_token_sent,
    )


# =============================================================================
# Test: Phase Registration
# =============================================================================


class TestPhaseRegistration:
    """Tests for phase registration and unregistration."""

    def test_register_phase_stores_context(self, callback_handler):
        """Register phase should store callback context."""
        progress = MagicMock()
        progress.all_credits_returned_event = asyncio.Event()
        lifecycle = MagicMock()
        stop_checker = MagicMock()
        strategy = MagicMock()

        callback_handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=progress,
            lifecycle=lifecycle,
            stop_checker=stop_checker,
            strategy=strategy,
        )

        assert CreditPhase.PROFILING in callback_handler._phase_handlers

    def test_unregister_phase_removes_context(self, registered_handler):
        """Unregister phase should remove callback context."""
        registered_handler.unregister_phase(CreditPhase.PROFILING)

        assert CreditPhase.PROFILING not in registered_handler._phase_handlers

    def test_unregister_nonexistent_phase_is_safe(self, callback_handler):
        """Unregistering a non-existent phase should not raise."""
        callback_handler.unregister_phase(CreditPhase.WARMUP)  # Not registered

    def test_can_register_multiple_phases(self, callback_handler):
        """Can register multiple phases simultaneously."""
        for phase in [CreditPhase.WARMUP, CreditPhase.PROFILING]:
            callback_handler.register_phase(
                phase=phase,
                progress=MagicMock(all_credits_returned_event=asyncio.Event()),
                lifecycle=MagicMock(is_complete=False),
                stop_checker=MagicMock(),
                strategy=MagicMock(),
            )

        assert CreditPhase.WARMUP in callback_handler._phase_handlers
        assert CreditPhase.PROFILING in callback_handler._phase_handlers


# =============================================================================
# Test: Credit Return - Basic Flow
# =============================================================================


class TestCreditReturnBasicFlow:
    """Tests for basic credit return handling."""

    async def test_on_credit_return_increments_returned_count(
        self, registered_handler, mock_progress
    ):
        """Credit return should increment returned count."""
        credit = make_credit()
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_progress.increment_returned.assert_called_once_with(
            credit.is_final_turn,
            False,  # cancelled=False
        )

    async def test_on_credit_return_tracks_cancelled_status(
        self, registered_handler, mock_progress
    ):
        """Credit return should track cancelled status."""
        credit = make_credit()
        credit_return = make_credit_return(credit, cancelled=True)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_progress.increment_returned.assert_called_once_with(
            credit.is_final_turn,
            True,  # cancelled=True
        )

    async def test_on_credit_return_releases_session_slot_on_final_turn(
        self, registered_handler, mock_concurrency
    ):
        """Should release session slot when final turn returns."""
        credit = make_credit(turn_index=2, num_turns=3)  # Final turn
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_concurrency.release_session_slot.assert_called_once_with(
            CreditPhase.PROFILING
        )

    async def test_on_credit_return_does_not_release_session_on_non_final_turn(
        self, registered_handler, mock_concurrency
    ):
        """Should NOT release session slot on non-final turn."""
        credit = make_credit(turn_index=0, num_turns=3)  # Not final
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_concurrency.release_session_slot.assert_not_called()


# =============================================================================
# Test: Credit Return - TTFT Handling
# =============================================================================


class TestCreditReturnTTFTHandling:
    """Tests for TTFT-related handling in credit returns."""

    async def test_tracks_prefill_release_when_ttft_never_arrived(
        self, registered_handler, mock_progress
    ):
        """Should track prefill release when first_token_sent is False."""
        credit = make_credit()
        credit_return = make_credit_return(credit, first_token_sent=False)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_progress.increment_prefill_released.assert_called_once()

    async def test_does_not_track_prefill_release_when_ttft_arrived(
        self, registered_handler, mock_progress
    ):
        """Should NOT track prefill release when first_token_sent is True."""
        credit = make_credit()
        credit_return = make_credit_return(credit, first_token_sent=True)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_progress.increment_prefill_released.assert_not_called()

    async def test_releases_prefill_slot_when_ttft_never_arrived(
        self, registered_handler, mock_concurrency
    ):
        """Should release prefill slot when first_token_sent is False."""
        credit = make_credit()
        credit_return = make_credit_return(credit, first_token_sent=False)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_concurrency.release_prefill_slot.assert_called_once_with(
            CreditPhase.PROFILING
        )

    async def test_does_not_release_prefill_slot_when_ttft_arrived(
        self, registered_handler, mock_concurrency
    ):
        """Should NOT release prefill slot when first_token_sent is True."""
        credit = make_credit()
        credit_return = make_credit_return(credit, first_token_sent=True)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_concurrency.release_prefill_slot.assert_not_called()


# =============================================================================
# Test: Credit Return - Final Return Handling
# =============================================================================


class TestCreditReturnFinalHandling:
    """Tests for final return handling."""

    async def test_sets_event_on_final_return(self, registered_handler, mock_progress):
        """Should set all_credits_returned_event on final return."""
        mock_progress.increment_returned.return_value = True  # Final return
        credit = make_credit()
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        assert mock_progress.all_credits_returned_event.is_set()

    async def test_does_not_set_event_on_non_final_return(
        self, registered_handler, mock_progress
    ):
        """Should NOT set event on non-final return."""
        mock_progress.increment_returned.return_value = False  # Not final
        credit = make_credit()
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        assert not mock_progress.all_credits_returned_event.is_set()

    async def test_releases_in_flight_session_slots_on_final_return(
        self,
        callback_handler,
        mock_concurrency,
    ):
        """Should release in-flight session slots on final return."""
        # Create progress with 2 in-flight sessions
        progress = MagicMock()
        progress.all_credits_returned_event = asyncio.Event()
        progress.increment_returned = MagicMock(return_value=True)  # Final return
        progress.increment_prefill_released = MagicMock()
        progress.in_flight_sessions = 2

        callback_handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=progress,
            lifecycle=MagicMock(is_complete=False),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=False)),
            strategy=MagicMock(handle_credit_return=AsyncMock()),
        )

        credit = make_credit(turn_index=0, num_turns=1)  # Final turn
        credit_return = make_credit_return(credit)

        await callback_handler.on_credit_return("worker-1", credit_return)

        # Should release 2 in-flight session slots + 1 for final turn
        assert mock_concurrency.release_session_slot.call_count == 3


# =============================================================================
# Test: Credit Return - Next Turn Dispatch
# =============================================================================


class TestNextTurnDispatch:
    """Tests for next turn dispatch via strategy."""

    async def test_dispatches_next_turn_when_not_final(
        self, registered_handler, mock_strategy
    ):
        """Should dispatch next turn when credit is not final turn."""
        credit = make_credit(turn_index=0, num_turns=3)  # Not final
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_strategy.handle_credit_return.assert_called_once_with(credit)

    async def test_dispatches_for_final_turn_when_can_send(
        self, registered_handler, mock_strategy
    ):
        """Should dispatch to strategy for final turn when can_send_any_turn is True.

        Strategy's handle_credit_return returns early for final turns.
        """
        credit = make_credit(turn_index=2, num_turns=3)  # Final turn
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        # Called because can_send_any_turn() is True - strategy handles final turn internally
        mock_strategy.handle_credit_return.assert_called_once_with(credit)

    async def test_does_not_dispatch_when_stop_condition_reached(
        self, registered_handler, mock_strategy, mock_stop_checker
    ):
        """Should NOT dispatch next turn when stop condition reached."""
        mock_stop_checker.can_send_any_turn.return_value = False
        credit = make_credit(turn_index=0, num_turns=3)  # Not final
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_strategy.handle_credit_return.assert_not_called()


# =============================================================================
# Test: Credit Return - Unregistered Phase
# =============================================================================


class TestUnregisteredPhaseHandling:
    """Tests for handling credits from unregistered phases."""

    async def test_ignores_credit_return_for_unregistered_phase(self, callback_handler):
        """Should silently ignore returns for unregistered phases."""
        credit = make_credit(phase=CreditPhase.WARMUP)  # Not registered
        credit_return = make_credit_return(credit)

        # Should not raise
        await callback_handler.on_credit_return("worker-1", credit_return)

    async def test_ignores_credit_return_after_phase_unregistered(
        self, registered_handler
    ):
        """Should ignore returns after phase is unregistered."""
        registered_handler.unregister_phase(CreditPhase.PROFILING)

        credit = make_credit()
        credit_return = make_credit_return(credit)

        # Should not raise
        await registered_handler.on_credit_return("worker-1", credit_return)


# =============================================================================
# Test: Credit Return - Phase Complete
# =============================================================================


class TestPhaseCompleteHandling:
    """Tests for handling credits after phase is complete."""

    async def test_ignores_credit_return_after_phase_complete(
        self, registered_handler, mock_lifecycle, mock_progress
    ):
        """Should ignore late returns after phase is complete."""
        mock_lifecycle.is_complete = True
        credit = make_credit()
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        # Should not increment counts
        mock_progress.increment_returned.assert_not_called()


# =============================================================================
# Test: First Token (TTFT) Handling
# =============================================================================


class TestFirstTokenHandling:
    """Tests for TTFT event handling."""

    async def test_on_first_token_tracks_prefill_release(
        self, registered_handler, mock_progress
    ):
        """Should track prefill release on TTFT."""
        first_token = FirstToken(
            credit_id=1,
            phase=CreditPhase.PROFILING,
            ttft_ns=1000000,
        )

        await registered_handler.on_first_token(first_token)

        mock_progress.increment_prefill_released.assert_called_once()

    async def test_on_first_token_releases_prefill_slot(
        self, registered_handler, mock_concurrency
    ):
        """Should release prefill slot on TTFT."""
        first_token = FirstToken(
            credit_id=1,
            phase=CreditPhase.PROFILING,
            ttft_ns=1000000,
        )

        await registered_handler.on_first_token(first_token)

        mock_concurrency.release_prefill_slot.assert_called_once_with(
            CreditPhase.PROFILING
        )

    async def test_ignores_first_token_for_unregistered_phase(self, callback_handler):
        """Should ignore TTFT for unregistered phase."""
        first_token = FirstToken(
            credit_id=1,
            phase=CreditPhase.WARMUP,  # Not registered
            ttft_ns=1000000,
        )

        # Should not raise
        await callback_handler.on_first_token(first_token)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_handles_warmup_phase(self, callback_handler):
        """Should handle WARMUP phase correctly."""
        progress = MagicMock()
        progress.all_credits_returned_event = asyncio.Event()
        progress.increment_returned = MagicMock(return_value=False)
        progress.increment_prefill_released = MagicMock()
        progress.in_flight_sessions = 0

        callback_handler.register_phase(
            phase=CreditPhase.WARMUP,
            progress=progress,
            lifecycle=MagicMock(is_complete=False),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
            strategy=MagicMock(handle_credit_return=AsyncMock()),
        )

        credit = make_credit(phase=CreditPhase.WARMUP)
        credit_return = make_credit_return(credit)

        await callback_handler.on_credit_return("worker-1", credit_return)

        progress.increment_returned.assert_called_once()

    async def test_handles_single_turn_conversation(
        self, registered_handler, mock_concurrency, mock_strategy
    ):
        """Should handle single-turn conversation correctly."""
        credit = make_credit(turn_index=0, num_turns=1)  # Single turn
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        # Should release session slot (final turn)
        mock_concurrency.release_session_slot.assert_called_once()
        # Should dispatch to strategy for cleanup (final turn allows tracking cleanup)
        mock_strategy.handle_credit_return.assert_called_once_with(credit)

    async def test_concurrent_returns_from_different_workers(
        self, registered_handler, mock_progress
    ):
        """Should handle concurrent returns from different workers."""
        credits = [make_credit(credit_id=i) for i in range(5)]
        credit_returns = [make_credit_return(c) for c in credits]

        # Simulate concurrent returns
        tasks = [
            registered_handler.on_credit_return(f"worker-{i}", cr)
            for i, cr in enumerate(credit_returns)
        ]
        await asyncio.gather(*tasks)

        assert mock_progress.increment_returned.call_count == 5

    @pytest.mark.parametrize(
        "cancelled,first_token_sent",
        [
            (False, True),   # Normal completion
            (False, False),  # No TTFT (error path)
            (True, True),    # Cancelled with TTFT
            (True, False),   # Cancelled before TTFT
        ],
    )  # fmt: skip
    async def test_all_return_state_combinations(
        self,
        registered_handler,
        mock_progress,
        mock_concurrency,
        cancelled: bool,
        first_token_sent: bool,
    ):
        """Should handle all combinations of cancelled/first_token_sent."""
        credit = make_credit()
        credit_return = make_credit_return(
            credit, cancelled=cancelled, first_token_sent=first_token_sent
        )

        await registered_handler.on_credit_return("worker-1", credit_return)

        # Should always increment returned
        mock_progress.increment_returned.assert_called_once_with(
            credit.is_final_turn, cancelled
        )
        # Should release prefill only if no TTFT
        if not first_token_sent:
            mock_concurrency.release_prefill_slot.assert_called_once()
        else:
            mock_concurrency.release_prefill_slot.assert_not_called()
