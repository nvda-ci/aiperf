# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhasePublisher.

Clean, behavior-focused tests for:
- Phase lifecycle event publishing (start, sending complete, complete)
- Progress updates
- Message construction
"""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.models import CreditPhaseStats
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.publisher import PhasePublisher

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def publisher(mock_pub_client: MagicMock) -> PhasePublisher:
    """PhasePublisher instance with mocked dependencies."""
    return PhasePublisher(
        pub_client=mock_pub_client,
        service_id="timing-manager-001",
    )


# =============================================================================
# Phase Start Tests
# =============================================================================


@pytest.mark.asyncio
class TestPublishPhaseStart:
    """Tests for publish_phase_start method."""

    async def test_creates_phase_start_message(
        self,
        publisher: PhasePublisher,
        mock_pub_client: MagicMock,
        sample_phase_config: CreditPhaseConfig,
        sample_phase_stats: CreditPhaseStats,
    ):
        """Creates CreditPhaseStartMessage with correct data."""
        await publisher.publish_phase_start(sample_phase_config, sample_phase_stats)

        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]

        assert isinstance(msg, CreditPhaseStartMessage)
        assert msg.service_id == "timing-manager-001"
        assert msg.stats is sample_phase_stats
        assert msg.config is sample_phase_config


# =============================================================================
# Phase Sending Complete Tests
# =============================================================================


@pytest.mark.asyncio
class TestPublishPhaseSendingComplete:
    """Tests for publish_phase_sending_complete method."""

    async def test_creates_sending_complete_message(
        self,
        publisher: PhasePublisher,
        mock_pub_client: MagicMock,
        sample_phase_stats: CreditPhaseStats,
    ):
        """Creates CreditPhaseSendingCompleteMessage with correct data."""
        await publisher.publish_phase_sending_complete(sample_phase_stats)

        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]

        assert isinstance(msg, CreditPhaseSendingCompleteMessage)
        assert msg.service_id == "timing-manager-001"
        assert msg.stats is sample_phase_stats


# =============================================================================
# Phase Complete Tests
# =============================================================================


@pytest.mark.asyncio
class TestPublishPhaseComplete:
    """Tests for publish_phase_complete method."""

    async def test_creates_phase_complete_message(
        self,
        publisher: PhasePublisher,
        mock_pub_client: MagicMock,
        sample_phase_stats: CreditPhaseStats,
    ):
        """Creates CreditPhaseCompleteMessage with correct data."""
        await publisher.publish_phase_complete(sample_phase_stats)

        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]

        assert isinstance(msg, CreditPhaseCompleteMessage)
        assert msg.service_id == "timing-manager-001"
        assert msg.stats is sample_phase_stats


# =============================================================================
# Progress Tests
# =============================================================================


@pytest.mark.asyncio
class TestPublishProgress:
    """Tests for publish_progress method."""

    async def test_creates_progress_message(
        self,
        publisher: PhasePublisher,
        mock_pub_client: MagicMock,
        sample_phase_stats: CreditPhaseStats,
    ):
        """Creates CreditPhaseProgressMessage with correct data."""
        await publisher.publish_progress(sample_phase_stats)

        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]

        assert isinstance(msg, CreditPhaseProgressMessage)
        assert msg.service_id == "timing-manager-001"
        assert msg.stats is sample_phase_stats


# =============================================================================
# Credits Complete Tests
# =============================================================================


@pytest.mark.asyncio
class TestPublishCreditsComplete:
    """Tests for publish_credits_complete method."""

    async def test_creates_credits_complete_message(
        self,
        publisher: PhasePublisher,
        mock_pub_client: MagicMock,
    ):
        """Creates CreditsCompleteMessage with service_id only."""
        await publisher.publish_credits_complete()

        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]

        assert isinstance(msg, CreditsCompleteMessage)
        assert msg.service_id == "timing-manager-001"


# =============================================================================
# Initialization Tests
# =============================================================================


class TestPhasePublisherInitialization:
    """Tests for PhasePublisher initialization."""

    def test_stores_pub_client(self, mock_pub_client: MagicMock):
        """Stores pub_client reference."""
        publisher = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="test-id",
        )
        assert publisher._pub_client is mock_pub_client

    def test_stores_service_id(self, mock_pub_client: MagicMock):
        """Stores service_id."""
        publisher = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="custom-service-id",
        )
        assert publisher._service_id == "custom-service-id"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
class TestPhasePublisherIntegration:
    """Integration tests for PhasePublisher."""

    async def test_all_lifecycle_events_use_same_service_id(
        self,
        mock_pub_client: MagicMock,
        sample_phase_config: CreditPhaseConfig,
        sample_phase_stats: CreditPhaseStats,
    ):
        """All lifecycle events include the same service_id."""
        service_id = "consistent-service-id"
        publisher = PhasePublisher(
            pub_client=mock_pub_client,
            service_id=service_id,
        )

        # Publish all event types
        await publisher.publish_phase_start(sample_phase_config, sample_phase_stats)
        await publisher.publish_phase_sending_complete(sample_phase_stats)
        await publisher.publish_phase_complete(sample_phase_stats)
        await publisher.publish_progress(sample_phase_stats)
        await publisher.publish_credits_complete()

        # Verify all messages have same service_id
        assert mock_pub_client.publish.call_count == 5
        for call in mock_pub_client.publish.call_args_list:
            msg = call[0][0]
            assert msg.service_id == service_id

    async def test_different_stats_produce_different_messages(
        self,
        mock_pub_client: MagicMock,
    ):
        """Different stats objects produce different messages."""
        publisher = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="test-id",
        )

        config1 = CreditPhaseConfig(
            phase=CreditPhase.WARMUP,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=10,
        )
        stats1 = CreditPhaseStats(
            phase=CreditPhase.WARMUP,
            requests_sent=10,
            requests_completed=5,
            requests_cancelled=0,
            final_requests_sent=10,
            start_ns=1000,
        )

        config2 = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=100,
        )
        stats2 = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            requests_sent=100,
            requests_completed=90,
            requests_cancelled=2,
            final_requests_sent=100,
            start_ns=2000,
        )

        await publisher.publish_phase_start(config1, stats1)
        await publisher.publish_phase_start(config2, stats2)

        calls = mock_pub_client.publish.call_args_list
        msg1 = calls[0][0][0]
        msg2 = calls[1][0][0]

        assert msg1.stats.phase == CreditPhase.WARMUP
        assert msg2.stats.phase == CreditPhase.PROFILING
        assert msg1.stats.requests_sent == 10
        assert msg2.stats.requests_sent == 100


# =============================================================================
# Parameterized Phase Tests
# =============================================================================


@pytest.mark.asyncio
class TestPhasePublisherParameterized:
    """Parameterized tests for different phases."""

    @pytest.mark.parametrize(
        "phase",
        [CreditPhase.WARMUP, CreditPhase.PROFILING],
    )
    async def test_publish_phase_start_all_phases(
        self, mock_pub_client: MagicMock, phase: CreditPhase
    ):
        """publish_phase_start works for all phase types."""
        publisher = PhasePublisher(pub_client=mock_pub_client, service_id="test-id")
        config = CreditPhaseConfig(
            phase=phase, timing_mode=TimingMode.REQUEST_RATE, total_expected_requests=10
        )
        stats = CreditPhaseStats(
            phase=phase,
            requests_sent=5,
            requests_completed=3,
            requests_cancelled=0,
            final_requests_sent=5,
            start_ns=1000,
        )

        await publisher.publish_phase_start(config, stats)

        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseStartMessage)
        assert msg.stats.phase == phase
        assert msg.config.phase == phase

    @pytest.mark.parametrize(
        "phase",
        [CreditPhase.WARMUP, CreditPhase.PROFILING],
    )
    async def test_publish_phase_complete_all_phases(
        self, mock_pub_client: MagicMock, phase: CreditPhase
    ):
        """publish_phase_complete works for all phase types."""
        publisher = PhasePublisher(pub_client=mock_pub_client, service_id="test-id")
        stats = CreditPhaseStats(
            phase=phase,
            requests_sent=10,
            requests_completed=10,
            requests_cancelled=0,
            final_requests_sent=10,
            start_ns=1000,
        )

        await publisher.publish_phase_complete(stats)

        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseCompleteMessage)
        assert msg.stats.phase == phase

    @pytest.mark.parametrize(
        "phase",
        [CreditPhase.WARMUP, CreditPhase.PROFILING],
    )
    async def test_publish_progress_all_phases(
        self, mock_pub_client: MagicMock, phase: CreditPhase
    ):
        """publish_progress works for all phase types."""
        publisher = PhasePublisher(pub_client=mock_pub_client, service_id="test-id")
        stats = CreditPhaseStats(
            phase=phase,
            requests_sent=5,
            requests_completed=2,
            requests_cancelled=1,
            final_requests_sent=5,
            start_ns=1000,
        )

        await publisher.publish_progress(stats)

        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseProgressMessage)
        assert msg.stats.phase == phase

    @pytest.mark.parametrize(
        "phase",
        [CreditPhase.WARMUP, CreditPhase.PROFILING],
    )
    async def test_publish_sending_complete_all_phases(
        self, mock_pub_client: MagicMock, phase: CreditPhase
    ):
        """publish_phase_sending_complete works for all phase types."""
        publisher = PhasePublisher(pub_client=mock_pub_client, service_id="test-id")
        stats = CreditPhaseStats(
            phase=phase,
            requests_sent=10,
            requests_completed=5,
            requests_cancelled=0,
            final_requests_sent=10,
            start_ns=1000,
        )

        await publisher.publish_phase_sending_complete(stats)

        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseSendingCompleteMessage)
        assert msg.stats.phase == phase


# =============================================================================
# Stats Field Tests
# =============================================================================


@pytest.mark.asyncio
class TestPhasePublisherStatsFields:
    """Tests for stats field propagation."""

    @pytest.mark.parametrize(
        "requests_sent,requests_completed,requests_cancelled",
        [
            (0, 0, 0),
            (100, 50, 10),
            (1000, 999, 1),
            (1, 1, 0),
        ],
    )  # fmt: skip
    async def test_progress_stats_preserved(
        self,
        mock_pub_client: MagicMock,
        requests_sent: int,
        requests_completed: int,
        requests_cancelled: int,
    ):
        """Progress stats are preserved in message."""
        publisher = PhasePublisher(pub_client=mock_pub_client, service_id="test-id")
        stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            requests_sent=requests_sent,
            requests_completed=requests_completed,
            requests_cancelled=requests_cancelled,
            final_requests_sent=requests_sent,
            start_ns=1000,
        )

        await publisher.publish_progress(stats)

        msg = mock_pub_client.publish.call_args[0][0]
        assert msg.stats.requests_sent == requests_sent
        assert msg.stats.requests_completed == requests_completed
        assert msg.stats.requests_cancelled == requests_cancelled

    @pytest.mark.parametrize(
        "total_expected",
        [1, 10, 100, 1000, 10000],
    )
    async def test_config_total_expected_preserved(
        self, mock_pub_client: MagicMock, total_expected: int
    ):
        """Config total_expected_requests is preserved in message."""
        publisher = PhasePublisher(pub_client=mock_pub_client, service_id="test-id")
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=total_expected,
        )
        stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            requests_sent=0,
            requests_completed=0,
            requests_cancelled=0,
            final_requests_sent=0,
            start_ns=1000,
        )

        await publisher.publish_phase_start(config, stats)

        msg = mock_pub_client.publish.call_args[0][0]
        assert msg.config.total_expected_requests == total_expected
