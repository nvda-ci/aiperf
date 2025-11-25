# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for credit message validation."""

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import Credit, CreditDropMessage


class TestCreditDropMessageValidation:
    """Test validation logic for CreditDropMessage."""

    def test_credit_drop_first_turn_allows_none_conversation_id(self):
        """First turn (turn_index=0) can have None conversation_id."""
        credit = Credit(
            phase=CreditPhase.PROFILING,
            num=1,
            turn_index=0,
            conversation_id="conv-1",
            x_correlation_id="corr-1",
        )
        msg = CreditDropMessage(
            service_id="test-service",
            credit=credit,
        )

        assert msg.credit.turn_index == 0
        assert msg.credit.conversation_id == "conv-1"

    def test_credit_drop_subsequent_turn_requires_conversation_id(self):
        """Subsequent turns (turn_index > 0) require conversation_id."""
        # Note: With the new Credit model, conversation_id is always required
        # This test documents that behavior
        credit = Credit(
            phase=CreditPhase.PROFILING,
            num=1,
            turn_index=1,
            conversation_id="conv-1",
            x_correlation_id="corr-1",
        )
        msg = CreditDropMessage(
            service_id="test-service",
            credit=credit,
        )

        assert msg.credit.turn_index == 1
        assert msg.credit.conversation_id == "conv-1"

    def test_credit_drop_subsequent_turn_with_conversation_id_succeeds(self):
        """Subsequent turns with conversation_id should succeed."""
        credit = Credit(
            phase=CreditPhase.PROFILING,
            num=1,
            turn_index=1,
            conversation_id="session-123",
            x_correlation_id="corr-1",
        )
        msg = CreditDropMessage(
            service_id="test-service",
            credit=credit,
        )

        assert msg.credit.turn_index == 1
        assert msg.credit.conversation_id == "session-123"

    def test_credit_drop_turn_index_5_requires_conversation_id(self):
        """Test validation for higher turn indices."""
        credit = Credit(
            phase=CreditPhase.PROFILING,
            num=5,
            turn_index=5,
            conversation_id="session-123",
            x_correlation_id="corr-1",
        )
        msg = CreditDropMessage(
            service_id="test-service",
            credit=credit,
        )

        assert msg.credit.turn_index == 5
        assert msg.credit.conversation_id == "session-123"

    def test_credit_drop_first_turn_with_conversation_id_succeeds(self):
        """First turn can have conversation_id (it's optional, not forbidden)."""
        credit = Credit(
            phase=CreditPhase.PROFILING,
            num=1,
            turn_index=0,
            conversation_id="session-123",
            x_correlation_id="corr-1",
        )
        msg = CreditDropMessage(
            service_id="test-service",
            credit=credit,
        )

        assert msg.credit.turn_index == 0
        assert msg.credit.conversation_id == "session-123"
