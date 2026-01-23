# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid

from aiperf.common.enums import CreditPhase, RequestRateMode, TimingMode
from aiperf.common.messages import CreditReturnMessage, ProfileCancelCommand
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.request_rate_strategy import RequestRateStrategy
from tests.unit.timing.conftest import profiling_phase_stats_from_config


def test_steady_state_triggers_profile_cancel(mock_credit_manager):
    config = TimingManagerConfig(
        timing_mode=TimingMode.REQUEST_RATE,
        request_count=2,
        request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
        concurrency=1,
        steady_state=True,
    )
    phase_stats = profiling_phase_stats_from_config(config)

    strategy = RequestRateStrategy(config, mock_credit_manager)
    strategy.phase_stats[CreditPhase.PROFILING] = phase_stats

    asyncio.run(
        strategy._on_credit_return(
            CreditReturnMessage(
                service_id="worker-1",
                phase=CreditPhase.PROFILING,
                credit_drop_id=str(uuid.uuid4()),
                requests_sent=1,
            )
        )
    )
    asyncio.run(
        strategy._on_credit_return(
            CreditReturnMessage(
                service_id="worker-1",
                phase=CreditPhase.PROFILING,
                credit_drop_id=str(uuid.uuid4()),
                requests_sent=1,
            )
        )
    )

    assert strategy._steady_state_stop_event.is_set()
    assert any(
        isinstance(message, ProfileCancelCommand)
        and message.reason == "steady_state"
        for message in mock_credit_manager.publish_calls
    )
