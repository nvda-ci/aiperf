# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple test that runs the aiperf profile command with mocked services."""

from collections import defaultdict

import pytest

# Import aiperf modules first to avoid circular import issues.
from aiperf.common.enums import CommunicationBackend, ServiceRunType, TransportType
from aiperf.common.factories import (
    CommunicationFactory,
    ServiceManagerFactory,
    TransportFactory,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.credit.messages import CreditReturn

# Import harness modules for registration side effects.
# MockServiceManager overrides ServiceManagerFactory to run services in-process.
# FakeTransport overrides TransportFactory to bypass HTTP entirely.
# MockCommunication overrides CommunicationFactory to bypass ZMQ.
from aiperf.credit.structs import Credit
from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.harness import (
    FakeCommunication,  # noqa: F401
    FakeServiceManager,  # noqa: F401
    FakeTransport,  # noqa: F401
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
def test_timing_harness_mocks_registered():
    """Verify harness mocks are registered with high priority."""
    # Verify FakeTransport overrides HTTP transport
    http_class = TransportFactory.get_class_from_type(TransportType.HTTP)
    assert http_class.__name__ == "FakeTransport"

    # Verify MockServiceManager overrides multiprocessing manager
    mp_class = ServiceManagerFactory.get_class_from_type(ServiceRunType.MULTIPROCESSING)
    assert mp_class.__name__ == "FakeServiceManager"

    # Verify MockCommunication overrides ZMQ IPC backend
    ipc_class = CommunicationFactory.get_class_from_type(CommunicationBackend.ZMQ_IPC)
    assert ipc_class.__name__ == "FakeCommunication"

    assert Tokenizer.from_pretrained.__qualname__ == "FakeTokenizer.from_pretrained"


@pytest.mark.component_integration
class TestCLIProfile:
    """Tests for CLI profile command."""

    def test_profile_command_runs(self, cli: AIPerfCLI):
        """Basic test that aiperf profile command runs with mocked services."""

        concurrency = 13
        workers_max = 3
        request_count = 100
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model gpt2 \
                --endpoint-type chat \
                --request-count {request_count} \
                --concurrency {concurrency} \
                --osl 2 \
                --isl 2 \
                --extra-inputs ignore_eos:true \
                --request-cancellation-rate 50 \
                --workers-max {workers_max} \
                --random-seed 42 \
                --ui simple \
                --streaming
            """
        )
        # With seed 42 and 50% cancellation rate, expect 47 completed requests
        # (53 cancelled out of 100 total due to random timing)
        assert result.request_count == 47
        runner_result: AIPerfRunnerResultWithSharedBus = result.runner_result
        assert runner_result.shared_bus is not None
        credits = [p for p in runner_result.payloads_by_type(Credit, sent=True)]
        credit_returns = [
            p.payload for p in runner_result.payloads_by_type(CreditReturn, sent=True)
        ]
        assert len(credits) == len(credit_returns)
        credits_by_worker = defaultdict(list)
        for credit in credits:
            credits_by_worker[credit.receiver_identity].append(credit)
