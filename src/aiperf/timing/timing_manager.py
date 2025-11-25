# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    MessageType,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_command,
    on_message,
    on_stop,
)
from aiperf.common.messages import (
    CommandAcknowledgedResponse,
    CommandMessage,
    DatasetConfiguredNotification,
    ProfileCancelCommand,
    ProfileConfigureCommand,
)
from aiperf.common.messages.credit_messages import Credit
from aiperf.common.models import DatasetMetadata
from aiperf.common.protocols import (
    ServiceProtocol,
)
from aiperf.timing.config import (
    TimingManagerConfig,
)
from aiperf.timing.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
)
from aiperf.timing.credit_manager import CreditPhaseMessagesMixin
from aiperf.timing.sticky_router import StickyCreditRouter


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.TIMING_MANAGER)
class TimingManager(BaseComponentService, CreditPhaseMessagesMixin):
    """
    The TimingManager service is responsible to generate the schedule and issuing
    timing credits for requests.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )
        self.debug("Timing manager __init__")
        self.config = TimingManagerConfig.from_user_config(self.user_config)

        self._dataset_configured_event = asyncio.Event()
        self._dataset_metadata: DatasetMetadata | None = None

        # StickyCreditRouter handles everything: routing, sending, returns, worker lifecycle
        # Created early to handle worker connections immediately, as well as attaching to the lifecycle
        self.sticky_router: StickyCreditRouter = StickyCreditRouter(
            service_config=service_config
        )
        self.attach_child_lifecycle(self.sticky_router)

        # Dataset request client for strategies to fetch conversations
        self.dataset_request_client = self.comms.create_request_client(
            CommAddress.DATASET_MANAGER_PROXY_FRONTEND
        )

        self._credit_issuing_strategy: CreditIssuingStrategy | None = None

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured_notification(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Handle the dataset configured notification."""
        self.debug(f"Received dataset configured notification: {message}")
        self._dataset_metadata = message.metadata
        self._dataset_configured_event.set()

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the timing manager."""
        self.info(
            "Waiting for dataset to be configured before configuring credit issuing strategy"
        )
        await asyncio.wait_for(
            self._dataset_configured_event.wait(),
            timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
        )

        if not self._dataset_metadata:
            raise InvalidStateError("Dataset metadata is not available")

        self.debug(f"Configuring credit issuing strategy for {self.service_id}")
        self.info(f"Using {self.config.timing_mode.title()} strategy")
        self._credit_issuing_strategy = CreditIssuingStrategyFactory.create_instance(
            self.config.timing_mode,
            config=self.config,
            credit_manager=self,
            dataset_metadata=self._dataset_metadata,
        )

        self.sticky_router.set_return_callback(
            self._credit_issuing_strategy._on_credit_return
        )
        self.info("Linked StickyCreditRouter to Strategy for credit returns")

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")
        self.debug(
            lambda: f"Timing manager configured with credit issuing strategy: {self._credit_issuing_strategy}"
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message: CommandMessage) -> None:
        """Start the timing manager and issue credit drops according to the configured strategy."""
        self.debug("Starting profiling")

        self.debug("Waiting for timing manager to be initialized")
        await self.initialized_event.wait()
        self.debug("Timing manager initialized, starting profiling")

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")

        self.execute_async(self._credit_issuing_strategy.start())
        self.info(
            f"Credit issuing strategy for {self.config.timing_mode.title()} started"
        )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        self.debug(lambda: f"Received profile cancel command: {message}")
        await self.publish(
            CommandAcknowledgedResponse.from_command_message(message, self.service_id)
        )
        if self._credit_issuing_strategy:
            await self._credit_issuing_strategy.stop()

    async def send_credit(self, credit: Credit) -> None:
        """Route credit via StickyCreditRouter.

        This method is called by credit issuing strategies to send credits to workers.
        It delegates to the SmartRouter which handles load balancing and sticky routing.

        Args:
            credit: Credit to route to a worker
        """
        await self.sticky_router.send_credit(self.service_id, credit)

    @on_stop
    async def _timing_manager_stop(self) -> None:
        """Stop the timing manager."""
        self.debug("Stopping timing manager")
        if self._credit_issuing_strategy:
            await self._credit_issuing_strategy.stop()
        await self.cancel_all_tasks()


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TimingManager)


if __name__ == "__main__":
    main()
