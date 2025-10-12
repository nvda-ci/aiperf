# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clean event-driven multiprocess service manager."""

import asyncio
import uuid
from multiprocessing import Process

from pydantic import ConfigDict, Field

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import (
    DEFAULT_MULTIPROCESS_MONITOR_INTERVAL,
    DEFAULT_SERVICE_REGISTRATION_TIMEOUT,
    TASK_CANCEL_TIMEOUT_SHORT,
)
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ServiceRunType
from aiperf.common.exceptions import AIPerfError
from aiperf.common.factories import ServiceFactory, ServiceManagerFactory
from aiperf.common.hooks import background_task
from aiperf.common.messages import ServiceFailedMessage
from aiperf.common.mixins import MessageBusClientMixin
from aiperf.common.models import ServiceRunInfo
from aiperf.common.protocols import ServiceManagerProtocol
from aiperf.common.types import ServiceTypeT
from aiperf.controller.base_service_manager import BaseServiceManager
from aiperf.controller.registration_manager import RegistrationManager


class MultiProcessRunInfo(ServiceRunInfo):
    """Service metadata with additional process handle information."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    process: Process | None = Field(default=None)


@implements_protocol(ServiceManagerProtocol)
@ServiceManagerFactory.register(ServiceRunType.MULTIPROCESSING)
class MultiProcessServiceManager(BaseServiceManager, MessageBusClientMixin):
    """Multiprocess service manager with event-driven registration."""

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        service_config: ServiceConfig,
        user_config: UserConfig,
        log_queue: "asyncio.Queue | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(required_services, service_config, user_config, **kwargs)
        self.log_queue = log_queue
        self._registry: RegistrationManager = RegistrationManager()

    @property
    def registry(self) -> RegistrationManager:
        return self._registry

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Start service processes."""
        service_class = ServiceFactory.get_class_from_type(service_type)

        for _ in range(num_replicas):
            service_id = f"{service_type}_{uuid.uuid4().hex[:8]}"

            info = MultiProcessRunInfo(
                service_type=service_type,
                service_id=service_id,
                required=service_type in self.required_services,
                process=Process(
                    target=bootstrap_and_run_service,
                    name=f"{service_type}_process",
                    kwargs={
                        "service_class": service_class,
                        "service_id": service_id,
                        "service_config": self.service_config,
                        "user_config": self.user_config,
                        "log_queue": self.log_queue,
                    },
                    daemon=True,
                ),
            )

            await self._registry.expect(info)
            info.process.start()
            self.debug(f"{service_type} started (pid {info.process.pid})")

    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]:
        """Stop specific services."""
        services = await self._registry.get_services(service_type)

        targets = [
            s
            for s in services
            if isinstance(s, MultiProcessRunInfo)
            and (service_id is None or s.service_id == service_id)
        ]

        if not targets:
            return []

        results = await asyncio.gather(
            *[self._terminate_process(s) for s in targets],
            return_exceptions=True,
        )
        return results

    async def shutdown_all_services(self) -> list[BaseException | None]:
        """Gracefully terminate all services."""
        services = await self._registry.get_services()
        processes = [s for s in services if isinstance(s, MultiProcessRunInfo)]

        return await asyncio.gather(
            *[self._terminate_process(p) for p in processes],
            return_exceptions=True,
        )

    async def kill_all_services(self) -> list[BaseException | None]:
        """Force kill all services."""
        services = await self._registry.get_services()
        processes = [s for s in services if isinstance(s, MultiProcessRunInfo)]

        for proc in processes:
            if proc.process and proc.process.is_alive():
                proc.process.kill()

        return await asyncio.gather(
            *[self._terminate_process(p) for p in processes],
            return_exceptions=True,
        )

    async def wait_for_all_services_registration(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = DEFAULT_SERVICE_REGISTRATION_TIMEOUT,
    ) -> None:
        """Wait for all services to register."""
        self.debug("Waiting for service registration...")

        try:
            await self._registry.wait_for_all(timeout=timeout_seconds)
            self.info("All services registered")
        except asyncio.TimeoutError as e:
            missing = await self._registry.get_missing()
            registered, expected = await self._registry.get_progress()

            lines = [
                f"Timeout: {registered}/{expected} services registered",
                "Missing:",
            ]

            for sid in missing:
                info = await self._registry.get_service(sid)
                service_type_name = info.service_type if info else "unknown"
                lines.append(f"  • {sid} ({service_type_name})")

            msg = "\n".join(lines)
            self.error(msg)
            raise AIPerfError(msg) from e

    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float,
    ) -> None:
        """Not implemented for multiprocessing."""
        self.warning("Start waiting not implemented for multiprocessing")

    async def _terminate_process(self, info: MultiProcessRunInfo) -> None:
        """Terminate a process with timeout."""
        if not info.process or not info.process.is_alive():
            return

        await self._registry.remove(info.service_id)
        info.process.terminate()

        try:
            await asyncio.to_thread(
                info.process.join, timeout=TASK_CANCEL_TIMEOUT_SHORT
            )
            self.debug(lambda: f"{info.service_type} stopped (pid {info.process.pid})")
        except TimeoutError:
            self.warning(
                f"{info.service_type} timeout, killing (pid {info.process.pid})"
            )
            info.process.kill()

    @background_task(interval=DEFAULT_MULTIPROCESS_MONITOR_INTERVAL, immediate=False)
    async def _monitor_health(self) -> None:
        """Monitor processes and publish alerts if any process exits."""
        services = await self._registry.get_services()
        for service in services:
            if (
                not isinstance(service, MultiProcessRunInfo)
                or not service.process
                or service.process.is_alive()
            ):
                continue

            if self.stop_requested:
                return

            await self._registry.mark_failed(
                service.service_id,
                f"Process exited with code {service.process.exitcode}",
            )
            if service.required:
                await self.publish(
                    ServiceFailedMessage(
                        service_id=service.service_id,
                        reason=f"Process exited with code {service.process.exitcode}",
                    )
                )
