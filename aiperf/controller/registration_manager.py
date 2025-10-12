# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Registration Manager for tracking and coordinating service registration.

Provides event-driven registration tracking with no polling overhead.
"""

import asyncio
import time

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ServiceRunInfo
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.types import ServiceTypeT


class RegistrationManager(AIPerfLoggerMixin):
    """
    Event-driven service registration tracker.

    Tracks exact service IDs and uses asyncio.Event for efficient waiting.
    Thread-safe for concurrent access.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the registration manager."""
        super().__init__(**kwargs)
        self._lock = asyncio.Lock()
        self._all_registered = asyncio.Event()

        self._expected: set[str] = set()
        self._registered: set[str] = set()
        self._services: dict[str, ServiceRunInfo] = {}
        self._failures: dict[str, ServiceRunInfo] = {}

    async def expect(self, *services: ServiceRunInfo) -> None:
        """
        Add services to the expected list with their metadata.

        Call this when starting new services that need to register.

        Args:
            *services: Service metadata for services that must register
        """
        async with self._lock:
            for service in services:
                self._expected.add(service.service_id)
                self._services[service.service_id] = service
            # Clear the event since we now expect more services
            if self._registered != self._expected:
                self._all_registered.clear()

    async def remove(self, *service_ids: str) -> None:
        """
        Remove service IDs from expected list.

        Call this when services are stopped and no longer need to be tracked.

        Args:
            service_ids: Service IDs to stop expecting
        """
        async with self._lock:
            self._expected.difference_update(set(service_ids))
            # Also remove from registered and services if present
            for service_id in service_ids:
                self._registered.discard(service_id)
                self._services.pop(service_id, None)
            # Check if we've now met expectations
            if self._registered == self._expected:
                self._all_registered.set()

    async def register(self, service_id: str, state: LifecycleState) -> bool:
        """
        Register a service by ID and state.

        Args:
            service_id: ID of the service
            state: State of the service

        Returns:
            True if newly registered, False if duplicate, or unexpected
        """
        async with self._lock:
            if service_id not in self._expected:
                self.warning(
                    f"Received registration for unexpected service: {service_id}"
                )
                return False

            if service_id in self._registered:
                return False

            info = self._services[service_id]
            info.registration_status = ServiceRegistrationStatus.REGISTERED
            info.state = state
            info.last_seen = time.time_ns()
            info.first_seen = time.time_ns()

            self._registered.add(service_id)

            if self._registered == self._expected:
                self._all_registered.set()

            return True

    async def update_service(
        self, service_id: str, timestamp: int, state: LifecycleState
    ) -> None:
        """Update a service by ID and state."""
        async with self._lock:
            info = self._services[service_id]
            info.last_seen = timestamp
            info.state = state

    async def mark_failed(self, service_id: str, error: ErrorDetails) -> None:
        """Mark a service as failed with a reason and error details."""
        async with self._lock:
            if service_id not in self._services:
                self.warning(f"Received failure for unknown service: {service_id}")
                return

            info = self._services[service_id]
            info.state = LifecycleState.FAILED
            info.registration_status = ServiceRegistrationStatus.UNREGISTERED
            info.error = error
            self._failures[service_id] = info

        await self.remove(service_id)

    async def wait_for_all(self, timeout: float | None = None) -> None:
        """
        Wait for all expected services to register.

        Args:
            timeout: Maximum wait time in seconds

        Raises:
            asyncio.TimeoutError: On timeout
            asyncio.CancelledError: If task is cancelled
        """
        if not self._expected:
            return

        await asyncio.wait_for(self._all_registered.wait(), timeout=timeout)

    async def get_services(
        self, service_type: ServiceTypeT | None = None
    ) -> list[ServiceRunInfo]:
        """Get registered services, optionally filtered by type."""
        async with self._lock:
            if service_type:
                return [
                    s for s in self._services.values() if s.service_type == service_type
                ]
            return list(self._services.values())

    async def get_progress(self) -> tuple[int, int]:
        """Get (registered_count, expected_count)."""
        async with self._lock:
            return (len(self._registered), len(self._expected))

    async def get_missing(self) -> set[str]:
        """Get service IDs that haven't registered."""
        async with self._lock:
            return self._expected - self._registered

    async def get_failures(self) -> dict[str, ServiceRunInfo]:
        """Get failed services and their reasons."""
        async with self._lock:
            return self._failures.copy()

    async def is_registered(self, service_id: str) -> bool:
        """Check if a service is registered."""
        async with self._lock:
            return service_id in self._registered

    async def get_service(self, service_id: str) -> ServiceRunInfo | None:
        """Get service info by ID."""
        async with self._lock:
            return self._services.get(service_id)

    async def reset(self) -> None:
        """Reset to initial state."""
        async with self._lock:
            self._all_registered.clear()
            self._expected.clear()
            self._registered.clear()
            self._services.clear()
            self._failures.clear()
