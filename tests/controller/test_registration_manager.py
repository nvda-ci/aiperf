# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus, ServiceType
from aiperf.common.models import ErrorDetails, ServiceRunInfo
from aiperf.controller.registration_manager import RegistrationManager


@pytest.fixture
def manager() -> RegistrationManager:
    """Create a fresh RegistrationManager instance for each test."""
    return RegistrationManager()


@pytest.fixture
def service_info() -> ServiceRunInfo:
    """Create a sample ServiceRunInfo for testing."""
    return ServiceRunInfo(
        service_type=ServiceType.WORKER,
        service_id="test_worker_1",
    )


@pytest.fixture
def multiple_services() -> list[ServiceRunInfo]:
    """Create multiple ServiceRunInfo objects for testing."""
    return [
        ServiceRunInfo(
            service_type=ServiceType.WORKER,
            service_id=f"worker_{i}",
        )
        for i in range(3)
    ]


async def expect_and_register(
    manager: RegistrationManager, *services: ServiceRunInfo
) -> None:
    """Helper to expect and register services in one call."""
    await manager.expect(*services)
    for service in services:
        await manager.register(service.service_id, LifecycleState.RUNNING)


async def assert_progress(
    manager: RegistrationManager, registered: int, expected: int
) -> None:
    """Helper to assert registration progress."""
    reg, exp = await manager.get_progress()
    assert reg == registered and exp == expected


class TestRegistrationManagerBasics:
    """Test basic registration manager functionality."""

    @pytest.mark.asyncio
    async def test_initial_state(self, manager: RegistrationManager):
        """Verify manager starts in clean initial state."""
        await assert_progress(manager, 0, 0)
        assert await manager.get_missing() == set()
        assert await manager.get_failures() == {}

    @pytest.mark.asyncio
    async def test_expect_single_service(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test expecting a single service adds it to tracking."""
        await manager.expect(service_info)
        await assert_progress(manager, 0, 1)
        assert await manager.get_missing() == {service_info.service_id}

    @pytest.mark.asyncio
    async def test_expect_multiple_services(
        self, manager: RegistrationManager, multiple_services: list[ServiceRunInfo]
    ):
        """Test expecting multiple services in one call."""
        await manager.expect(*multiple_services)
        await assert_progress(manager, 0, 3)
        assert await manager.get_missing() == {s.service_id for s in multiple_services}

    @pytest.mark.asyncio
    async def test_expect_clears_all_registered_event(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test that expecting services doesn't set the event until all register."""
        assert not manager._all_registered.is_set()
        await manager.expect(service_info)
        assert not manager._all_registered.is_set()


class TestServiceRegistration:
    """Test service registration flows."""

    @pytest.mark.asyncio
    async def test_register_expected_service(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test registering a service updates all state correctly."""
        await manager.expect(service_info)
        result = await manager.register(service_info.service_id, LifecycleState.RUNNING)

        assert result is True
        assert await manager.is_registered(service_info.service_id)
        await assert_progress(manager, 1, 1)

        # Verify service info is updated correctly
        info = await manager.get_service(service_info.service_id)
        assert info.registration_status == ServiceRegistrationStatus.REGISTERED
        assert info.state == LifecycleState.RUNNING
        assert info.last_seen > 0 and info.first_seen > 0

    @pytest.mark.asyncio
    async def test_register_unexpected_service_warns(
        self, manager: RegistrationManager, service_info: ServiceRunInfo, caplog
    ):
        """Test that registering an unexpected service returns False and warns."""
        result = await manager.register(service_info.service_id, LifecycleState.RUNNING)

        assert result is False
        # Check that a warning was logged using pytest's caplog fixture
        assert any(
            "unexpected service" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    @pytest.mark.asyncio
    async def test_register_duplicate_service(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test that registering the same service twice returns False."""
        await manager.expect(service_info)
        first = await manager.register(service_info.service_id, LifecycleState.RUNNING)
        second = await manager.register(service_info.service_id, LifecycleState.RUNNING)

        assert first is True
        assert second is False

    @pytest.mark.asyncio
    async def test_register_all_services_sets_event(
        self, manager: RegistrationManager, multiple_services: list[ServiceRunInfo]
    ):
        """Test that registering all expected services sets the all_registered event."""
        await manager.expect(*multiple_services)

        # Register all but one
        for service in multiple_services[:-1]:
            await manager.register(service.service_id, LifecycleState.RUNNING)
            assert not manager._all_registered.is_set()

        # Register the last one
        await manager.register(multiple_services[-1].service_id, LifecycleState.RUNNING)
        assert manager._all_registered.is_set()


class TestServiceRemoval:
    """Test service removal functionality."""

    @pytest.mark.asyncio
    async def test_remove_expected_service(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test removing an expected service."""
        await manager.expect(service_info)
        await manager.remove(service_info.service_id)

        registered, expected = await manager.get_progress()
        assert expected == 0
        assert service_info.service_id not in await manager.get_missing()

    @pytest.mark.asyncio
    async def test_remove_registered_service(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test removing a service that was already registered."""
        await manager.expect(service_info)
        await manager.register(service_info.service_id, LifecycleState.RUNNING)
        await manager.remove(service_info.service_id)

        assert not await manager.is_registered(service_info.service_id)
        assert await manager.get_service(service_info.service_id) is None

    @pytest.mark.asyncio
    async def test_remove_sets_event_when_expectations_met(
        self, manager: RegistrationManager, multiple_services: list[ServiceRunInfo]
    ):
        """Test that removing services can set all_registered if expectations are met."""
        await manager.expect(*multiple_services)

        # Register all services
        for service in multiple_services:
            await manager.register(service.service_id, LifecycleState.RUNNING)

        assert manager._all_registered.is_set()

        # Add an extra expectation
        extra_service = ServiceRunInfo(
            service_type=ServiceType.WORKER, service_id="extra"
        )
        await manager.expect(extra_service)
        assert not manager._all_registered.is_set()

        # Remove the extra expectation - should set event again
        await manager.remove(extra_service.service_id)
        assert manager._all_registered.is_set()

    @pytest.mark.asyncio
    async def test_remove_multiple_services(
        self, manager: RegistrationManager, multiple_services: list[ServiceRunInfo]
    ):
        """Test removing multiple services at once."""
        await manager.expect(*multiple_services)
        service_ids = [s.service_id for s in multiple_services[:2]]

        await manager.remove(*service_ids)

        registered, expected = await manager.get_progress()
        assert expected == 1
        missing = await manager.get_missing()
        assert multiple_services[2].service_id in missing
        assert not any(sid in missing for sid in service_ids)


class TestFailureHandling:
    """Test failure tracking and handling."""

    @pytest.mark.asyncio
    async def test_mark_failed_with_error(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test marking a service as failed with error details."""
        await manager.expect(service_info)
        error = ErrorDetails.from_exception(ValueError("Test error"))

        await manager.mark_failed(service_info.service_id, error)

        failures = await manager.get_failures()
        assert service_info.service_id in failures
        assert failures[service_info.service_id].error == error
        assert failures[service_info.service_id].state == LifecycleState.FAILED

    @pytest.mark.asyncio
    async def test_mark_failed_updates_status(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test that marking failed updates registration status."""
        await manager.expect(service_info)
        await manager.register(service_info.service_id, LifecycleState.RUNNING)

        error = ErrorDetails.from_exception(RuntimeError("Crash"))
        await manager.mark_failed(service_info.service_id, error)

        # Service should be removed from expected/registered
        assert not await manager.is_registered(service_info.service_id)
        registered, expected = await manager.get_progress()
        assert registered == 0
        assert expected == 0

    @pytest.mark.asyncio
    async def test_mark_failed_unknown_service_warns(
        self, manager: RegistrationManager, caplog
    ):
        """Test that marking unknown service as failed logs a warning."""
        error = ErrorDetails.from_exception(Exception("Error"))
        await manager.mark_failed("unknown_service", error)

        # Check that a warning was logged using pytest's caplog fixture
        assert any(
            "unknown service" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    @pytest.mark.asyncio
    async def test_mark_failed_removes_from_tracking(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test that failed services are removed from expected set."""
        await manager.expect(service_info)
        error = ErrorDetails.from_exception(Exception("Error"))

        await manager.mark_failed(service_info.service_id, error)

        missing = await manager.get_missing()
        assert service_info.service_id not in missing


class TestWaitForAll:
    """Test waiting for all services to register."""

    @pytest.mark.asyncio
    async def test_wait_for_all_succeeds_when_registered(
        self, manager: RegistrationManager, multiple_services: list[ServiceRunInfo]
    ):
        """Test that wait_for_all succeeds when all services register."""
        await manager.expect(*multiple_services)

        # Register services in background
        async def register_services():
            await asyncio.sleep(0.01)
            for service in multiple_services:
                await manager.register(service.service_id, LifecycleState.RUNNING)

        task = asyncio.create_task(register_services())

        # Should complete without timeout
        await manager.wait_for_all(timeout=1.0)
        await task

    @pytest.mark.asyncio
    async def test_wait_for_all_times_out(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test that wait_for_all times out when services don't register."""
        await manager.expect(service_info)

        with pytest.raises(asyncio.TimeoutError):
            await manager.wait_for_all(timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_all_no_timeout(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test wait_for_all without timeout parameter."""
        await manager.expect(service_info)

        async def register_later():
            await asyncio.sleep(0.01)
            await manager.register(service_info.service_id, LifecycleState.RUNNING)

        task = asyncio.create_task(register_later())
        await manager.wait_for_all()
        await task

    @pytest.mark.asyncio
    async def test_wait_for_all_completes_immediately_when_ready(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test that wait_for_all returns immediately if all already registered."""
        await manager.expect(service_info)
        await manager.register(service_info.service_id, LifecycleState.RUNNING)

        # Should return immediately
        await manager.wait_for_all(timeout=1.0)


class TestServiceQueries:
    """Test service query methods."""

    @pytest.mark.asyncio
    async def test_get_services_all(
        self, manager: RegistrationManager, multiple_services: list[ServiceRunInfo]
    ):
        """Test getting all registered services."""
        await expect_and_register(manager, *multiple_services)

        services = await manager.get_services()
        assert len(services) == 3
        assert {s.service_id for s in services} == {
            s.service_id for s in multiple_services
        }

    @pytest.mark.asyncio
    async def test_get_services_filtered_by_type(self, manager: RegistrationManager):
        """Test filtering services by type."""
        services = [
            ServiceRunInfo(service_type=ServiceType.WORKER, service_id="worker_1"),
            ServiceRunInfo(service_type=ServiceType.WORKER, service_id="worker_2"),
            ServiceRunInfo(
                service_type=ServiceType.DATASET_MANAGER, service_id="dataset_1"
            ),
        ]
        await expect_and_register(manager, *services)

        workers = await manager.get_services(service_type=ServiceType.WORKER)
        assert len(workers) == 2
        assert all(s.service_type == ServiceType.WORKER for s in workers)

        datasets = await manager.get_services(service_type=ServiceType.DATASET_MANAGER)
        assert len(datasets) == 1 and datasets[0].service_id == "dataset_1"

    @pytest.mark.asyncio
    async def test_get_service_by_id_and_nonexistent(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test getting service by ID and handling nonexistent services."""
        # Nonexistent service returns None
        assert await manager.get_service("nonexistent") is None

        # Existing service returns the service
        await expect_and_register(manager, service_info)
        service = await manager.get_service(service_info.service_id)
        assert service is not None and service.service_id == service_info.service_id

    @pytest.mark.asyncio
    async def test_is_registered(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test checking if service is registered."""
        await manager.expect(service_info)
        assert not await manager.is_registered(service_info.service_id)

        await manager.register(service_info.service_id, LifecycleState.RUNNING)
        assert await manager.is_registered(service_info.service_id)

    @pytest.mark.asyncio
    async def test_get_missing_services(
        self, manager: RegistrationManager, multiple_services: list[ServiceRunInfo]
    ):
        """Test getting list of missing services."""
        await manager.expect(*multiple_services)

        # Register only first service
        await manager.register(multiple_services[0].service_id, LifecycleState.RUNNING)

        missing = await manager.get_missing()
        assert len(missing) == 2
        assert multiple_services[0].service_id not in missing
        assert multiple_services[1].service_id in missing
        assert multiple_services[2].service_id in missing


class TestServiceUpdates:
    """Test service update functionality."""

    @pytest.mark.asyncio
    async def test_update_service_state(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test updating service state and timestamp."""
        await manager.expect(service_info)
        await manager.register(service_info.service_id, LifecycleState.RUNNING)

        original_info = await manager.get_service(service_info.service_id)
        original_timestamp = original_info.last_seen

        # Update the service
        new_timestamp = original_timestamp + 1000
        await manager.update_service(
            service_info.service_id, new_timestamp, LifecycleState.STOPPING
        )

        updated_info = await manager.get_service(service_info.service_id)
        assert updated_info.state == LifecycleState.STOPPING
        assert updated_info.last_seen == new_timestamp
        assert updated_info.first_seen == original_info.first_seen


class TestConcurrency:
    """Test thread-safety and concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_registrations(
        self, manager: RegistrationManager, multiple_services: list[ServiceRunInfo]
    ):
        """Test that concurrent registrations are handled correctly."""
        await manager.expect(*multiple_services)

        # Register all services concurrently
        tasks = [
            manager.register(service.service_id, LifecycleState.RUNNING)
            for service in multiple_services
        ]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)
        registered, expected = await manager.get_progress()
        assert registered == expected == 3

    @pytest.mark.asyncio
    async def test_concurrent_expect_and_register(self, manager: RegistrationManager):
        """Test concurrent expects and registers don't cause issues."""

        async def expect_and_register(i: int):
            service = ServiceRunInfo(
                service_type=ServiceType.WORKER, service_id=f"worker_{i}"
            )
            await manager.expect(service)
            await asyncio.sleep(0.001)
            await manager.register(service.service_id, LifecycleState.RUNNING)

        # Run multiple expect/register cycles concurrently
        await asyncio.gather(*[expect_and_register(i) for i in range(5)])

        registered, expected = await manager.get_progress()
        assert registered == expected == 5


class TestReset:
    """Test reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_all_state(
        self, manager: RegistrationManager, multiple_services: list[ServiceRunInfo]
    ):
        """Test that reset clears all internal state."""
        await manager.expect(*multiple_services)
        for service in multiple_services:
            await manager.register(service.service_id, LifecycleState.RUNNING)

        error = ErrorDetails.from_exception(Exception("Test"))
        await manager.mark_failed(multiple_services[0].service_id, error)

        # Reset everything
        await manager.reset()

        # Verify clean state
        registered, expected = await manager.get_progress()
        assert registered == 0
        assert expected == 0
        assert await manager.get_missing() == set()
        assert await manager.get_services() == []
        assert await manager.get_failures() == {}

    @pytest.mark.asyncio
    async def test_reset_clears_event(self, manager: RegistrationManager):
        """Test that reset clears the all_registered event."""
        service = ServiceRunInfo(service_type=ServiceType.WORKER, service_id="worker_1")
        await manager.expect(service)
        await manager.register(service.service_id, LifecycleState.RUNNING)

        # Event should be set
        assert manager._all_registered.is_set()

        await manager.reset()

        # Event should be cleared
        assert not manager._all_registered.is_set()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_expect(self, manager: RegistrationManager):
        """Test calling expect with no services."""
        await manager.expect()
        registered, expected = await manager.get_progress()
        assert registered == 0
        assert expected == 0

    @pytest.mark.asyncio
    async def test_empty_remove(self, manager: RegistrationManager):
        """Test calling remove with no service IDs."""
        await manager.remove()
        # Should not raise

    @pytest.mark.asyncio
    async def test_wait_for_all_with_no_expectations(
        self, manager: RegistrationManager
    ):
        """Test wait_for_all when no services are expected."""
        # Should complete immediately since 0 == 0
        await manager.wait_for_all(timeout=0.1)

    @pytest.mark.asyncio
    async def test_expect_same_service_twice(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test expecting the same service multiple times."""
        await manager.expect(service_info)
        await manager.expect(service_info)

        registered, expected = await manager.get_progress()
        # Should still only expect it once
        assert expected == 1

    @pytest.mark.asyncio
    async def test_register_after_failure(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test registering a service after it was marked as failed."""
        await manager.expect(service_info)
        error = ErrorDetails.from_exception(Exception("Failed"))
        await manager.mark_failed(service_info.service_id, error)

        # Try to register - should fail since it's no longer expected
        result = await manager.register(service_info.service_id, LifecycleState.RUNNING)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_failures_returns_copy(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test that get_failures returns a copy, not the internal dict."""
        await manager.expect(service_info)
        error = ErrorDetails.from_exception(Exception("Error"))
        await manager.mark_failed(service_info.service_id, error)

        failures1 = await manager.get_failures()
        failures2 = await manager.get_failures()

        # Should be equal but not the same object
        assert failures1 == failures2
        assert failures1 is not failures2

    @pytest.mark.asyncio
    async def test_multiple_state_updates(
        self, manager: RegistrationManager, service_info: ServiceRunInfo
    ):
        """Test updating service state multiple times."""
        await manager.expect(service_info)
        await manager.register(service_info.service_id, LifecycleState.RUNNING)

        states = [
            LifecycleState.RUNNING,
            LifecycleState.STOPPING,
            LifecycleState.STOPPED,
        ]

        for i, state in enumerate(states):
            await manager.update_service(service_info.service_id, 1000 + i, state)

        info = await manager.get_service(service_info.service_id)
        assert info.state == LifecycleState.STOPPED
        assert info.last_seen == 1000 + len(states) - 1
