# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from multiprocessing import Process
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.common.enums import LifecycleState, ServiceType
from aiperf.common.exceptions import AIPerfError
from aiperf.common.messages import ServiceFailedMessage
from aiperf.controller.multiprocess_service_manager import (
    MultiProcessRunInfo,
    MultiProcessServiceManager,
)
from tests.conftest import real_sleep


def create_service_info(
    service_type: ServiceType = ServiceType.WORKER,
    service_id: str | None = None,
    process: Mock | None = None,
    required: bool = True,
) -> MultiProcessRunInfo:
    """Helper to create MultiProcessRunInfo for tests."""
    if service_id is None:
        service_id = f"{service_type}_test"
    return MultiProcessRunInfo.model_construct(
        service_type=service_type,
        service_id=service_id,
        process=process,
        required=required,
    )


@pytest.fixture
def mock_comms():
    """Reusable mock communication object."""
    mock = Mock()
    mock.create_sub_client = Mock(return_value=Mock())
    mock.create_pub_client = Mock(return_value=Mock())
    return mock


@pytest.fixture
def mock_process_factory():
    """Patches Process creation for testing service startup."""
    from contextlib import contextmanager

    @contextmanager
    def _factory():
        with (
            patch(
                "aiperf.controller.multiprocess_service_manager.Process"
            ) as mock_process_cls,
            patch(
                "aiperf.controller.multiprocess_service_manager.bootstrap_and_run_service"
            ),
            patch(
                "aiperf.controller.multiprocess_service_manager.ServiceFactory.get_class_from_type",
                return_value=Mock,
            ),
        ):
            mock_process = Mock(
                spec=Process, pid=99999, is_alive=Mock(return_value=True)
            )
            mock_process_cls.return_value = mock_process

            yield {
                "process_cls": mock_process_cls,
                "process": mock_process,
            }

    return _factory


@pytest.fixture
def manager(service_config, user_config, mock_comms) -> MultiProcessServiceManager:
    """Create manager instance with mocked communication."""
    with patch(
        "aiperf.common.mixins.communication_mixin.CommunicationFactory.get_or_create_instance",
        return_value=mock_comms,
    ):
        manager = MultiProcessServiceManager(
            required_services={
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.TIMING_MANAGER: 1,
            },
            service_config=service_config,
            user_config=user_config,
        )
        manager.publish = AsyncMock()
        return manager


@pytest.fixture
def mock_process_alive() -> Mock:
    """Mock process that is alive."""
    return Mock(
        spec=Process, is_alive=Mock(return_value=True), pid=12345, exitcode=None
    )


@pytest.fixture
def mock_process_dead() -> Mock:
    """Mock process that is dead."""
    return Mock(spec=Process, is_alive=Mock(return_value=False), pid=54321, exitcode=1)


@pytest.fixture
def service_info_alive(mock_process_alive: Mock) -> MultiProcessRunInfo:
    """Service info with alive process."""
    return create_service_info(service_id="worker_alive", process=mock_process_alive)


@pytest.fixture
def service_info_dead(mock_process_dead: Mock) -> MultiProcessRunInfo:
    """Service info with dead process."""
    return create_service_info(service_id="worker_dead", process=mock_process_dead)


class TestMultiProcessServiceManagerInitialization:
    """Test manager initialization and configuration."""

    def test_initialization_with_required_services(
        self, service_config, user_config, mock_comms
    ) -> None:
        """Verify manager initializes with required services."""
        with patch(
            "aiperf.common.mixins.communication_mixin.CommunicationFactory.get_or_create_instance",
            return_value=mock_comms,
        ):
            manager = MultiProcessServiceManager(
                required_services={
                    ServiceType.DATASET_MANAGER: 2,
                    ServiceType.WORKER: 5,
                },
                service_config=service_config,
                user_config=user_config,
            )

            assert manager.required_services[ServiceType.DATASET_MANAGER] == 2
            assert manager.required_services[ServiceType.WORKER] == 5
            assert hasattr(manager, "registry")


class TestServiceStartup:
    """Test service process startup functionality."""

    @pytest.mark.asyncio
    async def test_run_service_starts_single_process(
        self, manager: MultiProcessServiceManager, mock_process_factory
    ) -> None:
        """Verify run_service starts a process and adds to registry."""
        with mock_process_factory() as mocks:
            await manager.run_service(ServiceType.WORKER, num_replicas=1)

            mocks["process"].start.assert_called_once()
            missing = await manager.registry.get_missing()
            assert len(missing) == 1
            assert any("worker" in sid.lower() for sid in missing)

    @pytest.mark.asyncio
    async def test_run_service_starts_multiple_replicas(
        self, manager: MultiProcessServiceManager, mock_process_factory
    ) -> None:
        """Verify run_service starts multiple replicas."""
        with mock_process_factory() as mocks:
            num_replicas = 3
            await manager.run_service(ServiceType.WORKER, num_replicas=num_replicas)

            assert mocks["process_cls"].call_count == num_replicas
            missing = await manager.registry.get_missing()
            assert len(missing) == num_replicas

    @pytest.mark.asyncio
    async def test_run_service_generates_unique_service_ids(
        self, manager: MultiProcessServiceManager, mock_process_factory
    ) -> None:
        """Verify each service instance gets unique ID."""
        with mock_process_factory():
            await manager.run_service(ServiceType.WORKER, num_replicas=3)

            service_ids = list(await manager.registry.get_missing())
            assert len(service_ids) == len(set(service_ids))
            assert all(sid.startswith("worker") for sid in service_ids)


class TestServiceRegistration:
    """Test service registration flows with RegistrationManager."""

    @pytest.mark.asyncio
    async def test_wait_for_all_services_registration_success(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify wait completes when all services register."""
        service = create_service_info(service_id="worker_test")
        await manager.registry.expect(service)
        await manager.registry.register(service.service_id, LifecycleState.RUNNING)

        await manager.wait_for_all_services_registration(
            stop_event=asyncio.Event(),
            timeout_seconds=1.0,
        )

    @pytest.mark.asyncio
    async def test_wait_for_all_services_registration_timeout(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify timeout raises AIPerfError with missing services."""
        asyncio.sleep = real_sleep

        service = create_service_info(service_id="worker_timeout")
        await manager.registry.expect(service)

        with pytest.raises(AIPerfError, match=r"Timeout: 0/1"):
            await manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(),
                timeout_seconds=0.1,
            )

    @pytest.mark.asyncio
    async def test_wait_for_all_services_registration_partial_registration(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify timeout error shows which services are missing."""
        asyncio.sleep = real_sleep

        services = [create_service_info(service_id=f"worker_{i}") for i in range(3)]
        await manager.registry.expect(*services)
        await manager.registry.register(services[0].service_id, LifecycleState.RUNNING)

        with pytest.raises(AIPerfError) as exc_info:
            await manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(),
                timeout_seconds=0.1,
            )

        error_msg = str(exc_info.value)
        assert "1/3" in error_msg
        assert "worker_1" in error_msg
        assert "worker_2" in error_msg


class TestServiceShutdown:
    """Test service shutdown and termination."""

    @pytest.mark.asyncio
    async def test_stop_service_terminates_process(
        self,
        manager: MultiProcessServiceManager,
        service_info_alive: MultiProcessRunInfo,
    ) -> None:
        """Verify stop_service terminates process and removes from registry."""
        await manager.registry.expect(service_info_alive)
        await manager.registry.register(
            service_info_alive.service_id, LifecycleState.RUNNING
        )

        with patch.object(
            manager, "_terminate_process", new_callable=AsyncMock
        ) as mock_terminate:
            await manager.stop_service(
                ServiceType.WORKER,
                service_id=service_info_alive.service_id,
            )

            mock_terminate.assert_called_once()
            assert (
                mock_terminate.call_args[0][0].service_id
                == service_info_alive.service_id
            )

    @pytest.mark.asyncio
    async def test_stop_service_by_type_without_service_id(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify stop_service stops all services of given type."""
        services = [
            create_service_info(
                service_id=f"worker_{i}",
                process=Mock(is_alive=Mock(return_value=True), pid=1000 + i),
                required=False,
            )
            for i in range(3)
        ]

        for service in services:
            await manager.registry.expect(service)
            await manager.registry.register(service.service_id, LifecycleState.RUNNING)

        with patch.object(
            manager, "_terminate_process", new_callable=AsyncMock
        ) as mock_terminate:
            await manager.stop_service(ServiceType.WORKER)
            assert mock_terminate.call_count == 3

    @pytest.mark.asyncio
    async def test_stop_service_returns_empty_list_when_no_matching_services(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify stop_service returns empty when no services match."""
        results = await manager.stop_service(
            ServiceType.WORKER,
            service_id="nonexistent_service",
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_shutdown_all_services_terminates_all_processes(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify shutdown_all_services terminates all registered services."""
        services = [
            create_service_info(
                service_id=f"worker_{i}",
                process=Mock(is_alive=Mock(return_value=True), pid=2000 + i),
                required=False,
            )
            for i in range(2)
        ]

        for service in services:
            await manager.registry.expect(service)
            await manager.registry.register(service.service_id, LifecycleState.RUNNING)

        with patch.object(
            manager, "_terminate_process", new_callable=AsyncMock
        ) as mock_terminate:
            results = await manager.shutdown_all_services()
            assert mock_terminate.call_count == 2
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_kill_all_services_force_kills_processes(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify kill_all_services uses kill() instead of terminate()."""
        services = [
            create_service_info(
                service_id=f"worker_{i}",
                process=Mock(
                    is_alive=Mock(return_value=True), kill=Mock(), pid=3000 + i
                ),
                required=False,
            )
            for i in range(2)
        ]

        for service in services:
            await manager.registry.expect(service)
            await manager.registry.register(service.service_id, LifecycleState.RUNNING)

        with patch.object(manager, "_terminate_process", new_callable=AsyncMock):
            await manager.kill_all_services()
            for service in services:
                service.process.kill.assert_called_once()


class TestProcessTermination:
    """Test internal process termination logic."""

    @pytest.mark.asyncio
    async def test_terminate_process_graceful_shutdown(
        self,
        manager: MultiProcessServiceManager,
        service_info_alive: MultiProcessRunInfo,
    ) -> None:
        """Verify _terminate_process performs graceful termination."""
        await manager.registry.expect(service_info_alive)
        await manager.registry.register(
            service_info_alive.service_id, LifecycleState.RUNNING
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock):
            await manager._terminate_process(service_info_alive)

            service_info_alive.process.terminate.assert_called_once()
            assert not await manager.registry.is_registered(
                service_info_alive.service_id
            )

    @pytest.mark.asyncio
    async def test_terminate_process_kill_on_timeout(
        self,
        manager: MultiProcessServiceManager,
        service_info_alive: MultiProcessRunInfo,
    ) -> None:
        """Verify _terminate_process kills process if join times out."""
        await manager.registry.expect(service_info_alive)
        await manager.registry.register(
            service_info_alive.service_id, LifecycleState.RUNNING
        )

        with patch(
            "asyncio.to_thread", new_callable=AsyncMock, side_effect=TimeoutError
        ):
            await manager._terminate_process(service_info_alive)
            service_info_alive.process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_process_handles_already_dead_process(
        self,
        manager: MultiProcessServiceManager,
        service_info_dead: MultiProcessRunInfo,
    ) -> None:
        """Verify _terminate_process handles already dead processes gracefully."""
        await manager._terminate_process(service_info_dead)

        service_info_dead.process.terminate.assert_not_called()
        service_info_dead.process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_terminate_process_handles_none_process(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify _terminate_process handles service with None process."""
        service_info = create_service_info(
            service_id="worker_none", process=None, required=False
        )
        await manager._terminate_process(service_info)


class TestHealthMonitoring:
    """Test background health monitoring functionality."""

    @pytest.mark.asyncio
    async def test_monitor_health_detects_dead_process(
        self,
        manager: MultiProcessServiceManager,
        service_info_dead: MultiProcessRunInfo,
    ) -> None:
        """Verify monitor detects dead processes and marks as failed."""
        await manager.registry.expect(service_info_dead)
        await manager.registry.register(
            service_info_dead.service_id, LifecycleState.RUNNING
        )

        await manager._monitor_health()

        failures = await manager.registry.get_failures()
        assert service_info_dead.service_id in failures

    @pytest.mark.asyncio
    async def test_monitor_health_publishes_failure_message_for_required_services(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify monitor publishes ServiceFailedMessage for required services."""
        service = create_service_info(
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dataset_dead",
            process=Mock(is_alive=Mock(return_value=False), exitcode=1, pid=4000),
        )

        await manager.registry.expect(service)
        await manager.registry.register(service.service_id, LifecycleState.RUNNING)

        await manager._monitor_health()

        manager.publish.assert_called_once()
        published_msg = manager.publish.call_args[0][0]
        assert isinstance(published_msg, ServiceFailedMessage)
        assert published_msg.service_id == service.service_id
        assert "code 1" in published_msg.reason.lower()

    @pytest.mark.asyncio
    async def test_monitor_health_does_not_publish_for_optional_services(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify monitor does not publish failures for optional services."""
        service = create_service_info(
            service_id="worker_optional_dead",
            process=Mock(is_alive=Mock(return_value=False), exitcode=2, pid=5000),
            required=False,
        )

        await manager.registry.expect(service)
        await manager.registry.register(service.service_id, LifecycleState.RUNNING)

        await manager._monitor_health()
        manager.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_monitor_health_skips_when_stop_requested(
        self,
        manager: MultiProcessServiceManager,
        service_info_dead: MultiProcessRunInfo,
    ) -> None:
        """Verify monitor exits early when stop is requested."""
        await manager.registry.expect(service_info_dead)
        await manager.registry.register(
            service_info_dead.service_id, LifecycleState.RUNNING
        )
        manager.stop_requested = True

        await manager._monitor_health()

        failures = await manager.registry.get_failures()
        assert service_info_dead.service_id not in failures


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_stop_service_calls(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify concurrent stop_service calls are handled safely."""
        services = [
            create_service_info(
                service_id=f"worker_{i}",
                process=Mock(
                    is_alive=Mock(return_value=True),
                    terminate=Mock(),
                    kill=Mock(),
                    pid=6000 + i,
                ),
                required=False,
            )
            for i in range(3)
        ]

        for service in services:
            await manager.registry.expect(service)
            await manager.registry.register(service.service_id, LifecycleState.RUNNING)

        with patch("asyncio.to_thread", new_callable=AsyncMock):
            results = await asyncio.gather(
                manager.stop_service(ServiceType.WORKER),
                manager.stop_service(ServiceType.WORKER),
                return_exceptions=True,
            )
            assert all(not isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_shutdown_with_no_services(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify shutdown operations work when no services running."""
        results = await manager.shutdown_all_services()
        assert results == []

        results = await manager.kill_all_services()
        assert results == []

    @pytest.mark.asyncio
    async def test_monitor_health_with_mixed_process_states(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Verify monitor handles mix of alive, dead, and None processes."""
        services = [
            create_service_info(
                service_id="worker_alive",
                process=Mock(is_alive=Mock(return_value=True), pid=7001),
            ),
            create_service_info(
                service_id="worker_dead",
                process=Mock(is_alive=Mock(return_value=False), exitcode=1, pid=7002),
            ),
            create_service_info(
                service_id="worker_none",
                process=None,
                required=False,
            ),
        ]

        for service in services:
            await manager.registry.expect(service)
            if service.process:
                await manager.registry.register(
                    service.service_id, LifecycleState.RUNNING
                )

        await manager._monitor_health()

        failures = await manager.registry.get_failures()
        assert "worker_dead" in failures
        assert "worker_alive" not in failures


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_service_lifecycle(
        self, manager: MultiProcessServiceManager, mock_process_factory
    ) -> None:
        """Test complete lifecycle: start -> register -> monitor -> shutdown."""
        with mock_process_factory() as mocks:
            await manager.run_service(ServiceType.WORKER, num_replicas=1)

            services = await manager.registry.get_services()
            assert len(services) == 1
            service = services[0]
            await manager.registry.register(service.service_id, LifecycleState.RUNNING)

            assert await manager.registry.is_registered(service.service_id)

            await manager._monitor_health()
            failures = await manager.registry.get_failures()
            assert len(failures) == 0

            with patch("asyncio.to_thread", new_callable=AsyncMock):
                await manager.shutdown_all_services()
                mocks["process"].terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_failure_detection_and_reporting(
        self, manager: MultiProcessServiceManager
    ) -> None:
        """Test failure detection publishes appropriate messages."""
        service = create_service_info(
            service_type=ServiceType.TIMING_MANAGER,
            service_id="timing_test",
            process=Mock(is_alive=Mock(return_value=False), exitcode=137, pid=9000),
        )

        await manager.registry.expect(service)
        await manager.registry.register(service.service_id, LifecycleState.RUNNING)

        await manager._monitor_health()

        failures = await manager.registry.get_failures()
        assert service.service_id in failures

        assert manager.publish.call_count == 1
        msg = manager.publish.call_args[0][0]
        assert msg.service_id == service.service_id
        assert "137" in msg.reason
