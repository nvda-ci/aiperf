# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TimingManager service."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.enums import TimingMode
from aiperf.common.environment import Environment
from aiperf.common.messages import (
    CommandMessage,
    DatasetConfiguredNotification,
    ProfileCancelCommand,
    ProfileConfigureCommand,
)
from aiperf.common.models import DatasetMetadata, MemoryMapClientMetadata
from aiperf.timing.manager import TimingManager
from tests.unit.timing.conftest import create_mock_dataset_metadata_with_schedule


def make_mock_client_metadata() -> MemoryMapClientMetadata:
    """Create mock client metadata for testing."""
    return MemoryMapClientMetadata(
        data_file_path=Path("/tmp/test_data.mmap"),
        index_file_path=Path("/tmp/test_index.mmap"),
        conversation_count=3,
        total_size_bytes=1024,
    )


# =============================================================================
# Module-Level Fixtures (shared across all test classes)
# =============================================================================


@pytest.fixture
def timing_manager_user_config():
    """User config fixture for timing manager tests."""
    return UserConfig.model_construct(
        endpoint=MagicMock(),
        _timing_mode=TimingMode.REQUEST_RATE,
    )


@pytest.fixture
def create_timing_manager(service_config):
    """Factory fixture to create TimingManager instances."""

    def _create(user_config: UserConfig) -> TimingManager:
        return TimingManager(
            service_config=service_config,
            user_config=user_config,
            service_id="test-timing-manager",
        )

    return _create


@pytest.fixture
def configured_timing_manager(create_timing_manager, timing_manager_user_config):
    """Create a configured timing manager with mock orchestrator."""

    async def async_noop(*args, **kwargs):
        return None

    manager = create_timing_manager(timing_manager_user_config)
    manager._phase_orchestrator = MagicMock()
    manager._phase_orchestrator.start = MagicMock(side_effect=async_noop)
    manager._phase_orchestrator.stop = MagicMock(side_effect=async_noop)
    manager._phase_orchestrator.cancel = MagicMock(side_effect=async_noop)
    manager.initialized_event.set()
    return manager


# =============================================================================
# Dataset Configuration Tests
# =============================================================================


class TestTimingManagerDatasetConfiguration:
    """Test suite for TimingManager dataset configuration via notification."""

    @pytest.fixture
    def user_config_fixed_schedule(self):
        """User config with fixed schedule timing mode."""
        return UserConfig.model_construct(
            endpoint=MagicMock(),
            _timing_mode=TimingMode.FIXED_SCHEDULE,
        )

    @pytest.fixture
    def user_config_request_rate(self):
        """User config with request rate timing mode."""
        return UserConfig.model_construct(
            endpoint=MagicMock(),
            _timing_mode=TimingMode.REQUEST_RATE,
        )

    @pytest.fixture
    def mock_dataset_metadata(self) -> DatasetMetadata:
        """Create mock dataset metadata with schedule."""
        return create_mock_dataset_metadata_with_schedule(
            schedule=[(0, "conv1"), (100, "conv2"), (200, "conv3")]
        )

    @pytest.mark.asyncio
    async def test_profile_configure_waits_for_dataset_notification_fixed_schedule(
        self,
        create_timing_manager,
        user_config_fixed_schedule,
        mock_dataset_metadata,
    ):
        """Test that profile configure command waits for dataset notification for fixed schedule mode."""
        manager = create_timing_manager(user_config_fixed_schedule)

        mock_engine = MagicMock()

        async def async_init(*args, **kwargs):
            return None

        mock_engine.initialize = async_init

        with patch(
            "aiperf.timing.manager.PhaseOrchestrator",
            return_value=mock_engine,
        ) as mock_orchestrator:
            import asyncio

            configure_task = asyncio.create_task(
                manager._profile_configure_command(
                    ProfileConfigureCommand.model_construct(
                        service_id="test-system-controller",
                        config={},
                    )
                )
            )

            await asyncio.sleep(0.1)

            await manager._on_dataset_configured_notification(
                DatasetConfiguredNotification(
                    service_id="test-dataset-manager",
                    metadata=mock_dataset_metadata,
                    client_metadata=make_mock_client_metadata(),
                )
            )

            await configure_task

            assert manager._dataset_metadata == mock_dataset_metadata
            mock_orchestrator.assert_called_once()
            call_kwargs = mock_orchestrator.call_args.kwargs
            assert "dataset_metadata" in call_kwargs
            assert call_kwargs["dataset_metadata"] == mock_dataset_metadata

    @pytest.mark.asyncio
    async def test_profile_configure_waits_for_dataset_notification_request_rate(
        self,
        create_timing_manager,
        user_config_request_rate,
        mock_dataset_metadata,
    ):
        """Test that profile configure command waits for dataset notification for request rate mode."""
        manager = create_timing_manager(user_config_request_rate)

        mock_engine = MagicMock()

        async def async_init(*args, **kwargs):
            return None

        mock_engine.initialize = async_init

        with patch(
            "aiperf.timing.manager.PhaseOrchestrator",
            return_value=mock_engine,
        ) as mock_orchestrator:
            import asyncio

            configure_task = asyncio.create_task(
                manager._profile_configure_command(
                    ProfileConfigureCommand.model_construct(
                        service_id="test-system-controller",
                        config={},
                    )
                )
            )

            await asyncio.sleep(0.1)

            await manager._on_dataset_configured_notification(
                DatasetConfiguredNotification(
                    service_id="test-dataset-manager",
                    metadata=mock_dataset_metadata,
                    client_metadata=make_mock_client_metadata(),
                )
            )

            await configure_task

            assert manager._dataset_metadata == mock_dataset_metadata
            mock_orchestrator.assert_called_once()
            call_kwargs = mock_orchestrator.call_args.kwargs
            assert "dataset_metadata" in call_kwargs
            assert call_kwargs["dataset_metadata"] == mock_dataset_metadata

    @pytest.mark.asyncio
    async def test_dataset_configuration_timeout(
        self, create_timing_manager, user_config_fixed_schedule
    ):
        """Test that profile configure command times out if dataset notification is not received."""
        manager = create_timing_manager(user_config_fixed_schedule)

        import asyncio

        with (
            patch.object(Environment.DATASET, "CONFIGURATION_TIMEOUT", 0.1),
            pytest.raises(asyncio.TimeoutError),
        ):
            await manager._profile_configure_command(
                ProfileConfigureCommand.model_construct(
                    service_id="test-system-controller",
                    config={},
                )
            )

    @pytest.mark.asyncio
    async def test_dataset_notification_before_configure(
        self,
        create_timing_manager,
        user_config_fixed_schedule,
        mock_dataset_metadata,
    ):
        """Test that dataset notification can be received before profile configure command."""
        manager = create_timing_manager(user_config_fixed_schedule)

        await manager._on_dataset_configured_notification(
            DatasetConfiguredNotification(
                service_id="test-dataset-manager",
                metadata=mock_dataset_metadata,
                client_metadata=make_mock_client_metadata(),
            )
        )

        assert manager._dataset_metadata == mock_dataset_metadata

        mock_engine = MagicMock()

        async def async_init(*args, **kwargs):
            return None

        mock_engine.initialize = async_init

        with patch(
            "aiperf.timing.manager.PhaseOrchestrator",
            return_value=mock_engine,
        ) as mock_orchestrator:
            await manager._profile_configure_command(
                ProfileConfigureCommand.model_construct(
                    service_id="test-system-controller",
                    config={},
                )
            )

            mock_orchestrator.assert_called_once()
            call_kwargs = mock_orchestrator.call_args.kwargs
            assert "dataset_metadata" in call_kwargs
            assert call_kwargs["dataset_metadata"] == mock_dataset_metadata


# =============================================================================
# Garbage Collection Tests
# =============================================================================


class TestTimingManagerGarbageCollection:
    """Test suite for TimingManager garbage collection control."""

    @pytest.mark.asyncio
    async def test_gc_disabled_on_profiling_start(self, configured_timing_manager):
        """Test that garbage collection is collected, frozen, and disabled when profiling starts."""
        with patch("aiperf.timing.manager.gc") as mock_gc:
            await configured_timing_manager._on_start_profiling(
                CommandMessage.model_construct(service_id="test-controller")
            )

            assert mock_gc.collect.called
            assert mock_gc.freeze.called
            assert mock_gc.disable.called

            calls = mock_gc.method_calls
            call_names = [c[0] for c in calls]
            collect_idx = call_names.index("collect")
            freeze_idx = call_names.index("freeze")
            disable_idx = call_names.index("disable")
            assert collect_idx < freeze_idx < disable_idx

    @pytest.mark.asyncio
    async def test_gc_enabled_on_stop(self, configured_timing_manager):
        """Test that garbage collection is unfrozen and re-enabled when timing manager stops."""
        with patch("aiperf.timing.manager.gc") as mock_gc:
            await configured_timing_manager._timing_manager_stop()

            assert mock_gc.unfreeze.called
            assert mock_gc.enable.called

            calls = mock_gc.method_calls
            call_names = [c[0] for c in calls]
            unfreeze_idx = call_names.index("unfreeze")
            enable_idx = call_names.index("enable")
            assert unfreeze_idx < enable_idx

    @pytest.mark.asyncio
    async def test_gc_enabled_on_stop_without_strategy(
        self, create_timing_manager, timing_manager_user_config
    ):
        """Test that GC is re-enabled even if no strategy was configured."""
        manager = create_timing_manager(timing_manager_user_config)

        with patch("aiperf.timing.manager.gc") as mock_gc:
            await manager._timing_manager_stop()

            assert mock_gc.unfreeze.called
            assert mock_gc.enable.called


# =============================================================================
# Cancel Command Tests
# =============================================================================


class TestTimingManagerCancelCommand:
    """Tests for PROFILE_CANCEL command handling."""

    @pytest.mark.asyncio
    async def test_cancel_calls_orchestrator_cancel(self, configured_timing_manager):
        """Cancel command calls orchestrator.cancel()."""
        await configured_timing_manager._handle_profile_cancel_command(
            ProfileCancelCommand.model_construct(service_id="test-controller")
        )

        configured_timing_manager._phase_orchestrator.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_without_orchestrator_is_safe(
        self, create_timing_manager, timing_manager_user_config
    ):
        """Cancel command is safe when no orchestrator is configured."""
        manager = create_timing_manager(timing_manager_user_config)

        # Should not raise
        await manager._handle_profile_cancel_command(
            ProfileCancelCommand.model_construct(service_id="test-controller")
        )

    @pytest.mark.asyncio
    async def test_cancel_is_idempotent(self, configured_timing_manager):
        """Cancel command can be called multiple times."""
        await configured_timing_manager._handle_profile_cancel_command(
            ProfileCancelCommand.model_construct(service_id="test-controller")
        )
        await configured_timing_manager._handle_profile_cancel_command(
            ProfileCancelCommand.model_construct(service_id="test-controller")
        )

        assert configured_timing_manager._phase_orchestrator.cancel.call_count == 2


# =============================================================================
# Start Profiling Error Handling Tests
# =============================================================================


class TestTimingManagerStartProfilingErrors:
    """Tests for PROFILE_START command error handling."""

    @pytest.mark.asyncio
    async def test_start_profiling_without_orchestrator_raises(
        self, create_timing_manager, timing_manager_user_config
    ):
        """Start profiling raises InvalidStateError when no orchestrator configured."""
        from aiperf.common.exceptions import InvalidStateError

        manager = create_timing_manager(timing_manager_user_config)

        with pytest.raises(InvalidStateError, match="No phase orchestrator configured"):
            await manager._on_start_profiling(
                CommandMessage.model_construct(service_id="test-controller")
            )

    @pytest.mark.asyncio
    async def test_start_profiling_starts_orchestrator(
        self, create_timing_manager, timing_manager_user_config
    ):
        """Start profiling calls orchestrator.start()."""

        async def async_noop(*args, **kwargs):
            return None

        manager = create_timing_manager(timing_manager_user_config)
        manager._phase_orchestrator = MagicMock()
        manager._phase_orchestrator.start = async_noop

        with patch("aiperf.timing.manager.gc"):
            await manager._on_start_profiling(
                CommandMessage.model_construct(service_id="test-controller")
            )

        assert manager._phase_orchestrator is not None


# =============================================================================
# Dataset Configuration Error Tests
# =============================================================================


class TestTimingManagerDatasetErrors:
    """Tests for dataset configuration error handling."""

    @pytest.mark.asyncio
    async def test_configure_raises_when_event_set_but_no_metadata(
        self, create_timing_manager, timing_manager_user_config
    ):
        """Configure raises InvalidStateError if event set but metadata is None."""
        from aiperf.common.exceptions import InvalidStateError

        manager = create_timing_manager(timing_manager_user_config)
        manager._dataset_configured_event.set()  # Set event but no metadata

        with pytest.raises(
            InvalidStateError, match="Dataset metadata is not available"
        ):
            await manager._profile_configure_command(
                ProfileConfigureCommand.model_construct(
                    service_id="test-controller",
                    config={},
                )
            )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestTimingManagerInitialization:
    """Tests for TimingManager initialization."""

    def test_creates_timing_config_from_user_config(
        self, create_timing_manager, timing_manager_user_config
    ):
        """TimingConfig is created from UserConfig."""
        manager = create_timing_manager(timing_manager_user_config)

        assert manager.config is not None
        assert manager.config.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE

    def test_creates_phase_publisher(
        self, create_timing_manager, timing_manager_user_config
    ):
        """PhasePublisher is created on init."""
        manager = create_timing_manager(timing_manager_user_config)

        assert manager.phase_publisher is not None

    def test_creates_sticky_router(
        self, create_timing_manager, timing_manager_user_config
    ):
        """StickyCreditRouter is created on init."""
        manager = create_timing_manager(timing_manager_user_config)

        assert manager.sticky_router is not None

    def test_no_orchestrator_initially(
        self, create_timing_manager, timing_manager_user_config
    ):
        """No orchestrator exists until PROFILE_CONFIGURE."""
        manager = create_timing_manager(timing_manager_user_config)

        assert manager._phase_orchestrator is None

    def test_dataset_event_not_set_initially(
        self, create_timing_manager, timing_manager_user_config
    ):
        """Dataset configured event is not set initially."""
        manager = create_timing_manager(timing_manager_user_config)

        assert not manager._dataset_configured_event.is_set()
