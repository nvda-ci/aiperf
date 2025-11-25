# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TimingManager service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import TimingMode
from aiperf.common.environment import Environment
from aiperf.common.messages import (
    CommandMessage,
    DatasetConfiguredNotification,
    ProfileConfigureCommand,
)
from aiperf.common.models import DatasetMetadata
from aiperf.timing.timing_manager import TimingManager
from tests.unit.timing.conftest import create_mock_dataset_metadata_with_schedule


class TestTimingManagerDatasetConfiguration:
    """Test suite for TimingManager dataset configuration via notification."""

    def _create_timing_manager(
        self, service_config: ServiceConfig, user_config: UserConfig
    ) -> TimingManager:
        """Create a TimingManager instance (ZMQ is globally mocked)."""
        return TimingManager(
            service_config=service_config,
            user_config=user_config,
            service_id="test-timing-manager",
        )

    @pytest.fixture
    def service_config(self):
        """Service configuration fixture."""
        return ServiceConfig()

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
        service_config,
        user_config_fixed_schedule,
        mock_dataset_metadata,
    ):
        """Test that profile configure command waits for dataset notification for fixed schedule mode."""
        manager = self._create_timing_manager(
            service_config, user_config_fixed_schedule
        )

        with patch(
            "aiperf.timing.timing_manager.CreditIssuingStrategyFactory.create_instance"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()

            # Start configure command in background (it will wait for notification)
            import asyncio

            configure_task = asyncio.create_task(
                manager._profile_configure_command(
                    ProfileConfigureCommand.model_construct(
                        service_id="test-system-controller",
                        config={},
                    )
                )
            )

            # Wait a bit to ensure the command is waiting
            await asyncio.sleep(0.1)

            # Send dataset configured notification
            await manager._on_dataset_configured_notification(
                DatasetConfiguredNotification(
                    service_id="test-dataset-manager",
                    metadata=mock_dataset_metadata,
                )
            )

            # Wait for configure command to complete
            await configure_task

            # Verify the dataset metadata was set
            assert manager._dataset_metadata == mock_dataset_metadata

            # Verify the strategy factory was called with dataset metadata
            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args.kwargs
            assert "dataset_metadata" in call_kwargs
            assert call_kwargs["dataset_metadata"] == mock_dataset_metadata

    @pytest.mark.asyncio
    async def test_profile_configure_waits_for_dataset_notification_request_rate(
        self,
        service_config,
        user_config_request_rate,
        mock_dataset_metadata,
    ):
        """Test that profile configure command waits for dataset notification for request rate mode."""
        manager = self._create_timing_manager(service_config, user_config_request_rate)

        with patch(
            "aiperf.timing.timing_manager.CreditIssuingStrategyFactory.create_instance"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()

            # Start configure command in background (it will wait for notification)
            import asyncio

            configure_task = asyncio.create_task(
                manager._profile_configure_command(
                    ProfileConfigureCommand.model_construct(
                        service_id="test-system-controller",
                        config={},
                    )
                )
            )

            # Wait a bit to ensure the command is waiting
            await asyncio.sleep(0.1)

            # Send dataset configured notification
            await manager._on_dataset_configured_notification(
                DatasetConfiguredNotification(
                    service_id="test-dataset-manager",
                    metadata=mock_dataset_metadata,
                )
            )

            # Wait for configure command to complete
            await configure_task

            # Verify the dataset metadata was set
            assert manager._dataset_metadata == mock_dataset_metadata

            # Verify the strategy factory was called with dataset metadata
            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args.kwargs
            assert "dataset_metadata" in call_kwargs
            assert call_kwargs["dataset_metadata"] == mock_dataset_metadata

    @pytest.mark.asyncio
    async def test_dataset_configuration_timeout(
        self, service_config, user_config_fixed_schedule
    ):
        """Test that profile configure command times out if dataset notification is not received."""
        manager = self._create_timing_manager(
            service_config, user_config_fixed_schedule
        )

        # Mock the timeout to be very short for testing
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
        service_config,
        user_config_fixed_schedule,
        mock_dataset_metadata,
    ):
        """Test that dataset notification can be received before profile configure command."""
        manager = self._create_timing_manager(
            service_config, user_config_fixed_schedule
        )

        # Send dataset configured notification BEFORE configure command
        await manager._on_dataset_configured_notification(
            DatasetConfiguredNotification(
                service_id="test-dataset-manager",
                metadata=mock_dataset_metadata,
            )
        )

        # Verify the dataset metadata was set
        assert manager._dataset_metadata == mock_dataset_metadata

        # Now send configure command - it should proceed immediately
        with patch(
            "aiperf.timing.timing_manager.CreditIssuingStrategyFactory.create_instance"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()

            await manager._profile_configure_command(
                ProfileConfigureCommand.model_construct(
                    service_id="test-system-controller",
                    config={},
                )
            )

            # Verify the strategy factory was called with dataset metadata
            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args.kwargs
            assert "dataset_metadata" in call_kwargs
            assert call_kwargs["dataset_metadata"] == mock_dataset_metadata


class TestTimingManagerGarbageCollection:
    """Test suite for TimingManager garbage collection control."""

    def _create_timing_manager(
        self, service_config: ServiceConfig, user_config: UserConfig
    ) -> TimingManager:
        """Create a TimingManager instance (ZMQ is globally mocked)."""
        return TimingManager(
            service_config=service_config,
            user_config=user_config,
            service_id="test-timing-manager",
        )

    @pytest.fixture
    def service_config(self):
        """Service configuration fixture."""
        return ServiceConfig()

    @pytest.fixture
    def user_config(self):
        """User config fixture."""
        return UserConfig.model_construct(
            endpoint=MagicMock(),
            _timing_mode=TimingMode.REQUEST_RATE,
        )

    @pytest.fixture
    def configured_manager(self, service_config, user_config):
        """Create a configured timing manager with strategy."""
        manager = self._create_timing_manager(service_config, user_config)
        manager._credit_issuing_strategy = MagicMock()
        manager._credit_issuing_strategy.start = AsyncMock()
        manager._credit_issuing_strategy.stop = AsyncMock()
        manager.initialized_event.set()
        return manager

    @pytest.mark.asyncio
    async def test_gc_disabled_on_profiling_start(self, configured_manager):
        """Test that garbage collection is collected, frozen, and disabled when profiling starts."""
        with patch("aiperf.timing.timing_manager.gc") as mock_gc:
            await configured_manager._on_start_profiling(
                CommandMessage.model_construct(service_id="test-controller")
            )

            assert mock_gc.collect.called
            assert mock_gc.freeze.called
            assert mock_gc.disable.called

            # Verify correct order: collect -> freeze -> disable
            calls = mock_gc.method_calls
            call_names = [c[0] for c in calls]
            collect_idx = call_names.index("collect")
            freeze_idx = call_names.index("freeze")
            disable_idx = call_names.index("disable")
            assert collect_idx < freeze_idx < disable_idx

    @pytest.mark.asyncio
    async def test_gc_enabled_on_stop(self, configured_manager):
        """Test that garbage collection is unfrozen and re-enabled when timing manager stops."""
        with patch("aiperf.timing.timing_manager.gc") as mock_gc:
            await configured_manager._timing_manager_stop()

            assert mock_gc.unfreeze.called
            assert mock_gc.enable.called

            # Verify correct order: unfreeze -> enable
            calls = mock_gc.method_calls
            call_names = [c[0] for c in calls]
            unfreeze_idx = call_names.index("unfreeze")
            enable_idx = call_names.index("enable")
            assert unfreeze_idx < enable_idx

    @pytest.mark.asyncio
    async def test_gc_enabled_on_stop_without_strategy(
        self, service_config, user_config
    ):
        """Test that GC is re-enabled even if no strategy was configured."""
        manager = self._create_timing_manager(service_config, user_config)

        with patch("aiperf.timing.timing_manager.gc") as mock_gc:
            await manager._timing_manager_stop()

            assert mock_gc.unfreeze.called
            assert mock_gc.enable.called
