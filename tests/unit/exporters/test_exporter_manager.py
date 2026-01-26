# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.exporter_manager import ExporterManager
from aiperf.plugin.enums import (
    EndpointType,
)


@pytest.fixture
def endpoint_config():
    return EndpointConfig(type=EndpointType.CHAT, streaming=True, model_names=["gpt2"])


@pytest.fixture
def output_config(tmp_path):
    return OutputConfig(artifact_directory=tmp_path)


@pytest.fixture
def sample_records():
    return [
        MetricResult(
            tag="Latency",
            unit="ms",
            avg=10.0,
            header="test-header",
        )
    ]


@pytest.fixture
def mock_user_config(endpoint_config, output_config):
    config = UserConfig(endpoint=endpoint_config, output=output_config)
    return config


class TestExporterManager:
    @pytest.mark.asyncio
    async def test_export(
        self, endpoint_config, output_config, sample_records, mock_user_config
    ):
        # Create a mock exporter instance
        mock_instance = MagicMock()
        mock_instance.export = AsyncMock()
        mock_class = MagicMock(return_value=mock_instance)

        # Create a mock PluginType that returns our mock class when loaded
        mock_plugin_type = MagicMock()
        mock_plugin_type.load.return_value = mock_class

        with patch(
            "aiperf.exporters.exporter_manager.plugins.list_types",
            return_value=[mock_plugin_type],
        ):
            manager = ExporterManager(
                results=ProfileResults(
                    records=sample_records,
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                    was_cancelled=False,
                    error_summary=[],
                ),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )
            await manager.export_data()

        mock_plugin_type.load.assert_called_once()
        mock_class.assert_called_once()
        mock_instance.export.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_export_console(
        self, endpoint_config, output_config, sample_records, mock_user_config
    ):
        from rich.console import Console

        # Create mock exporter instances for each console exporter type
        mock_instances = []
        mock_classes = []
        mock_plugin_types = []

        for _ in range(2):  # Simulate two console exporters
            instance = MagicMock()
            instance.export = AsyncMock()
            mock_class = MagicMock(return_value=instance)
            mock_plugin_type = MagicMock()
            mock_plugin_type.load.return_value = mock_class

            mock_instances.append(instance)
            mock_classes.append(mock_class)
            mock_plugin_types.append(mock_plugin_type)

        with patch(
            "aiperf.exporters.exporter_manager.plugins.list_types",
            return_value=mock_plugin_types,
        ):
            manager = ExporterManager(
                results=ProfileResults(
                    records=sample_records,
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                    was_cancelled=False,
                    error_summary=[],
                ),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )
            await manager.export_console(Console())

        for mock_plugin_type, mock_class, mock_instance in zip(
            mock_plugin_types, mock_classes, mock_instances, strict=False
        ):
            mock_plugin_type.load.assert_called_once()
            mock_class.assert_called_once()
            mock_instance.export.assert_awaited_once()
