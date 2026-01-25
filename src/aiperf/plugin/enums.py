# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Dynamic Plugin Enums

All plugin-based type enums are generated here dynamically from the PluginRegistry.
This ensures:
- No circular imports (plugin module loads registry before any plugin modules)
- Enums loaded AFTER all types are registered
- Works with pydantic validation
- Works with cyclopts CLI parsing
- Supports hardcoded checks (e.g., ui_type == UIType.DASHBOARD)
- Can be extended at runtime if new plugins are registered

Import these types from aiperf.plugin.enums or aiperf.plugin.
"""

from typing import TYPE_CHECKING

from aiperf.common import plugin_registry
from aiperf.common.enums.base_enums import ExtensibleStrEnum

# ============================================================================
# Plugin Protocol Categories
# ============================================================================

# Create enum with all plugin category names from the registry
_all_plugin_categories = plugin_registry.list_categories()
if not TYPE_CHECKING:
    PluginCategory = ExtensibleStrEnum(
        "PluginCategory",
        {
            category.replace("-", "_").upper(): category
            for category in _all_plugin_categories
        },
    )
"""
Dynamic enum for plugin categories.

Members are auto-generated from registered plugin categories in registry.yaml.
Example: PluginCategory.ENDPOINT, PluginCategory.UI, PluginCategory.TRANSPORT, etc.
"""

# ============================================================================
# UI Types
# ============================================================================

UIType = plugin_registry.create_enum("ui", "UIType")
"""
Dynamic enum for UI implementations.

Members are auto-generated from registered UI plugins.
Example: UIType.DASHBOARD, UIType.SIMPLE, UIType.NONE
"""

# ============================================================================
# Transport Types
# ============================================================================

TransportType = plugin_registry.create_enum("transport", "TransportType")
"""
Dynamic enum for transport implementations.

Members are auto-generated from registered transport plugins.
Example: TransportType.HTTP
"""

# ============================================================================
# Endpoint Types
# ============================================================================

EndpointType = plugin_registry.create_enum("endpoint", "EndpointType")
"""
Dynamic enum for endpoint implementations.

Members are auto-generated from registered endpoint plugins.
Example: EndpointType.CHAT, EndpointType.COMPLETIONS, EndpointType.EMBEDDINGS
"""

# ============================================================================
# Exporter Types
# ============================================================================

DataExporterType = plugin_registry.create_enum("data_exporter", "DataExporterType")
"""
Dynamic enum for data exporter implementations.

Members are auto-generated from registered data_exporter plugins.
Example: DataExporterType.CSV, DataExporterType.JSON
"""

ConsoleExporterType = plugin_registry.create_enum(
    "console_exporter", "ConsoleExporterType"
)
"""
Dynamic enum for console exporter implementations.

Members are auto-generated from registered console_exporter plugins.
Example: ConsoleExporterType.ERRORS, ConsoleExporterType.METRICS
"""

# ============================================================================
# Dataset Types
# ============================================================================

DatasetSamplingStrategy = plugin_registry.create_enum(
    "dataset_sampler", "DatasetSamplingStrategy"
)
"""
Dynamic enum for dataset sampling strategies.

Members are auto-generated from registered dataset_sampler plugins.
Example: DatasetSamplingStrategy.RANDOM, DatasetSamplingStrategy.SEQUENTIAL
"""

CustomDatasetType = plugin_registry.create_enum(
    "custom_dataset_loader", "CustomDatasetType"
)
"""
Dynamic enum for custom dataset loader implementations.

Members are auto-generated from registered custom_dataset_loader plugins.
Example: CustomDatasetType.SINGLE_TURN, CustomDatasetType.MULTI_TURN
"""

ComposerType = plugin_registry.create_enum("dataset_composer", "ComposerType")
"""
Dynamic enum for dataset composer implementations.

Members are auto-generated from registered dataset_composer plugins.
Example: ComposerType.SYNTHETIC, ComposerType.CUSTOM
"""

DatasetBackingStoreType = plugin_registry.create_enum(
    "dataset_backing_store", "DatasetBackingStoreType"
)
"""
Dynamic enum for dataset backing store implementations.

Members are auto-generated from registered dataset_backing_store plugins.
Example: DatasetBackingStoreType.MEMORY_MAP
"""

DatasetClientStoreType = plugin_registry.create_enum(
    "dataset_client_store", "DatasetClientStoreType"
)
"""
Dynamic enum for dataset client store implementations.

Members are auto-generated from registered dataset_client_store plugins.
Example: DatasetClientStoreType.MEMORY_MAP
"""

# ============================================================================
# Post-Processor Types
# ============================================================================

RecordProcessorType = plugin_registry.create_enum(
    "record_processor", "RecordProcessorType"
)
"""
Dynamic enum for record processor implementations.

Members are auto-generated from registered record_processor plugins.
Example: RecordProcessorType.METRIC_RECORD, RecordProcessorType.RAW_RECORD_WRITER
"""

ResultsProcessorType = plugin_registry.create_enum(
    "results_processor", "ResultsProcessorType"
)
"""
Dynamic enum for results processor implementations.

Members are auto-generated from registered results_processor plugins.
Example: ResultsProcessorType.METRIC_RESULTS, ResultsProcessorType.TIMESLICE
"""

# ============================================================================
# Timing Types
# ============================================================================

TimingMode = plugin_registry.create_enum("timing_strategy", "TimingMode")
"""
Dynamic enum for timing strategy implementations.

Members are auto-generated from registered timing_strategy plugins.
Example: TimingMode.REQUEST_RATE, TimingMode.FIXED_SCHEDULE
"""

ArrivalPattern = plugin_registry.create_enum("arrival_pattern", "ArrivalPattern")
"""
Dynamic enum for arrival pattern (interval generator) implementations.

Members are auto-generated from registered arrival_pattern plugins.
Example: ArrivalPattern.POISSON, ArrivalPattern.CONSTANT
"""

RampType = plugin_registry.create_enum("ramp", "RampType")
"""
Dynamic enum for ramp strategy implementations.

Members are auto-generated from registered ramp plugins.
Example: RampType.LINEAR, RampType.EXPONENTIAL
"""

# ============================================================================
# Service Types
# ============================================================================

ServiceRunType = plugin_registry.create_enum("service_manager", "ServiceRunType")
"""
Dynamic enum for service manager implementations.

Members are auto-generated from registered service_manager plugins.
Example: ServiceRunType.MULTIPROCESSING, ServiceRunType.KUBERNETES
"""

ServiceType = plugin_registry.create_enum("service", "ServiceType")
"""
Dynamic enum for service implementations.

Members are auto-generated from registered service plugins.
Example: ServiceType.DATASET_MANAGER, ServiceType.WORKER, ServiceType.SYSTEM_CONTROLLER
"""

# ============================================================================
# ZMQ Communication Types
# ============================================================================

CommClientType = plugin_registry.create_enum("communication_client", "CommClientType")
"""
Dynamic enum for ZMQ communication client implementations.

Members are auto-generated from registered communication_client plugins.
Example: CommClientType.PUB, CommClientType.SUB, CommClientType.REQUEST
"""

CommunicationBackend = plugin_registry.create_enum(
    "communication", "CommunicationBackend"
)
"""
Dynamic enum for ZMQ communication backend implementations.

Members are auto-generated from registered communication plugins.
Example: CommunicationBackend.ZMQ_IPC, CommunicationBackend.ZMQ_TCP
"""

ZMQProxyType = plugin_registry.create_enum("zmq_proxy", "ZMQProxyType")
"""
Dynamic enum for ZMQ proxy implementations.

Members are auto-generated from registered zmq_proxy plugins.
Example: ZMQProxyType.DEALER_ROUTER, ZMQProxyType.XPUB_XSUB
"""

# ============================================================================
# Plot Types
# ============================================================================

PlotType = plugin_registry.create_enum("plot", "PlotType")
"""
Dynamic enum for plot handler implementations.

Members are auto-generated from registered plot plugins.
Example: PlotType.SCATTER, PlotType.HISTOGRAM, PlotType.TIMESLICE
"""

__all__ = [
    "ArrivalPattern",
    "CommClientType",
    "CommunicationBackend",
    "ComposerType",
    "ConsoleExporterType",
    "CustomDatasetType",
    "DataExporterType",
    "DatasetBackingStoreType",
    "DatasetClientStoreType",
    "DatasetSamplingStrategy",
    "EndpointType",
    "PlotType",
    "PluginCategory",
    "RampType",
    "RecordProcessorType",
    "ResultsProcessorType",
    "ServiceRunType",
    "ServiceType",
    "TimingMode",
    "TransportType",
    "UIType",
    "ZMQProxyType",
]
