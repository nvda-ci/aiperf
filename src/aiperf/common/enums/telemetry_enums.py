# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class GPUTelemetryCollectorType(CaseInsensitiveStrEnum):
    """GPU telemetry collector implementation type."""

    DCGM = "dcgm"
    """Collects GPU telemetry metrics from DCGM Prometheus exporter."""

    PYNVML = "pynvml"
    """Collects GPU telemetry metrics using the pynvml Python library."""


class GPUTelemetryMode(CaseInsensitiveStrEnum):
    """GPU telemetry display mode."""

    SUMMARY = "summary"
    REALTIME_DASHBOARD = "realtime_dashboard"
