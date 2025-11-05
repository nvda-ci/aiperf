# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants specific to GPU telemetry collection."""

from aiperf.common.metrics import (
    METRIC_NAME_ACRONYMS,
    format_metric_display_name,
    infer_metric_unit,
)

# Re-export for backward compatibility
__all__ = [
    "METRIC_NAME_ACRONYMS",
    "format_metric_display_name",
    "infer_metric_unit",
    "SCALING_FACTORS",
    "DCGM_TO_FIELD_MAPPING",
]


# Unit conversion scaling factors
SCALING_FACTORS = {
    "energy_consumption": 1e-9,  # mJ to MJ
    "gpu_memory_used": 1.048576 * 1e-3,  # MiB to GB
}

# DCGM field mapping to telemetry record fields
# This mapping is required because DCGM uses specific field IDs that need translation
DCGM_TO_FIELD_MAPPING = {
    "DCGM_FI_DEV_POWER_USAGE": "gpu_power_usage",
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": "energy_consumption",
    "DCGM_FI_DEV_GPU_UTIL": "gpu_utilization",
    "DCGM_FI_DEV_FB_USED": "gpu_memory_used",
    "DCGM_FI_DEV_GPU_TEMP": "gpu_temperature",
    "DCGM_FI_DEV_XID_ERRORS": "xid_errors",
    "DCGM_FI_DEV_POWER_VIOLATION": "power_violation",
}
