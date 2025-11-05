# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants specific to server metrics collection."""

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
]


# Unit conversion scaling factors
# Maps field names to their scaling multipliers
SCALING_FACTORS = {
    # Convert 0.0-1.0 decimal percentages to 0-100 percentages
    "kvstats_gpu_cache_usage_percent": 100.0,
    "kvstats_gpu_prefix_cache_hit_rate": 100.0,
    "cpu_usage_percent": 100.0,
    "vllm_kv_cache_usage_perc": 100.0,
}
