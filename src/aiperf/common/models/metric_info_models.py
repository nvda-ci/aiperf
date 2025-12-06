# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Info metric data model for static Prometheus _info metrics."""

from __future__ import annotations

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class InfoMetricData(AIPerfBaseModel):
    """Complete data for an info metric including label data.

    Info metrics (ending in _info) contain static system information that doesn't
    change over time. We store only the labels (not values) since the labels contain
    the actual information and values are typically just 1.0.
    """

    description: str = Field(description="Metric description from HELP text")
    labels: list[dict[str, str]] = Field(
        description="List of label keys and values as reported by the Prometheus endpoint"
    )
