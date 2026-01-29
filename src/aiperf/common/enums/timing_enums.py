# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class CreditPhase(CaseInsensitiveStrEnum):
    """The type of credit phase. This is used to identify which phase of the
    benchmark the credit is being used in, for tracking and reporting purposes."""

    WARMUP = "warmup"
    """The credit phase while the warmup is active. This is used to warm up the model and
    ensure that the model is ready to be profiled."""

    PROFILING = "profiling"
    """The credit phase while profiling is active. This is the primary phase of the
    benchmark, and what is used to calculate the final results."""
