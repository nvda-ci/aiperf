# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ConnectionReuseStrategy(CaseInsensitiveStrEnum):
    """Transport connection reuse strategy. Controls how and when connections are reused across requests."""

    POOLED = "pooled"
    """Connections are pooled and reused across all requests"""

    NEVER = "never"
    """New connection for each request, closed after response"""

    STICKY_USER_SESSIONS = "sticky-user-sessions"
    """Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing)"""
