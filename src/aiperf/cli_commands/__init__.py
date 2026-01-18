# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for AIPerf."""

from aiperf.cli_commands.analyze_trace import (
    STAT_COLUMNS,
    analyze_app,
    analyze_trace,
)
from aiperf.cli_commands.docs import (
    docs_app,
    docs_command,
)
from aiperf.cli_commands.help import (
    help_app,
    help_command,
)

__all__ = [
    "STAT_COLUMNS",
    "analyze_app",
    "analyze_trace",
    "docs_app",
    "docs_command",
    "help_app",
    "help_command",
]
