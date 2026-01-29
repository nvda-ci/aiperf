# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf CLI subcommands.

This module provides CLI subcommands for AIPerf.
"""

from aiperf.cli_commands.analyze_trace import (
    STAT_COLUMNS,
    analyze_app,
    analyze_trace,
)
from aiperf.cli_commands.plugins_cli import (
    console,
    plugins_app,
    plugins_cli_command,
    run_validate,
    show_categories_overview,
    show_category_types,
    show_packages_detailed,
    show_type_details,
)

__all__ = [
    "STAT_COLUMNS",
    "analyze_app",
    "analyze_trace",
    "console",
    "plugins_app",
    "plugins_cli_command",
    "run_validate",
    "show_categories_overview",
    "show_category_types",
    "show_packages_detailed",
    "show_type_details",
]
