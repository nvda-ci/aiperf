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
    ensure_registry_loaded,
    get_all_categories,
    plugins,
    plugins_app,
    run_validate,
    show_category_types,
    show_overview,
    show_packages,
    show_type_details,
)

__all__ = [
    "STAT_COLUMNS",
    "analyze_app",
    "analyze_trace",
    "console",
    "ensure_registry_loaded",
    "get_all_categories",
    "plugins",
    "plugins_app",
    "run_validate",
    "show_category_types",
    "show_overview",
    "show_packages",
    "show_type_details",
]
