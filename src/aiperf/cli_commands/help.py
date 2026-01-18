# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for viewing CLI options help."""

from pathlib import Path

from cyclopts import App

help_app = App(name="help", help="View AIPerf CLI options reference")


@help_app.default
def help_command() -> None:
    """View AIPerf CLI options in an interactive terminal interface.

    Opens the CLI options reference document with find-in-page enabled
    for quick option lookup. Use Ctrl+F or start typing to search.

    Examples:
        # Open CLI options reference
        aiperf help
    """
    from aiperf.ui.docs.docs_viewer import DocsViewerApp

    # Locate docs directory - check installed package location first, then dev repo root
    package_docs = Path(__file__).parent.parent / "docs"

    if package_docs.exists():
        # Installed wheel - docs bundled in package
        docs_dir = package_docs
    else:
        # Development mode (editable install) - look for docs at repo root
        # Walk up from src/aiperf/cli_commands/ to find repo root with docs/
        repo_root = Path(__file__).parent.parent.parent.parent
        dev_docs = repo_root / "docs"
        if dev_docs.exists():
            docs_dir = dev_docs
        else:
            error_msg = (
                f"Documentation not found.\n\n"
                f"Checked locations:\n"
                f"  - {package_docs} (installed package)\n"
                f"  - {dev_docs} (development repo)\n\n"
                "If you installed from source, ensure the docs/ folder exists.\n"
                "If the problem persists, please file an issue at:\n"
                "  https://github.com/NVIDIA/AIPerf/issues"
            )
            raise FileNotFoundError(error_msg)

    # Create and run the app with cli_options.md, find bar open, sidebar hidden
    app = DocsViewerApp(
        docs_dir,
        initial_file="cli_options.md",
        initial_find=True,
        sidebar_hidden=True,
    )
    app.run()
