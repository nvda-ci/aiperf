# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for viewing documentation."""

from pathlib import Path

from cyclopts import App

docs_app = App(name="docs", help="View AIPerf documentation in the terminal")


@docs_app.default
def docs_command(
    file: str | None = None,
    search: str | None = None,
) -> None:
    """View AIPerf documentation in an interactive terminal interface.

    Opens a rich terminal UI with sidebar navigation, full-text search, and markdown
    rendering. Navigate using arrow keys, search with Ctrl+S, and toggle sidebar with Ctrl+T.

    Examples:
        # Open documentation browser (shows index.md)
        aiperf docs

        # Open specific document
        aiperf docs --file tutorials/multi-turn.md

        # Search documentation
        aiperf docs --search "request rate"

    Args:
        file: Specific documentation file to open (relative to docs root)
        search: Search query to filter documentation
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

    # Create and run the app
    app = DocsViewerApp(docs_dir, initial_file=file, initial_search=search)
    app.run()
