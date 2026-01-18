# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Search modal for documentation."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView


class SearchModal(ModalScreen[tuple[Path, int] | None]):
    """Modal screen for searching documentation."""

    CSS = """
    SearchModal {
        align: center middle;
    }

    #search-container {
        width: 80;
        height: 30;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }

    #search-input {
        margin-bottom: 1;
    }

    #search-results {
        height: 1fr;
        border: solid $accent;
    }

    #no-results {
        padding: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(self, initial_query: str | None = None) -> None:
        """Initialize the search modal.

        Args:
            initial_query: Optional initial search query
        """
        super().__init__()
        self.initial_query = initial_query or ""

    def compose(self) -> ComposeResult:
        """Compose the search modal layout."""
        with Vertical(id="search-container"):
            yield Input(
                placeholder="Search documentation... (min 2 characters)",
                value=self.initial_query,
                id="search-input",
            )
            yield ListView(id="search-results")

    def on_mount(self) -> None:
        """Focus the input when mounted."""
        self.query_one("#search-input", Input).focus()

        # Perform initial search if query provided
        if self.initial_query:
            self._perform_search(self.initial_query)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update search results as user types.

        Args:
            event: Input change event
        """
        if len(event.value) < 2:
            # Clear results if query too short
            results_list = self.query_one("#search-results", ListView)
            results_list.clear()
            if event.value:
                results_list.append(
                    ListItem(Label("Type at least 2 characters to search..."))
                )
            return

        self._perform_search(event.value)

    def _perform_search(self, query: str) -> None:
        """Perform search and update results.

        Args:
            query: Search query string
        """
        from aiperf.ui.docs.docs_viewer import DocsViewerApp

        app = self.app
        if not isinstance(app, DocsViewerApp):
            return

        results = app.search_index.search(query)
        results_list = self.query_one("#search-results", ListView)
        results_list.clear()

        if not results:
            results_list.append(ListItem(Label("No results found.", id="no-results")))
            return

        for result in results:
            # Format: filename:line - preview
            relative_path = result.file.relative_to(app.docs_dir)
            label = f"{relative_path}:{result.line_number} - {result.line_content}"
            list_item = ListItem(Label(label))
            list_item.result_file = result.file
            list_item.result_line = result.line_number
            results_list.append(list_item)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle result selection.

        Args:
            event: ListView selection event
        """
        if hasattr(event.item, "result_file"):
            # Return selected file and line number
            self.dismiss((event.item.result_file, event.item.result_line))
        else:
            # No valid result (e.g., "No results" message)
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in search input.

        Args:
            event: Input submit event
        """
        # Select first result if available
        results_list = self.query_one("#search-results", ListView)
        if results_list.children:
            first_item = results_list.children[0]
            if hasattr(first_item, "result_file"):
                self.dismiss((first_item.result_file, first_item.result_line))

    def on_key(self, event) -> None:
        """Handle key presses.

        Args:
            event: Key event
        """
        if event.key == "escape":
            self.dismiss(None)
