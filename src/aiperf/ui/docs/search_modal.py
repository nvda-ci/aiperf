# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Search modal for documentation."""

import re
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static


class SearchResultItem(ListItem):
    """ListItem that stores search result data."""

    def __init__(
        self,
        content: Static,
        result_file: Path,
        result_line: int,
        result_query: str,
    ) -> None:
        """Initialize the search result item.

        Args:
            content: The static widget to display
            result_file: Path to the matching file
            result_line: Line number of the match
            result_query: The search query that produced this result
        """
        super().__init__(content)
        self.result_file = result_file
        self.result_line = result_line
        self.result_query = result_query


class SearchModal(ModalScreen[tuple[Path, int, str] | None]):
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

    #results-count {
        height: 1;
        text-align: right;
        color: $text-muted;
        margin-top: 1;
    }

    #results-count.hidden {
        display: none;
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
            yield Static("", id="results-count", classes="hidden")

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
            results_count = self.query_one("#results-count", Static)
            results_list.clear()
            results_count.add_class("hidden")
            if event.value:
                results_list.append(
                    ListItem(Label("Type at least 2 characters to search..."))
                )
            return

        self._perform_search(event.value)

    def _format_markdown_content(self, content: str) -> str:
        """Convert markdown syntax to Rich markup.

        Args:
            content: Raw content string

        Returns:
            Content with Rich markup for backticks, bold, and italic
        """
        # Escape any existing Rich markup brackets
        content = content.replace("[", r"\[").replace("]", r"\]")

        # Convert backticks to code style (cyan)
        content = re.sub(r"`([^`]+)`", r"[bold cyan]\1[/bold cyan]", content)

        # Convert **bold** to bold (must come before single *)
        content = re.sub(r"\*\*([^*]+)\*\*", r"[bold]\1[/bold]", content)

        # Convert _italic_ to italic (word boundaries to avoid underscores in identifiers)
        content = re.sub(
            r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])", r"[italic]\1[/italic]", content
        )

        return content

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
        results_count = self.query_one("#results-count", Static)
        results_list.clear()

        if not results:
            results_list.append(ListItem(Label("No results found.", id="no-results")))
            results_count.add_class("hidden")
            return

        # Count unique files
        unique_files = len({result.file for result in results})

        for result in results:
            # Format with separate styles for filename vs content
            relative_path = result.file.relative_to(app.docs_dir)

            # Style: path and line number subdued, content with markdown formatting
            styled_content = self._format_markdown_content(result.line_content)
            label_text = (
                f"[dim]{relative_path}:{result.line_number}[/dim] {styled_content}"
            )

            list_item = SearchResultItem(
                Static(label_text),
                result_file=result.file,
                result_line=result.line_number,
                result_query=query,
            )
            results_list.append(list_item)

        # Update results count
        results_count.update(
            f"Found {len(results)} results across {unique_files} files"
        )
        results_count.remove_class("hidden")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle result selection.

        Args:
            event: ListView selection event
        """
        if isinstance(event.item, SearchResultItem):
            # Return selected file, line number, and search query
            self.dismiss(
                (
                    event.item.result_file,
                    event.item.result_line,
                    event.item.result_query,
                )
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in search input.

        Args:
            event: Input submit event
        """
        # Select first result if available
        results_list = self.query_one("#search-results", ListView)
        if results_list.children:
            first_item = results_list.children[0]
            if isinstance(first_item, SearchResultItem):
                self.dismiss(
                    (
                        first_item.result_file,
                        first_item.result_line,
                        first_item.result_query,
                    )
                )

    def on_key(self, event: Key) -> None:
        """Handle key presses.

        Args:
            event: Key event
        """
        if event.key == "escape":
            self.dismiss(None)
