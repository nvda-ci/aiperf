# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main documentation viewer application."""

import contextlib
import re
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Click, Key
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Markdown,
    Static,
    Tree,
)
from textual.widgets._markdown import MarkdownFence

from aiperf.ui.docs.search_index import DocsSearchIndex
from aiperf.ui.docs.search_modal import SearchModal
from aiperf.ui.docs.sidebar import DocsSidebar

# Maximum history size for back/forward navigation
MAX_HISTORY_SIZE = 50


class DocsMarkdown(Markdown):
    """Custom Markdown widget that doesn't open links in browser."""

    def _on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Override to prevent default browser opening.

        Just post the event without opening browser - let app handle it.
        """
        # Don't call super() - that's what opens the browser
        # The event will still bubble up to the app for handling
        event.prevent_default()


class FindBar(Horizontal):
    """Find bar for searching within the current document."""

    DEFAULT_CSS = """
    FindBar {
        height: 4;
        background: $surface;
        border-top: solid $primary;
        dock: bottom;
    }

    FindBar.hidden {
        display: none;
    }

    FindBar Input {
        width: 1fr;
        margin: 0 1;
    }

    FindBar #find-status {
        width: auto;
        min-width: 10;
        padding: 0 1;
        text-style: bold;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the find bar."""
        super().__init__(**kwargs)
        self._matches: list[tuple[Static, int]] = []  # (widget, char_offset)
        self._current_match_idx: int = -1
        self._last_query: str = ""
        self._highlighted_widget: Static | None = None

    def compose(self) -> ComposeResult:
        """Compose the find bar layout."""
        yield Input(
            placeholder="Find in page... (Enter=next, Shift+Enter=prev)",
            id="find-input",
        )
        yield Static("", id="find-status")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes for live search.

        Args:
            event: Input change event
        """
        query = event.value.strip()
        if len(query) < 2:
            self._clear_matches()
            return

        self._search(query)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key to go to next match.

        Args:
            event: Input submit event
        """
        # Check if Shift was held (for previous match)
        # Since we can't easily detect shift in submitted, use a different approach
        if self._matches:
            self._goto_next_match()

    def on_key(self, event: Key) -> None:
        """Handle key events for navigation.

        Args:
            event: Key event
        """
        if event.key == "escape":
            self._clear_highlight()
            self.add_class("hidden")
            event.stop()
        elif event.key == "f3":
            self._goto_next_match()
            event.stop()
        elif event.key == "shift+f3":
            self._goto_prev_match()
            event.stop()

    def _search(self, query: str) -> None:
        """Search for query in the markdown content.

        Args:
            query: Search query string
        """
        self._last_query = query
        self._matches.clear()
        self._current_match_idx = -1

        # Get the markdown widget from parent app
        try:
            markdown = self.app.query_one("#markdown", DocsMarkdown)
        except LookupError:
            return

        # Search through all text-containing widgets in the markdown
        query_lower = query.lower()
        for widget in markdown.walk_children():
            # Check if widget has renderable text content
            if hasattr(widget, "renderable"):
                text = str(widget.renderable)
                if query_lower in text.lower():
                    # Find all occurrences in this widget
                    start = 0
                    while True:
                        idx = text.lower().find(query_lower, start)
                        if idx == -1:
                            break
                        self._matches.append((widget, idx))
                        start = idx + 1

        # Go to first match if any
        if self._matches:
            self._current_match_idx = 0
            self._scroll_to_current_match()

        self._update_status()

    def _clear_matches(self) -> None:
        """Clear all matches."""
        self._clear_highlight()
        self._matches.clear()
        self._current_match_idx = -1
        self._last_query = ""
        self._update_status()

    def _clear_highlight(self) -> None:
        """Remove highlight from currently highlighted widget."""
        if self._highlighted_widget is not None:
            with contextlib.suppress(Exception):
                self._highlighted_widget.remove_class("find-highlight")
            self._highlighted_widget = None

    def _update_status(self) -> None:
        """Update the status display."""
        status = self.query_one("#find-status", Static)
        if not self._matches:
            if self._last_query:
                status.update("No matches")
            else:
                status.update("")
        else:
            status.update(f"{self._current_match_idx + 1}/{len(self._matches)}")

    def _goto_next_match(self) -> None:
        """Go to the next match."""
        if not self._matches:
            return
        self._current_match_idx = (self._current_match_idx + 1) % len(self._matches)
        self._update_status()
        self._scroll_to_current_match()

    def _goto_prev_match(self) -> None:
        """Go to the previous match."""
        if not self._matches:
            return
        self._current_match_idx = (self._current_match_idx - 1) % len(self._matches)
        self._update_status()
        self._scroll_to_current_match()

    def _scroll_to_current_match(self) -> None:
        """Scroll to the current match and highlight it."""
        if self._current_match_idx < 0 or self._current_match_idx >= len(self._matches):
            return

        widget, _ = self._matches[self._current_match_idx]
        try:
            # Clear previous highlight
            self._clear_highlight()

            # Highlight the current match widget
            widget.add_class("find-highlight")
            self._highlighted_widget = widget

            # Get the scrollable container
            content = self.app.query_one("#content", ScrollableContainer)
            # Scroll the widget into view
            content.scroll_to_widget(widget, animate=False)
        except LookupError:
            pass  # Widget not found in DOM

    def focus_input(self) -> None:
        """Focus the find input."""
        self.query_one("#find-input", Input).focus()


class HelpScreen(ModalScreen[None]):
    """Modal screen showing keyboard shortcuts help."""

    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-container {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #help-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        padding-bottom: 1;
    }

    #help-content {
        height: auto;
    }

    .help-section {
        padding-top: 1;
        color: $secondary;
        text-style: bold;
    }

    .help-row {
        height: 1;
    }

    .help-key {
        width: 20;
        color: $warning;
    }

    .help-desc {
        width: 1fr;
    }

    #help-footer {
        text-align: center;
        padding-top: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the help screen layout."""
        with Vertical(id="help-container"):
            yield Static("Keyboard Shortcuts", id="help-title")
            with Vertical(id="help-content"):
                # Navigation
                yield Static("Navigation", classes="help-section")
                yield self._help_row("j / ↓", "Scroll down")
                yield self._help_row("k / ↑", "Scroll up")
                yield self._help_row("d / PgDn", "Scroll half page down")
                yield self._help_row("u / PgUp", "Scroll half page up")
                yield self._help_row("g / Home", "Go to top")
                yield self._help_row("G / End", "Go to bottom")
                yield self._help_row("Alt+← / [", "Go back in history")
                yield self._help_row("Alt+→ / ]", "Go forward in history")

                # Search
                yield Static("Search", classes="help-section")
                yield self._help_row("Ctrl+S", "Search all documents")
                yield self._help_row("Ctrl+F", "Find in current page")
                yield self._help_row("n", "Next search match")
                yield self._help_row("N", "Previous search match")

                # View
                yield Static("View", classes="help-section")
                yield self._help_row("Ctrl+T", "Toggle sidebar")
                yield self._help_row("w", "Toggle word wrap / scroll")
                yield self._help_row("?", "Show this help")

                # Actions
                yield Static("Actions", classes="help-section")
                yield self._help_row("Enter", "Open link / select item")
                yield self._help_row("Click code", "Copy code block")
                yield self._help_row("Escape", "Close modal / clear search")
                yield self._help_row("Ctrl+C / q", "Quit")

            yield Static("Press ? or Escape to close", id="help-footer")

    def _help_row(self, key: str, description: str) -> Horizontal:
        """Create a help row with key and description.

        Args:
            key: Keyboard shortcut
            description: Description of the action

        Returns:
            Horizontal container with key and description
        """
        row = Horizontal(classes="help-row")
        row.compose_add_child(Static(key, classes="help-key"))
        row.compose_add_child(Static(description, classes="help-desc"))
        return row

    def on_key(self, event: Key) -> None:
        """Handle any key press to close help.

        Args:
            event: Key event
        """
        self.dismiss(None)


class TableOfContents(Tree[str]):
    """Table of contents tree for current document headings."""

    DEFAULT_CSS = """
    TableOfContents {
        height: 1fr;
        padding: 0;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the TOC tree."""
        super().__init__("Contents", **kwargs)
        self._slug_to_node: dict[str, object] = {}

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Strip markdown syntax from heading text, keeping plain text.

        Args:
            text: Raw heading text with potential markdown

        Returns:
            Plain text with markdown syntax removed
        """
        # Remove images ![alt](url) entirely
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
        # Convert links [text](url) to just text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Clean up any extra whitespace
        return " ".join(text.split())

    @staticmethod
    def _style_title(text: str) -> str:
        """Apply Rich markup styling to title text.

        Args:
            text: Raw heading text

        Returns:
            Text with Rich markup for code spans
        """
        # First strip markdown syntax
        text = TableOfContents._strip_markdown(text)
        # Convert `code` to styled [bold cyan]code[/] markup
        return re.sub(r"`([^`]+)`", r"[bold cyan]\1[/]", text)

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert heading text to anchor slug (Strategy 0 - markdown-it-py style).

        Args:
            text: Heading text

        Returns:
            URL-safe anchor slug
        """
        # First strip markdown syntax
        text = TableOfContents._strip_markdown(text)
        return re.sub(r"[^\w\s-]", "", text.lower()).strip().replace(" ", "-")

    def update_toc(self, content: str) -> None:
        """Update TOC by parsing markdown content for headings.

        Args:
            content: Raw markdown content
        """
        self.clear()
        self.root.expand()
        self._slug_to_node.clear()

        # First, remove fenced code blocks to avoid matching # comments as headings
        # Match ```...``` or ~~~...~~~ blocks (with optional language identifier)
        code_block_pattern = re.compile(
            r"^(`{3,}|~{3,}).*?^\1", re.MULTILINE | re.DOTALL
        )
        content_without_code = code_block_pattern.sub("", content)

        # Also remove HTML comments (e.g., SPDX license headers)
        html_comment_pattern = re.compile(r"<!--.*?-->", re.DOTALL)
        content_without_code = html_comment_pattern.sub("", content_without_code)

        # Parse headings from markdown content (outside of code blocks)
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        headings = [
            (len(m.group(1)), m.group(2).strip())
            for m in heading_pattern.finditer(content_without_code)
        ]

        if not headings:
            self.root.add_leaf("No headings found")
            return

        # Find the minimum heading level to use as base indentation
        min_level = min(level for level, _ in headings)

        # Add all headings as leaves (clickable) with visual indentation
        for level, label in headings:
            # Style the title for display (convert `code` to Rich markup)
            styled_title = self._style_title(label)

            # Add indentation based on heading level
            indent = "  " * (level - min_level)
            display_title = f"{indent}{styled_title}"

            # Add as leaf so it's clickable
            leaf = self.root.add_leaf(display_title)
            # Store the original label text for slugify attempts
            leaf.data = label


class SidebarFrame(Vertical):
    """Framed container for the sidebar navigation split vertically."""

    DEFAULT_CSS = """
    SidebarFrame {
        width: 36;
        height: 100%;
        border: solid $primary;
        background: $surface;
    }

    SidebarFrame.hidden {
        display: none;
    }

    SidebarFrame .sidebar-section {
        height: 1fr;
        border-bottom: solid $primary-darken-2;
    }

    SidebarFrame .sidebar-section:last-child {
        border-bottom: none;
    }

    SidebarFrame .sidebar-label {
        height: 1;
        padding: 0 1;
        background: $primary-darken-1;
        color: $text;
        text-style: bold;
    }

    SidebarFrame DocsSidebar {
        height: 1fr;
        padding: 0;
        scrollbar-gutter: stable;
    }

    SidebarFrame TableOfContents {
        height: 1fr;
        padding: 0;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self, docs_dir: Path, **kwargs) -> None:
        """Initialize the sidebar frame."""
        super().__init__(**kwargs)
        self.docs_dir = docs_dir

    def compose(self) -> ComposeResult:
        """Compose the sidebar frame with vertical split."""
        with Vertical(classes="sidebar-section"):
            yield Static("Documents", classes="sidebar-label")
            yield DocsSidebar(self.docs_dir, id="sidebar-tree")
        with Vertical(classes="sidebar-section"):
            yield Static("Contents", classes="sidebar-label")
            yield TableOfContents(id="toc-tree")


class DocsViewerApp(App):
    """Documentation viewer TUI application."""

    CSS = """
    #main-container {
        width: 100%;
        height: 100%;
    }

    #content-frame {
        width: 1fr;
        height: 100%;
        border: solid $accent;
        background: $surface;
    }

    #content-header {
        height: 1;
        width: 100%;
    }

    #content-title {
        width: 1fr;
        padding: 0 1;
        background: $accent;
        color: $text;
        text-style: bold;
    }

    #progress-indicator {
        width: auto;
        min-width: 8;
        padding: 0 1;
        background: $accent;
        color: $text-muted;
        text-align: right;
    }

    #content {
        height: 1fr;
        padding: 0 1;
    }

    #content.horizontal-scroll {
        overflow-x: auto;
    }

    #content.horizontal-scroll #markdown {
        width: auto;
        min-width: 100%;
    }

    #markdown {
        height: auto;
    }

    Markdown H1 {
        color: $primary;
        text-style: bold;
    }

    Markdown H2 {
        color: $secondary;
        text-style: bold;
    }

    Markdown Code {
        background: $panel;
        color: $warning;
    }

    .find-highlight {
        background: $warning 30%;
    }

    .search-highlight {
        background: $success 30%;
    }
    """

    BINDINGS = [
        # Quit
        Binding("ctrl+c", "quit", "Quit"),
        Binding("q", "quit", "Quit", show=False),
        # Search
        Binding("ctrl+s", "search", "Search"),
        Binding("ctrl+f", "find", "Find"),
        Binding("n", "find_next", "Next", show=False),
        Binding("N", "find_prev", "Prev", show=False),
        # View
        Binding("ctrl+t", "toggle_sidebar", "Nav"),
        Binding("w", "toggle_wrap", "Wrap", show=False),
        Binding("escape", "clear_search", "Clear"),
        Binding("question_mark", "help", "Help"),
        # Vim-style scrolling
        Binding("j", "scroll_down", "↓", show=False),
        Binding("k", "scroll_up", "↑", show=False),
        Binding("d", "scroll_half_down", "½↓", show=False),
        Binding("u", "scroll_half_up", "½↑", show=False),
        Binding("g", "scroll_top", "Top", show=False),
        Binding("G", "scroll_bottom", "Bot", show=False),
        Binding("home", "scroll_top", "Top"),
        Binding("end", "scroll_bottom", "Bottom"),
        # History navigation
        Binding("alt+left", "go_back", "Back", show=False),
        Binding("alt+right", "go_forward", "Fwd", show=False),
        Binding("left_square_bracket", "go_back", "Back", show=False),
        Binding("right_square_bracket", "go_forward", "Fwd", show=False),
    ]

    def __init__(
        self,
        docs_dir: Path,
        initial_file: str | None = None,
        initial_search: str | None = None,
        initial_find: bool = False,
        sidebar_hidden: bool = False,
    ) -> None:
        """Initialize the docs viewer app.

        Args:
            docs_dir: Root directory containing documentation files
            initial_file: Optional initial file to open (relative to docs_dir)
            initial_search: Optional initial search query
            initial_find: Open the find bar on startup
            sidebar_hidden: Start with sidebar hidden
        """
        super().__init__()
        self.docs_dir = docs_dir
        self.initial_file = initial_file
        self.initial_search = initial_search
        self.initial_find = initial_find
        self._start_sidebar_hidden = sidebar_hidden
        self.current_doc: Path | None = None
        self.search_index = DocsSearchIndex(docs_dir)
        self.sidebar_visible = not sidebar_hidden
        self._last_clicked_fence: MarkdownFence | None = None
        # Navigation history for back/forward
        self._history: list[Path] = []
        self._history_index: int = -1
        self._navigating_history: bool = False
        # Last search query for highlighting
        self._last_search_query: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        with Horizontal(id="main-container"):
            yield SidebarFrame(self.docs_dir, id="sidebar-frame")
            with Vertical(id="content-frame"):
                with Horizontal(id="content-header"):
                    yield Static("Document", id="content-title")
                    yield Static("0%", id="progress-indicator")
                with ScrollableContainer(id="content"):
                    yield DocsMarkdown(id="markdown")
                yield FindBar(id="find-bar", classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Get version from package metadata
        try:
            from importlib.metadata import PackageNotFoundError, version

            pkg_version = version("aiperf")
        except PackageNotFoundError:
            pkg_version = "unknown"
        self.title = f"AIPerf {pkg_version} Documentation"

        # Handle initial file or search
        if self.initial_search:
            self.action_search()
        elif self.initial_file:
            file_path = self.docs_dir / self.initial_file
            if file_path.exists():
                self._load_document(file_path)
            else:
                self._show_error(f"File not found: {self.initial_file}")
                self._load_document(self.docs_dir / "index.md")
        else:
            # Load index.md by default
            self._load_document(self.docs_dir / "index.md")

        # Open find bar if requested
        if self.initial_find:
            self.action_find()

        # Hide sidebar if requested
        if self._start_sidebar_hidden:
            sidebar_frame = self.query_one("#sidebar-frame", SidebarFrame)
            sidebar_frame.add_class("hidden")

    def on_tree_node_selected(self, event: Tree.NodeSelected[Path | str]) -> None:
        """Handle tree node selection.

        Args:
            event: Tree node selection event
        """
        if not event.node.data:
            return

        # Check if it's a document path (from DocsSidebar) - only load files, not folders
        if isinstance(event.node.data, Path) and event.node.data.is_file():
            self._load_document(event.node.data)
        # Check if it's a heading slug (from TableOfContents)
        elif isinstance(event.node.data, str):
            self._scroll_to_heading(event.node.data)

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle link clicks within markdown documents.

        Args:
            event: Link click event with href attribute
        """
        # Stop the event completely to prevent default browser opening
        event.prevent_default()
        event.stop()

        href = event.href

        # Handle file:// URLs (Textual converts relative links to absolute file:// paths)
        if href.startswith("file://"):
            # Extract the path and any anchor
            parsed = urlparse(href)
            file_path = Path(unquote(parsed.path))
            anchor = parsed.fragment  # The part after #

            # If it's just an anchor (file path matches current or is the docs dir)
            if anchor and (
                not file_path.suffix or file_path == self.docs_dir.resolve()
            ):
                markdown = self.query_one("#markdown", DocsMarkdown)
                if not markdown.goto_anchor(anchor):
                    self.notify(
                        f"Anchor '{anchor}' not found", severity="warning", timeout=3
                    )
                return

            # It's a file link
            if file_path.exists() and file_path.suffix == ".md":
                # Security check
                try:
                    file_path.relative_to(self.docs_dir.resolve())
                except ValueError:
                    self.notify(
                        "Link points outside documentation", severity="error", timeout=5
                    )
                    return

                self._load_document(file_path)
                if anchor:
                    markdown = self.query_one("#markdown", DocsMarkdown)
                    if not markdown.goto_anchor(anchor):
                        self.notify(
                            f"Anchor '{anchor}' not found",
                            severity="warning",
                            timeout=3,
                        )
            else:
                self.notify(
                    f"Document not found: {file_path.name}",
                    severity="warning",
                    timeout=5,
                )
            return

        # Anchor link within current document
        if href.startswith("#"):
            slug = href[1:]  # Remove the # prefix
            markdown = self.query_one("#markdown", DocsMarkdown)
            if not markdown.goto_anchor(slug):
                self.notify(f"Anchor '{slug}' not found", severity="warning", timeout=3)
            return

        # External links - open in browser
        if href.startswith(("http://", "https://")):
            try:
                webbrowser.open(href)
                self.notify(f"Opened in browser: {href}", timeout=3)
            except Exception:
                self.notify(f"Could not open: {href}", severity="warning", timeout=5)
            return

        if href.startswith("mailto:"):
            try:
                webbrowser.open(href)
                self.notify("Opening email client...", timeout=2)
            except Exception:
                self.notify(f"Email: {href[7:]}", timeout=5)
            return

        # Relative file link - resolve relative to current document
        if self.current_doc is None:
            self.notify("No document loaded", severity="warning", timeout=3)
            return

        # Handle anchor in relative link (e.g., "file.md#heading")
        anchor = None
        if "#" in href:
            href, anchor = href.split("#", 1)

        # Resolve relative path from current document's directory
        current_dir = self.current_doc.parent
        target_path = (current_dir / href).resolve()

        # Security check: ensure target is within docs_dir
        try:
            target_path.relative_to(self.docs_dir.resolve())
        except ValueError:
            self.notify(
                "Link points outside documentation", severity="error", timeout=5
            )
            return

        if target_path.exists() and target_path.suffix == ".md":
            self._load_document(target_path)
            # If there was an anchor, scroll to it after loading
            if anchor:
                markdown = self.query_one("#markdown", DocsMarkdown)
                if not markdown.goto_anchor(anchor):
                    self.notify(
                        f"Anchor '{anchor}' not found", severity="warning", timeout=3
                    )
        else:
            self.notify(f"Document not found: {href}", severity="warning", timeout=5)

    def _scroll_to_heading(self, heading_text: str) -> None:
        """Scroll the markdown viewer to a heading.

        Args:
            heading_text: The original heading text to find
        """
        markdown = self.query_one("#markdown", DocsMarkdown)
        slug = TableOfContents._slugify(heading_text)

        if not markdown.goto_anchor(slug):
            self.notify(f"Anchor '{slug}' not found", severity="warning", timeout=3)

    def _load_document(self, doc_path: Path, search_text: str | None = None) -> None:
        """Load and display a documentation file.

        Args:
            doc_path: Path to the documentation file
            search_text: Optional text to search for and scroll to
        """
        try:
            if not doc_path.exists():
                self._show_error(f"Document not found: {doc_path.name}")
                return

            # Check file size (warn if > 10MB)
            file_size = doc_path.stat().st_size
            if file_size > 10_000_000:
                self.notify(
                    "Large file - rendering may be slow", severity="warning", timeout=5
                )

            # Read file content
            content = doc_path.read_text(encoding="utf-8")

            # Process image references to show placeholders
            content = self._process_images(content)

            # Update markdown widget
            markdown = self.query_one("#markdown", DocsMarkdown)
            markdown.update(content)

            # Update state
            self.current_doc = doc_path
            relative_path = doc_path.relative_to(self.docs_dir)
            self.sub_title = str(relative_path)

            # Update content frame title
            content_title = self.query_one("#content-title", Static)
            content_title.update(str(relative_path))

            # Update the table of contents by parsing the markdown content
            toc = self.query_one("#toc-tree", TableOfContents)
            toc.update_toc(content)

            # Scroll to top of document first
            content_container = self.query_one("#content", ScrollableContainer)
            content_container.scroll_home(animate=False)

            # Reset progress indicator
            self._update_progress()

            # Add to history (unless we're navigating history)
            if not self._navigating_history:
                self._add_to_history(doc_path)

            # Store and handle search text for highlighting
            self._last_search_query = search_text
            if search_text:
                self._scroll_to_text(search_text)
                self._highlight_search_terms(search_text)

        except UnicodeDecodeError:
            self._show_error("File encoding error - cannot display")
        except Exception as e:
            self._show_error(f"Error loading document: {e}")

    def _process_images(self, content: str) -> str:
        """Replace image references with placeholders.

        Args:
            content: Markdown content

        Returns:
            Processed content with image placeholders
        """
        # Match ![alt](path) and replace with [Image: path]
        pattern = r"!\[([^\]]*)\]\(([^\)]+)\)"
        replacement = r"[Image: \2]"
        return re.sub(pattern, replacement, content)

    def _show_error(self, message: str) -> None:
        """Show an error message in the viewer.

        Args:
            message: Error message to display
        """
        markdown = self.query_one("#markdown", DocsMarkdown)
        markdown.update(f"# Error\n\n{message}")
        self.notify(message, severity="error", timeout=10)

    def _scroll_to_text(self, search_text: str) -> None:
        """Scroll to the first occurrence of text in the rendered markdown.

        Args:
            search_text: Text to search for and scroll to
        """

        def do_scroll() -> None:
            try:
                markdown = self.query_one("#markdown", DocsMarkdown)
                container = self.query_one("#content", ScrollableContainer)
                search_lower = search_text.lower()

                # Search through rendered widgets for matching text
                for widget in markdown.walk_children():
                    if hasattr(widget, "renderable"):
                        text = str(widget.renderable)
                        if search_lower in text.lower():
                            # Found a match - scroll to this widget
                            container.scroll_to_widget(widget, animate=False)
                            return
            except LookupError:
                pass  # Widgets not yet in DOM

        self.call_after_refresh(do_scroll)

    def action_search(self) -> None:
        """Open the search modal."""
        self.push_screen(SearchModal(self.initial_search), self._handle_search_result)
        # Clear initial search after first use
        self.initial_search = None

    def _handle_search_result(self, result: tuple[Path, int, str] | None) -> None:
        """Handle search result selection.

        Args:
            result: Tuple of (file_path, line_number, search_query) or None if cancelled
        """
        if result:
            file_path, line_number, search_query = result
            self._load_document(file_path, search_text=search_query)

    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        sidebar_frame = self.query_one("#sidebar-frame", SidebarFrame)
        self.sidebar_visible = not self.sidebar_visible

        if self.sidebar_visible:
            sidebar_frame.remove_class("hidden")
        else:
            sidebar_frame.add_class("hidden")

    def action_toggle_wrap(self) -> None:
        """Toggle horizontal scrolling for wide content."""
        content = self.query_one("#content", ScrollableContainer)
        if content.has_class("horizontal-scroll"):
            content.remove_class("horizontal-scroll")
            self.notify("Word wrap enabled", timeout=2)
        else:
            content.add_class("horizontal-scroll")
            self.notify("Horizontal scroll enabled", timeout=2)

    def action_clear_search(self) -> None:
        """Clear search and return to normal view."""
        # If a modal is open (more than just the default screen), close it
        if len(self.screen_stack) > 1:
            self.pop_screen()
        # Also hide the find bar if visible and clear its highlight
        find_bar = self.query_one("#find-bar", FindBar)
        if not find_bar.has_class("hidden"):
            find_bar._clear_highlight()
            find_bar.add_class("hidden")

    def action_find(self) -> None:
        """Open or focus the find bar."""
        find_bar = self.query_one("#find-bar", FindBar)
        find_bar.remove_class("hidden")
        find_bar.focus_input()

    def action_scroll_top(self) -> None:
        """Scroll to the top of the document."""
        content = self.query_one("#content", ScrollableContainer)
        content.scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        """Scroll to the bottom of the document."""
        content = self.query_one("#content", ScrollableContainer)
        content.scroll_end(animate=False)

    def on_click(self, event: Click) -> None:
        """Handle click events to detect code block clicks for copying.

        Args:
            event: Click event
        """
        # Walk up from the clicked widget to see if it's inside a MarkdownFence
        widget = event.widget
        while widget is not None:
            if isinstance(widget, MarkdownFence):
                self._copy_code_block(widget)
                event.stop()
                return
            widget = widget.parent

    def _copy_code_block(self, fence: MarkdownFence) -> None:
        """Copy code block content to clipboard.

        Args:
            fence: The MarkdownFence widget containing the code
        """
        # MarkdownFence stores the code in its `code` attribute
        if hasattr(fence, "code") and fence.code:
            self.copy_to_clipboard(fence.code)
            # Show brief notification
            self.notify("Code copied to clipboard", timeout=2)

    # --- Vim-style navigation actions ---

    def action_scroll_down(self) -> None:
        """Scroll down one line (vim j)."""
        content = self.query_one("#content", ScrollableContainer)
        content.scroll_relative(y=3)
        self._update_progress()

    def action_scroll_up(self) -> None:
        """Scroll up one line (vim k)."""
        content = self.query_one("#content", ScrollableContainer)
        content.scroll_relative(y=-3)
        self._update_progress()

    def action_scroll_half_down(self) -> None:
        """Scroll down half a page (vim d)."""
        content = self.query_one("#content", ScrollableContainer)
        content.scroll_relative(y=content.size.height // 2)
        self._update_progress()

    def action_scroll_half_up(self) -> None:
        """Scroll up half a page (vim u)."""
        content = self.query_one("#content", ScrollableContainer)
        content.scroll_relative(y=-(content.size.height // 2))
        self._update_progress()

    # --- Progress tracking ---

    def _update_progress(self) -> None:
        """Update the reading progress indicator."""

        def do_update() -> None:
            try:
                content = self.query_one("#content", ScrollableContainer)
                progress = self.query_one("#progress-indicator", Static)

                # Calculate progress percentage
                max_scroll = content.max_scroll_y
                if max_scroll > 0:
                    pct = int((content.scroll_y / max_scroll) * 100)
                    progress.update(f"{pct}%")
                else:
                    progress.update("100%")
            except LookupError:
                pass

        self.call_after_refresh(do_update)

    def on_scroll(self) -> None:
        """Handle scroll events to update progress."""
        self._update_progress()

    # --- History navigation ---

    def _add_to_history(self, doc_path: Path) -> None:
        """Add a document to the navigation history.

        Args:
            doc_path: Path to the document
        """
        # If we're in the middle of history, truncate forward history
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]

        # Don't add duplicates consecutively
        if self._history and self._history[-1] == doc_path:
            return

        # Add to history
        self._history.append(doc_path)

        # Limit history size
        if len(self._history) > MAX_HISTORY_SIZE:
            self._history = self._history[-MAX_HISTORY_SIZE:]

        self._history_index = len(self._history) - 1

    def action_go_back(self) -> None:
        """Go back in navigation history."""
        if self._history_index > 0:
            self._history_index -= 1
            self._navigating_history = True
            self._load_document(self._history[self._history_index])
            self._navigating_history = False
        else:
            self.notify("No previous page", timeout=2)

    def action_go_forward(self) -> None:
        """Go forward in navigation history."""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._navigating_history = True
            self._load_document(self._history[self._history_index])
            self._navigating_history = False
        else:
            self.notify("No next page", timeout=2)

    # --- Help screen ---

    def action_help(self) -> None:
        """Show the help screen with keyboard shortcuts."""
        self.push_screen(HelpScreen())

    # --- Find navigation ---

    def action_find_next(self) -> None:
        """Go to next find match (n key)."""
        find_bar = self.query_one("#find-bar", FindBar)
        if not find_bar.has_class("hidden"):
            find_bar._goto_next_match()
        elif self._last_search_query:
            # Use the last search query if find bar is hidden
            self._scroll_to_text(self._last_search_query)

    def action_find_prev(self) -> None:
        """Go to previous find match (N key)."""
        find_bar = self.query_one("#find-bar", FindBar)
        if not find_bar.has_class("hidden"):
            find_bar._goto_prev_match()

    # --- Search highlighting ---

    def _highlight_search_terms(self, search_text: str) -> None:
        """Highlight search terms in the document.

        Args:
            search_text: Text to highlight
        """

        def do_highlight() -> None:
            try:
                markdown = self.query_one("#markdown", DocsMarkdown)
                search_lower = search_text.lower()

                # Find and highlight widgets containing the search text
                for widget in markdown.walk_children():
                    if hasattr(widget, "renderable"):
                        text = str(widget.renderable)
                        if search_lower in text.lower():
                            widget.add_class("search-highlight")
            except LookupError:
                pass

        self.call_after_refresh(do_highlight)
