# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main documentation viewer application."""

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Footer,
    Header,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)
from textual.widgets._markdown import MarkdownFence

from aiperf.ui.docs.search_index import DocsSearchIndex
from aiperf.ui.docs.search_modal import SearchModal
from aiperf.ui.docs.sidebar import DocsSidebar


class DocsMarkdown(Markdown):
    """Custom Markdown widget that doesn't open links in browser."""

    def _on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Override to prevent default browser opening.

        Just post the event without opening browser - let app handle it.
        """
        # Don't call super() - that's what opens the browser
        # The event will still bubble up to the app for handling
        event.prevent_default()


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
        # Map slug -> tree node for quick lookup during scroll sync
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

            # Map slug to node for scroll sync
            slug = self._slugify(label)
            self._slug_to_node[slug] = leaf

    def highlight_by_slug(self, slug: str) -> bool:
        """Highlight a TOC entry by its slug.

        Args:
            slug: The anchor slug to highlight

        Returns:
            True if found and highlighted, False otherwise
        """
        if slug in self._slug_to_node:
            node = self._slug_to_node[slug]
            self.select_node(node)
            self.scroll_to_node(node)
            return True
        return False


class SidebarFrame(Vertical):
    """Framed container for the sidebar navigation with tabs."""

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

    SidebarFrame TabbedContent {
        height: 1fr;
    }

    SidebarFrame TabPane {
        padding: 0;
    }

    SidebarFrame ContentSwitcher {
        height: 1fr;
    }

    SidebarFrame DocsSidebar {
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
        """Compose the sidebar frame with tabs."""
        with TabbedContent(id="sidebar-tabs"):
            with TabPane("Docs", id="docs-tab"):
                yield DocsSidebar(self.docs_dir, id="sidebar-tree")
            with TabPane("TOC", id="toc-tab"):
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

    #content-title {
        height: 1;
        padding: 0 1;
        background: $accent;
        color: $text;
        text-style: bold;
    }

    #content {
        height: 1fr;
        padding: 0 1;
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
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+s", "search", "Search"),
        Binding("ctrl+t", "toggle_sidebar", "Nav"),
        Binding("escape", "clear_search", "Clear"),
        Binding("home", "scroll_top", "Top"),
        Binding("end", "scroll_bottom", "Bottom"),
    ]

    # Track the last clicked code block for potential keyboard copy
    _last_clicked_fence: MarkdownFence | None = None

    def __init__(
        self,
        docs_dir: Path,
        initial_file: str | None = None,
        initial_search: str | None = None,
    ) -> None:
        """Initialize the docs viewer app.

        Args:
            docs_dir: Root directory containing documentation files
            initial_file: Optional initial file to open (relative to docs_dir)
            initial_search: Optional initial search query
        """
        super().__init__()
        self.docs_dir = docs_dir
        self.initial_file = initial_file
        self.initial_search = initial_search
        self.current_doc: Path | None = None
        self.search_index = DocsSearchIndex(docs_dir)
        self.sidebar_visible = True

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        with Horizontal(id="main-container"):
            yield SidebarFrame(self.docs_dir, id="sidebar-frame")
            with Vertical(id="content-frame"):
                yield Static("Document", id="content-title")
                with ScrollableContainer(id="content"):
                    yield DocsMarkdown(id="markdown")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Get version from package metadata
        try:
            from importlib.metadata import version

            pkg_version = version("aiperf")
        except Exception:
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

    def on_tree_node_selected(self, event) -> None:
        """Handle tree node selection.

        Args:
            event: Tree node selection event
        """
        if not event.node.data:
            return

        # Check if it's a document path (from DocsSidebar) - only load files, not folders
        if isinstance(event.node.data, Path) and event.node.data.is_file():
            self._load_document(event.node.data)
            # Switch to TOC tab when opening a document from sidebar
            self.query_one("#sidebar-tabs", TabbedContent).active = "toc-tab"
        # Check if it's a heading slug (from TableOfContents)
        elif isinstance(event.node.data, str):
            self._scroll_to_heading(event.node.data)

    def on_scrollable_container_scroll_end(self, _event) -> None:
        """Sync TOC highlight when scrolling ends."""
        self._sync_toc_to_scroll()

    def _sync_toc_to_scroll(self) -> None:
        """Find the topmost visible heading and highlight it in TOC."""
        try:
            markdown = self.query_one("#markdown", DocsMarkdown)
            content = self.query_one("#content", ScrollableContainer)
            toc = self.query_one("#toc-tree", TableOfContents)

            # Query all heading widgets (H1-H6)
            headings = list(
                markdown.query(
                    "MarkdownH1, MarkdownH2, MarkdownH3, MarkdownH4, MarkdownH5, MarkdownH6"
                )
            )
            if not headings:
                return

            # Find the heading closest to the top of the visible area
            scroll_y = content.scroll_y
            best_heading = None
            best_distance = float("inf")

            for heading in headings:
                # Get heading position relative to the scrollable container
                heading_y = heading.region.y - content.region.y + scroll_y
                distance = abs(heading_y - scroll_y)

                # Prefer headings at or above the scroll position
                if heading_y <= scroll_y + 5:  # Small buffer for headings near top
                    if best_heading is None or heading_y > (
                        best_heading.region.y - content.region.y + scroll_y
                    ):
                        best_heading = heading
                elif distance < best_distance and best_heading is None:
                    best_distance = distance
                    best_heading = heading

            if best_heading and best_heading.id:
                toc.highlight_by_slug(best_heading.id)
        except Exception:
            # Silently ignore errors during scroll sync
            pass

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

        # External links
        if href.startswith(("http://", "https://", "mailto:")):
            self.notify(f"External link: {href}", timeout=5)
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

    def _load_document(self, doc_path: Path, jump_to_line: int | None = None) -> None:
        """Load and display a documentation file.

        Args:
            doc_path: Path to the documentation file
            jump_to_line: Optional line number to scroll to
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

            # Scroll to top of document
            content_container = self.query_one("#content", ScrollableContainer)
            content_container.scroll_home(animate=False)

            # Scroll to line if specified
            if jump_to_line:
                # Note: MarkdownViewer doesn't have direct line scrolling
                # This is a limitation of the widget
                self.notify(f"Navigated to {relative_path}:{jump_to_line}", timeout=3)

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

    def action_search(self) -> None:
        """Open the search modal."""
        self.push_screen(SearchModal(self.initial_search), self._handle_search_result)
        # Clear initial search after first use
        self.initial_search = None

    def _handle_search_result(self, result: tuple[Path, int] | None) -> None:
        """Handle search result selection.

        Args:
            result: Tuple of (file_path, line_number) or None if cancelled
        """
        if result:
            file_path, line_number = result
            self._load_document(file_path, jump_to_line=line_number)

    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        sidebar_frame = self.query_one("#sidebar-frame", SidebarFrame)
        self.sidebar_visible = not self.sidebar_visible

        if self.sidebar_visible:
            sidebar_frame.remove_class("hidden")
        else:
            sidebar_frame.add_class("hidden")

    def action_clear_search(self) -> None:
        """Clear search and return to normal view."""
        # If a modal is open, close it
        if self.screen_stack:
            self.pop_screen()

    def action_scroll_top(self) -> None:
        """Scroll to the top of the document."""
        content = self.query_one("#content", ScrollableContainer)
        content.scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        """Scroll to the bottom of the document."""
        content = self.query_one("#content", ScrollableContainer)
        content.scroll_end(animate=False)

    def on_click(self, event) -> None:
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
