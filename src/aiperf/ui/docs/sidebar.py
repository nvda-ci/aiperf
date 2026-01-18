# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sidebar navigation widget for documentation."""

from pathlib import Path

from textual.widgets import Tree
from textual.widgets.tree import TreeNode


class DocsSidebar(Tree[Path]):
    """Sidebar navigation for documentation files."""

    def __init__(self, docs_dir: Path, **kwargs) -> None:
        """Initialize the sidebar.

        Args:
            docs_dir: Root directory containing documentation files
            **kwargs: Additional arguments passed to Tree
        """
        super().__init__("Documentation", **kwargs)
        self.docs_dir = docs_dir
        self.file_nodes: dict[Path, TreeNode[Path]] = {}

    def on_mount(self) -> None:
        """Populate the tree when mounted."""
        self._populate_tree(self.root, self.docs_dir)
        self.root.expand()

    def _populate_tree(self, node: TreeNode[Path], path: Path) -> None:
        """Recursively populate tree with markdown files.

        Args:
            node: Current tree node
            path: Current filesystem path
        """
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except (PermissionError, OSError):
            # Skip directories that can't be read
            return

        for item in items:
            # Skip hidden files
            if item.name.startswith("."):
                continue

            if item.is_dir():
                # Skip media/images directories
                if item.name in ("media", "images", "extracted_images"):
                    continue

                # Add directory node
                dir_node = node.add(self._format_label(item.name), expand=False)
                dir_node.data = item
                self._populate_tree(dir_node, item)

                # Remove directory node if it has no children
                if not dir_node.children:
                    dir_node.remove()

            elif item.suffix == ".md":
                # Add markdown file as leaf
                leaf = node.add_leaf(self._format_label(item.stem))
                leaf.data = item
                self.file_nodes[item] = leaf

    def _format_label(self, name: str) -> str:
        """Convert kebab-case or snake_case to Title Case.

        Args:
            name: File or directory name

        Returns:
            Formatted label string
        """
        return name.replace("-", " ").replace("_", " ").title()
