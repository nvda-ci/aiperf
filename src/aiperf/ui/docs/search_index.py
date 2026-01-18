# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Search index for documentation files."""

from pathlib import Path

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel


class SearchResult(AIPerfBaseModel):
    """A single search result."""

    file: Path = Field(description="Path to the matching file")
    line_number: int = Field(description="Line number of the match")
    line_content: str = Field(description="Content of the matching line")
    relevance: float = Field(description="Relevance score for sorting")


class DocsSearchIndex:
    """Simple full-text search index for documentation."""

    def __init__(self, docs_dir: Path) -> None:
        """Initialize the search index.

        Args:
            docs_dir: Root directory containing documentation files
        """
        self.docs_dir = docs_dir
        self.index: dict[Path, list[str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Index all markdown files."""
        for md_file in self.docs_dir.rglob("*.md"):
            # Skip media/images directories
            if any(
                part in md_file.parts
                for part in ("media", "images", "extracted_images")
            ):
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
                self.index[md_file] = content.splitlines(keepends=True)
            except (OSError, UnicodeDecodeError):
                # Skip files that can't be read or have encoding issues
                continue

    def search(self, query: str, limit: int = 50) -> list[SearchResult]:
        """Search for query in all documents.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects sorted by relevance
        """
        if not query:
            return []

        results: list[SearchResult] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for file, lines in self.index.items():
            for i, line in enumerate(lines, 1):
                line_lower = line.lower()
                if query_lower not in line_lower:
                    continue

                # Calculate relevance score
                relevance = 1.0

                # Headers score higher
                if line.strip().startswith("#"):
                    relevance = 2.0
                # Exact word matches score higher than substrings
                elif any(word in line_lower.split() for word in query_words):
                    relevance = 1.5

                # Create result with preview (first 80 chars of content)
                line_content = line.strip()
                if len(line_content) > 80:
                    line_content = line_content[:77] + "..."

                results.append(
                    SearchResult(
                        file=file,
                        line_number=i,
                        line_content=line_content,
                        relevance=relevance,
                    )
                )

        # Sort by relevance (descending), then by file name
        results.sort(key=lambda x: (-x.relevance, x.file.name))
        return results[:limit]
