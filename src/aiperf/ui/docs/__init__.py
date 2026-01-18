# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Documentation viewer UI components."""

from aiperf.ui.docs.docs_viewer import (
    MAX_HISTORY_SIZE,
    ContentScroll,
    DocsMarkdown,
    DocsViewerApp,
    FindBar,
    HelpScreen,
    SidebarFrame,
    TableOfContents,
)
from aiperf.ui.docs.search_index import (
    DocsSearchIndex,
    SearchResult,
)
from aiperf.ui.docs.search_modal import (
    SearchModal,
    SearchResultItem,
)
from aiperf.ui.docs.sidebar import (
    DocsSidebar,
)

__all__ = [
    "ContentScroll",
    "DocsMarkdown",
    "DocsSearchIndex",
    "DocsSidebar",
    "DocsViewerApp",
    "FindBar",
    "HelpScreen",
    "MAX_HISTORY_SIZE",
    "SearchModal",
    "SearchResult",
    "SearchResultItem",
    "SidebarFrame",
    "TableOfContents",
]
