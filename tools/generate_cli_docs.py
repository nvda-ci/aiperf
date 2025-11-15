#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate CLI documentation using cyclopts."""

import sys
from pathlib import Path


def main() -> None:
    """Generate markdown documentation for the AIPerf CLI.

    Exits with code 0 on success, 1 on failure.
    """
    # Add src to path to import aiperf
    root_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(root_dir / "src"))

    try:
        from aiperf.cli import app

        docs = app.generate_docs(output_format="markdown")
        output_path = root_dir / "docs" / "cli_options.md"

        copyright_header = """<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

"""
        with open(output_path, "w") as f:
            f.write(copyright_header)
            f.write(docs)

        print(f"Successfully generated CLI documentation at {output_path}")
        sys.exit(0)

    except Exception as e:
        # Exit with code 1 so pre-commit fails
        print(f"Error generating CLI documentation: {e!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
