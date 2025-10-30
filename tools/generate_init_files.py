#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Cross-platform script to generate __init__.py files using mkinit.

This script replaces generate_init_files.sh for Windows compatibility.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Generate __init__.py files and fix formatting."""
    # Determine venv path
    venv_path = os.environ.get("VENV_PATH", ".venv")

    # Check if we're in a virtual environment and activate it if available
    if sys.platform == "win32":
        activate_script = Path(venv_path) / "Scripts" / "activate"
        python_executable = Path(venv_path) / "Scripts" / "python.exe"
    else:
        activate_script = Path(venv_path) / "bin" / "activate"
        python_executable = Path(venv_path) / "bin" / "python"

    # Use the venv python if available, otherwise use current python
    if python_executable.exists():
        python_cmd = str(python_executable)
    else:
        python_cmd = sys.executable

    print(f"Using Python: {python_cmd}")

    # Run mkinit
    print("Running mkinit...")
    try:
        subprocess.run(
            [
                python_cmd,
                "-m",
                "mkinit",
                "--write",
                "--black",
                "--nomods",
                "--recursive",
                "src/aiperf",
            ],
            check=True,
        )
        print("mkinit completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running mkinit: {e}")
        return 1

    # Fix imports in __init__.py files using ruff
    print("Fixing imports in __init__.py files...")
    try:
        # Find all __init__.py files
        init_files = list(Path("src/aiperf").rglob("__init__.py"))

        if init_files:
            # Run ruff check --fix on each file
            for init_file in init_files:
                subprocess.run(
                    [python_cmd, "-m", "ruff", "check", "--fix", str(init_file)],
                    check=False,  # Don't fail if ruff has warnings
                )
            print(f"Fixed {len(init_files)} __init__.py files")
        else:
            print("No __init__.py files found")
    except subprocess.CalledProcessError as e:
        print(f"Error running ruff: {e}")
        return 1

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
