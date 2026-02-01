#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate environment variable documentation for AIPerf.

This script parses the environment.py file to extract Pydantic Settings classes
and generates markdown documentation for all environment variables.

Usage:
    python tools/generate_env_vars_docs.py              # Generate docs
    python tools/generate_env_vars_docs.py --check      # Verify docs are up-to-date
    python tools/generate_env_vars_docs.py --verbose    # Show detailed progress
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.traceback import Traceback

# =============================================================================
# Console Setup
# =============================================================================

console = Console()
error_console = Console(stderr=True)

# =============================================================================
# Paths
# =============================================================================

ENV_FILE = Path("src/aiperf/common/environment.py")
OUTPUT_FILE = Path("docs/environment_variables.md")

# =============================================================================
# Rich Output Helpers
# =============================================================================


def print_section(title: str) -> None:
    """Print a section header."""
    console.print(f"\n[bold cyan]━━━ {title} ━━━[/]")


def print_generated(path: Path) -> None:
    """Print generated file message."""
    console.print(f"  [green]✓[/] Generated [cyan]{path}[/]")


def print_up_to_date(message: str) -> None:
    """Print up-to-date message."""
    console.print(f"  [dim]✓[/] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"  [yellow]⚠[/] {message}")


def print_step(message: str, timing_ms: float | None = None) -> None:
    """Print a step with optional timing."""
    if timing_ms is not None:
        console.print(f"  [dim]•[/] {message} [dim]({timing_ms:.0f}ms)[/]")
    else:
        console.print(f"  [dim]•[/] {message}")


# =============================================================================
# Error Classes
# =============================================================================


class GeneratorError(Exception):
    """Base exception for generator errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ParseError(GeneratorError):
    """Error parsing source file."""


def print_error(error: Exception, verbose: bool = False) -> None:
    """Print error with optional traceback."""
    if isinstance(error, GeneratorError):
        console.print(
            Panel(
                f"[red]{error.message}[/]\n\n"
                + "\n".join(f"[dim]{k}:[/] {v}" for k, v in error.details.items()),
                title="[red]Error[/]",
                border_style="red",
            )
        )
    else:
        console.print(f"  [red]✗[/] {error}")

    if verbose:
        console.print()
        error_console.print(
            Traceback.from_exception(type(error), error, error.__traceback__)
        )


# =============================================================================
# File Writing
# =============================================================================


def write_if_changed(path: Path, content: str) -> bool:
    """Write file only if content has changed.

    Returns True if file was written, False if unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = path.read_text()
        if existing == content:
            return False

    path.write_text(content)
    return True


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FieldDefinition:
    """Represents a Pydantic field definition."""

    name: str
    default: str
    description: str
    constraints: list[str]


@dataclass
class SettingsClass:
    """Represents a settings class."""

    name: str
    docstring: str
    env_prefix: str
    fields: list[FieldDefinition]


# =============================================================================
# Text Helpers
# =============================================================================


def _normalize_text(text: str) -> str:
    """Normalize text by replacing newlines with spaces and stripping whitespace."""
    return " ".join(text.strip().split())


# =============================================================================
# AST Parsing
# =============================================================================


def _extract_field_call_args(
    node: ast.Call,
) -> tuple[str | None, list[str], str | None]:
    """Extract default, constraints, and description from a Field() call.

    Returns:
        Tuple of (default, constraints, description)
    """
    default = None
    constraints = []
    description = None

    # First positional argument is the default
    if node.args:
        default = ast.unparse(node.args[0])

    # Process keyword arguments
    for keyword in node.keywords:
        if keyword.arg == "default":
            default = ast.unparse(keyword.value)
        elif keyword.arg == "description":
            # Extract string literal
            if isinstance(keyword.value, ast.Constant):
                description = keyword.value.value
        elif keyword.arg in ["ge", "le", "gt", "lt", "min_length", "max_length"]:
            # Extract constraint
            constraint_val = ast.unparse(keyword.value)
            symbol_map = {
                "ge": "≥",
                "le": "≤",
                "gt": ">",
                "lt": "<",
                "min_length": "min length:",
                "max_length": "max length:",
            }
            symbol = symbol_map.get(keyword.arg, keyword.arg)
            constraints.append(f"{symbol} {constraint_val}")

    return default, constraints, description


def _parse_field_definition(node: ast.AnnAssign) -> FieldDefinition | None:
    """Parse a field definition from an annotated assignment."""
    if not isinstance(node.target, ast.Name):
        return None

    field_name = node.target.id
    default = "—"
    constraints = []
    description = "—"

    # Check if the value is a Field() call
    if isinstance(node.value, ast.Call):
        if isinstance(node.value.func, ast.Name) and node.value.func.id == "Field":
            default_val, field_constraints, field_desc = _extract_field_call_args(
                node.value
            )
            if default_val:
                default = default_val
            if field_constraints:
                constraints = field_constraints
            if field_desc:
                description = _normalize_text(field_desc)
    elif node.value:
        # Direct assignment without Field()
        default = ast.unparse(node.value)

    return FieldDefinition(
        name=field_name,
        default=default,
        description=description,
        constraints=constraints,
    )


def _extract_env_prefix(class_node: ast.ClassDef) -> str:
    """Extract the env_prefix from the model_config."""
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "model_config"
                    and isinstance(item.value, ast.Call)
                ):
                    # Find env_prefix in the SettingsConfigDict call
                    for keyword in item.value.keywords:
                        if keyword.arg == "env_prefix" and isinstance(
                            keyword.value, ast.Constant
                        ):
                            return keyword.value.value
    return "AIPERF_"


def _parse_settings_class(class_node: ast.ClassDef) -> SettingsClass | None:
    """Parse a settings class definition."""
    # Skip non-settings classes (must inherit from BaseSettings)
    is_settings = False
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseSettings":
            is_settings = True
            break

    if not is_settings:
        return None

    # Extract docstring
    docstring = ast.get_docstring(class_node) or ""
    docstring = _normalize_text(docstring) if docstring else ""

    # Extract env_prefix
    env_prefix = _extract_env_prefix(class_node)

    # Extract fields
    fields = []
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign):
            field_def = _parse_field_definition(node)
            if (
                field_def
                and not field_def.name.startswith("_")
                and "default_factory" not in field_def.default
            ):
                # Skip private fields and nested settings fields
                fields.append(field_def)

    return SettingsClass(
        name=class_node.name,
        docstring=docstring,
        env_prefix=env_prefix,
        fields=fields,
    )


def parse_environment_file(file_path: Path) -> list[SettingsClass]:
    """Parse the environment.py file and extract settings classes."""
    content = file_path.read_text()
    tree = ast.parse(content)

    settings_classes = []

    # Find all class definitions
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name.startswith("_")
            and node.name.endswith("Settings")
        ):
            # Parse the settings class
            settings_class = _parse_settings_class(node)
            if settings_class and settings_class.fields:
                settings_classes.append(settings_class)

    return settings_classes


# =============================================================================
# Markdown Formatting
# =============================================================================


def _format_default_value(default: str) -> str:
    """Format default value for display in markdown."""
    if default == "—":
        return default

    # Clean up some common patterns
    if default.startswith("[") and default.endswith("]"):
        # Format lists
        items = re.findall(r'"([^"]*)"', default)
        if items:
            if len(items) <= 3:
                return "[" + ", ".join(f"`{item}`" for item in items) + "]"
            else:
                return "[" + ", ".join(f"`{item}`" for item in items[:3]) + ", ...]"
        return f"`{default}`"

    # Wrap in backticks
    return f"`{default}`"


def _format_constraints(constraints: list[str]) -> str:
    """Format constraints for display in markdown."""
    if not constraints:
        return "—"
    return ", ".join(constraints)


def generate_markdown_docs(settings_classes: list[SettingsClass]) -> str:
    """Generate markdown documentation for environment variables."""
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "-->",
        "",
        "# Environment Variables",
        "",
        "AIPerf can be configured using environment variables with the `AIPERF_` prefix.",
        "All settings are organized into logical subsystems for better discoverability.",
        "",
        "**Pattern:** `AIPERF_{SUBSYSTEM}_{SETTING_NAME}`",
        "",
        "**Examples:**",
        "```bash",
        "export AIPERF_HTTP_CONNECTION_LIMIT=5000",
        "export AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.8",
        "export AIPERF_ZMQ_RCVTIMEO=600000",
        "```",
        "",
        "> [!WARNING]",
        "> Environment variable names, default values, and definitions are subject to change.",
        "> These settings may be modified, renamed, or removed in future releases. Always refer to the",
        "> documentation for your specific release version and test thoroughly when upgrading AIPerf.",
        "",
    ]

    # Sort by env_prefix for consistent output, except DEV which should be last
    settings_classes = sorted(
        settings_classes,
        key=lambda x: x.env_prefix if x.env_prefix != "AIPERF_DEV_" else "Z" * 100,
    )

    for settings_class in settings_classes:
        # Extract section name from env_prefix (e.g., AIPERF_DATASET_ -> DATASET)
        section_name = settings_class.env_prefix.replace("AIPERF_", "").replace("_", "")

        # Add section header
        lines.append(f"## {section_name}")
        lines.append("")
        if settings_class.docstring:
            lines.append(settings_class.docstring)
            lines.append("")

        # Create table
        lines.append("| Environment Variable | Default | Constraints | Description |")
        lines.append("|----------------------|---------|-------------|-------------|")

        for field in settings_class.fields:
            env_var = f"{settings_class.env_prefix}{field.name}"
            default = _format_default_value(field.default)
            constraints = _format_constraints(field.constraints)
            description = field.description.replace("|", "\\|")  # Escape pipes

            lines.append(f"| `{env_var}` | {default} | {constraints} | {description} |")

        lines.append("")

    return "\n".join(line.rstrip() for line in lines)


# =============================================================================
# Main CLI
# =============================================================================


def main() -> int:
    """Generate environment variable documentation."""
    parser = argparse.ArgumentParser(
        description="Generate environment variable documentation for AIPerf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/generate_env_vars_docs.py              # Generate docs
  python tools/generate_env_vars_docs.py --check      # Verify docs are up-to-date
  python tools/generate_env_vars_docs.py --verbose    # Show detailed progress
""",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if docs are up-to-date (exit 1 if not)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress and full tracebacks",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()

    print_section("Environment Variables Documentation")

    # Check source file exists
    if not ENV_FILE.exists():
        print_error(
            ParseError(
                f"Source file not found: {ENV_FILE}",
                {"hint": "Ensure you're running from the project root"},
            ),
            verbose=args.verbose,
        )
        return 1

    # Parse environment file
    start = time.perf_counter()
    try:
        settings_classes = parse_environment_file(ENV_FILE)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not settings_classes:
            print_warning("No settings classes found")
            return 1

        total_fields = sum(len(sc.fields) for sc in settings_classes)

        if args.verbose:
            print_step(
                f"Parsed {len(settings_classes)} settings classes ({total_fields} fields)",
                elapsed_ms,
            )
            for sc in settings_classes:
                print_step(f"  {sc.name}: {len(sc.fields)} fields ({sc.env_prefix}*)")
    except SyntaxError as e:
        print_error(
            ParseError(
                "Failed to parse environment.py",
                {"error": str(e), "line": e.lineno},
            ),
            verbose=args.verbose,
        )
        return 1
    except Exception as e:
        print_error(
            ParseError("Failed to parse environment.py", {"error": str(e)}),
            verbose=args.verbose,
        )
        return 1

    # Generate markdown
    start = time.perf_counter()
    try:
        markdown_content = generate_markdown_docs(settings_classes)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if args.verbose:
            print_step(
                f"Generated markdown ({len(markdown_content)} bytes)", elapsed_ms
            )
    except Exception as e:
        print_error(
            ParseError("Failed to generate markdown", {"error": str(e)}),
            verbose=args.verbose,
        )
        return 1

    # Write output
    start = time.perf_counter()
    if args.check:
        # Check mode - verify file is up-to-date
        if OUTPUT_FILE.exists():
            existing = OUTPUT_FILE.read_text()
            if existing == markdown_content:
                elapsed_ms = (time.perf_counter() - start) * 1000
                print_up_to_date(
                    f"{OUTPUT_FILE} is up-to-date [dim]({elapsed_ms:.0f}ms)[/]"
                )
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000
                console.print(
                    f"  [yellow]⚠[/] {OUTPUT_FILE} needs updating [dim]({elapsed_ms:.0f}ms)[/]"
                )
                total_elapsed = time.perf_counter() - total_start
                console.print()
                console.print(
                    f"[bold yellow]1[/] file(s) would be updated. [dim]({total_elapsed:.2f}s)[/]"
                )
                console.print("Run without [cyan]--check[/] to apply.")
                return 1
        else:
            console.print(f"  [yellow]⚠[/] {OUTPUT_FILE} does not exist")
            total_elapsed = time.perf_counter() - total_start
            console.print()
            console.print(
                f"[bold yellow]1[/] file(s) would be created. [dim]({total_elapsed:.2f}s)[/]"
            )
            console.print("Run without [cyan]--check[/] to apply.")
            return 1
    else:
        # Write mode
        if write_if_changed(OUTPUT_FILE, markdown_content):
            elapsed_ms = (time.perf_counter() - start) * 1000
            print_generated(OUTPUT_FILE)
            if args.verbose:
                print_step(f"Wrote file [dim]({elapsed_ms:.0f}ms)[/]")
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000
            print_up_to_date(
                f"{OUTPUT_FILE.name} is up-to-date [dim]({elapsed_ms:.0f}ms)[/]"
            )

    # Summary
    total_elapsed = time.perf_counter() - total_start
    console.print()

    console.print(
        f"[bold green]✓[/] Documented [bold]{len(settings_classes)}[/] subsystems "
        f"with [bold]{total_fields}[/] environment variables. [dim]({total_elapsed:.2f}s)[/]"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
