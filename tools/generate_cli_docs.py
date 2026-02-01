#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate CLI docs for AIPerf.

This script generates markdown documentation for AIPerf CLI commands by
introspecting the cyclopts application and extracting parameter information.

Usage:
    python tools/generate_cli_docs.py              # Generate docs
    python tools/generate_cli_docs.py --check      # Verify docs are up-to-date
    python tools/generate_cli_docs.py --verbose    # Show detailed progress
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from inspect import isclass
from io import StringIO
from pathlib import Path
from typing import Any, get_origin

from rich.console import Console
from rich.panel import Panel
from rich.traceback import Traceback

# =============================================================================
# Console Setup
# =============================================================================

console = Console()
error_console = Console(stderr=True)

# =============================================================================
# Output Paths
# =============================================================================

OUTPUT_FILE = Path("docs/cli_options.md")

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


class CLIExtractionError(GeneratorError):
    """Error extracting CLI data."""


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
class ParameterInfo:
    """Information about a CLI parameter."""

    display_name: str
    long_options: str
    short: str
    description: str
    required: bool
    type_suffix: str
    default_value: str = ""
    choices: list[str] | None = None
    choice_descriptions: dict[str, str] | None = None
    constraints: list[str] | None = None


# =============================================================================
# Text Extraction Helpers
# =============================================================================


def _normalize_text(text: str) -> str:
    """Normalize text by replacing newlines with spaces and stripping whitespace."""
    return " ".join(text.strip().split())


def _extract_enum_member_docstrings(enum_class: type[Enum]) -> dict[str, str]:
    """Extract docstrings for enum members by parsing the source code."""
    try:
        source = inspect.getsource(enum_class)
        tree = ast.parse(source)

        class_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == enum_class.__name__:
                class_def = node
                break

        if not class_def:
            return {}

        member_docs = {}
        i = 0
        while i < len(class_def.body):
            node = class_def.body[i]

            if isinstance(node, ast.Assign):  # noqa: SIM102
                if node.targets and isinstance(node.targets[0], ast.Name):
                    member_name = node.targets[0].id

                    if i + 1 < len(class_def.body):
                        next_node = class_def.body[i + 1]
                        if isinstance(next_node, ast.Expr) and isinstance(  # noqa: SIM102
                            next_node.value, ast.Constant
                        ):
                            if isinstance(next_node.value.value, str):
                                docstring = next_node.value.value.strip()
                                member_docs[member_name] = docstring
            i += 1

        return member_docs
    except (OSError, TypeError, SyntaxError):
        return {}


# =============================================================================
# Pydantic Constraint Extraction
# =============================================================================


def _build_constraint_map(
    model_class: type, _visited: set[type] | None = None
) -> dict[str, list[str]]:
    """Build a map of field names to their constraints from a Pydantic model."""
    from pydantic import BaseModel

    if _visited is None:
        _visited = set()
    if model_class in _visited:
        return {}
    _visited.add(model_class)

    constraint_map_result: dict[str, list[str]] = {}

    constraint_symbols = {
        "ge": "≥",
        "le": "≤",
        "gt": ">",
        "lt": "<",
        "min_length": "min length:",
        "max_length": "max length:",
    }

    for field_name, field_info in model_class.model_fields.items():
        constraints = []

        for attr_name, symbol in constraint_symbols.items():
            if hasattr(field_info, attr_name):
                value = getattr(field_info, attr_name)
                if value is not None:
                    constraints.append(f"{symbol} {value}")

        if hasattr(field_info, "metadata") and field_info.metadata:
            for metadata_item in field_info.metadata:
                for attr_name, symbol in constraint_symbols.items():
                    if hasattr(metadata_item, attr_name):
                        value = getattr(metadata_item, attr_name)
                        if value is not None and f"{symbol} {value}" not in constraints:
                            constraints.append(f"{symbol} {value}")

        if constraints:
            constraint_map_result[field_name] = constraints

    for _field_name, field_info in model_class.model_fields.items():
        annotation = field_info.annotation
        if hasattr(annotation, "__origin__"):
            args = getattr(annotation, "__args__", ())
            for arg in args:
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    nested_constraints = _build_constraint_map(arg, _visited)
                    for nested_name, nested_vals in nested_constraints.items():
                        if nested_name not in constraint_map_result:
                            constraint_map_result[nested_name] = nested_vals
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested_constraints = _build_constraint_map(annotation, _visited)
            for nested_name, nested_vals in nested_constraints.items():
                if nested_name not in constraint_map_result:
                    constraint_map_result[nested_name] = nested_vals

    return constraint_map_result


# =============================================================================
# Cyclopts Extraction
# =============================================================================


def extract_plain_text(obj: Any) -> str:
    """Extract plain text from cyclopts objects."""
    from cyclopts.help import InlineText

    if isinstance(obj, InlineText):
        console_temp = Console(file=StringIO(), record=True, width=1000)
        console_temp.print(obj)
        text = console_temp.export_text(clear=False, styles=False)
        return _normalize_text(text.replace("\r", ""))
    return str(obj) if obj else ""


def get_type_suffix(hint: Any) -> str:
    """Get type suffix for parameter hints."""
    type_mapping: dict[type, str] = {
        bool: "",
        int: " <int>",
        float: " <float>",
        list: " <list>",
        tuple: " <list>",
        set: " <list>",
    }

    lookup_type = hint if hint in type_mapping else get_origin(hint)
    return type_mapping.get(lookup_type, " <str>")


def _extract_display_name(arg: Any) -> str:
    """Extract display name from argument."""
    name = arg.names[0].lstrip("-").replace("-", " ").title()
    return f"{name} _(Required)_" if arg.required else name


def _extract_default_value(arg: Any) -> str:
    """Extract default value from argument."""
    from cyclopts.field_info import FieldInfo

    if not arg.show_default:
        return ""

    default = arg.field_info.default
    if default is FieldInfo.empty or default is None:
        return ""

    if callable(arg.show_default):
        return str(arg.show_default(default))

    return str(default)


def _extract_choices(arg: Any) -> tuple[list[str] | None, dict[str, str] | None]:
    """Extract choices and their descriptions from argument."""
    if not arg.parameter.show_choices:
        return None, None

    enum_class = None
    if isclass(arg.hint) and issubclass(arg.hint, Enum):
        enum_class = arg.hint
    elif get_origin(arg.hint) in (list, tuple, set):
        args = getattr(arg.hint, "__args__", ())
        if args and isclass(args[0]) and issubclass(args[0], Enum):
            enum_class = args[0]

    if enum_class:
        choices = []
        descriptions = {}

        member_docstrings = _extract_enum_member_docstrings(enum_class)

        for member_name, member_value in enum_class.__members__.items():
            choice_str = f"`{member_value}`"
            choices.append(choice_str)

            if member_name in member_docstrings:
                doc = _normalize_text(member_docstrings[member_name])
                if doc:
                    descriptions[choice_str] = doc

        return choices, descriptions if descriptions else None
    return None, None


def _extract_constraints(
    arg: Any, constraint_map: dict[str, list[str]]
) -> list[str] | None:
    """Extract constraints from argument using pre-built constraint map."""
    field_name = arg.names[0].lstrip("-").replace("-", "_")
    return constraint_map.get(field_name)


def _split_argument_names(names: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split argument names into short and long options."""
    long_opts = [name for name in names if name.startswith("--")]
    short_opts = [
        name for name in names if name not in long_opts and name.startswith("-")
    ]
    return short_opts, long_opts


def _create_parameter_info(
    arg: Any, constraint_map: dict[str, list[str]]
) -> ParameterInfo:
    """Create ParameterInfo from cyclopts argument."""
    short_opts, long_opts = _split_argument_names(arg.names)
    choices, choice_descriptions = _extract_choices(arg)

    return ParameterInfo(
        display_name=_extract_display_name(arg),
        long_options=" --".join(long_opts),
        short=" ".join(short_opts),
        description=extract_plain_text(arg.parameter.help),
        required=arg.required,
        type_suffix=get_type_suffix(arg.hint),
        default_value=_extract_default_value(arg),
        choices=choices,
        choice_descriptions=choice_descriptions,
        constraints=_extract_constraints(arg, constraint_map),
    )


def _extract_parameter_groups(
    argument_collection: Any, constraint_map: dict[str, list[str]]
) -> dict[str, list[ParameterInfo]]:
    """Extract parameter groups from argument collection."""
    groups: dict[str, list[ParameterInfo]] = defaultdict(list)

    for arg in argument_collection.filter_by(show=True):
        group_name = arg.parameter.group[0].name
        param_info = _create_parameter_info(arg, constraint_map)
        groups[group_name].append(param_info)

    return dict(groups)


def extract_command_info(app: Any) -> list[tuple[str, str]]:
    """Extract available commands and their descriptions."""
    skip_commands = {"--help", "-h", "--version"}
    commands = []

    for name, command_obj in app._commands.items():
        if name in skip_commands:
            continue

        help_text = command_obj.help if hasattr(command_obj, "help") else ""
        if callable(help_text):
            help_text = help_text()
        if help_text:
            help_text = extract_plain_text(help_text).split("\n")[0].strip()

        commands.append((name, help_text))
    return commands


def extract_help_data(app: Any, subcommand: str) -> dict[str, list[ParameterInfo]]:
    """Extract structured help data from the CLI."""
    from typing import get_type_hints

    from cyclopts.bind import normalize_tokens
    from pydantic import BaseModel

    tokens = normalize_tokens(subcommand)
    _, apps, _ = app.parse_commands(tokens)

    constraint_map: dict[str, list[str]] = {}
    func = apps[-1].default_command
    if func:
        hints = get_type_hints(func, include_extras=False)
        for _param_name, hint_type in hints.items():
            if isinstance(hint_type, type) and issubclass(hint_type, BaseModel):
                constraint_map.update(_build_constraint_map(hint_type))
            elif hasattr(hint_type, "__args__"):
                args = getattr(hint_type, "__args__", ())
                for arg in args:
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        constraint_map.update(_build_constraint_map(arg))

    argument_collection = apps[-1].assemble_argument_collection(parse_docstring=True)
    return _extract_parameter_groups(argument_collection, constraint_map)


# =============================================================================
# Markdown Formatting
# =============================================================================


def _format_parameter_header(param: ParameterInfo) -> str:
    """Format parameter header and extract aliases."""
    all_opts = []

    if param.short:
        all_opts.append(f"`{param.short}`")

    for option in param.long_options.split(" --"):
        if option := option.strip():
            if not option.startswith("--"):
                option = f"--{option.lower().replace(' ', '-')}"
            all_opts.append(f"`{option}`")

    if not all_opts:
        return ""

    primary = ", ".join(all_opts)
    type_display = f" `{param.type_suffix.strip()}`" if param.type_suffix else ""
    required_tag = " _(Required)_" if param.required else ""

    return f"#### {primary}{type_display}{required_tag}"


def _format_parameter_body(param: ParameterInfo) -> list[str]:
    """Format parameter description and metadata as markdown list."""
    import re

    lines = [f"{_normalize_text(param.description).rstrip('.')}."]

    is_bool_flag = param.type_suffix == "" or param.type_suffix == " <bool>"
    has_negative = "--no-" in param.long_options

    if is_bool_flag and not has_negative:
        lines.append("<br>_Flag (no value required)_")

    if param.constraints:
        lines.append(f"<br>_Constraints: {', '.join(param.constraints)}_")

    if param.choices:
        if param.choice_descriptions:
            lines.append("")
            lines.append("**Choices:**")
            lines.append("")
            lines.append("| | | |")
            lines.append("|-------|:-------:|-------------|")
            for choice in param.choices:
                desc = param.choice_descriptions.get(choice, "")
                choice_value = choice.strip("`")
                is_default = False

                if param.default_value and param.default_value != "False":
                    default_str = str(param.default_value)
                    if default_str.startswith("[") and default_str.endswith("]"):
                        enum_values = re.findall(r"\.(\w+)", default_str)
                        quoted_values = re.findall(r"'([^']+)'", default_str)
                        all_values = enum_values + quoted_values
                        is_default = any(
                            choice_value.lower() == val.lower() for val in all_values
                        )
                    else:
                        is_default = choice_value == param.default_value

                default_marker = "_default_" if is_default else ""

                if desc:
                    lines.append(f"| {choice} | {default_marker} | {desc} |")
                else:
                    lines.append(f"| {choice} | {default_marker} | |")
        else:
            lines.append(f"<br>_Choices: [{', '.join(param.choices)}]_")
            if param.default_value and param.default_value != "False":
                lines.append(f"<br>_Default: `{param.default_value}`_")
    elif param.default_value and param.default_value != "False":
        lines.append(f"<br>_Default: `{param.default_value}`_")

    lines.append("")
    return lines


def generate_markdown_docs(
    app: Any,
    commands_data: dict[str, dict[str, list[ParameterInfo]]],
) -> str:
    """Generate markdown documentation from help data."""
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "-->",
        "",
        "# Command Line Options",
        "",
    ]

    if commands_data:
        lines.append("## `aiperf` Commands")
        lines.append("")
        commands = extract_command_info(app)
        for name, description in commands:
            if name in commands_data:
                command_anchor = f"aiperf-{name}".lower().replace(" ", "-")
                lines.append(f"### [`{name}`](#{command_anchor})")
                lines.append("")
                lines.append(description)
                lines.append("")

                parameter_groups = commands_data[name]
                skip_group_header = len(parameter_groups) == 1 and list(
                    parameter_groups.keys()
                )[0] in ("Parameters", "Options", "General")

                if not skip_group_header:
                    group_links = []
                    for group_name in parameter_groups:
                        group_anchor = (
                            group_name.lower()
                            .replace(" ", "-")
                            .replace("(", "")
                            .replace(")", "")
                        )
                        group_links.append(f"[{group_name}](#{group_anchor})")
                    lines.append(" • ".join(group_links))
                    lines.append("")

    for command_name, parameter_groups in commands_data.items():
        lines.append("<hr>")
        lines.append("")
        lines.append(f"## `aiperf {command_name}`")
        lines.append("")

        command_obj = app._commands.get(command_name)
        if command_obj and hasattr(command_obj, "help"):
            help_text = command_obj.help
            if callable(help_text):
                help_text = help_text()
            if help_text:
                full_text = extract_plain_text(help_text)
                text_lines = full_text.split("\n")

                description_lines = []
                examples_start_idx = None
                examples_end_idx = None

                for i, line in enumerate(text_lines):
                    if line.strip().lower().startswith("examples:"):
                        examples_start_idx = i
                        break
                    elif line.strip().lower().startswith("args:"):
                        break
                    else:
                        description_lines.append(line)

                description = "\n".join(description_lines).strip()
                if description:
                    paragraphs = description.split("\n\n")
                    for para in paragraphs:
                        if para.strip():
                            para_lines = para.split("\n")
                            min_indent = min(
                                (
                                    len(line) - len(line.lstrip())
                                    for line in para_lines
                                    if line.strip()
                                ),
                                default=0,
                            )
                            normalized_lines = [
                                line[min_indent:] if len(line) > min_indent else line
                                for line in para_lines
                            ]
                            lines.append("\n".join(normalized_lines).strip())
                            lines.append("")

                if examples_start_idx is not None:
                    for i in range(examples_start_idx + 1, len(text_lines)):
                        if text_lines[i].strip().lower().startswith("args:"):
                            examples_end_idx = i
                            break

                    if examples_end_idx is None:
                        examples_end_idx = len(text_lines)

                    example_lines = text_lines[
                        examples_start_idx + 1 : examples_end_idx
                    ]

                    while example_lines and not example_lines[0].strip():
                        example_lines.pop(0)
                    while example_lines and not example_lines[-1].strip():
                        example_lines.pop()

                    if example_lines:
                        non_empty_lines = [
                            line for line in example_lines if line.strip()
                        ]
                        if non_empty_lines:
                            min_indent = min(
                                len(line) - len(line.lstrip())
                                for line in non_empty_lines
                            )
                            normalized_examples = [
                                line[min_indent:] if len(line) > min_indent else line
                                for line in example_lines
                            ]

                            lines.append("**Examples:**")
                            lines.append("")
                            lines.append("```bash")
                            lines.extend(normalized_examples)
                            lines.append("```")
                            lines.append("")

        skip_group_header = len(parameter_groups) == 1 and list(
            parameter_groups.keys()
        )[0] in ("Parameters", "Options", "General")

        for group_name, parameters in parameter_groups.items():
            if not skip_group_header:
                lines.append(f"### {group_name}")
                lines.append("")

            for param in parameters:
                if header := _format_parameter_header(param):
                    lines.append(header)
                    lines.append("")
                    lines.extend(_format_parameter_body(param))

    return "\n".join(line.rstrip() for line in lines)


# =============================================================================
# Main CLI
# =============================================================================


def main() -> int:
    """Generate CLI documentation."""
    parser = argparse.ArgumentParser(
        description="Generate CLI documentation for AIPerf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/generate_cli_docs.py              # Generate docs
  python tools/generate_cli_docs.py --check      # Verify docs are up-to-date
  python tools/generate_cli_docs.py --verbose    # Show detailed progress
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

    print_section("CLI Documentation")

    # Import app lazily
    start = time.perf_counter()
    try:
        # Add src to path so we can import aiperf.cli
        sys.path.insert(0, "src")
        from aiperf.cli import app

        elapsed_ms = (time.perf_counter() - start) * 1000
        if args.verbose:
            print_step("Loaded CLI app", elapsed_ms)
    except ImportError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print_error(
            CLIExtractionError(
                "Failed to import aiperf.cli",
                {
                    "error": str(e),
                    "hint": "Ensure aiperf is installed: uv pip install -e .",
                },
            ),
            verbose=args.verbose,
        )
        return 1

    # Extract command info
    start = time.perf_counter()
    try:
        commands = extract_command_info(app)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if args.verbose:
            print_step(f"Found {len(commands)} commands", elapsed_ms)
    except Exception as e:
        print_error(
            CLIExtractionError("Failed to extract command info", {"error": str(e)}),
            verbose=args.verbose,
        )
        return 1

    # Extract help data for each command
    commands_data: dict[str, dict[str, list[ParameterInfo]]] = {}
    failed_commands = []

    for command_name, _ in commands:
        start = time.perf_counter()
        try:
            commands_data[command_name] = extract_help_data(app, command_name)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if args.verbose:
                param_count = sum(
                    len(params) for params in commands_data[command_name].values()
                )
                print_step(
                    f"Extracted `{command_name}` ({param_count} params)", elapsed_ms
                )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if args.verbose:
                print_step(f"[red]Failed[/] `{command_name}` ({elapsed_ms:.0f}ms)")
            failed_commands.append((command_name, str(e)))

    if failed_commands:
        console.print()
        for cmd, err in failed_commands:
            print_warning(f"Could not extract help data for '{cmd}': {err}")

    # Generate markdown
    start = time.perf_counter()
    try:
        markdown_content = generate_markdown_docs(app, commands_data)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if args.verbose:
            print_step(
                f"Generated markdown ({len(markdown_content)} bytes)", elapsed_ms
            )
    except Exception as e:
        print_error(
            CLIExtractionError("Failed to generate markdown", {"error": str(e)}),
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

    total_params = sum(
        sum(len(params) for params in cmd_data.values())
        for cmd_data in commands_data.values()
    )

    console.print(
        f"[bold green]✓[/] Documented [bold]{len(commands_data)}[/] commands "
        f"with [bold]{total_params}[/] parameters. [dim]({total_elapsed:.2f}s)[/]"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
