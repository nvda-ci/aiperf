# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin management CLI commands.

Simple interface to explore AIPerf plugins:

    aiperf plugins                       # Show all categories
    aiperf plugins endpoint              # List endpoint types
    aiperf plugins endpoint openai       # Details about openai endpoint
    aiperf plugins --packages            # List installed plugin packages
    aiperf plugins --validate            # Validate registry.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cyclopts
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aiperf.plugin import plugin_registry

# ==============================================================================
# CLI Application Setup
# ==============================================================================

plugins_app = cyclopts.App(
    name="plugins",
    help="Explore AIPerf plugins: aiperf plugins [category] [type]",
)

console = Console()


# ==============================================================================
# Helper Functions
# ==============================================================================


def get_all_categories() -> list[str]:
    """Get all category names from registry."""
    return plugin_registry.list_categories()


def ensure_registry_loaded() -> None:
    """Ensure plugin registry is loaded."""
    _ = plugin_registry.list_categories()


def show_overview() -> None:
    """Show all categories with type counts."""
    all_categories = get_all_categories()

    if not all_categories:
        console.print("[yellow]No categories found[/yellow]")
        return

    table = Table(title="Plugin Categories")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Types", style="green", justify="right")
    table.add_column("Description", style="dim")

    descriptions = {
        "arrival_pattern": "Request timing intervals",
        "endpoint": "LLM API endpoint types",
        "timing_strategy": "Request scheduling",
        "data_exporter": "Export formats (CSV, JSON)",
        "transport": "HTTP transport types",
        "dataset_composer": "Dataset creation",
        "dataset_sampler": "Conversation sampling",
        "record_processor": "Record processing",
        "results_processor": "Results aggregation",
        "service": "AIPerf services",
        "ui": "User interfaces",
    }

    for cat in all_categories:
        type_count = len(plugin_registry.list_types(cat))
        desc = descriptions.get(cat, "")
        table.add_row(cat, str(type_count), desc)

    console.print(table)
    console.print(
        "\n[dim]Usage: aiperf plugins <category> to see available types[/dim]"
    )


def show_category_types(category: str) -> None:
    """List all types for a category."""
    lazy_types = plugin_registry.list_types(category)

    if not lazy_types:
        console.print(f"[yellow]Unknown category: {category}[/yellow]")
        console.print("\n[dim]Available categories:[/dim]")
        for cat in get_all_categories():
            console.print(f"  {cat}")
        return

    table = Table(title=f"{category} types")
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Description")

    for lt in lazy_types:
        table.add_row(lt.type_name, lt.description or "[dim]-[/dim]")

    console.print(table)
    console.print(f"\n[dim]Usage: aiperf plugins {category} <type> for details[/dim]")


def show_type_details(category: str, type_name: str) -> None:
    """Show details about a specific type."""
    lazy_types = plugin_registry.list_types(category)
    lazy_type = next((lt for lt in lazy_types if lt.type_name == type_name), None)

    if not lazy_type:
        console.print(f"[red]Not found: {category}:{type_name}[/red]")
        if lazy_types:
            console.print(f"\n[dim]Available {category} types:[/dim]")
            for lt in lazy_types:
                console.print(f"  {lt.type_name}")
        return

    info = f"""[bold]Type:[/bold] {lazy_type.type_name}
[bold]Category:[/bold] {lazy_type.category}
[bold]Package:[/bold] {lazy_type.package_name}
[bold]Class:[/bold] {lazy_type.class_path}

{lazy_type.description or "[dim]No description[/dim]"}"""

    console.print(Panel(info, title=f"{category}:{type_name}", border_style="cyan"))


def show_packages(builtin_only: bool = False) -> None:
    """List all installed plugin packages."""
    packages = plugin_registry.list_packages(builtin_only=builtin_only)

    if not packages:
        console.print("[yellow]No plugins found[/yellow]")
        return

    table = Table(title="Installed Plugins")
    table.add_column("Plugin", style="cyan")
    table.add_column("Source", style="magenta")

    for pkg in packages:
        try:
            metadata = plugin_registry.get_package_metadata(pkg)
            source = "built-in" if metadata.get("builtin", False) else "external"
        except KeyError:
            source = "unknown"
        table.add_row(pkg, source)

    console.print(table)


def run_validate(registry_file: Path | None) -> None:
    """Validate a registry.yaml file."""
    import sys

    try:
        scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from validate_registry import (
            validate_alphabetical_order,
            validate_class_paths,
            validate_no_duplicates,
            validate_plugin_metadata,
            validate_schema_version,
            validate_yaml_syntax,
        )
    except ImportError:
        console.print("[red]Error: validate_registry.py not found[/red]")
        return

    if registry_file is None:
        registry_file = Path(__file__).parent.parent / "registry.yaml"

    if not registry_file.exists():
        console.print(f"[red]File not found: {registry_file}[/red]")
        return

    console.print(f"[bold]Validating:[/bold] {registry_file}\n")

    validators = [
        ("YAML syntax", lambda r: (validate_yaml_syntax(registry_file), r)[0]),
        ("Schema version", validate_schema_version),
        ("Plugin metadata", validate_plugin_metadata),
        ("Alphabetical order", validate_alphabetical_order),
        ("No duplicates", validate_no_duplicates),
        ("Class paths", validate_class_paths),
    ]

    # First validate YAML
    is_valid, registry, error = validate_yaml_syntax(registry_file)
    if not is_valid:
        console.print(f"[red]YAML Error: {error}[/red]")
        return

    console.print("[green]✓[/green] YAML syntax")

    all_passed = True
    for name, validator in validators[1:]:
        errors = validator(registry)
        if errors:
            console.print(f"[red]✗[/red] {name}")
            for err in errors:
                console.print(f"    {err}")
            all_passed = False
        else:
            console.print(f"[green]✓[/green] {name}")

    if all_passed:
        console.print("\n[bold green]All checks passed[/bold green]")
    else:
        console.print("\n[bold red]Validation failed[/bold red]")


# ==============================================================================
# Main Command
# ==============================================================================


@plugins_app.default
def plugins(
    category: Annotated[
        str | None,
        cyclopts.Parameter(help="Category to explore (e.g., endpoint)"),
    ] = None,
    type_name: Annotated[
        str | None,
        cyclopts.Parameter(help="Type name for details (e.g., openai)"),
    ] = None,
    *,
    packages: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--packages", "-p"], help="List installed plugin packages"
        ),
    ] = False,
    validate: Annotated[
        bool,
        cyclopts.Parameter(
            name=["--validate", "-v"], help="Validate built-in registry.yaml"
        ),
    ] = False,
) -> None:
    """Explore AIPerf plugins.

    Examples:
        aiperf plugins                       # Show categories
        aiperf plugins endpoint              # List endpoint types
        aiperf plugins endpoint openai       # Details about openai
        aiperf plugins --packages            # List plugin packages
        aiperf plugins --validate            # Validate registry
    """
    ensure_registry_loaded()

    if packages:
        show_packages()
        return

    if validate:
        run_validate(None)
        return

    if category is None:
        show_overview()
    elif type_name is None:
        show_category_types(category)
    else:
        show_type_details(category, type_name)
