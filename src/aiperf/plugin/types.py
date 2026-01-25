# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)

# ==============================================================================
# Type Definitions
# ==============================================================================


class PackageMetadata(TypedDict, total=False):
    """Package metadata from YAML manifest."""

    name: str
    version: str
    description: str
    author: str
    license: str
    homepage: str
    builtin: bool


class ManifestData(TypedDict, total=False):
    """YAML manifest structure."""

    schema_version: str
    plugin: dict[str, Any]


# TypeSpec uses functional form because 'class' is a Python keyword.
TypeSpec = TypedDict(
    "TypeSpec",
    {
        "class": str,  # Fully qualified class path (module:Class)
        "description": str,
        "priority": int,
    },
    total=False,
)


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class PluginError(Exception):
    """Base exception for plugin system errors."""


class TypeNotFoundError(PluginError):
    """Type not found in category. Includes available types in error message."""

    def __init__(self, category: str, type_name: str, available: list[str]) -> None:
        self.category = category
        self.type_name = type_name
        self.available = available

        available_str = "\n".join(f"  • {name}" for name in sorted(available))
        super().__init__(
            f"Type '{type_name}' not found for category '{category}'.\n"
            f"Available types:\n{available_str}"
        )


# ==============================================================================
# Implementation Classes
# ==============================================================================


@dataclass(frozen=True, slots=True)
class TypeEntry:
    """Lazy-loading type entry with metadata. Call load() to import the class."""

    category: str = field(metadata={"description": "Category identifier"})
    type_name: str = field(metadata={"description": "Type name"})
    package_name: str = field(metadata={"description": "Package providing this type"})
    class_path: str = field(
        metadata={"description": "Fully qualified class path (module:Class)"}
    )
    priority: int = field(
        default=0, metadata={"description": "Conflict resolution priority"}
    )
    description: str = field(
        default="", metadata={"description": "Human-readable description"}
    )
    metadata: PackageMetadata = field(
        default_factory=dict, metadata={"description": "Package metadata"}
    )
    loaded_class: type | None = field(
        default=None, metadata={"description": "Cached class after loading"}
    )
    is_builtin: bool = field(
        default=False, metadata={"description": "Whether this is built-in"}
    )

    def load(self) -> type:
        """Import and return the class (cached after first call)."""
        # Return cached class if already loaded
        if self.loaded_class is not None:
            return self.loaded_class

        # Validate and parse class path using structural pattern matching
        module_path, _, class_name = self.class_path.rpartition(":")
        if not module_path or not class_name:
            raise ValueError(
                f"Invalid class_path format: {self.class_path}\n"
                f"Expected format: 'module.path:ClassName'\n"
                f"Example: 'aiperf.endpoints.openai:OpenAIEndpoint'"
            )

        # Import and cache the class
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            # Cache for future calls (thread-safe with frozen dataclass)
            object.__setattr__(self, "loaded_class", cls)

            _logger.debug(
                lambda: f"Loaded {self.category}:{self.type_name} from {self.class_path}"
            )

            return cls

        except ImportError as e:
            # Raise enriched ImportError for backward compatibility
            raise ImportError(
                f"Failed to import module for {self.category}:{self.type_name} from '{self.class_path}'\n"
                f"Reason: {e!r}\n"
                f"Tip: Check that the module is installed and importable"
            ) from e
        except AttributeError as e:
            # Raise enriched AttributeError for backward compatibility
            raise AttributeError(
                f"Class '{class_name}' not found for {self.category}:{self.type_name} from '{self.class_path}'\n"
                f"Reason: {e!r}\n"
                f"Tip: Check that the class name is spelled correctly and exported from the module"
            ) from e

    def validate(self, check_class: bool = False) -> tuple[bool, str | None]:
        """Validate class is loadable without importing. Returns (is_valid, error_message)."""
        # Already loaded means it's valid
        if self.loaded_class is not None:
            return True, None

        # Validate class_path format
        parts = self.class_path.split(":")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return (
                False,
                f"Invalid class_path format: {self.class_path} (expected 'module:ClassName')",
            )

        module_path, class_name = parts

        # Check if module exists without importing it
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                return False, f"Module not found: {module_path}"
        except ModuleNotFoundError as e:
            return False, f"Module not found: {module_path} ({e})"
        except Exception as e:
            return False, f"Error checking module {module_path}: {e}"

        # Optionally verify class exists via AST (no code execution)
        if check_class and spec is not None and spec.origin is not None:
            try:
                source_path = Path(spec.origin)
                if source_path.suffix == ".py" and source_path.exists():
                    source = source_path.read_text(encoding="utf-8")
                    tree = ast.parse(source)

                    # Look for class definition or import/assignment
                    class_found = False
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == class_name:
                            class_found = True
                            break
                        # Also check for imports that might bring in the class
                        if isinstance(node, ast.ImportFrom) and node.names:
                            for alias in node.names:
                                if (
                                    alias.name == class_name
                                    or alias.asname == class_name
                                ):
                                    class_found = True
                                    break

                    if not class_found:
                        return False, f"Class '{class_name}' not found in {module_path}"
            except SyntaxError as e:
                return False, f"Syntax error in {module_path}: {e}"
            except Exception as e:
                # AST parsing failed, but module exists - don't fail validation
                _logger.debug(lambda err=e: f"Could not verify class via AST: {err}")

        return True, None
