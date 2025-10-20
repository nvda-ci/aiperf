<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# DEP 001: Lazy-Loaded Plugin Registry System

**Status**: Draft

**Authors**: AIPerf Engineering Team

**Category**: Architecture

**Replaces**: N/A

**Replaced By**: N/A

**Sponsor**: [TBD]

**Required Reviewers**: [TBD]

**Review Date**: [TBD]

**Pull Request**: [TBD]

**Implementation PR / Tracking Issue**: [TBD]

# Summary

Implement a lazy-loaded, unified plugin registry system for AIPerf component discovery and instantiation. The system uses Protocols as type identifiers, decouples implementation registration from loading via string paths, supports multiple plugin discovery mechanisms (entry points and environment variables), and provides identifier-based implementation selection with co-located registration in package `__init__.py` files.

# Motivation

The current factory system requires decorator-based registration scattered across multiple files and deletes lower-priority implementations, preventing users from accessing alternative implementations. The system lacks:

1. Centralized visibility into available implementations per package
2. Support for late-binding implementation selection
3. Flexible plugin distribution mechanisms
4. Decoupling of registration from class loading
5. Co-located registration with implementation organization
6. Unified plugin discovery mechanism
7. User-friendly implementation override mechanisms

This proposal addresses these gaps by introducing a lazy-loaded, protocol-based registry system with co-located registrations in package `__init__.py` files, supporting multiple discovery mechanisms while maintaining backward compatibility.

## Goals

* Co-locate registration with package organization (no separate registry files)
* Enable lazy loading of implementation classes via string paths
* Support multiple plugin discovery mechanisms
* Allow users to select between available implementations without code changes
* Provide clear visibility into available implementations via identifiers
* Use Protocols as type-safe, self-documenting type identifiers
* Reduce friction for plugin development and maintenance
* Provide single unified registry for all component types

### Non Goals

* Replace existing Protocol definitions
* Require changes to implementation classes
* Support complex multi-version compatibility
* Provide automatic fallback mechanisms
* Implement configuration file parsing (environment variables only)

## Requirements

### REQ 1: Lazy Loading

All implementation classes **MUST** be loaded on first use, not at registration time. The registry **MUST** store only module paths and class names as strings, not actual class references, until instantiation.

### REQ 2: Co-Located Registration

Implementation registrations **MUST** be located in package `__init__.py` files alongside their respective implementations. Registrations **MUST** use `register_lazy()` with string paths and **MUST NOT** use decorators.

### REQ 3: Protocol-Based Typing

Implementation types **MUST** be identified by their Protocol interface (e.g., `EndpointProtocol`, `TransportProtocol`). Protocols **MUST** serve as both type identifiers and structural contracts. The registry **MUST** support any Protocol type without requiring predefined enumerations.

### REQ 4: Unified Registry

A single `PluginRegistry` class **MUST** manage all component types. The registry **MUST** accept Protocol types as the first parameter to `register_lazy()` and **MUST NOT** require separate factory classes per component type.

### REQ 5: Identifier-Based Selection

Implementation selection within a Protocol type **MUST** use identifiers (e.g., "openai_chat", "nvidia_optimized"). Users **MUST** be able to select implementations by identifier without importing classes.

### REQ 6: Plugin Discovery from Entry Points

The system **MUST** support plugin discovery via entry points in the `aiperf.plugins` group. Each entry point value **MUST** be a module path to a plugin package (e.g., `my_plugin`).

### REQ 7: Plugin Discovery from Environment Variables

The system **MUST** support plugin discovery via the `AIPERF_PLUGIN_MODULES` environment variable. The variable **MUST** contain comma-separated module paths that are imported at discovery time.

### REQ 8: Priority Resolution

When multiple implementations exist for the same Protocol type, the system **MUST** prioritize them in this order:
1. Runtime overrides (user-selected)
2. Plugin implementations (is_plugin=True)
3. Builtin implementations (is_plugin=False)

### REQ 9: Backward Compatibility

Existing factory class APIs **MUST** continue to function. The registry system **MUST** work alongside existing decorator-based registration during transition period.

# Proposal

## Overview

The lazy-loaded plugin registry system consists of three layers:

1. **Builtin Registration** (Package Init Time): Co-located in `__init__.py` files using string paths
2. **Plugin Discovery** (Startup Time): Entry points or environment variables automatically import plugin packages
3. **Runtime Selection** (Usage Time): User can override implementation selection via identifiers

All three layers use a single unified `PluginRegistry` with Protocols as type identifiers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION STARTUP                        │
│                                                                  │
│  import aiperf                  (loads aiperf package)          │
│    ↓ imports subpackages                                        │
│  from aiperf import endpoints   (__init__.py registrations)    │
│  from aiperf import transports  (__init__.py registrations)    │
│                                                                  │
│  PluginDiscovery.discover_all() (discovers & imports plugins)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │                                             │
        ├──────────────────┬──────────────────────────┤
        │                  │                          │
        ▼                  ▼                          ▼
   ┌─────────┐        ┌──────────┐           ┌────────────┐
   │ BUILTIN │        │ PLUGINS  │           │  RUNTIME   │
   │REGISTRY │        │DISCOVERY │           │ OVERRIDE   │
   └─────────┘        └──────────┘           └────────────┘
        │                  │                        │
        ▼                  ▼                        ▼
   __init__.py        pip install OR      use_implementation_by_id()
   (endpoints,        environment vars     (user switches anytime)
    transports,       (simple plugins)
    ui, etc)
```

## Layer 1: Builtin Registration (Co-Located in __init__.py)

Each package provides registrations in its `__init__.py`:

```python
# aiperf/endpoints/__init__.py
from aiperf.common.plugins import PluginRegistry
from aiperf.common.protocols import EndpointProtocol

# Register all builtin endpoints using string paths (lazy loaded)
PluginRegistry.register_lazy(
    EndpointProtocol,
    "chat",
    "aiperf.endpoints.openai_chat",
    "ChatEndpoint",
    identifier="openai_chat",
    is_plugin=False
)

PluginRegistry.register_lazy(
    EndpointProtocol,
    "embeddings",
    "aiperf.endpoints.openai_embeddings",
    "EmbeddingsEndpoint",
    identifier="openai_embeddings",
    is_plugin=False
)

# Re-export for public API
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
__all__ = ["ChatEndpoint", "EmbeddingsEndpoint"]
```

Root package imports subpackages to trigger registration:

```python
# aiperf/__init__.py
# Importing subpackages triggers their __init__.py registrations
from aiperf import endpoints
from aiperf import transports
from aiperf import ui

# Re-export public APIs
from aiperf.endpoints import ChatEndpoint, EmbeddingsEndpoint
# ... etc
```

## Layer 2: Plugin Discovery

Plugins register via two mechanisms:

### Entry Points (Packaged Distribution)

```toml
# my_plugin/pyproject.toml
[project.entry-points."aiperf.plugins"]
my_plugin = "my_plugin"
```

Plugin provides registrations in its `__init__.py`:

```python
# my_plugin/__init__.py
from aiperf.common.plugins import PluginRegistry
from aiperf.common.protocols import EndpointProtocol, TransportProtocol

# Register all plugin implementations
PluginRegistry.register_lazy(
    EndpointProtocol,
    "chat",
    "my_plugin.endpoints",
    "OptimizedChatEndpoint",
    identifier="nvidia_optimized",
    is_plugin=True
)

PluginRegistry.register_lazy(
    TransportProtocol,
    "grpc",
    "my_plugin.transports",
    "OptimizedGRPCTransport",
    identifier="nvidia_grpc",
    is_plugin=True
)
```

### Environment Variables (Simple/Local Plugins)

```bash
export AIPERF_PLUGIN_MODULES="my_custom_plugin,experimental_features"
```

Plugins use same `__init__.py` structure. No `pyproject.toml` required.

### Discovery Mechanism

```python
@classmethod
def discover_all(cls) -> None:
    """Load plugins from entry points and environment variables"""
    import os
    import importlib
    from importlib.metadata import entry_points

    loaded = set()

    # Entry points: import plugin packages
    for ep in entry_points().select(group="aiperf.plugins"):
        importlib.import_module(ep.value)  # Triggers plugin/__init__.py
        loaded.add(ep.value)

    # Environment variables: import plugin modules
    env_plugins = os.getenv("AIPERF_PLUGIN_MODULES", "")
    for module_path in env_plugins.split(","):
        module_path = module_path.strip()
        if module_path and module_path not in loaded:
            importlib.import_module(module_path)
```

## Layer 3: Runtime Selection

Users select implementations by Protocol and identifier:

```python
from aiperf.common.plugins import PluginRegistry
from aiperf.common.protocols import EndpointProtocol

# See all available endpoints
implementations = PluginRegistry.list_implementations(EndpointProtocol)
# [
#   {"id": "nvidia_optimized", "module": "my_plugin.endpoints", "is_plugin": True, "selected": True},
#   {"id": "openai_chat", "module": "aiperf.endpoints.openai_chat", "is_plugin": False},
# ]

# Switch to a specific implementation
PluginRegistry.use_implementation_by_id(EndpointProtocol, "openai_chat")

# Create instance (lazy loads if needed)
endpoint = PluginRegistry.create_instance(EndpointProtocol, **kwargs)
```

## Unified Registry Structure

```
PluginRegistry = {
    EndpointProtocol: {
        "chat": [
            {"identifier": "nvidia_optimized", "module": "my_plugin.endpoints", ...},
            {"identifier": "openai_chat", "module": "aiperf.endpoints.openai_chat", ...},
        ],
        "embeddings": [
            {"identifier": "nvidia_embeddings", "module": "my_plugin.endpoints", ...},
            {"identifier": "openai_embeddings", "module": "aiperf.endpoints.openai_embeddings", ...},
        ]
    },
    TransportProtocol: {
        "http": [
            {"identifier": "http", "module": "aiperf.transports.http_transport", ...},
        ],
        "grpc": [
            {"identifier": "nvidia_grpc", "module": "my_plugin.transports", "is_plugin": True, ...},
            {"identifier": "grpc", "module": "aiperf.transports.grpc_transport", ...},
        ]
    }
}
```

Lookup is Protocol → identifier within that Protocol.

# Implementation Details

## Core PluginRegistry Methods

```python
@classmethod
def register_lazy(
    cls,
    protocol: Type[ProtocolT],
    class_type: str,
    module_path: str,
    class_name: str,
    identifier: str | None = None,
    is_plugin: bool = False
) -> None:
    """Register a lazy-loaded implementation under a Protocol type"""
    # If identifier not provided, generate from class_name
    if identifier is None:
        identifier = class_name

    metadata = {
        "identifier": identifier,
        "module_path": module_path,
        "class_name": class_name,
        "is_plugin": is_plugin,
        "_loaded_class": None,  # Cache after first load
    }

    if protocol not in cls._registry:
        cls._registry[protocol] = {}

    if class_type not in cls._registry[protocol]:
        cls._registry[protocol][class_type] = []

    cls._registry[protocol][class_type].append(metadata)
    cls._update_selected(protocol, class_type)

@classmethod
def _get_class_lazy(cls, module_path: str, class_name: str) -> type:
    """Load class on first access"""
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

@classmethod
def use_implementation_by_id(
    cls,
    protocol: Type[ProtocolT],
    identifier: str
) -> None:
    """Select implementation by identifier"""
    # Find and select the implementation with matching identifier
    for class_type_impls in cls._registry.get(protocol, {}).values():
        for metadata in class_type_impls:
            if metadata["identifier"] == identifier:
                cls._selected[(protocol, metadata["class_type"])] = metadata
                return

    raise FactoryCreationError(f"Implementation {identifier!r} not found")

@classmethod
def list_implementations(
    cls, protocol: Type[ProtocolT]
) -> list[dict]:
    """List all implementations for a Protocol"""
    result = []
    for class_type, impls in cls._registry.get(protocol, {}).items():
        for metadata in impls:
            result.append({
                "id": metadata["identifier"],
                "name": metadata["class_name"],
                "module": metadata["module_path"],
                "class_type": class_type,
                "is_plugin": metadata["is_plugin"],
                "is_selected": cls._selected.get((protocol, class_type)) == metadata,
            })
    return result

@classmethod
def create_instance(
    cls, protocol: Type[ProtocolT], **kwargs
) -> ProtocolT:
    """Create instance - loads class lazily if needed"""
    # Get the selected implementation for this protocol
    selected = cls._selected.get(protocol)
    if selected is None:
        raise FactoryCreationError(f"No implementation selected for {protocol.__name__}")

    impl_class, metadata = selected

    # Load class if not already loaded
    if impl_class is None:
        impl_class = cls._get_class_lazy(metadata["module_path"], metadata["class_name"])
        metadata["_loaded_class"] = impl_class
        cls._selected[protocol] = (impl_class, metadata)

    return impl_class(**kwargs)
```

## Project Structure

```
aiperf/
├── __init__.py                         # Imports subpackages (triggers registrations)
├── endpoints/
│   ├── __init__.py                     # ← Endpoint registrations here
│   ├── openai_chat.py                  # ChatEndpoint (clean, no decorators)
│   ├── openai_embeddings.py            # EmbeddingsEndpoint (clean)
│   └── ...
├── transports/
│   ├── __init__.py                     # ← Transport registrations here
│   ├── http_transport.py               # HTTPTransport (clean)
│   ├── grpc_transport.py               # GRPCTransport (clean)
│   └── ...
├── ui/
│   ├── __init__.py                     # ← UI registrations here
│   └── ...
└── common/
    ├── protocols.py                    # EndpointProtocol, TransportProtocol, etc
    ├── plugins.py                      # PluginRegistry implementation
    └── ...

# Plugin example
my_plugin/
├── __init__.py                         # ← Plugin registrations here
├── endpoints.py
├── transports.py
└── pyproject.toml
    [project.entry-points."aiperf.plugins"]
    my_plugin = "my_plugin"
```

## Deferred to Implementation

* Protocol discovery and validation mechanisms
* Caching strategies for loaded classes
* Error handling and logging levels
* Configuration file support (beyond environment variables)

# Implementation Phases

## Phase 0: PluginRegistry Implementation

**Release Target**: Current sprint

**Effort Estimate**: 2-3 engineers, 1 week

**Supported API / Behavior:**

* `register_lazy(protocol, class_type, module_path, class_name, identifier, is_plugin)` - Register lazy-loaded implementation under Protocol
* `discover_all()` - Discover plugins from entry points and environment variables
* `use_implementation_by_id(protocol, identifier)` - Select implementation by identifier
* `list_implementations(protocol)` - List available implementations for Protocol
* `create_instance(protocol, **kwargs)` - Create instance with lazy loading on first use

**Not Supported:**

* Configuration file parsing (future phase)
* Protocol validation and discovery (future phase)

## Phase 1: Co-Located Registration Migration

**Release Target**: Following phase

**Effort Estimate**: 2 engineers, 1 week

**Supported API / Behavior:**

* Add `PluginRegistry.register_lazy()` calls to each package `__init__.py`
* Update `aiperf/__init__.py` to import subpackages (triggers registration)
* Registrations for all builtin endpoints, transports, UI components

**Not Supported:**

* Removal of existing decorator support (maintain backward compatibility)
* Plugin distribution updates (Phase 2)

## Phase 2: Plugin Distribution

**Release Target**: Following phase

**Effort Estimate**: 1-2 engineers, 1 week

**Supported API / Behavior:**

* Entry point support for plugin discovery
* Environment variable support for plugin modules
* Documentation for plugin developers

**Not Supported:**

* Automatic plugin installation
* Plugin marketplace or registry

# Related Proposals

N/A

# Alternate Solutions

## Alt 1: Entry Points Only

**Pros:**

* Standard Python packaging mechanism
* Single discovery method
* Works well for packaged distributions

**Cons:**

* Requires pyproject.toml for all plugins
* Complex for local/simple plugin development
* Friction for prototyping

**Reason Rejected:**

Entry points are appropriate for distributed packages but create unnecessary friction for simple/local plugin development. The hybrid approach supports both use cases.

## Alt 2: Scatter Decorators with Dynamic Loading

**Pros:**

* Minimal registry file changes
* Decorators remain in implementation files
* Straightforward to understand

**Cons:**

* Requires importing all implementation modules to trigger registration
* Defeats purpose of lazy loading
* No centralized visibility into available implementations
* Difficult to discover what's available

**Reason Rejected:**

Scattered decorators prevent the primary benefit of lazy loading (deferred class import). Centralized registry provides better visibility and maintainability.

## Alt 3: YAML/TOML Configuration Files

**Pros:**

* Explicit configuration
* Human-readable format
* Supports complex scenarios

**Cons:**

* Additional configuration file format to maintain
* Duplication of code location (registry.py + config file)
* Learning curve for users
* Harder to validate at development time

**Reason Rejected:**

Environment variables provide sufficient configuration for simple cases while avoiding format proliferation. Configuration files can be added in future phases if needed.

# Background

The AIPerf factory system currently uses decorator-based registration that requires importing all implementation classes upfront, even if they are not used. This creates startup latency and makes plugin development friction-filled. Additionally, the system deletes lower-priority implementations, preventing users from accessing alternatives without code changes.

The proposal addresses these limitations through lazy loading, centralized registration, and flexible plugin discovery mechanisms that support both packaged distributions (via entry points) and simple local plugins (via environment variables).

## References

* [PEP 659: Python entry points](https://peps.python.org/pep-0659/)
* [RFC 2119: Key words](https://datatracker.ietf.org/doc/html/rfc2119)

## Terminology & Definitions

| **Term** | **Definition** |
| :---- | :---- |
| **Class Type** | Categorization of implementations (e.g., "chat", "embeddings"). Represents the interface/protocol that implementations conform to. |
| **Identifier** | Unique identifier for an implementation within a class_type. Used for user selection and discovery. |
| **Lazy Loading** | Deferring class instantiation until first use rather than importing at registration time. |
| **Plugin** | Third-party or optional implementation registered via entry points or environment variables. |
| **Registry** | Centralized Python file containing registration declarations for a component package. |

## Acronyms & Abbreviations

**DEP**: Design Enhancement Proposal
**REQ**: Requirement
**pyproject.toml**: Python project configuration file following PEP 517/518
