<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Unified PluginRegistry Design

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PluginRegistry (Singleton)                  │
│                  Protocol-Based Unified Registry                │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   Entry Points      Environment Variables     __init__.py
   (pyproject.toml)  (AIPERF_PLUGIN_MODULES)   (Builtins)
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    PluginRegistry.register_lazy()
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
  Internal Registry Structure          Selection Logic
  dict[Protocol →                       (plugins > builtins)
    dict[identifier →
      (PluginMetadata, cached_class)]]
```

## Data Flow

### 1. Registration Phase (at startup)

```
┌──────────────────────────┐
│ aiperf/__init__.py       │
│ from aiperf import *     │ ◄─── Triggers subpackage imports
└──────────────┬───────────┘
               │
        ┌──────┴──────────────────────┬───────────────┐
        │                             │               │
        ▼                             ▼               ▼
  endpoints/        transports/         ui/
  __init__.py       __init__.py      __init__.py
        │                  │              │
        └──────────┬───────┴────────┬─────┘
                   │               │
        PluginRegistry.register_lazy(
            protocol=EndpointProtocol,
            identifier="openai_chat",
            module_path="aiperf.endpoints.openai_chat",
            class_name="OpenAIChat",
            is_plugin=False,  ◄─── Builtins
            package_name="aiperf"
        )
```

### 2. Discovery Phase (after registration)

```
Somewhere in startup (e.g., main.py or app.py):

    PluginRegistry.discover_all(EndpointProtocol)
            │
            ├──► Load from entry points: EntryPoint → register_lazy(is_plugin=True)
            │
            └──► Load from env: AIPERF_PLUGIN_MODULES → import module (which calls register_lazy)
```

### 3. Selection Phase (automatic after each register)

```
PluginRegistry._update_selected(protocol)
            │
        ┌───┴──────────────────────────┐
        │                              │
        ▼                              ▼
   Filter by is_plugin              Filter by is_plugin
        (True)                           (False)
        │                              │
   Third-party plugins           Built-in implementations
   (HIGHEST priority)            (lower priority)
        │
        └──► Pick first (registration order) ──► _selected[Protocol] = identifier
```

### 4. Usage Phase (lazy loading on demand)

```
PluginRegistry.create_instance(EndpointProtocol, config=...)
            │
            ├──► Get selected identifier from _selected[Protocol]
            │
            └──► Call _get_class_lazy(Protocol, identifier)
                        │
                        ├──► Is class already cached?
                        │    YES: return cached_class
                        │
                        └──► NO: import module, extract class, cache it, return
                                    │
                                    ▼
                            importlib.import_module(module_path)
                            getattr(module, class_name)
                            cache in _registry[Protocol][identifier]
```

## Registry Internal Structure

```python
_registry: dict[type[Protocol], dict[str, tuple[PluginMetadata, type[Any] | None]]] = {
    EndpointProtocol: {
        "openai_chat": (
            PluginMetadata(
                identifier="openai_chat",
                protocol=EndpointProtocol,
                module_path="aiperf.endpoints.openai_chat",
                class_name="OpenAIChat",
                is_plugin=False,
                package_name="aiperf",
                version="0.1.0",
                description="..."
            ),
            None  # ◄─── Lazy loaded on first use
        ),
        "nvidia_optimized_chat": (
            PluginMetadata(
                identifier="nvidia_optimized_chat",
                protocol=EndpointProtocol,
                module_path="nvidia_endpoints.optimized",
                class_name="OptimizedChatEndpoint",
                is_plugin=True,  # ◄─── Third-party plugin
                package_name="nvidia-optimized-endpoints",
                version="1.2.0",
                description="..."
            ),
            OptimizedChatEndpoint  # ◄─── Cached after first load
        ),
    },
    TransportProtocol: {
        # ... similar structure
    }
}

_selected: dict[type[Protocol], str] = {
    EndpointProtocol: "nvidia_optimized_chat",  # Plugins override builtins!
    TransportProtocol: "http_transport",
}
```

## Key Methods & Signatures

```python
class PluginRegistry:
    # Registration
    @classmethod
    def register_lazy(
        cls,
        protocol: type[ProtocolT],
        identifier: str,
        module_path: str,
        class_name: str,
        is_plugin: bool = False,
        package_name: str = "aiperf",
        version: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register without loading. Call from __init__.py or discovery."""

    # Discovery
    @classmethod
    def discover_all(
        cls,
        protocol: type[Protocol],
        entry_point_group: str | None = None
    ) -> None:
        """Discover from entry points + AIPERF_PLUGIN_MODULES env var."""

    # Selection
    @classmethod
    def use_implementation_by_id(
        cls,
        protocol: type[Protocol],
        identifier: str
    ) -> None:
        """Explicitly select an implementation."""

    @classmethod
    def _update_selected(cls, protocol: type[Protocol]) -> None:
        """Auto-select best (plugins > builtins). Called after register_lazy."""

    # Usage
    @classmethod
    def create_instance(
        cls,
        protocol: type[Protocol],
        **kwargs: Any
    ) -> Any:
        """Create instance of selected implementation (lazy loads class)."""

    @classmethod
    def get_class(cls, protocol: type[Protocol]) -> type[Any]:
        """Get the selected class (lazy loads if needed)."""

    # Inspection
    @classmethod
    def list_implementations(
        cls,
        protocol: type[Protocol]
    ) -> list[dict[str, Any]]:
        """List all available implementations for a protocol."""

    # Internal
    @classmethod
    def _get_class_lazy(
        cls,
        protocol: type[Protocol],
        identifier: str
    ) -> type[Any]:
        """Lazy load + cache a class."""

    @classmethod
    def _load_entrypoint_plugin(
        cls,
        protocol: type[Protocol],
        ep: Any
    ) -> None:
        """Helper: load single entry point."""

    @classmethod
    def _load_env_plugin_module(
        cls,
        protocol: type[Protocol],
        module_name: str
    ) -> None:
        """Helper: import module that calls register_lazy()."""
```

## PluginMetadata Model

```python
class PluginMetadata(BaseModel):
    identifier: str              # "openai_chat"
    protocol: type[Protocol]     # EndpointProtocol
    module_path: str             # "aiperf.endpoints.openai_chat"
    class_name: str              # "OpenAIChat"
    is_plugin: bool              # False (builtin) or True (third-party)
    package_name: str            # "aiperf" or "nvidia-endpoints"
    version: str | None          # "0.1.0"
    description: str | None      # Extracted from package metadata
```

## Registration Workflow

### Builtins (in `aiperf/endpoints/__init__.py`)

```python
from aiperf.common.plugins import PluginRegistry
from aiperf.common.protocols import EndpointProtocol

# Register without importing (lazy loading!)
PluginRegistry.register_lazy(
    protocol=EndpointProtocol,
    identifier="openai_chat",
    module_path="aiperf.endpoints.openai_chat",
    class_name="OpenAIChat",
    is_plugin=False,
    package_name="aiperf",
)

PluginRegistry.register_lazy(
    protocol=EndpointProtocol,
    identifier="anthropic_chat",
    module_path="aiperf.endpoints.anthropic_chat",
    class_name="AnthropicChat",
    is_plugin=False,
    package_name="aiperf",
)
```

### Entry Points (in `pyproject.toml`)

```toml
[project.entry-points."EndpointProtocol"]
nvidia_optimized_chat = "nvidia_endpoints.optimized:OptimizedChatEndpoint"
```

Entry point loader will extract and call:
```python
PluginRegistry.register_lazy(
    protocol=EndpointProtocol,
    identifier="nvidia_optimized_chat",
    module_path="nvidia_endpoints.optimized",
    class_name="OptimizedChatEndpoint",
    is_plugin=True,
    package_name="nvidia-optimized-endpoints",
    version="1.2.0",
    description="Optimized chat endpoint..."
)
```

### Environment Plugins (in plugin's `__init__.py`)

Set: `AIPERF_PLUGIN_MODULES=my_custom_plugin.endpoints`

Then in `my_custom_plugin/endpoints/__init__.py`:
```python
from aiperf.common.plugins import PluginRegistry
from aiperf.common.protocols import EndpointProtocol

PluginRegistry.register_lazy(
    protocol=EndpointProtocol,
    identifier="custom_chat",
    module_path="my_custom_plugin.endpoints.custom",
    class_name="CustomChat",
    is_plugin=True,
    package_name="my-custom-plugin",
)
```

## Priority System

**Selection order (highest to lowest priority):**

```
1. User explicitly calls use_implementation_by_id()  ◄─── HIGHEST
2. Third-party plugins (is_plugin=True)
3. Built-in implementations (is_plugin=False)       ◄─── LOWEST
```

When multiple plugins registered:
- Among plugins: First registered wins
- Among builtins: First registered wins
- Plugins always beat builtins

## Thread Safety

```
_instance_lock: threading.Lock

register_lazy()          ✓ Thread-safe (uses lock)
_update_selected()       ✓ Thread-safe (uses lock)
use_implementation_by_id() ✓ Thread-safe (uses lock)
_get_class_lazy()        ✓ Thread-safe (uses lock for caching)
list_implementations()   ✓ Read-only (no lock needed)
create_instance()        ✓ Thread-safe (uses get_class which uses lock)
```

## Example Usage

```python
from aiperf.common.plugins import PluginRegistry
from aiperf.common.protocols import EndpointProtocol

# At startup
PluginRegistry.discover_all(EndpointProtocol)

# List what's available
for impl in PluginRegistry.list_implementations(EndpointProtocol):
    print(f"{impl['identifier']}: {impl['class_name']} "
          f"({'plugin' if impl['is_plugin'] else 'builtin'}) "
          f"[selected={impl['is_selected']}]")

# Create an instance with selected (plugin overrides builtin)
endpoint = PluginRegistry.create_instance(EndpointProtocol, config=my_config)

# Override to use specific one
PluginRegistry.use_implementation_by_id(EndpointProtocol, "openai_chat")
endpoint = PluginRegistry.create_instance(EndpointProtocol, config=my_config)

# Inspect available
available = PluginRegistry.list_implementations(EndpointProtocol)
print(available)
```

## Key Design Principles

✅ **Lazy Loading** - Classes only loaded when first used (not at registration)
✅ **Protocol-Based** - Uses Python Protocols as type identifiers (type-safe)
✅ **Unified Registry** - Single PluginRegistry for all component types
✅ **Co-Located Registration** - Registrations live in each package's `__init__.py`
✅ **Seamless Overrides** - Plugins automatically win over builtins (same identifier)
✅ **Flexible Discovery** - Entry points + environment variables
✅ **Thread-Safe** - All mutable operations protected by lock
✅ **Backward Compatible** - Coexists with existing AIPerfFactory (during transition)
