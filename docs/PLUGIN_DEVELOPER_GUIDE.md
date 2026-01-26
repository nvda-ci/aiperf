<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Plugin System Developer Guide

A practical guide to understanding and using the AIPerf plugin system.

**Document Version**: 1.0
**Date**: January 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Concepts](#core-concepts)
4. [Using Plugins](#using-plugins)
5. [Creating Plugins](#creating-plugins)
6. [Plugin Categories](#plugin-categories)
7. [Advanced Topics](#advanced-topics)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The AIPerf plugin system provides a **YAML-first, type-safe factory pattern** for extensibility. It enables:

- **Selecting implementations** - Choose one endpoint, transport, or strategy from registered options
- **Iterating all implementations** - Run all exporters, processors, or loaders
- **Extending AIPerf** - Add custom endpoints, exporters, timing strategies without modifying core code
- **Overriding built-ins** - Replace default implementations with higher-priority alternatives

### Key Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Type-safe** | 21 `@overload` signatures provide IDE autocomplete and type inference |
| **YAML-first** | Declarative plugin manifests with rich metadata |
| **Lazy loading** | Classes loaded only when first accessed |
| **Priority-based** | Conflict resolution via priority + external-beats-builtin rules |
| **Entry point discovery** | Automatic discovery via `pyproject.toml` entry points |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      plugins.py (Facade)                     │
│  - 21 type-safe @overload signatures                        │
│  - Module-level functions: get_class, list_types, etc.      │
├─────────────────────────────────────────────────────────────┤
│                  _plugin_registry.py (Core)                  │
│  - PluginRegistry singleton                                 │
│  - YAML loading, entry point discovery                      │
│  - Priority-based conflict resolution                       │
├─────────────────────────────────────────────────────────────┤
│                     types.py (Models)                        │
│  - TypeEntry: frozen Pydantic model per registered type     │
│  - CategoryMetadata: protocol, enum, description            │
│  - PluginError, TypeNotFoundError                           │
├─────────────────────────────────────────────────────────────┤
│                    enums.py (Runtime Enums)                  │
│  - PluginType: dynamically generated from categories    │
│  - Type enums: EndpointType, TransportType, etc.            │
│  - ExtensibleStrEnum: works as both enum and string         │
├─────────────────────────────────────────────────────────────┤
│                   schema/ (Validation)                       │
│  - Pydantic models for YAML validation                      │
│  - JSON Schema generation for IDE support                   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Discovery (on import)
   pyproject.toml entry points → discover_plugins() → load_registry()

2. Registration
   plugins.yaml → Pydantic validation → TypeEntry → _types dict

3. Retrieval
   plugins.get_class(category, name)
       → _registry.get_class()
       → TypeEntry.load()  # lazy import
       → cached class returned
```

---

## Core Concepts

### TypeEntry

Every registered plugin type is represented by a `TypeEntry`:

```python
class TypeEntry(BaseModel):
    model_config = ConfigDict(frozen=True)  # Immutable

    category: str      # e.g., "endpoint"
    name: str          # e.g., "chat"
    package: str       # e.g., "aiperf"
    class_path: str    # e.g., "aiperf.endpoints.openai_chat:ChatEndpoint"
    priority: int      # Conflict resolution (higher wins)
    description: str   # Human-readable description
    metadata: dict     # Category-specific config (e.g., url_schemes)
    loaded_class: type | None  # Cached after load()

    @property
    def is_builtin(self) -> bool:
        return self.package == "aiperf"

    def load(self) -> type:
        """Lazy load and cache the class."""
```

### PluginType Enum

`PluginType` is dynamically generated from `categories.yaml`:

```python
from aiperf.plugin.enums import PluginType

# These are equivalent (ExtensibleStrEnum)
PluginType.ENDPOINT == "endpoint"  # True
plugins.get_class(PluginType.ENDPOINT, "chat")
plugins.get_class("endpoint", "chat")  # Both work
```

### Conflict Resolution

When multiple plugins register the same `(category, name)`:

```
1. Higher priority wins
   new.priority > existing.priority → new wins

2. Equal priority: external beats built-in
   new.is_external AND existing.is_builtin → new wins

3. Equal priority, same type: first wins (logs warning)
```

---

## Using Plugins

### Import

```python
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, EndpointType, TransportType
```

### Get a Single Implementation (Factory Pattern)

The most common pattern - select ONE implementation by name:

```python
# Type-safe (IDE knows return type is type[EndpointProtocol])
EndpointClass = plugins.get_class(PluginType.ENDPOINT, "chat")
endpoint = EndpointClass(model_endpoint=model_endpoint)

# String form also works
TransportClass = plugins.get_class("transport", "aiohttp")
transport = TransportClass(model_endpoint=model_endpoint)
```

**Real example from `inference_client.py`:**

```python
# Create endpoint and transport from config
EndpointClass = plugins.get_class("endpoint", self.model_endpoint.endpoint.type)
self.endpoint = EndpointClass(model_endpoint=self.model_endpoint)

TransportClass = plugins.get_class("transport", self.model_endpoint.transport)
self.transport = TransportClass(model_endpoint=self.model_endpoint)
```

### Iterate All Implementations (Hook-like Pattern)

Run ALL registered implementations in a category:

```python
# Iterate with lazy loading
for entry in plugins.list_types("data_exporter"):
    try:
        ExporterClass = entry.load()  # Lazy load
        exporter = ExporterClass(exporter_config=config)
        await exporter.export()
    except DataExporterDisabled:
        continue  # Skip disabled exporters
```

**Real example from `exporter_manager.py`:**

```python
async def export_data(self) -> None:
    for exporter_type in plugins.list_types("data_exporter"):
        try:
            ExporterClass = exporter_type.load()
            exporter: DataExporterProtocol = ExporterClass(
                exporter_config=self._exporter_config,
            )
        except DataExporterDisabled:
            self.debug(f"Data exporter {exporter_type} is disabled")
            continue

        task = asyncio.create_task(exporter.export())
        self._tasks.add(task)
```

### Auto-Detect Type from URL

For transports, detect the appropriate type from URL scheme:

```python
# Detect transport type from URL
if not model_endpoint.transport:
    model_endpoint.transport = plugins.detect_type_from_url(
        "transport",
        model_endpoint.endpoint.base_url,  # e.g., "http://localhost:8000"
    )
# Returns "aiohttp" for http/https URLs
```

### Use Dynamic Enums

Type enums are generated from registered plugins:

```python
from aiperf.plugin.enums import EndpointType, TimingMode

# Use in Pydantic configs for validation
class EndpointConfig(BaseConfig):
    type: EndpointType = Field(description="Endpoint type")

# EndpointType.CHAT, EndpointType.COMPLETIONS, etc. are generated
# from plugins registered in the "endpoint" category
```

### Access Metadata

Get plugin and category metadata:

```python
# Package metadata
meta = plugins.get_package_metadata("aiperf")
print(meta.version, meta.author, meta.description)

# Category metadata
cat_meta = plugins.get_category_metadata("endpoint")
print(cat_meta["protocol"])  # EndpointProtocol
print(cat_meta["enum"])      # EndpointType

# Type metadata
entry = plugins.get_type("endpoint", "chat")
print(entry.metadata)  # {"endpoint_path": "/v1/chat/completions", ...}
```

### Reverse Lookup (Class → Name)

Find the registered name for a class:

```python
from aiperf.endpoints.openai_chat import ChatEndpoint

name = plugins.find_registered_name("endpoint", ChatEndpoint)
# Returns "chat"
```

---

## Creating Plugins

### Step 1: Create Package Structure

```
aiperf-my-plugin/
├── pyproject.toml
├── src/
│   └── aiperf_my_plugin/
│       ├── __init__.py
│       ├── plugins.yaml      # Plugin manifest
│       └── endpoints.py      # Your implementations
```

### Step 2: Define Entry Point

In `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "aiperf-my-plugin"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = ["aiperf>=2.0.0"]

[project.entry-points."aiperf.plugins"]
my_plugin = "aiperf_my_plugin:plugins.yaml"
```

### Step 3: Create Plugin Manifest

In `src/aiperf_my_plugin/plugins.yaml`:

```yaml
schema_version: "1.0"

plugin:
  name: aiperf-my-plugin
  version: 1.0.0
  description: My custom AIPerf plugin
  author: Your Name
  license: Apache-2.0

# Register implementations by category
endpoint:
  my_custom:
    class: aiperf_my_plugin.endpoints:MyCustomEndpoint
    description: My custom endpoint implementation
    priority: 10  # Higher than built-in (0) to override
    metadata:
      endpoint_path: /v1/custom
      supports_streaming: true
```

### Step 4: Implement the Protocol

In `src/aiperf_my_plugin/endpoints.py`:

```python
from aiperf.common.models import ModelEndpointInfo, RequestInfo
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.endpoints.base_endpoint import BaseEndpoint

class MyCustomEndpoint(BaseEndpoint):
    """My custom endpoint implementation."""

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None:
        super().__init__(model_endpoint, **kwargs)

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        return EndpointMetadata(
            endpoint_path="/v1/custom",
            supports_streaming=True,
        )

    def format_payload(self, request_info: RequestInfo) -> dict:
        """Format the request payload."""
        return {
            "prompt": request_info.prompt,
            "max_tokens": request_info.max_tokens,
        }

    def parse_response(self, response: dict) -> dict:
        """Parse the API response."""
        return {
            "text": response.get("output", ""),
            "tokens": response.get("usage", {}).get("total_tokens", 0),
        }
```

### Step 5: Install and Verify

```bash
# Install in development mode
pip install -e .

# Verify registration
aiperf plugins list-implementations endpoint

# Should show your plugin with priority 10
```

---

## Plugin Categories

AIPerf defines 20+ plugin categories. Here are the most commonly extended:

### Endpoints

Implement LLM API request/response formatting.

```yaml
endpoint:
  my_endpoint:
    class: my_plugin:MyEndpoint
    description: Custom LLM endpoint
    metadata:
      endpoint_path: /v1/generate
      supports_streaming: true
```

**Protocol:** `EndpointProtocol`
**Base class:** `BaseEndpoint`
**Key methods:** `format_payload()`, `parse_response()`, `metadata()`

### Transports

Implement HTTP client behavior.

```yaml
transport:
  my_transport:
    class: my_plugin:MyTransport
    description: Custom HTTP transport
    metadata:
      url_schemes: ["http", "https"]
```

**Protocol:** `TransportProtocol`
**Base class:** `BaseTransport`
**Key methods:** `send_request()`

### Data Exporters

Export benchmark results to files or external systems.

```yaml
data_exporter:
  my_exporter:
    class: my_plugin:MyExporter
    description: Export to custom format
```

**Protocol:** `DataExporterProtocol`
**Key methods:** `export()`, `get_export_info()`

### Timing Strategies

Control request scheduling.

```yaml
timing_strategy:
  my_strategy:
    class: my_plugin:MyTimingStrategy
    description: Custom timing strategy
```

**Protocol:** `TimingStrategyProtocol`
**Key methods:** `get_next_send_time()`, `process_completion()`

### Results Processors

Process and aggregate benchmark results.

```yaml
results_processor:
  my_processor:
    class: my_plugin:MyProcessor
    description: Custom results processing
```

**Protocol:** `ResultsProcessorProtocol`
**Key methods:** `process_results()`

### Full Category List

| Category | Description | Protocol |
|----------|-------------|----------|
| `endpoint` | LLM API formatting | `EndpointProtocol` |
| `transport` | HTTP clients | `TransportProtocol` |
| `timing_strategy` | Request scheduling | `TimingStrategyProtocol` |
| `arrival_pattern` | Request arrival patterns | `ArrivalPatternProtocol` |
| `ramp` | Load ramping | `RampProtocol` |
| `dataset_backing_store` | Dataset storage | `DatasetBackingStoreProtocol` |
| `dataset_sampler` | Prompt sampling | `DatasetSamplerProtocol` |
| `dataset_composer` | Dataset composition | `DatasetComposerProtocol` |
| `custom_dataset_loader` | Custom dataset formats | `CustomDatasetLoaderProtocol` |
| `record_processor` | Record processing | `RecordProcessorProtocol` |
| `results_processor` | Results aggregation | `ResultsProcessorProtocol` |
| `data_exporter` | File/system export | `DataExporterProtocol` |
| `console_exporter` | Console output | `ConsoleExporterProtocol` |
| `service` | AIPerf services | `ServiceProtocol` |
| `service_manager` | Service orchestration | `ServiceManagerProtocol` |
| `communication` | ZMQ backends | `CommunicationProtocol` |
| `ui` | User interfaces | `AIPerfUIProtocol` |
| `plot` | Plotting | `PlotProtocol` |

---

## Advanced Topics

### Programmatic Registration

Register plugins at runtime without YAML:

```python
from aiperf.plugin import plugins

plugins.register(
    category="endpoint",
    name="dynamic_endpoint",
    cls=MyDynamicEndpoint,
    priority=10,
    is_builtin=False,
)
```

### Loading Additional Registries

Load plugin manifests from arbitrary paths:

```python
from pathlib import Path
from aiperf.plugin import plugins

plugins.load_registry(Path("/path/to/custom/plugins.yaml"))
```

### Creating Custom Enums

Generate enums from registered types:

```python
# Create enum from all types in a category
MyEndpointType = plugins.create_enum(PluginType.ENDPOINT, "MyEndpointType")
# MyEndpointType.CHAT, MyEndpointType.COMPLETIONS, etc.
```

### Accessing Class by Full Path

Access any registered implementation by its full class path (even if not the "winner"):

```python
# By name - returns highest priority implementation
cls = plugins.get_class("endpoint", "chat")

# By class path - access specific implementation directly
cls = plugins.get_class("endpoint", "aiperf.endpoints.openai_chat:ChatEndpoint")
other_cls = plugins.get_class("endpoint", "my_plugin.endpoints:MyChat")
```

### Validation Without Import

Validate plugin manifests without importing code (AST-based):

```python
# Validate all registered plugins
results = plugins.validate_all(check_class=True)
for category, types in results.items():
    for name, (valid, error) in types.items():
        if not valid:
            print(f"{category}.{name}: {error}")
```

### Protocol Enforcement

Categories define their expected protocol in `categories.yaml`:

```yaml
endpoint:
  protocol: aiperf.common.protocols:EndpointProtocol
  metadata_class: aiperf.common.models.metadata:EndpointMetadata
  enum: EndpointType
  description: HTTP endpoint handlers for LLM APIs
```

The plugin system validates that implementations conform to the protocol (via `isinstance` with `runtime_checkable`).

---

## Testing

### Reset Registry Between Tests

```python
import pytest
from aiperf.plugin import plugins

@pytest.fixture(autouse=True)
def reset_plugin_registry():
    """Reset plugin registry before each test."""
    yield
    plugins.reset()  # Clear all registrations
```

### Test Plugin Registration

```python
def test_my_plugin_registers():
    """Verify plugin is discoverable."""
    # Trigger discovery
    plugins.discover_plugins()

    # Check registration
    entry = plugins.get_type("endpoint", "my_custom")
    assert entry is not None
    assert entry.package == "aiperf-my-plugin"
    assert entry.priority == 10
```

### Test Plugin Implementation

```python
@pytest.mark.asyncio
async def test_my_endpoint_formats_payload():
    """Test endpoint payload formatting."""
    from aiperf_my_plugin.endpoints import MyCustomEndpoint

    endpoint = MyCustomEndpoint(model_endpoint=mock_model_endpoint)

    payload = endpoint.format_payload(request_info)

    assert "prompt" in payload
    assert payload["max_tokens"] == request_info.max_tokens
```

### Mock Plugin Dependencies

```python
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_model_endpoint():
    """Create mock model endpoint."""
    endpoint = Mock()
    endpoint.endpoint.type = "chat"
    endpoint.endpoint.base_url = "http://localhost:8000"
    endpoint.transport = "aiohttp"
    return endpoint
```

---

## Troubleshooting

### Plugin Not Found

**Symptom:** `TypeNotFoundError: Type 'my_endpoint' not found in category 'endpoint'`

**Causes:**
1. Entry point not configured in `pyproject.toml`
2. Plugin not installed
3. Typo in category or name

**Debug:**
```bash
# List all registered plugins
aiperf plugins list-plugins

# List types in category
aiperf plugins list-implementations endpoint

# Validate plugin manifest
aiperf plugins validate /path/to/plugins.yaml
```

### Wrong Implementation Selected

**Symptom:** Built-in used instead of your plugin

**Cause:** Priority too low

**Fix:** Increase priority in `plugins.yaml`:
```yaml
endpoint:
  chat:
    class: my_plugin:MyChat
    priority: 10  # Higher than built-in (0)
```

**Debug:**
```python
entry = plugins.get_type("endpoint", "chat")
print(f"Selected: {entry.class_path} (priority={entry.priority})")
```

### Import Error on Load

**Symptom:** `ImportError` when calling `plugins.get_class()`

**Causes:**
1. Missing dependency
2. Syntax error in plugin code
3. Wrong class path in `plugins.yaml`

**Debug:**
```python
# Test import directly
from aiperf_my_plugin.endpoints import MyEndpoint

# Validate without importing
results = plugins.validate_all(check_class=True)
```

### Entry Point Not Discovered

**Symptom:** Plugin not in `aiperf plugins list-plugins`

**Causes:**
1. Wrong entry point group (must be `aiperf.plugins`)
2. Package not installed
3. Entry point value format wrong

**Fix:** Verify `pyproject.toml`:
```toml
[project.entry-points."aiperf.plugins"]
my_plugin = "aiperf_my_plugin:plugins.yaml"
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           Must be "module.path:filename"
```

**Debug:**
```python
from importlib.metadata import entry_points
eps = entry_points(group="aiperf.plugins")
for ep in eps:
    print(f"{ep.name}: {ep.value}")
```

---

## API Reference

### Module: `aiperf.plugin.plugins`

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_class` | `(category, name_or_class_path) -> type` | Get class by name (21 typed overloads) |
| `get_type` | `(category, name) -> TypeEntry` | Get TypeEntry metadata |
| `list_types` | `(category) -> list[TypeEntry]` | List all types in category |
| `list_categories` | `(include_internal=True) -> list[str]` | List all categories |
| `list_packages` | `(builtin_only=False) -> list[str]` | List loaded packages |
| `get_package_metadata` | `(name) -> PackageInfo` | Get package info |
| `get_category_metadata` | `(category) -> CategoryMetadata` | Get category info |
| `find_registered_name` | `(category, cls) -> str \| None` | Reverse lookup |
| `detect_type_from_url` | `(category, url) -> str \| None` | Match URL scheme |
| `create_enum` | `(category, name) -> type` | Generate StrEnum |
| `register` | `(category, name, cls, *, priority, is_builtin)` | Runtime registration |
| `load_registry` | `(path: Path)` | Load additional YAML |
| `validate_all` | `(check_class=False) -> dict` | Validate all plugins |
| `reset` | `()` | Reset for testing |

### Class: `TypeEntry`

| Property/Method | Type | Description |
|-----------------|------|-------------|
| `category` | `str` | Category identifier |
| `name` | `str` | Type name |
| `package` | `str` | Providing package |
| `class_path` | `str` | `module:ClassName` |
| `priority` | `int` | Conflict resolution priority |
| `description` | `str` | Human-readable description |
| `metadata` | `dict` | Category-specific metadata |
| `is_builtin` | `bool` | True if `package == "aiperf"` |
| `load()` | `type` | Lazy load and cache class |
| `validate()` | `(bool, str \| None)` | Validate without import |

---

## See Also

- [CLI Quick Start](CLI_PLUGINS_QUICKSTART.md) - Plugin CLI commands
- [Best Practices](PLUGIN_BEST_PRACTICES.md) - Detailed development patterns
- [categories.yaml](../src/aiperf/plugin/categories.yaml) - Category definitions
- [plugins.yaml](../src/aiperf/plugin/plugins.yaml) - Built-in plugin manifest
