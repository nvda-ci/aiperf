<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Lazy-Loaded Plugin Registry System - Implementation Summary

## Overview

A complete redesign of the AIPerf factory system for component discovery and instantiation. The system eliminates scattered decorators, enables lazy loading, supports flexible plugin discovery (entry points + environment variables), and provides identifier-based implementation selection.

## Key Design Decisions

### Single Centralized Registry

All builtin implementations registered in one file: `aiperf/aiperf_registry.py`

```python
# aiperf/aiperf_registry.py - Single source of truth
EndpointFactory.register_lazy("chat", "aiperf.endpoints.openai_chat", "ChatEndpoint", identifier="openai_chat")
EndpointFactory.register_lazy("embeddings", "aiperf.endpoints.openai_embeddings", "EmbeddingsEndpoint", identifier="openai_embeddings")
TransportFactory.register_lazy("http", "aiperf.transports.http_transport", "HTTPTransport", identifier="http")
# ... all registrations
```

### Lazy Loading

Classes loaded on first use, not at registration time. Registry stores only module paths and class names.

### Two-Level Lookup

- **Level 1**: `class_type` (e.g., "chat", "embeddings") - Categorizes implementations
- **Level 2**: `identifier` (e.g., "openai_chat", "nvidia_optimized") - Uniquely identifies implementation

### Flexible Plugin Discovery

Two mechanisms support different use cases:

**Entry Points** (Packaged plugins):
```toml
[project.entry-points."aiperf.plugins"]
my_plugin = "my_plugin.registry"
```

**Environment Variables** (Simple/local plugins):
```bash
export AIPERF_PLUGIN_MODULES="my_plugin.registry,experimental.registry"
```

### Priority Resolution

When multiple implementations exist:
1. Runtime override (user-selected)
2. Plugin implementations (is_plugin=True)
3. Builtin implementations (is_plugin=False)

## Core API

### Registration (Development)

```python
EndpointFactory.register_lazy(
    class_type="chat",
    module_path="aiperf.endpoints.openai_chat",
    class_name="ChatEndpoint",
    identifier="openai_chat",
    is_plugin=False
)
```

### Discovery (Startup)

```python
import aiperf  # Registers builtins via aiperf/aiperf_registry.py

from aiperf.common.factories import PluginDiscovery
PluginDiscovery.discover_all()  # Load plugins from entry points + env vars
```

### Selection (Usage)

```python
# List available
implementations = EndpointFactory.list_implementations("chat")

# Switch implementation
EndpointFactory.use_implementation_by_id("chat", "openai_chat")

# Create instance (lazy loads if needed)
endpoint = EndpointFactory.create_instance("chat", model_endpoint=info)
```

## Plugin Structure

### Packaged Plugin (with entry point)

```
my_plugin/
├── pyproject.toml
│   [project.entry-points."aiperf.plugins"]
│   my_plugin = "my_plugin.registry"
├── my_plugin/
│   ├── __init__.py
│   ├── registry.py
│   │   EndpointFactory.register_lazy(..., is_plugin=True)
│   └── endpoints.py
│       class OptimizedChatEndpoint:
│           pass
```

### Local Plugin (with environment variable)

```bash
export AIPERF_PLUGIN_MODULES="my_custom_plugin.registry"
```

```
my_custom_plugin/
├── registry.py
│   EndpointFactory.register_lazy(..., is_plugin=True)
└── endpoints.py
    class CustomChatEndpoint:
        pass
```

## Implementation Phases

### Phase 0: Factory Enhancement
- Implement `register_lazy()` method
- Implement `use_implementation_by_id()` method
- Implement `discover_all()` method
- Update `create_instance()` for lazy loading

### Phase 1: Builtin Registry Migration
- Create `aiperf/aiperf_registry.py` with all builtin registrations
- Update `aiperf/__init__.py` to import aiperf_registry
- Remove scattered decorators (optional, backward compatible)

### Phase 2: Plugin Distribution
- Document entry point setup for plugin developers
- Document environment variable usage for local plugins
- Provide example plugin templates

## Benefits

| Aspect | Benefit |
|--------|---------|
| **Development** | No scattered decorators, instant feedback, no reinstall |
| **Maintainability** | Single registry file, easy to audit, clear source of truth |
| **User Experience** | Identifier-based selection, no code changes needed to override |
| **Plugin Support** | Multiple discovery mechanisms, both packaged and local |
| **Performance** | Lazy loading reduces startup time |
| **Flexibility** | Full control over implementation selection |

## File Structure

```
aiperf/
├── __init__.py
├── aiperf_registry.py                    ← SINGLE REGISTRY
├── endpoints/
│   ├── openai_chat.py            ← No decorators, clean
│   ├── openai_embeddings.py
│   └── ...
├── transports/
│   ├── http_transport.py         ← No decorators, clean
│   ├── grpc_transport.py
│   └── ...
└── common/
    └── factories.py              ← Enhanced with lazy loading
```

## Backward Compatibility

- Existing `create_instance()` API unchanged
- Existing `list_implementations()` API enhanced with identifiers
- Existing `get_class_from_type()` API unchanged
- Old decorator-based registration continues to work (optional support)

## Related Documentation

- `DEP_001_LAZY_LOADED_PLUGIN_REGISTRY_SYSTEM.md` - Full technical proposal
- `ARCHITECTURE_DIAGRAM.md` - Visual diagrams of system layers
- `CENTRALIZED_REGISTRY_GUIDE.md` - How centralized registration works
- Implementation in `aiperf/common/factories.py`

## Next Steps

1. Implement `register_lazy()` and related methods in factories.py
2. Create `aiperf/registry.py` with builtin registrations
3. Update `aiperf/__init__.py` to import registry
4. Implement plugin discovery mechanism
5. Update documentation for plugin developers
6. Test with example plugins
