<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Hybrid Registration: Best of Both Worlds

## The Problem Solved

- ❌ **All entry points**: Slow for development (reinstall needed)
- ❌ **All decorators**: Plugins don't integrate seamlessly
- ✅ **Hybrid**: Decorators for dev speed, entry points for user-friendly plugins

---

## The Approach

| Type | Registration | Why |
|------|--------------|-----|
| **Builtins** | `@Factory.register()` decorator | Fast dev iteration, instant changes |
| **Plugins** | Entry points in pyproject.toml | Seamless installation, automatic override |

---

## Your Project: Builtins as Decorators

Fast development, zero reinstalls needed:

```python
# aiperf/endpoints/openai_chat.py
from aiperf.common.factories import EndpointFactory

@EndpointFactory.register("chat")
class ChatEndpoint(BaseEndpoint):
    """Builtin endpoint - registered instantly on import"""
    pass

# aiperf/endpoints/openai_embeddings.py
@EndpointFactory.register("embeddings")
class EmbeddingsEndpoint(BaseEndpoint):
    """Builtin endpoint - registered instantly on import"""
    pass

# aiperf/endpoints/openai_completions.py
@EndpointFactory.register("completions")
class CompletionsEndpoint(BaseEndpoint):
    """Builtin endpoint - registered instantly on import"""
    pass
```

**No entry points needed for these!** Just decorators.

---

## Plugin Packages: Entry Points Only

External plugins register via entry points:

```toml
# nvidia_optimized_plugin/pyproject.toml
[project]
name = "nvidia-optimized-endpoints"

[project.entry-points."aiperf.endpoints.plugins"]
chat = "nvidia_endpoints.optimized:OptimizedChatEndpoint"
embeddings = "nvidia_endpoints.optimized:OptimizedEmbeddingsEndpoint"
completions = "nvidia_endpoints.optimized:OptimizedCompletionsEndpoint"
```

**NO code changes** when installing plugin. Entry point system handles it.

---

## At Application Startup

```python
# Your app initialization
from aiperf.common.factories import EndpointFactory, TransportFactory

# Step 1: Import modules (this triggers @register decorators)
import aiperf.endpoints  # ← Registers: chat, embeddings, completions (builtins)
import aiperf.transports  # ← Registers: http, grpc, http2, in_engine (builtins)

# Step 2: Load plugins from entry points
EndpointFactory.discover_plugins("aiperf.endpoints")    # ← Loads plugins only
TransportFactory.discover_plugins("aiperf.transports")  # ← Loads plugins only

# Now everything is available:
# - Builtins (from decorators)
# - Plugins (from entry points)
# - Plugins automatically override builtins with same name!

print(EndpointFactory.list_implementations("chat"))
# [
#   {'name': 'OptimizedChatEndpoint', 'is_plugin': True, 'is_selected': True},
#   {'name': 'ChatEndpoint', 'is_plugin': False, 'is_selected': False},
# ]
```

---

## Development Workflow

### Change a Builtin (INSTANT)

```python
# Edit aiperf/endpoints/openai_chat.py
@EndpointFactory.register("chat")
class ChatEndpoint(BaseEndpoint):
    async def format_payload(self, request_info):
        # Changed implementation
        pass
```

**Result:** Changes visible immediately on next run. No reinstall!

### Add a Plugin (User-friendly)

```toml
# In your plugin's pyproject.toml
[project.entry-points."aiperf.endpoints.plugins"]
chat = "my_plugin.endpoints:MyOptimizedChat"
```

**User experience:**
```bash
$ pip install my_plugin
# ✓ Automatically available
# ✓ Overrides builtin "chat" seamlessly
# ✓ No code changes needed
```

---

## Real Example: Complete Flow

### Step 1: Builtin Implementation
```python
# aiperf/endpoints/openai_chat.py
from aiperf.common.factories import EndpointFactory

@EndpointFactory.register("chat")
class ChatEndpoint(BaseEndpoint):
    def __init__(self, model_endpoint, **kwargs):
        self.model_endpoint = model_endpoint

    async def format_payload(self, request_info):
        return {"messages": request_info.turns[0]["content"]}
```

### Step 2: User Installs Plugin
```bash
$ pip install nvidia-optimized-endpoints
```

Plugin's pyproject.toml:
```toml
[project.entry-points."aiperf.endpoints.plugins"]
chat = "nvidia_endpoints:OptimizedChatEndpoint"
```

### Step 3: App Startup
```python
import aiperf.endpoints
EndpointFactory.discover_plugins("aiperf.endpoints")
```

### Step 4: Usage (Seamless!)
```python
# Gets OptimizedChatEndpoint by default (plugin, is_plugin=True)
endpoint = EndpointFactory.create_instance("chat", model_endpoint=info)

# User can still override if needed
from aiperf.endpoints import ChatEndpoint
EndpointFactory.use_implementation("chat", ChatEndpoint)
```

---

## API Summary

### For Builtin Developers (Internal)

```python
# In your implementation file
@EndpointFactory.register("chat")  # ← One decorator, instant registration
class ChatEndpoint(BaseEndpoint):
    pass

# Edit, save, run. No reinstall needed!
```

### For Plugin Developers (External)

```toml
# In plugin's pyproject.toml
[project.entry-points."aiperf.endpoints.plugins"]
chat = "my_plugin.endpoints:MyChat"  # ← Entry point, user installs and it works
```

### For Users

```python
# See what's available
EndpointFactory.list_implementations("chat")

# Switch if needed
EndpointFactory.use_implementation("chat", ChatEndpoint)

# Create instance
endpoint = EndpointFactory.create_instance("chat", ...)
```

---

## Advantages of Hybrid Approach

✅ **Fast development** - Decorators, no reinstall
✅ **User-friendly plugins** - Entry points, automatic discovery
✅ **Seamless override** - Same name, plugins win automatically
✅ **Zero friction** - Users just pip install
✅ **Simple for dev team** - Use decorators for internal work
✅ **Standard for plugins** - Use entry points, follows Python conventions
✅ **Both work together** - No conflicts, just works

---

## Migration Path

### If You Had Everything as Entry Points

```toml
# OLD: All implementations via entry points (slow for dev)
[project.entry-points."aiperf.endpoints"]
chat = "aiperf.endpoints.openai_chat:ChatEndpoint"
```

### Change To (Faster!)

```python
# NEW: Builtins as decorators
@EndpointFactory.register("chat")
class ChatEndpoint(BaseEndpoint):
    pass
```

**Then keep entry points for PLUGINS ONLY:**

```toml
# Plugin developers use this
[project.entry-points."aiperf.endpoints.plugins"]
chat = "plugin.endpoints:OptimizedChatEndpoint"
```

---

## FAQ

**Q: Do I have to remove existing decorators?**
A: No! Keep them. They work perfectly for builtins.

**Q: What about internal plugins?**
A: Still use decorators. Entry points are for external pip packages.

**Q: When should I use entry points?**
A: When the plugin is a separate package users install via pip.

**Q: Can I mix both in same package?**
A: Yes! Decorators for public API, entry points for plugins.

**Q: How do plugins override without names conflicting?**
A: Use SAME NAME in different group. `aiperf.endpoints` vs `aiperf.endpoints.plugins`

---

## The Sweet Spot

This hybrid approach is:
- **Fast for you** (decorators, instant changes)
- **Simple for users** (pip install, it just works)
- **Professional** (standard Python packaging)
- **Flexible** (users can still override)

Perfect!
