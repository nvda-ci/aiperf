<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Centralized Registry: Single Source of Truth

## The Problem with Scattered Decorators

```python
# Bad: Imports scattered across many files
# aiperf/endpoints/openai_chat.py
@EndpointFactory.register("chat")  ← Requires importing ChatEndpoint
class ChatEndpoint(BaseEndpoint):
    pass

# aiperf/endpoints/openai_embeddings.py
@EndpointFactory.register("embeddings")  ← Requires importing EmbeddingsEndpoint
class EmbeddingsEndpoint(BaseEndpoint):
    pass

# ...50 more files like this...
```

**Issues:**
- Must import all modules just to trigger registration
- No single place to see what's available
- Hard to manage, easy to forget
- Discovery nightmare

---

## The Solution: Centralized Registry

```python
# aiperf/endpoints/registry.py (SINGLE SOURCE OF TRUTH)
from aiperf.common.factories import EndpointFactory

# Import implementations (no decorators needed on them)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.endpoints.nim_rankings import RankingsEndpoint

# All registrations in ONE place
EndpointFactory.register("chat")(ChatEndpoint)
EndpointFactory.register("embeddings")(EmbeddingsEndpoint)
EndpointFactory.register("completions")(CompletionsEndpoint)
EndpointFactory.register("rankings")(RankingsEndpoint)
```

**Implementations are now clean (no decorators):**

```python
# aiperf/endpoints/openai_chat.py - CLEAN
class ChatEndpoint(BaseEndpoint):
    """Just the implementation, no decorators"""
    pass
```

---

## At Startup (One Line)

```python
import aiperf.endpoints  # ← Imports __init__.py which imports registry.py
                         # ← Triggers all registrations
EndpointFactory.discover_plugins("aiperf.endpoints")  # ← Load plugins
```

**Done!** All builtins registered + plugins loaded.

---

## Benefits

✅ **Centralized** - One file per component type
✅ **Clean implementations** - No decorators scattered
✅ **Single source of truth** - Look in one place
✅ **Easy discovery** - See all available implementations
✅ **Fast dev** - Edit registry, no reinstall
✅ **No import nightmare** - One import in __init__.py
✅ **Scalable** - Add new implementations easily

---

## Project Structure

```
aiperf/endpoints/
├── __init__.py           ← Imports registry.py
├── registry.py           ← All registrations HERE
├── openai_chat.py        ← Clean: just ChatEndpoint
├── openai_embeddings.py  ← Clean: just EmbeddingsEndpoint
├── openai_completions.py ← Clean: just CompletionsEndpoint
└── nim_rankings.py       ← Clean: just RankingsEndpoint
```

---

## File by File

### 1. Implementation Files (Clean, No Decorators)

```python
# aiperf/endpoints/openai_chat.py
from aiperf.endpoints.base_endpoint import BaseEndpoint

class ChatEndpoint(BaseEndpoint):
    """OpenAI Chat endpoint"""

    @classmethod
    def metadata(cls):
        return EndpointMetadata(...)

    async def format_payload(self, request_info):
        ...

    def parse_response(self, response):
        ...
```

### 2. Registry File (All Registrations)

```python
# aiperf/endpoints/registry.py
from aiperf.common.factories import EndpointFactory

# Import all implementations
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.endpoints.nim_rankings import RankingsEndpoint

# Register all builtins in ONE place
EndpointFactory.register("chat")(ChatEndpoint)
EndpointFactory.register("embeddings")(EmbeddingsEndpoint)
EndpointFactory.register("completions")(CompletionsEndpoint)
EndpointFactory.register("rankings")(RankingsEndpoint)

__all__ = [
    "ChatEndpoint",
    "EmbeddingsEndpoint",
    "CompletionsEndpoint",
    "RankingsEndpoint",
]
```

### 3. Package Init (Import Registry)

```python
# aiperf/endpoints/__init__.py
# Import registry to trigger all registrations
from aiperf.endpoints import registry  # ← This registers everything

# Re-export for convenience
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.endpoints.nim_rankings import RankingsEndpoint

__all__ = [
    "ChatEndpoint",
    "EmbeddingsEndpoint",
    "CompletionsEndpoint",
    "RankingsEndpoint",
]
```

### 4. Application Startup

```python
# my_app.py
from aiperf.common.factories import EndpointFactory, TransportFactory

# Step 1: Import packages (triggers registry.py which registers all)
import aiperf.endpoints
import aiperf.transports

# Step 2: Load plugins from entry points
EndpointFactory.discover_plugins("aiperf.endpoints")
TransportFactory.discover_plugins("aiperf.transports")

# Done! Everything is now available
endpoint = EndpointFactory.create_instance("chat", model_endpoint=info)
```

---

## Real Example: Adding New Endpoint

### Step 1: Create Implementation
```python
# aiperf/endpoints/my_new_endpoint.py
class MyNewEndpoint(BaseEndpoint):
    """New endpoint implementation"""
    pass
```

### Step 2: Add to Registry
```python
# aiperf/endpoints/registry.py
from aiperf.endpoints.my_new_endpoint import MyNewEndpoint

# Add one line:
EndpointFactory.register("my_new")(MyNewEndpoint)
```

### Step 3: Done!
```python
# Next startup automatically loads it
endpoint = EndpointFactory.create_instance("my_new", ...)
```

---

## Do the Same for Other Component Types

Same pattern for transports:

```python
# aiperf/transports/registry.py
from aiperf.common.factories import TransportFactory
from aiperf.transports.http_transport import HTTPTransport
from aiperf.transports.grpc_transport import GRPCTransport
from aiperf.transports.http2_transport import HTTP2Transport
from aiperf.transports.in_engine_transport import InEngineTransport

TransportFactory.register("http")(HTTPTransport)
TransportFactory.register("grpc")(GRPCTransport)
TransportFactory.register("http2")(HTTP2Transport)
TransportFactory.register("in_engine")(InEngineTransport)
```

```python
# aiperf/transports/__init__.py
from aiperf.transports import registry  # ← Trigger registrations

from aiperf.transports.http_transport import HTTPTransport
from aiperf.transports.grpc_transport import GRPCTransport
from aiperf.transports.http2_transport import HTTP2Transport
from aiperf.transports.in_engine_transport import InEngineTransport
```

---

## Development Workflow

### Quick Iteration: No Reinstall Needed

```python
# Change implementation
# aiperf/endpoints/openai_chat.py
class ChatEndpoint(BaseEndpoint):
    async def format_payload(self, request_info):
        # Fixed bug!
        pass

# Run app
# ✓ Changes visible immediately
# ✓ No reinstall needed
```

### Add New Implementation: One Change

```python
# Add to registry.py:
from aiperf.endpoints.my_endpoint import MyEndpoint
EndpointFactory.register("my")(MyEndpoint)

# Run app
# ✓ Automatically available
```

---

## Advantages Over Scattered Decorators

| Aspect | Scattered | Centralized |
|--------|-----------|-------------|
| **Registry location** | 50 different files | 1 file |
| **Easy to find** | ❌ Search code | ✅ Look at one file |
| **Easy to add** | ❌ Create new file + decorator | ✅ One line in registry |
| **Easy to remove** | ❌ Find and delete | ✅ Delete one line |
| **Import problem** | ❌ Must import all | ✅ Import one registry |
| **Discovery** | ❌ Nightmare | ✅ Clear and simple |

---

## Integration with Plugins

### At Startup
```python
import aiperf.endpoints  # Loads registry.py → registers builtins

EndpointFactory.discover_plugins("aiperf.endpoints")  # Load plugins
```

### Plugins Still Use Entry Points
```toml
# my_plugin/pyproject.toml
[project.entry-points."aiperf.endpoints.plugins"]
chat = "my_plugin.endpoints:OptimizedChat"
```

### Result
- Builtins: Registered via centralized registry.py
- Plugins: Loaded from entry points
- Plugins override builtins with same name
- Everything works seamlessly!

---

## For Each Component Type

| Component | Registry File | Import In |
|-----------|---------------|-----------|
| Endpoints | `aiperf/endpoints/registry.py` | `aiperf/endpoints/__init__.py` |
| Transports | `aiperf/transports/registry.py` | `aiperf/transports/__init__.py` |
| UI | `aiperf/ui/registry.py` | `aiperf/ui/__init__.py` |
| Exporters | `aiperf/exporters/registry.py` | `aiperf/exporters/__init__.py` |
| etc. | ... | ... |

---

## The Perfect Balance

✅ **No scattered decorators** - All in one registry file
✅ **Fast dev iteration** - Change, save, run
✅ **Single source of truth** - Look in one place
✅ **Scalable** - Easy to add/remove
✅ **Clean implementations** - No decorator noise
✅ **Works with plugins** - Entry points still apply
✅ **Professional** - Organized, maintainable

This is the sweet spot between decorators and entry points!

