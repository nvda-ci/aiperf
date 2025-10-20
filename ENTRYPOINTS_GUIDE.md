<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Entry Points with Seamless Plugin Override

## The Magic: Same Name = Seamless Override

When a plugin registers with the **same name** as a builtin, it **automatically overrides** it. Users ask for "chat" and get the plugin version!

---

## Your Project's pyproject.toml

```toml
[project]
name = "aiperf"
version = "0.1.0"

# Builtin implementations
[project.entry-points."aiperf.endpoints"]
chat = "aiperf.endpoints.openai_chat:ChatEndpoint"
embeddings = "aiperf.endpoints.openai_embeddings:EmbeddingsEndpoint"
completions = "aiperf.endpoints.openai_completions:CompletionsEndpoint"
rankings = "aiperf.endpoints.nim_rankings:RankingsEndpoint"

[project.entry-points."aiperf.transports"]
http = "aiperf.transports.http_transport:HTTPTransport"
grpc = "aiperf.transports.grpc_transport:GRPCTransport"
http2 = "aiperf.transports.http2_transport:HTTP2Transport"
in_engine = "aiperf.transports.in_engine_transport:InEngineTransport"
```

That's it for AIPerf core. No need for decorators anymore!

---

## External Plugin Package

A third-party plugin just adds their entry points:

```toml
# In nvidia_optimized_plugin/pyproject.toml
[project]
name = "nvidia-optimized-endpoints"
version = "1.0.0"

# Plugin implementations (SAME NAMES override builtins!)
[project.entry-points."aiperf.endpoints.plugins"]
chat = "nvidia_endpoints.optimized:OptimizedChatEndpoint"
embeddings = "nvidia_endpoints.optimized:OptimizedEmbeddingsEndpoint"
completions = "nvidia_endpoints.optimized:OptimizedCompletionsEndpoint"

[project.entry-points."aiperf.transports.plugins"]
grpc = "nvidia_transport.grpc:OptimizedGRPCTransport"
```

When user `pip install nvidia-optimized-endpoints`, those overrides are **automatically active**!

---

## In Your Application Code

```python
# At startup, discover all registered implementations
from aiperf.common.factories import EndpointFactory, TransportFactory

# Load all builtins and plugins from entry points
EndpointFactory.discover_from_entrypoints("aiperf.endpoints")
TransportFactory.discover_from_entrypoints("aiperf.transports")

# Now use them normally - plugins automatically preferred
endpoint = EndpointFactory.create_instance("chat", model_endpoint=info)
# ✓ Gets OptimizedChatEndpoint if nvidia plugin is installed
# ✓ Falls back to ChatEndpoint if not installed
```

---

## User Experience

### User 1: AIPerf Only
```bash
$ pip install aiperf
```
→ Gets builtin endpoints (ChatEndpoint, etc.)

### User 2: AIPerf + NVIDIA Plugin
```bash
$ pip install aiperf nvidia-optimized-endpoints
```
→ Builtins + Plugins installed
→ **Plugins automatically override** builtins
→ User asks for "chat" → gets OptimizedChatEndpoint

### User 3: Custom Override at Runtime
```python
# User wants to test with builtin
EndpointFactory.use_implementation("chat", ChatEndpoint)

# ✓ Now uses builtin, even if plugin is installed
```

---

## Real Example: Step by Step

### Step 1: Register Builtin
```toml
# aiperf/pyproject.toml
[project.entry-points."aiperf.endpoints"]
chat = "aiperf.endpoints.openai_chat:ChatEndpoint"
```

### Step 2: Plugin Registers Same Name
```toml
# nvidia_plugin/pyproject.toml
[project.entry-points."aiperf.endpoints.plugins"]
chat = "nvidia_plugin.endpoints:OptimizedChatEndpoint"
```

### Step 3: Auto-Discovery
```python
EndpointFactory.discover_from_entrypoints("aiperf.endpoints")
# Loads both:
# - ChatEndpoint (from aiperf)
# - OptimizedChatEndpoint (from nvidia_plugin, is_plugin=True)
```

### Step 4: Usage
```python
# Before plugin installed:
ep = EndpointFactory.create_instance("chat", ...)
# ✓ Gets ChatEndpoint

# After `pip install nvidia_plugin`:
ep = EndpointFactory.create_instance("chat", ...)
# ✓ Gets OptimizedChatEndpoint (seamless override!)
```

---

## Advantages of This Approach

✅ **Zero code duplication** - One name, not "nvidia_chat" vs "chat"
✅ **Seamless for users** - Just install, it works
✅ **Modern** - Uses PEP 659 entry points standard
✅ **No runtime registration** - All config in pyproject.toml
✅ **Multi-plugin friendly** - Multiple plugins can provide same name
✅ **Works with pip** - Standard Python packaging
✅ **Simple priority** - Plugins automatically win (is_plugin=True)

---

## Listing Available Options

Users can see what's installed:

```python
options = EndpointFactory.list_implementations("chat")
for opt in options:
    tag = "[PLUGIN]" if opt["is_plugin"] else ""
    active = "← ACTIVE" if opt["is_selected"] else ""
    print(f"{opt['name']} {tag} {active}")

# Output:
# OptimizedChatEndpoint [PLUGIN] ← ACTIVE
# ChatEndpoint
```

---

## Migration Path

### Before (Decorators)
```python
@EndpointFactory.register(EndpointType.CHAT)
class ChatEndpoint(BaseEndpoint):
    pass
```

### After (Entry Points)
```toml
# No decorators needed!
[project.entry-points."aiperf.endpoints"]
chat = "aiperf.endpoints.openai_chat:ChatEndpoint"
```

**Benefits:**
- Configuration-driven, not code-driven
- Plugins can override without code changes
- Standard Python packaging approach

---

## FAQ

**Q: What if multiple plugins register the same name?**
A: First one wins. Users can override with `use_implementation()`.

**Q: Do I have to remove decorators?**
A: No! Both work. Decorators are simpler for single packages. Entry points are better for plugins.

**Q: How do I test locally without installing?**
A: Develop mode: `pip install -e .` makes entry points available immediately.

**Q: What if user wants old behavior back?**
A: `EndpointFactory.use_implementation("chat", ChatEndpoint)` - one call.

---

## That's It!

Entry points handle registration. Your simple priority system (`is_plugin=True`) handles selection. Users get **seamless override** with zero friction.

🎉
