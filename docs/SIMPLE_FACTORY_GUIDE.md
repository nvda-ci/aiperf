<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Simplified Factory System - User Guide

## The Problem We Solved

**Before**: When two implementations registered for the same type, the lower-priority one got deleted.
- Users couldn't easily switch implementations
- Plugins couldn't guarantee they'd be used
- No way to override after registration

**Now**: All implementations coexist. Just pick which one you want to use.

---

## For Built-in Implementations (No Changes!)

```python
@EndpointFactory.register(EndpointType.CHAT)
class ChatEndpoint(BaseEndpoint):
    """Your endpoint - automatically available"""
    pass
```

---

## For Plugin Developers (Simple!)

Just add `is_plugin=True` to tell AIPerf this is important:

```python
@EndpointFactory.register(EndpointType.CHAT, is_plugin=True)
class OptimizedChatEndpoint(BaseEndpoint):
    """Plugin endpoints naturally take priority"""
    pass
```

**That's it!** Your plugin will be used by default over built-ins.

---

## For Users - Three Easy Ways to Choose

### Option 1: List and Pick (Programmatic)

```python
# See what's available
implementations = EndpointFactory.list_implementations(EndpointType.CHAT)
for impl in implementations:
    status = "✓ ACTIVE" if impl["is_selected"] else ""
    plugin_mark = "[PLUGIN]" if impl["is_plugin"] else ""
    print(f"{impl['name']} {plugin_mark} {status}")

# Output might look like:
# OptimizedChatEndpoint [PLUGIN] ✓ ACTIVE
# ChatEndpoint

# Switch if you want
EndpointFactory.use_implementation(EndpointType.CHAT, ChatEndpoint)
```

### Option 2: Use at Startup (Code)

```python
from aiperf.common.factories import EndpointFactory
from aiperf.endpoints import ChatEndpoint

# Use the built-in version
EndpointFactory.use_implementation(EndpointType.CHAT, ChatEndpoint)

# Now it's used by default
endpoint = EndpointFactory.create_instance(EndpointType.CHAT, ...)
```

### Option 3: Environment Variable (Ops Friendly)

```bash
# Set which implementation to use
export AIPERF_CHAT_ENDPOINT=ChatEndpoint

# Your app detects this and calls:
# EndpointFactory.use_implementation(EndpointType.CHAT, ChatEndpoint)
```

---

## Real Example: Plugin Takes Priority by Default

```python
# Built-in (priority: lower, by default)
@EndpointFactory.register(EndpointType.CHAT)
class ChatEndpoint(BaseEndpoint):
    pass

# Plugin from nvidia_plugin package (priority: higher!)
@EndpointFactory.register(EndpointType.CHAT, is_plugin=True)
class OptimizedChatEndpoint(BaseEndpoint):
    pass

# Usage:
endpoint = EndpointFactory.create_instance(EndpointType.CHAT, ...)
# ✓ Gets OptimizedChatEndpoint (because it's marked as plugin)

# But user can override:
EndpointFactory.use_implementation(EndpointType.CHAT, ChatEndpoint)
endpoint = EndpointFactory.create_instance(EndpointType.CHAT, ...)
# ✓ Now gets ChatEndpoint (user's choice)
```

---

## API Reference (Simple!)

### For Developers

```python
@Factory.register(TYPE)
@Factory.register(TYPE, is_plugin=True)  # Mark as plugin
class MyImplementation:
    pass
```

### For Users

```python
# See what's available
Factory.list_implementations(TYPE)

# Pick one
Factory.use_implementation(TYPE, SomeClass)

# Create using selected implementation
Factory.create_instance(TYPE, **kwargs)

# Get the currently selected class
Factory.get_class_from_type(TYPE)
```

---

## That's All You Need to Know!

✅ **Plugins get priority automatically**
✅ **Users can easily switch implementations**
✅ **No implementations get deleted**
✅ **Works with environment variables**
✅ **Simple API, no complexity**
