<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Quick Start - Factory System (2 Minutes)

## 🎯 The Core Idea

All implementations live together. Plugins win by default. Users pick if they want.

---

## 1️⃣ If You're Writing Built-in Code

```python
@EndpointFactory.register(EndpointType.CHAT)
class ChatEndpoint(BaseEndpoint):
    """Standard endpoint"""
    pass
```

**That's it.** Nothing changes from before.

---

## 2️⃣ If You're Writing a Plugin

```python
@EndpointFactory.register(EndpointType.CHAT, is_plugin=True)
class OptimizedChatEndpoint(BaseEndpoint):
    """Your plugin endpoint - automatically takes priority"""
    pass
```

**Boom.** Your plugin wins by default, but users can switch.

---

## 3️⃣ If You're a User/Ops Person

### See what's available:
```python
options = EndpointFactory.list_implementations(EndpointType.CHAT)
for opt in options:
    marker = " ← ACTIVE" if opt['is_selected'] else ""
    plugin_note = " [PLUGIN]" if opt['is_plugin'] else ""
    print(f"{opt['name']}{plugin_note}{marker}")
```

### Switch to one:
```python
EndpointFactory.use_implementation(EndpointType.CHAT, ChatEndpoint)
```

### Use it:
```python
endpoint = EndpointFactory.create_instance(EndpointType.CHAT, model_endpoint=info)
```

---

## 📊 What Happens Automatically

```
Registration order:  First: ChatEndpoint (builtin)
                     Second: OptimizedChatEndpoint (plugin, is_plugin=True)

Active by default:   OptimizedChatEndpoint ✓
                     (plugins always preferred)

Can the user switch? YES! EndpointFactory.use_implementation(...)
Can it be deleted?   NO! Both implementations stay alive
```

---

## 🚀 Real Example

```python
# Your plugin package (my_nvidia_plugin/endpoints.py)
@EndpointFactory.register(EndpointType.CHAT, is_plugin=True)
class NvidiaOptimizedChat(BaseEndpoint):
    def format_payload(self, request_info):
        # Optimized implementation
        pass

# User startup code
from aiperf.common.factories import EndpointFactory
from aiperf.common.enums import EndpointType

# ✓ Plugin is used by default
chat = EndpointFactory.create_instance(EndpointType.CHAT, ...)

# User wants to test with builtin? Easy:
from aiperf.endpoints import ChatEndpoint
EndpointFactory.use_implementation(EndpointType.CHAT, ChatEndpoint)

# ✓ Now builtin is used instead
chat = EndpointFactory.create_instance(EndpointType.CHAT, ...)
```

---

## ✅ Checklist

- [ ] Your built-in implementations register normally - ✓ works
- [ ] Your plugin adds `is_plugin=True` - ✓ gets priority
- [ ] Users can see options with `list_implementations()` - ✓ transparent
- [ ] Users can switch with `use_implementation()` - ✓ flexible
- [ ] No code deletions - ✓ safe
- [ ] Works with multiple implementations - ✓ scalable

---

## FAQ (30 seconds each)

**Q: Do I have to change my code?**
A: Nope! Existing code works as-is. Just add `is_plugin=True` if you want priority.

**Q: What if I have 3 implementations?**
A: All 3 live together. The plugin one is active. User picks which to use.

**Q: Can the user break anything?**
A: No. They can only switch between available implementations.

**Q: How does the ops team use this?**
A: Call one function at startup, or use an environment variable.

**Q: What about priority numbers?**
A: Gone! Just use `is_plugin=True` for the important one.

---

## That's All!

You now know the entire system. Go forth and build! 🚀
