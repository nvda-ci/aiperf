<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Before vs After - Factory System Comparison

## Problem Scenario

Two implementations want to register for the same type. One is built-in, one is a plugin.

---

## ❌ OLD SYSTEM (Before)

```python
# Built-in endpoint registers first
@EndpointFactory.register(EndpointType.CHAT, override_priority=0)
class ChatEndpoint(BaseEndpoint):
    pass

# Plugin endpoint registers second (higher priority)
@EndpointFactory.register(EndpointType.CHAT, override_priority=10)
class OptimizedChatEndpoint(BaseEndpoint):
    pass

# Result: ChatEndpoint is DELETED, only OptimizedChatEndpoint survives
# Warning logged: "overrides already registered class ChatEndpoint"

# User wants to test with ChatEndpoint? TOO BAD!
# Only option: Rewrite code and rebuild
```

### Problems with Old System
- 😞 Lower priority implementations get **deleted** completely
- 🤯 Complex priority numbers with unclear meaning
- 🔒 Fixed at registration time - can't change later
- ❌ No way to list what's available
- 😤 Users can't switch implementations
- 🧮 Have to think about numeric priorities

---

## ✅ NEW SYSTEM (After)

```python
# Built-in endpoint registers first
@EndpointFactory.register(EndpointType.CHAT)
class ChatEndpoint(BaseEndpoint):
    pass

# Plugin endpoint registers second (just mark it as plugin!)
@EndpointFactory.register(EndpointType.CHAT, is_plugin=True)
class OptimizedChatEndpoint(BaseEndpoint):
    pass

# Result: BOTH EXIST! OptimizedChatEndpoint is used by default (plugin priority)
# ✓ No deletion, no warnings, clean and simple

# User wants to test with ChatEndpoint? One line:
EndpointFactory.use_implementation(EndpointType.CHAT, ChatEndpoint)
# ✓ Done! Switches instantly

# User can see options:
options = EndpointFactory.list_implementations(EndpointType.CHAT)
# [
#   {'name': 'OptimizedChatEndpoint', 'is_plugin': True, 'is_selected': True},
#   {'name': 'ChatEndpoint', 'is_plugin': False, 'is_selected': False},
# ]
```

### Benefits of New System
- 😊 All implementations **coexist**, nothing deleted
- 💡 Semantic flag `is_plugin=True` - clear intent
- ⚡ Switch implementations anytime
- 👀 See what's available with `list_implementations()`
- 🎯 Users have full control
- 🧠 No complex priority math needed

---

## Side-by-Side Comparison

### Scenario 1: Plugin Should Win

| Aspect | OLD | NEW |
|--------|-----|-----|
| **Setup** | Set priority=10 vs priority=0 | Just add `is_plugin=True` |
| **Result** | Built-in deleted ❌ | Both exist, plugin used ✅ |
| **Clear intent?** | 🤔 Magic numbers | 🎯 Self-documenting |
| **Can list options?** | ❌ No API | ✅ `list_implementations()` |

### Scenario 2: User Wants Built-in

| Aspect | OLD | NEW |
|--------|-----|-----|
| **Change** | 😱 Modify code, rebuild | ✅ One function call |
| **Time to implement** | 30 min + build | 10 seconds |
| **Can revert quickly?** | ❌ Need to rebuild again | ✅ One more call |
| **For ops team?** | ❌ Requires dev | ✅ Can use ENV var |

### Scenario 3: Three Implementations

| Aspect | OLD | NEW |
|--------|-----|-----|
| **What happens** | Only highest priority survives, 2 deleted | All 3 available ✅ |
| **Can inspect?** | ❌ Others are gone | ✅ See all 3, pick any |
| **Switch between** | 😢 Impossible | ✅ Easy |

---

## Code Examples

### Plugin Developer

```python
# OLD: Confusing priorities
@MyFactory.register(MyType.THING, override_priority=50)
class PluginImpl:
    pass

# NEW: Clear intent
@MyFactory.register(MyType.THING, is_plugin=True)
class PluginImpl:
    pass
```

### Test/Ops Team

```python
# OLD: Not possible without code changes
# "We need to test with the other implementation"
# → Talk to dev → dev modifies code → rebuilds → 30 min later...

# NEW: Done instantly
EndpointFactory.use_implementation(EndpointType.CHAT, ChatEndpoint)
# Or from ops:
export AIPERF_ENDPOINT_CHAT=ChatEndpoint
# Your app loads this and switches
```

### Introspection

```python
# OLD: No way to know what's available
# You'd have to read the code

# NEW: See everything
for impl in EndpointFactory.list_implementations(EndpointType.CHAT):
    print(f"{impl['name']} - {'PLUGIN' if impl['is_plugin'] else 'builtin'}")
    if impl['is_selected']:
        print("  ^ Currently active")
```

---

## Migration Guide

### If you have existing code...

```python
# OLD
@Factory.register(MyType, override_priority=5)
class MyImpl:
    pass

# NEW (Option 1: Keep as-is, just remove priority)
@Factory.register(MyType)
class MyImpl:
    pass

# NEW (Option 2: If it's a plugin)
@Factory.register(MyType, is_plugin=True)
class MyImpl:
    pass
```

**No breaking changes!** Your existing code works as-is.

---

## The Killer Feature: User Control

### OLD Approach
```
Developer hardcodes priority at build time
    ↓
User stuck with that choice
    ↓
Need a different implementation?
    ↓
😢 Contact dev, wait for rebuild
```

### NEW Approach
```
All implementations available at runtime
    ↓
User can pick any one instantly
    ↓
Need a different implementation?
    ↓
✅ One function call or env var
```

---

## Public API Comparison

### OLD (Complex)
```python
@Factory.register(Type, override_priority=0)
@Factory.register(Type, override_priority=5, override_priority=10)
# That's basically it, but confusing
```

### NEW (Simple)
```python
@Factory.register(Type)                    # Default
@Factory.register(Type, is_plugin=True)    # Plugin (priority)

Factory.list_implementations(Type)         # See options
Factory.use_implementation(Type, Class)    # Pick one
Factory.create_instance(Type, **kwargs)    # Use it
```

**That's the entire API.** Clear, simple, powerful.

---

## The Bottom Line

✅ **Simpler to understand**
✅ **More flexible**
✅ **User-friendly**
✅ **No code deletions**
✅ **Backward compatible**
✅ **Production ready**

**All with** just one new flag: `is_plugin=True`
