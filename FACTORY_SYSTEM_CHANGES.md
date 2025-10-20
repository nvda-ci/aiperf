<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Factory System Enhancement - Summary

## What Changed?

The factory system now supports **multiple implementations per type** instead of the previous "last one wins, others deleted" approach.

### Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Multiple implementations | ❌ Only one survives | ✅ All coexist |
| Plugin priority | 📌 Manual priority numbers | ✅ Just set `is_plugin=True` |
| User override | ❌ Requires code change | ✅ One method call |
| After registration | ❌ Can't change priority | ✅ Can switch anytime |
| Learning curve | 📊 Complex priority system | 🎯 Dead simple |

---

## Three Line Summary

1. **For Plugin Developers**: Add `is_plugin=True` when registering → automatically takes priority
2. **For Users**: Call `Factory.use_implementation(Type, MyClass)` → override anytime
3. **No Breaking Changes**: Existing code works unchanged

---

## Migration Path

### Your Existing Code (Still Works!)

```python
# Before
@EndpointFactory.register(EndpointType.CHAT, override_priority=5)
class MyEndpoint(BaseEndpoint):
    pass

# Now becomes (just remove the priority parameter)
@EndpointFactory.register(EndpointType.CHAT)
class MyEndpoint(BaseEndpoint):
    pass

# Or if you're a plugin
@EndpointFactory.register(EndpointType.CHAT, is_plugin=True)
class MyEndpoint(BaseEndpoint):
    pass
```

---

## User Experience Flow

### Scenario: Plugin vs Built-in

```
Builtin ChatEndpoint registered
Nvidia plugin OptimizedChatEndpoint registered (is_plugin=True)
                            ↓
                   ✓ Uses OptimizedChatEndpoint
                   (plugin automatically preferred)
                            ↓
User wants to test with builtin:
    EndpointFactory.use_implementation(Type, ChatEndpoint)
                            ↓
                   ✓ Now uses ChatEndpoint
```

---

## Quick Reference

### Register (Developer)
```python
@Factory.register(TYPE)                    # Built-in
@Factory.register(TYPE, is_plugin=True)    # Plugin (auto-priority)
class MyImpl:
    pass
```

### Use (User/Ops)
```python
# See options
Factory.list_implementations(TYPE)

# Pick one
Factory.use_implementation(TYPE, MyClass)

# Create instance
instance = Factory.create_instance(TYPE, **kwargs)
```

---

## Design Principles

✅ **User Experience First**
- Minimal API
- Clear intent with `is_plugin=True`
- Easy override mechanism

✅ **No Surprises**
- Nothing gets deleted
- Plugins naturally take priority (sensible default)
- Users always in control

✅ **Backward Compatible**
- Existing registrations work as-is
- Gradual migration possible
- No code removal needed

✅ **Built for Real Use**
- Testing flexibility
- Environment variable support
- Works with CI/CD pipelines

---

## File Changes

- `aiperf/common/factories.py` - Simplified implementation
- `docs/SIMPLE_FACTORY_GUIDE.md` - User-friendly guide

No breaking changes to existing code!
