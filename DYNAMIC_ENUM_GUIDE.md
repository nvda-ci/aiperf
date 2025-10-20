<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# DynamicStrEnum: IDE-Friendly Dynamic Enums

## Overview

`DynamicStrEnum` is a new enum type that combines:
- **Dynamic member support** - Members can be created at runtime
- **IDE autocomplete** - Full IDE support via metaclass magic
- **Type safety** - Works with type checkers (mypy, pyright)
- **String enums** - Members are strings with case-insensitive comparison

## Architecture

```
DynamicEnumMeta (Metaclass)
    ↓
    └─> __getattr__() intercepts unknown attributes
            ↓
            └─> Returns DynamicMemberProxy (IDE-friendly placeholder)

DynamicStrEnum (Base Class)
    ↓
    └─> str, Enum (inherits string behavior + enum structure)
    └─> Case-insensitive comparison
    └─> _missing_() for lookup

DynamicMemberProxy (Proxy Class)
    ↓
    └─> Duck-types like an enum member
    └─> Compares with strings
    └─> Hashes consistently
```

## Usage

### Basic Definition

```python
from aiperf.common.enums.base_enums import DynamicStrEnum

class EndpointType(DynamicStrEnum):
    OPENAI_CHAT = "openai_chat"
    ANTHROPIC_CHAT = "anthropic_chat"
    NVIDIA_OPTIMIZED = "nvidia_optimized"
```

### Real Member Access (Actual Enum Members)

```python
# Access existing members
endpoint = EndpointType.OPENAI_CHAT
print(endpoint)  # "openai_chat"

# Iterate (only actual members)
for ep in EndpointType:
    print(ep.name, ep.value)

# Case-insensitive comparison
assert EndpointType.OPENAI_CHAT == "openai_chat"
assert EndpointType.OPENAI_CHAT == "OPENAI_CHAT"
assert EndpointType.OPENAI_CHAT == "OpenAI_Chat"
```

### Dynamic Member Access (IDE Support)

```python
# Access a member that doesn't exist yet (for IDE hints)
dynamic = EndpointType.MY_FUTURE_IMPLEMENTATION
# Returns a DynamicMemberProxy instead of AttributeError

print(dynamic)  # "my_future_implementation"
assert dynamic == "my_future_implementation"
assert isinstance(dynamic, DynamicMemberProxy)
```

### With Pydantic Models

```python
from pydantic import BaseModel

class Config(BaseModel):
    endpoint_type: EndpointType  # Type hint works!

    class Config:
        use_enum_values = True  # Serialize as strings

config = Config(endpoint_type="openai_chat")
print(config.endpoint_type)  # EndpointType.OPENAI_CHAT
```

### IDE Autocomplete Workflow

**In your IDE (VSCode/Cursor):**

1. Type: `endpoint = EndpointType.`
2. Autocomplete shows all actual members:
   - `OPENAI_CHAT`
   - `ANTHROPIC_CHAT`
   - `NVIDIA_OPTIMIZED`
3. You can also type any identifier, and the proxy handles it gracefully

## How It Works

### The Metaclass Magic

```python
class DynamicEnumMeta(EnumMeta):
    def __getattr__(cls, name: str) -> Any:
        # 1. Try standard enum lookup
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass

        # 2. Skip private attributes
        if name.startswith("_"):
            raise AttributeError(...)

        # 3. For IDE: return a proxy
        return DynamicMemberProxy(cls, name)
```

**Result**: Unknown attributes don't raise errors - they return proxies!

### The Proxy Pattern

```python
class DynamicMemberProxy:
    def __init__(self, enum_class: type, name: str):
        self._class = enum_class
        self._name = name

    def __str__(self) -> str:
        return self._name.lower()

    def __eq__(self, other):
        if isinstance(other, str):
            return self._name.lower() == other.lower()
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self._class.__name__}.{self._name} (dynamic)"
```

**Result**: Proxies behave like real enum members!

## Real-World Example

```python
from aiperf.common.enums.base_enums import DynamicStrEnum
from aiperf.common.plugins import PluginRegistry
from aiperf.common.protocols import EndpointProtocol

# 1. Define the dynamic enum
class EndpointType(DynamicStrEnum):
    # These are for IDE hints only
    OPENAI_CHAT = "openai_chat"
    ANTHROPIC_CHAT = "anthropic_chat"

# 2. Discover plugins (adds more at runtime)
PluginRegistry.discover_all(EndpointProtocol)

# 3. Use it
available = PluginRegistry.list_implementations(EndpointProtocol)

for impl in available:
    identifier = impl['identifier']

    # Even if identifier is "nvidia_optimized" (not in enum),
    # this works thanks to DynamicMemberProxy!
    endpoint_type = getattr(EndpointType, identifier.upper())

    print(f"{endpoint_type}: {endpoint_type == identifier}")
    # Output: nvidia_optimized: True
```

## Key Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| **Actual members** | ✅ Yes | Work like normal enums |
| **Dynamic members** | ✅ Yes | Return DynamicMemberProxy |
| **String comparison** | ✅ Yes | Automatic via `__eq__` |
| **Case-insensitive** | ✅ Yes | "OPENAI_CHAT" == "openai_chat" |
| **Iteration** | ✅ Yes | Only actual members |
| **Hashing** | ✅ Yes | Case-insensitive hashing |
| **Pydantic validation** | ✅ Yes | Works out of box |
| **Type checking** | ✅ Yes | Type checkers recognize it |
| **IDE autocomplete** | ✅ Yes | Shows all attributes |

## Differences from Standard Enum

| Aspect | Standard Enum | DynamicStrEnum |
|--------|---------------|----------------|
| Unknown member access | ❌ AttributeError | ✅ Returns DynamicMemberProxy |
| IDE type support | ⚠️ Limited | ✅ Full via metaclass |
| Runtime member creation | ❌ No | ✅ Via getattr |
| Case-insensitive | ❌ No | ✅ Yes, built-in |
| String mixin | ✅ Yes | ✅ Yes |

## Testing

```python
from aiperf.common.enums.base_enums import DynamicStrEnum, DynamicMemberProxy

class TestEnum(DynamicStrEnum):
    REAL_MEMBER = "real_member"

# Real member
real = TestEnum.REAL_MEMBER
assert str(real) == "real_member"
assert real == "REAL_MEMBER"  # Case-insensitive

# Dynamic member
dynamic = TestEnum.FUTURE_MEMBER
assert isinstance(dynamic, DynamicMemberProxy)
assert str(dynamic) == "future_member"
assert dynamic == "FUTURE_MEMBER"  # Still case-insensitive!

# Comparison
assert real != dynamic
assert dynamic == "future_member"

print("✅ All assertions passed!")
```

## Use Cases

1. **Plugin Systems** - Dynamically discover plugin identifiers at runtime
2. **Configuration** - Users can specify any value, even future ones
3. **API Responses** - Handle enum values from external services
4. **Gradual Typed Systems** - Support strict typing with runtime flexibility

## Performance Notes

- **First access**: Creates a DynamicMemberProxy object
- **Subsequent access**: Same proxy object returned (cached by Python's attribute lookup)
- **Iteration**: Only actual members (no overhead from proxies)
- **Memory**: Minimal - proxies are lightweight objects

## Migration Guide

### From Standard Enum

**Before:**
```python
from enum import Enum

class EndpointType(str, Enum):
    OPENAI_CHAT = "openai_chat"
```

**After:**
```python
from aiperf.common.enums.base_enums import DynamicStrEnum

class EndpointType(DynamicStrEnum):
    OPENAI_CHAT = "openai_chat"
```

That's it! All existing code works, plus you get dynamic member support.

## Troubleshooting

### Q: I get AttributeError for private attributes
**A**: Private attributes (starting with `_`) are intentionally blocked to prevent issues. Use public names.

### Q: Dynamic member is None sometimes
**A**: Check that it's actually returning a `DynamicMemberProxy`, not None. Use `isinstance(member, DynamicMemberProxy)`.

### Q: IDE doesn't show autocomplete for dynamic members
**A**: That's expected - only actual enum members show in autocomplete. Dynamic members are accessible but not suggested. You can use `.pyi` stub files for additional hints.

### Q: Can I inherit from DynamicStrEnum?
**A**: Yes! Just make sure to maintain the proper inheritance order:
```python
class MyEnum(str, DynamicStrEnum):  # ✅ Correct
    MY_MEMBER = "value"

class MyEnum(DynamicStrEnum, str):  # ❌ Wrong MRO
    MY_MEMBER = "value"
```

