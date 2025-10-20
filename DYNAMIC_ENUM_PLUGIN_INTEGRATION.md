<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# DynamicStrEnum + AIPerfPluginManager Integration

## Current Setup

You have:
1. **AIPerfPluginManager** - Discovers and manages plugins from entry points
2. **plugin_enums.py** - Dynamically creates enums from discovered plugins
3. **DynamicStrEnum** - New enum type with IDE support

Goal: Replace the dynamic enum generation with `DynamicStrEnum` for better IDE support.

## Integration Steps

### Step 1: Update `AIPerfPluginManager`

Add `PLUGIN_TYPES` constant to define which protocols to manage:

```python
# aiperf/common/plugins.py

from aiperf.common.protocols import AIPerfUIProtocol, EndpointProtocol, ServiceProtocol

class AIPerfPluginManager:
    """Factory for managing plugin mappings and lazy loading plugin classes."""

    # Define which protocols to manage
    PLUGIN_TYPES: list[type[Protocol]] = [
        AIPerfUIProtocol,
        EndpointProtocol,
        ServiceProtocol,
    ]

    _instance_lock: threading.Lock = threading.Lock()
    _logger: logging.Logger = logging.getLogger(__name__)

    # ... rest of implementation
```

### Step 2: Update `plugin_enums.py`

Replace the old dynamic generation with `DynamicStrEnum`:

```python
# aiperf/common/enums/plugin_enums.py

from aiperf.common.enums.base_enums import DynamicStrEnum
from aiperf.common.plugins import AIPerfPluginManager
from aiperf.common.protocols import AIPerfUIProtocol, EndpointProtocol, ServiceProtocol

# Initialize plugin manager once to get plugin names
_pm = AIPerfPluginManager()

# Define base enums with built-in registrations
class AIPerfUIType(DynamicStrEnum):
    """Available UI implementations."""
    GRADIO = "gradio"
    STREAMLIT = "streamlit"
    # Plugins can add more at runtime via DynamicMemberProxy!

class EndpointType(DynamicStrEnum):
    """Available endpoint implementations."""
    OPENAI_CHAT = "openai_chat"
    ANTHROPIC_CHAT = "anthropic_chat"
    # Plugins like "nvidia_optimized" will be accessible via proxy

class ServiceType(DynamicStrEnum):
    """Available service implementations."""
    # Add built-in services here
    pass

# Optional: Populate enums from discovered plugins (one-time, for built-in hints)
def _populate_builtin_enums():
    """Populate enum members from initially discovered plugins.

    This runs once at import time to add any built-in plugins to the enum.
    External plugins discovered later are still accessible via DynamicMemberProxy.
    """
    # This is optional - only do this if you want to pre-populate the enums
    # with discovered plugins. For built-ins, just define them in the class above.
    pass
```

### Step 3: Test Integration

```python
from aiperf.common.enums.plugin_enums import EndpointType
from aiperf.common.enums.base_enums import DynamicMemberProxy

# 1. Built-in members work with IDE autocomplete
builtin = EndpointType.OPENAI_CHAT
print(builtin)  # "openai_chat"
assert builtin == "openai_chat"

# 2. Dynamic members work at runtime (no IDE autocomplete, but still works!)
dynamic = EndpointType.NVIDIA_OPTIMIZED
print(dynamic)  # "nvidia_optimized"
assert isinstance(dynamic, DynamicMemberProxy)
assert dynamic == "nvidia_optimized"

# 3. Iteration only shows built-in members
for ep in EndpointType:
    print(f"Member: {ep}")
    # Output:
    # Member: openai_chat
    # Member: anthropic_chat
```

## Before & After Comparison

### BEFORE (Old Dynamic Generation)

```python
# plugin_enums.py
def _create_plugin_enum(plugin_type, enum_name):
    pm = AIPerfPluginManager()
    plugins = pm.list_plugin_names(plugin_type)
    enum_info = {
        name.replace("-", "_").upper(): BasePydanticEnumInfo(tag=name)
        for name in plugins
    }
    return BasePydanticBackedStrEnum(enum_name, enum_info)

AIPerfUIType = _create_plugin_enum(AIPerfUIProtocol, "AIPerfUIType")

# Problems:
# ❌ IDE doesn't know about enum members
# ❌ No autocomplete at all
# ❌ Type checkers confused
```

### AFTER (DynamicStrEnum Integration)

```python
# plugin_enums.py
class AIPerfUIType(DynamicStrEnum):
    """Available UI implementations."""
    GRADIO = "gradio"
    STREAMLIT = "streamlit"

# Benefits:
# ✅ IDE autocompletes built-in members
# ✅ Type checkers understand it
# ✅ Dynamic members still work at runtime (via proxy)
# ✅ Cleaner, more Pythonic code
```

## Architecture Diagram

```
┌─────────────────────────────────────┐
│    AIPerfPluginManager              │
│  - Discovers plugins from entry pts │
│  - Manages PLUGIN_TYPES             │
│  - Returns list of plugin names     │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
   Built-in     Discovered
   Endpoints    Endpoints
   (hardcoded)  (runtime)
        │             │
        └──────┬──────┘
               ▼
    ┌─────────────────────────┐
    │  EndpointType Enum      │
    │  (DynamicStrEnum)       │
    ├─────────────────────────┤
    │ Built-in (hardcoded):   │
    │ • OPENAI_CHAT           │ ◄─── IDE Autocomplete
    │ • ANTHROPIC_CHAT        │
    ├─────────────────────────┤
    │ Dynamic (at runtime):   │
    │ • NVIDIA_OPTIMIZED      │ ◄─── Via DynamicMemberProxy
    │ • CUSTOM_ENDPOINT       │      (no autocomplete)
    └─────────────────────────┘
```

## Usage Examples

### Example 1: Simple Access

```python
from aiperf.common.enums.plugin_enums import EndpointType

# Built-in - IDE shows autocomplete
endpoint = EndpointType.OPENAI_CHAT

# Plugin - Works at runtime, not in IDE
endpoint = EndpointType.NVIDIA_OPTIMIZED
```

### Example 2: Dynamic Discovery

```python
from aiperf.common.plugins import AIPerfPluginManager
from aiperf.common.enums.plugin_enums import EndpointType
from aiperf.common.protocols import EndpointProtocol

pm = AIPerfPluginManager()

# Get all discovered plugins
plugins = pm.list_plugin_names(EndpointProtocol)

for plugin_name in plugins:
    # Access via DynamicStrEnum (works for all!)
    endpoint_type = getattr(EndpointType, plugin_name.upper())

    print(f"{endpoint_type}: {plugin_name}")
```

### Example 3: With Pydantic

```python
from pydantic import BaseModel
from aiperf.common.enums.plugin_enums import EndpointType

class Config(BaseModel):
    endpoint_type: EndpointType

# Works for both built-in and dynamic!
config1 = Config(endpoint_type="openai_chat")
config2 = Config(endpoint_type="nvidia_optimized")
```

## Migration Checklist

- [ ] Add `PLUGIN_TYPES` to `AIPerfPluginManager`
- [ ] Import `DynamicStrEnum` in `plugin_enums.py`
- [ ] Replace `_create_plugin_enum()` calls with class definitions
- [ ] Define built-in members in each enum class
- [ ] Test that built-in members work with IDE autocomplete
- [ ] Test that plugin members work at runtime (via proxy)
- [ ] Test Pydantic validation with both types
- [ ] Update any documentation referencing old enums
- [ ] Commit changes

## FAQs

### Q: Do I need to define ALL plugins in the enum?

**A:** No! Define only the built-in/core ones. Plugins discovered at runtime are automatically accessible via `DynamicMemberProxy`.

```python
class EndpointType(DynamicStrEnum):
    # Only built-ins here
    OPENAI_CHAT = "openai_chat"
    ANTHROPIC_CHAT = "anthropic_chat"

    # Plugins like "nvidia_optimized" are still accessible:
    # EndpointType.NVIDIA_OPTIMIZED  # Works! (returns proxy)
```

### Q: Will Pydantic validation work for plugin members?

**A:** Yes! Pydantic validates the string value, not the enum member:

```python
class Config(BaseModel):
    endpoint: EndpointType

# Both work:
Config(endpoint="openai_chat")      # ✅ Built-in
Config(endpoint="nvidia_optimized")  # ✅ Plugin (even though not in enum!)
```

### Q: What about IDE autocomplete for plugins?

**A:** Built-in members get full autocomplete. For plugin autocomplete, create a `.pyi` stub file (optional):

```python
# plugin_enums.pyi
class EndpointType(DynamicStrEnum):
    OPENAI_CHAT: str
    ANTHROPIC_CHAT: str
    NVIDIA_OPTIMIZED: str  # Plugin hint for IDE
    CUSTOM_ENDPOINT: str   # Plugin hint for IDE
```

### Q: Do I need to regenerate stubs?

**A:** Only if you want autocomplete for plugins. Otherwise, the system works as-is with IDE hints for built-ins only.

## Performance Impact

- ✅ Same startup time (plugin discovery happens regardless)
- ✅ Same runtime performance (DynamicMemberProxy is lightweight)
- ✅ Slightly faster IDE responsiveness (clearer enum structure)
- ✅ No additional memory overhead

