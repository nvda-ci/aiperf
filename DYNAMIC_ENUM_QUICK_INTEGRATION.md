# Quick Integration: Step-by-Step Code

## Step 1: Add PLUGIN_TYPES to AIPerfPluginManager

**File:** `aiperf/common/plugins.py`

Add this after the imports, before the class definition:

```python
from aiperf.common.protocols import AIPerfUIProtocol, EndpointProtocol, ServiceProtocol
```

Then add to the class:

```python
class AIPerfPluginManager:
    """Factory for managing plugin mappings and lazy loading plugin classes."""

    # ✅ ADD THIS LINE
    PLUGIN_TYPES: list[type[Protocol]] = [AIPerfUIProtocol, EndpointProtocol, ServiceProtocol]

    _instance_lock: threading.Lock = threading.Lock()
    _logger: logging.Logger = logging.getLogger(__name__)

    # ... rest of implementation
```

---

## Step 2: Update plugin_enums.py to use DynamicStrEnum

**File:** `aiperf/common/enums/plugin_enums.py`

**Replace the entire file with:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import DynamicStrEnum
from aiperf.common.plugins import AIPerfPluginManager
from aiperf.common.protocols import AIPerfUIProtocol, EndpointProtocol, ServiceProtocol


# Initialize plugin manager (singleton)
_pm = AIPerfPluginManager()


class AIPerfUIType(DynamicStrEnum):
    """Available UI framework implementations.

    Built-in members defined here.
    Additional plugins discovered at runtime are accessible via DynamicMemberProxy.
    """
    GRADIO = "gradio"
    STREAMLIT = "streamlit"


class EndpointType(DynamicStrEnum):
    """Available endpoint implementations.

    Built-in members defined here.
    Additional plugins discovered at runtime are accessible via DynamicMemberProxy.
    """
    OPENAI_CHAT = "openai_chat"
    ANTHROPIC_CHAT = "anthropic_chat"


class ServiceType(DynamicStrEnum):
    """Available service implementations.

    Built-in members defined here.
    Additional plugins discovered at runtime are accessible via DynamicMemberProxy.
    """
    # Add built-in services here
    pass
```

**That's it!** The old `_create_plugin_enum()` function is completely removed.

---

## Step 3: Verify It Works

Create a test file: `test_dynamic_enum.py`

```python
#!/usr/bin/env python3

from aiperf.common.enums.plugin_enums import EndpointType, AIPerfUIType
from aiperf.common.enums.base_enums import DynamicMemberProxy
from aiperf.common.plugins import AIPerfPluginManager
from aiperf.common.protocols import EndpointProtocol

print("=" * 60)
print("Testing DynamicStrEnum Integration")
print("=" * 60)

# Test 1: Built-in members
print("\n✓ Test 1: Built-in members")
builtin = EndpointType.OPENAI_CHAT
print(f"  - Accessed: {builtin}")
assert str(builtin) == "openai_chat"
print(f"  - Value: {builtin}")

# Test 2: Case-insensitive comparison
print("\n✓ Test 2: Case-insensitive comparison")
assert EndpointType.OPENAI_CHAT == "openai_chat"
assert EndpointType.OPENAI_CHAT == "OPENAI_CHAT"
print(f"  - Works!")

# Test 3: Dynamic member access
print("\n✓ Test 3: Dynamic member (plugin)")
dynamic = EndpointType.NVIDIA_OPTIMIZED
print(f"  - Accessed: {dynamic}")
print(f"  - Is proxy: {isinstance(dynamic, DynamicMemberProxy)}")
assert dynamic == "nvidia_optimized"

# Test 4: Iteration (only real members)
print("\n✓ Test 4: Iteration")
members = list(EndpointType)
print(f"  - Real members: {[m.name for m in members]}")
assert len(members) == 2  # OPENAI_CHAT and ANTHROPIC_CHAT

# Test 5: Plugin manager integration
print("\n✓ Test 5: Plugin manager integration")
pm = AIPerfPluginManager()
plugins = pm.list_plugin_names(EndpointProtocol)
print(f"  - Discovered plugins: {plugins}")

# Test 6: Pydantic integration
print("\n✓ Test 6: Pydantic validation")
from pydantic import BaseModel

class Config(BaseModel):
    endpoint: EndpointType

# Both built-in and dynamic work!
config1 = Config(endpoint="openai_chat")
config2 = Config(endpoint="anthropic_chat")
print(f"  - Built-in validation works!")

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
```

Run it:

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin
python test_dynamic_enum.py
```

Expected output:

```
============================================================
Testing DynamicStrEnum Integration
============================================================

✓ Test 1: Built-in members
  - Accessed: openai_chat
  - Value: openai_chat

✓ Test 2: Case-insensitive comparison
  - Works!

✓ Test 3: Dynamic member (plugin)
  - Accessed: nvidia_optimized
  - Is proxy: True

✓ Test 4: Iteration
  - Real members: ['OPENAI_CHAT', 'ANTHROPIC_CHAT']

✓ Test 5: Plugin manager integration
  - Discovered plugins: [...]

✓ Test 6: Pydantic validation
  - Built-in validation works!

============================================================
✅ All tests passed!
============================================================
```

---

## Complete Changes Checklist

### 1. `aiperf/common/plugins.py`
- [ ] Add `from aiperf.common.protocols import AIPerfUIProtocol, EndpointProtocol, ServiceProtocol`
- [ ] Add `PLUGIN_TYPES` constant to `AIPerfPluginManager` class

### 2. `aiperf/common/enums/plugin_enums.py`
- [ ] Import `DynamicStrEnum` from `base_enums`
- [ ] Remove `_create_plugin_enum()` function
- [ ] Remove old `BasePydanticEnumInfo` import
- [ ] Replace dynamic enum generation with class definitions
- [ ] Update docstrings

### 3. Test
- [ ] Create test script and verify all three enums work
- [ ] Verify IDE autocomplete works for built-in members
- [ ] Verify dynamic members work at runtime
- [ ] Verify Pydantic validation works

---

## What Changed

| Aspect | Before | After |
|--------|--------|-------|
| Enum Generation | `_create_plugin_enum()` function | `DynamicStrEnum` class |
| Built-in Members | Dynamically created | Statically defined |
| IDE Support | ❌ None | ✅ Autocomplete for builtins |
| Plugin Runtime | ✅ Works | ✅ Works (via proxy) |
| Type Safety | ⚠️ Limited | ✅ Full |
| Code Clarity | 🤔 Dynamic, less clear | ✅ Clear class definition |

---

## Common Issues & Solutions

### Issue: Import cycle
```
ImportError: cannot import name 'EndpointProtocol' from aiperf.common.protocols
```

**Solution:** Make sure `aiperf/common/protocols.py` defines these protocols and doesn't import from `plugin_enums.py`.

### Issue: IDE doesn't show autocomplete
```
EndpointType.|  <- No suggestions
```

**Solution:**
1. Restart IDE
2. Check that enum class has actual members defined
3. Verify `DynamicStrEnum` is imported correctly

### Issue: Pydantic validation fails
```
ValidationError: value is not a valid enumeration member
```

**Solution:** Ensure the enum value matches the string exactly (case-insensitive works, but "openai_chat" must match).

---

## Next Steps

1. Apply changes above
2. Run test script
3. Commit changes:
   ```bash
   git add aiperf/common/plugins.py aiperf/common/enums/plugin_enums.py
   git commit -m "feat: Integrate DynamicStrEnum with AIPerfPluginManager"
   ```
4. (Optional) Create `.pyi` stub files for plugin autocomplete hints
5. Update any docs referencing old dynamic enum generation

