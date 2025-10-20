<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Hooks System - Quick Reference

## File Paths
```
/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/hooks.py
/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/mixins/hooks_mixin.py
/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/mixins/aiperf_lifecycle_mixin.py
/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/mixins/aiperf_logger_mixin.py
/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/protocols.py
/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/tests/test_hooks.py
```

## Available Decorators

### Lifecycle Hooks
- `@on_init` - During initialization
- `@on_start` - During start
- `@on_stop` - During stop
- `@on_state_change` - On state transition

### Message/Command Hooks
- `@on_message(*message_types)` - Specific message types
- `@on_command(*command_types)` - Specific command types
- `@on_request(*message_types)` - Request handling
- `@on_pull_message(*message_types)` - Pull client messages

### Progress/Metrics Hooks
- `@on_realtime_metrics` - Real-time metrics
- `@on_profiling_progress` - Profiling progress
- `@on_records_progress` - Records progress
- `@on_warmup_progress` - Warmup progress

### Worker Hooks
- `@on_worker_update` - Worker status update
- `@on_worker_status_summary` - Worker status summary

### Background Tasks
- `@background_task(interval=None, immediate=True, stop_on_error=False)`

### Class Declaration
- `@provides_hooks(*hook_types)` - Declare which hooks class provides

## Core Classes

**HooksMixin** - Base class for hooks functionality
- `get_hooks(*hook_types, reverse=False)` - Get registered hooks
- `run_hooks(*hook_types, reverse=False, **kwargs)` - Execute hooks
- `attach_hook(hook_type, func, params=None)` - Add hook dynamically

**AIPerfLifecycleMixin** - Lifecycle state machine
- Inherits from HooksMixin
- `async initialize()` - Initialize and run @on_init hooks
- `async start()` - Start and run @on_start hooks
- `async stop()` - Stop and run @on_stop hooks (reversed)

**AIPerfLoggerMixin** - Logging with lazy evaluation
- `.debug()`, `.info()`, `.warning()`, `.error()`, etc.
- Supports both strings and callables: `self.debug(lambda: f"msg {var}")`

## Validation & Error Handling

**Validation:**
- Hook type must be declared with `@provides_hooks`
- `UnsupportedHookError` raised if not declared
- Callable check on `attach_hook()`
- Parameter type validation in `for_each_hook_param()`

**Error Handling:**
- All exceptions collected in `AIPerfMultiError`
- Execution continues after hook failure
- Each error wrapped in `HookError` with context

## Hook Execution Order

1. **Base class hooks first** (reversed MRO)
2. **Definition order** (top to bottom)
3. **Reverse on stop** (cleanup in reverse)

Example:
```python
class Base(HooksMixin):
    @on_init
    async def base_hook(self): pass

class Child(Base):
    @on_init
    async def child_hook(self): pass

# Execution: base_hook() → child_hook()
```

## Key Attributes

Internal function attributes:
- `__aiperf_hook_type__` - Hook type enum
- `__aiperf_hook_params__` - Hook parameters
- `__provides_hooks__` - Set of hook types (class attribute)

## Common Patterns

### Pattern 1: Service with Lifecycle
```python
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.hooks import on_init, on_start, on_stop

class MyService(AIPerfLifecycleMixin):
    @on_init
    async def _init(self): pass

    @on_start
    async def _start(self): pass

    @on_stop
    async def _stop(self): pass
```

### Pattern 2: Background Task
```python
from aiperf.common.hooks import background_task
from aiperf.common.mixins import AIPerfLifecycleMixin

class TaskRunner(AIPerfLifecycleMixin):
    @background_task(interval=5.0, immediate=True)
    async def _periodic_task(self):
        # Runs every 5 seconds
        pass
```

### Pattern 3: Custom Hook Provider
```python
from aiperf.common.hooks import provides_hooks, AIPerfHook
from aiperf.common.mixins import HooksMixin

@provides_hooks(AIPerfHook.ON_MESSAGE)
class Handler(HooksMixin):
    @on_message(MessageType.STATUS)
    async def _handle_status(self, message): pass
```

### Pattern 4: Dynamic Hook
```python
handler = Handler()

async def my_handler(message):
    pass

handler.attach_hook(AIPerfHook.ON_MESSAGE, my_handler,
                    params=(MessageType.ERROR,))
```

## Testing

Run hook tests:
```bash
pytest tests/test_hooks.py -v
pytest tests/test_background_task.py -v
```

Key test patterns:
- `UnsupportedHookError` - Hook not provided
- `HookError` - Hook execution failure
- `AIPerfMultiError` - Multiple hook failures
- Hook order verification
- Reverse execution verification

## Exception Types

- **UnsupportedHookError** - Hook type not provided
- **HookError** - Single hook failure with context
- **AIPerfMultiError** - Multiple hook failures

## Parameter Resolution

Static:
```python
@on_message(MessageType.STATUS)
```

Dynamic:
```python
@on_message(lambda self: [MessageType.STATUS] if self.debug else [])
```

Parameters resolved at execution time using instance context.
