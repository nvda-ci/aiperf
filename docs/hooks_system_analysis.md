<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Hooks System - Complete Implementation Analysis

## 1. File Locations and Structure

### Core Files
- **Hooks Definition**: `/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/hooks.py`
- **HooksMixin Implementation**: `/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/mixins/hooks_mixin.py`
- **AIPerfLifecycleMixin**: `/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/mixins/aiperf_lifecycle_mixin.py`
- **AIPerfLoggerMixin**: `/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/mixins/aiperf_logger_mixin.py`
- **Protocols**: `/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/aiperf/common/protocols.py`
- **Tests**: `/home/anthony/nvidia/projects/aiperf/ajc/endpoint-plugin/tests/test_hooks.py`

---

## 2. Available Hooks Defined

All hooks are defined in the `AIPerfHook` enum (lines 41-57 of hooks.py):

```python
class AIPerfHook(CaseInsensitiveStrEnum):
    BACKGROUND_TASK = "@background_task"
    ON_COMMAND = "@on_command"
    ON_INIT = "@on_init"
    ON_MESSAGE = "@on_message"
    ON_REALTIME_METRICS = "@on_realtime_metrics"
    ON_PROFILING_PROGRESS = "@on_profiling_progress"
    ON_PULL_MESSAGE = "@on_pull_message"
    ON_RECORDS_PROGRESS = "@on_records_progress"
    ON_START = "@on_start"
    ON_STATE_CHANGE = "@on_state_change"
    ON_STOP = "@on_stop"
    ON_REQUEST = "@on_request"
    ON_WARMUP_PROGRESS = "@on_warmup_progress"
    ON_WORKER_STATUS_SUMMARY = "@on_worker_status_summary"
    ON_WORKER_UPDATE = "@on_worker_update"
```

**Total: 15 different hook types**

---

## 3. Hook Decorators Available

### Simple Decorators (No Parameters)
1. **`@on_init`** - Called during initialization phase
2. **`@on_start`** - Called during start phase
3. **`@on_stop`** - Called during stop phase
4. **`@on_state_change`** - Called when service state changes
5. **`@on_realtime_metrics`** - Called when realtime metrics received
6. **`@on_profiling_progress`** - Called on profiling progress update
7. **`@on_records_progress`** - Called on records progress update
8. **`@on_warmup_progress`** - Called on warmup progress update
9. **`@on_worker_status_summary`** - Called on worker status summary
10. **`@on_worker_update`** - Called on worker update

### Decorators with Parameters
1. **`@on_message(*message_types)`** - Listen to specific message types
2. **`@on_pull_message(*message_types)`** - Listen to pull messages of specific types
3. **`@on_request(*message_types)`** - Handle requests of specific types
4. **`@on_command(*command_types)`** - Handle commands of specific types

### Specialized Decorators
1. **`@background_task(interval=None, immediate=True, stop_on_error=False)`** - Define background tasks

### Class-Level Decorator
**`@provides_hooks(*hook_types)`** - Declare which hooks a class provides

---

## 4. LifecycleMixin Implementation

### File: `aiperf_lifecycle_mixin.py`

**Key Features:**
- Inherits from `TaskManagerMixin` and `HooksMixin`
- Decorated with `@provides_hooks()` for 5 lifecycle hooks
- Implements a state machine with lifecycle states
- Manages child lifecycles

**State Transitions:**
```
CREATED → INITIALIZING → INITIALIZED → STARTING → RUNNING
                                                       ↓
                                              STOPPING → STOPPED
                                                       ↓
                                              FAILED
```

**Provided Hooks:**
```python
@provides_hooks(
    AIPerfHook.ON_INIT,
    AIPerfHook.ON_START,
    AIPerfHook.ON_STOP,
    AIPerfHook.ON_STATE_CHANGE,
    AIPerfHook.BACKGROUND_TASK,
)
```

**Key Methods:**
- `async initialize()` - Transitions to INITIALIZED state, runs `ON_INIT` hooks
- `async start()` - Transitions to RUNNING state, runs `ON_START` hooks
- `async stop()` - Transitions to STOPPED state, runs `ON_STOP` hooks in reverse order
- `async _set_state(state)` - Changes state and runs `ON_STATE_CHANGE` hooks

**Built-in Hooks Used:**
- `@on_init async def _initialize_children()` - Initializes child lifecycles
- `@on_start async def _start_children()` - Starts child lifecycles
- `@on_start async def _start_background_tasks()` - Starts all background tasks
- `@on_stop async def _stop_children()` - Stops child lifecycles (in reverse)
- `@on_stop async def _stop_all_tasks()` - Stops all background tasks

---

## 5. LoggerMixin Implementation

### File: `aiperf_logger_mixin.py`

**Class: `AIPerfLoggerMixin`**
- Provides lazy-evaluated logging with f-strings
- Implements `AIPerfLoggerProtocol`

**Logging Methods:**
- `trace()`, `debug()`, `info()`, `notice()`, `warning()`, `success()`, `error()`, `exception()`, `critical()`
- All support both string messages and callable lambdas for lazy evaluation

**Properties:**
- `is_debug_enabled` - Check if debug logging is enabled
- `is_trace_enabled` - Check if trace logging is enabled
- `is_enabled_for(level)` - Check if logging level is enabled

**Example Usage:**
```python
self.debug(lambda: f"Processing {item} of {count}")
self.info("Simple message")
self.trace(lambda i=i: f"Binding: {i}")
```

---

## 6. HooksMixin Implementation

### File: `hooks_mixin.py`

**Key Responsibilities:**
1. Register hooks from decorated methods during `__init__`
2. Validate that hooks are provided by base classes
3. Allow dynamic hook attachment via `attach_hook()`
4. Execute hooks via `run_hooks()`
5. Retrieve hooks via `get_hooks()`

**Core Data Structures:**

```python
# Instance variables
self._provided_hook_types: set[HookType]  # Hook types this class provides
self._hooks: dict[HookType, list[Hook]]   # Actual hook callables
```

**Key Methods:**

1. **`__init__()`** (lines 41-76)
   - Iterates through MRO in reverse order (base classes first)
   - Collects all `@provides_hooks` declarations
   - Scans class methods for hook decorators
   - Creates `Hook` objects binding methods to instance
   - Validates all hooks are provided

2. **`_check_hook_type_is_provided(hook_type)`** (lines 78-94)
   - **Validation Method** - Ensures hook type is in `_provided_hook_types`
   - Raises `UnsupportedHookError` if not provided
   - Called during initialization and in `attach_hook()`

3. **`attach_hook(hook_type, func, params=None)`** (lines 96-114)
   - **Dynamic Hook Registration** - Add hooks at runtime
   - Validates hook type is provided
   - Validates function is callable
   - Appends to `_hooks` dictionary

4. **`get_hooks(*hook_types, reverse=False)`** (lines 116-128)
   - Retrieve all hooks of given type(s)
   - Optional reverse ordering (for cleanup/stop hooks)
   - Returns list of `Hook` objects

5. **`for_each_hook_param(*hook_types, self_obj, param_type, lambda_func, reverse=False)`** (lines 130-166)
   - Iterate hooks with parameter validation
   - Resolves callable parameters using `self_obj`
   - Type-checks each parameter
   - Useful for message/command type validation

6. **`async run_hooks(*hook_types, reverse=False, **kwargs)`** (lines 168-192)
   - **Execution Engine** - Run hooks sequentially
   - Collects all exceptions in `AIPerfMultiError`
   - Continues executing even if one hook fails
   - Passes kwargs to each hook function
   - Supports reverse execution order

---

## 7. Hook Model Classes

### Hook Class (lines 75-120)

```python
class Hook(BaseModel, Generic[HookParamsT]):
    func: Callable
    params: HookParamsT | Callable[[SelfT], HookParamsT] | None = None

    @property
    def hook_type(self) -> HookType
    @property
    def func_name(self) -> str
    @property
    def qualified_name(self) -> str

    def resolve_params(self, self_obj: SelfT) -> HookParamsT | None
    async def __call__(self, **kwargs) -> None
```

**Key Features:**
- Pydantic BaseModel for validation
- Supports both static and callable parameters
- `resolve_params()` handles dynamic parameter resolution
- `__call__()` handles both sync/async functions

### BackgroundTaskParams (lines 122-126)

```python
class BackgroundTaskParams(BaseModel):
    interval: float | Callable[[Any], float] | None = None
    immediate: bool = False
    stop_on_error: bool = False
```

### HookAttrs Class (lines 63-72)

**Constants for attribute names set on functions:**
```python
HOOK_TYPE = "__aiperf_hook_type__"           # String attribute
HOOK_PARAMS = "__aiperf_hook_params__"       # Parameter attribute
PROVIDES_HOOKS = "__provides_hooks__"        # Class attribute (set)
```

---

## 8. Decorator Implementation Details

### Generic Decorators (Internal)

**`_hook_decorator(hook_type, func)`** (lines 128-140)
- Simple decorator that sets `HOOK_TYPE` attribute on function

**`_hook_decorator_with_params(hook_type, params)`** (lines 143-161)
- Returns a decorator that sets both `HOOK_TYPE` and `HOOK_PARAMS` attributes

### Provides Hooks Decorator (lines 205-227)

```python
def provides_hooks(*hook_types: HookType) -> Callable:
    def decorator(cls: type[HooksMixinT]) -> type[HooksMixinT]:
        setattr(cls, HookAttrs.PROVIDES_HOOKS, set(hook_types))
        return cls
    return decorator
```

**Example Usage:**
```python
@provides_hooks(AIPerfHook.ON_MESSAGE, AIPerfHook.ON_STATE_CHANGE)
class MyService(HooksMixin):
    @on_message(MessageType.STATUS)
    async def _handle_status(self, message):
        pass
```

---

## 9. Hook Registration Process

### During Class Instantiation

1. **MRO Traversal** (lines 48-52)
   - Iterate `__mro__` in REVERSE order (base classes first)
   - Collect all `@provides_hooks` declarations
   - Build `_provided_hook_types` set

2. **Method Scanning** (lines 54-72)
   - Iterate through class `__dict__.values()`
   - Check for `HOOK_TYPE` attribute
   - **Validate** hook type is provided
   - Bind method to instance
   - Create `Hook` object with resolved params
   - Store in `_hooks` dictionary

### Ordering Guarantees

**Base class hooks execute before subclass hooks:**
```python
# MRO processing ensures:
# 1. GrandParent hooks first
# 2. Parent hooks second
# 3. Child hooks last
# Within each class: methods defined top-to-bottom
```

**Test Example:**
```python
class Hooks(HooksMixin):
    @on_init async def hook_2(self): pass    # Runs first
    @on_init async def hook_3(self): pass    # Runs second
    @on_init async def hook_1(self): pass    # Runs third

# Execution order: hook_2 → hook_3 → hook_1 (definition order)
```

---

## 10. Validation and Enforcement Mechanisms

### 1. Hook Type Validation (Critical)

**Method: `_check_hook_type_is_provided(hook_type)`**

- **When Called:**
  - During `__init__` for each decorated method (line 62)
  - When `attach_hook()` is called (line 113)

- **Enforcement:**
  - Checks if `hook_type in self._provided_hook_types`
  - Raises `UnsupportedHookError` if not provided
  - Error message includes class name and available hook types

- **Example Error:**
```
UnsupportedHookError: Hook @on_start is not provided by any base class of MyClass.
(Provided Hooks: [@on_init, @on_stop])
```

### 2. Callable Validation

**In `attach_hook()`** (line 110-111)
```python
if not callable(func):
    raise ValueError(f"Invalid hook function: {func}")
```

### 3. Parameter Type Validation

**In `for_each_hook_param()`** (lines 152-164)
```python
if not isinstance(params, Iterable):
    raise ValueError(f"Invalid hook params: {params}. Expected Iterable but got {type(params)}")

for param in params:
    if not isinstance(param, param_type):
        raise ValueError(f"Invalid hook param: {param}. Expected {param_type} but got {type(param)}")
```

### 4. Hook Execution Error Collection

**In `run_hooks()`** (lines 181-192)
```python
exceptions: list[Exception] = []
for hook in self.get_hooks(*hook_types, reverse=reverse):
    try:
        await hook(**kwargs)
    except Exception as e:
        exceptions.append(HookError(self.__class__.__name__, hook.func_name, e))
        # Log but continue

if exceptions:
    raise AIPerfMultiError(None, exceptions)
```

- **Continues execution** even after hook failure
- **Collects all errors** in `AIPerfMultiError`
- **Logs each error** with context

### 5. Background Task Validation

**In `_start_background_tasks()`** (lines 258-262)
```python
for hook in self.get_hooks(AIPerfHook.BACKGROUND_TASK):
    if not isinstance(hook.params, BackgroundTaskParams):
        raise AttributeError(
            f"Invalid hook parameters for {hook}: {hook.params}. "
            f"Expected BackgroundTaskParams."
        )
```

### 6. Test Coverage for Validation

From `tests/test_hooks.py`:

**Test: `test_unsupported_hook_decorator()`** (lines 160-172)
- Verifies `UnsupportedHookError` raised when using unsupported hook
- Exception raised during class instantiation

**Test: `test_unsupported_hook_error_attach_hook()`** (lines 362-383)
- Tests `attach_hook()` validation
- Ensures error message contains hook type, class name, available hooks

**Test: `test_unsupported_hook_error_message_content()`** (lines 386-402)
- Validates error messages are helpful and informative
- Includes available hook types in error

**Test: `test_hook_error_exception()`** (lines 291-313)
- Single hook failure raises `AIPerfMultiError`
- Contains `HookError` with class name, function name, and original exception

**Test: `test_multiple_hook_errors()`** (lines 317-348)
- Multiple hook failures collected
- All errors available via `exc_info.value.exceptions`

**Test: `test_hook_execution_continues_after_error()`** (lines 406-430)
- Successful hooks still execute even if one fails
- Proves error collection, not short-circuit

---

## 11. Exception Types

### HookError

Used to wrap hook execution exceptions with context:
```python
HookError(class_name, hook_func_name, original_exception)
```

Properties:
- `hook_class_name` - Class containing the hook
- `hook_func_name` - Name of the hook method
- `exception` - Original exception that occurred

### UnsupportedHookError

Raised when:
- Using a hook decorator on a class that doesn't provide that hook type
- Attaching a hook type that's not provided

### AIPerfMultiError

Wraps multiple exceptions:
- `exceptions: list[Exception]` - List of all collected errors
- Used by `run_hooks()` to report all hook failures

---

## 12. Hook Parameter Resolution

### Parameter Types

1. **Static Parameters**
```python
@on_message(MessageType.STATUS, MessageType.ERROR)
async def handle_message(self, message):
    pass
```

2. **Callable Parameters (Dynamic)**
```python
@on_message(lambda self: [MessageType.STATUS] if self.debug else [MessageType.ERROR])
async def handle_message(self, message):
    pass
```

### Resolution Process (lines 96-110)

```python
def resolve_params(self, self_obj: SelfT) -> HookParamsT | None:
    if self.params is None:
        return None

    # Handle tuple with single callable (from *args)
    if isinstance(self.params, Iterable) and len(self.params) == 1 and callable(self.params[0]):
        return self.params[0](self_obj)

    # Handle single callable
    if callable(self.params):
        return self.params(self_obj)

    # Return static params
    return self.params
```

---

## 13. Practical Examples

### Example 1: Simple Lifecycle Hooks

```python
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.hooks import on_init, on_start, on_stop

class MyService(AIPerfLifecycleMixin):
    @on_init
    async def _init_plugin(self):
        self.logger.info("Initializing...")

    @on_start
    async def _start_plugin(self):
        self.logger.info("Starting...")

    @on_stop
    async def _stop_plugin(self):
        self.logger.info("Stopping...")

# Usage
service = MyService()
await service.initialize()
await service.start()
await service.stop()
```

### Example 2: Background Tasks

```python
from aiperf.common.hooks import background_task
from aiperf.common.mixins import AIPerfLifecycleMixin

class DataCollector(AIPerfLifecycleMixin):
    @background_task(interval=5.0, immediate=True)
    async def _collect_data(self):
        # Runs every 5 seconds, starting immediately
        await self._fetch_and_process()

    async def _fetch_and_process(self):
        pass

# Automatically started/stopped with lifecycle
collector = DataCollector()
await collector.initialize()
await collector.start()  # Background task starts automatically
await collector.stop()   # Background task stops automatically
```

### Example 3: Custom Hooks

```python
from aiperf.common.hooks import provides_hooks, on_message, AIPerfHook
from aiperf.common.mixins import HooksMixin

@provides_hooks(AIPerfHook.ON_MESSAGE)
class MessageHandler(HooksMixin):
    @on_message(MessageType.STATUS)
    async def _handle_status(self, message):
        self.logger.info(f"Status: {message.status}")

handler = MessageHandler()
# Get registered hooks
hooks = handler.get_hooks(AIPerfHook.ON_MESSAGE)
```

### Example 4: Hook Attachment at Runtime

```python
handler = MessageHandler()

async def dynamic_handler(message):
    print(f"Dynamic: {message}")

handler.attach_hook(
    AIPerfHook.ON_MESSAGE,
    dynamic_handler,
    params=(MessageType.ERROR,)
)
```

---

## 14. Real-World Usage in AIPerf

### TelemetryManager (gpu_telemetry_manager.py)

```python
@provides_hooks(AIPerfHook.ON_INIT, AIPerfHook.ON_COMMAND)
class TelemetryManager(BaseComponentService):

    @on_init
    async def _initialize(self) -> None:
        """Initialize telemetry manager."""
        pass

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(self, message):
        """Configure collectors based on command."""
        # Setup collectors...

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message):
        """Start all collectors."""
        # Start collectors...
```

---

## 15. Key Design Principles

1. **Inheritance-Based Registration**
   - Hooks collected from entire MRO
   - Base class hooks run before subclass hooks
   - No manual registration needed

2. **Type Safety**
   - Must declare hooks with `@provides_hooks`
   - Runtime validation prevents misuse
   - Clear error messages guide developers

3. **Error Resilience**
   - All exceptions collected and reported
   - Execution continues despite failures
   - Detailed error context preserved

4. **Lazy Evaluation**
   - Parameters can be dynamic via callables
   - Resolved at execution time
   - Supports context-dependent behavior

5. **Async/Sync Agnostic**
   - Hook functions can be sync or async
   - Sync functions run in thread pool
   - Transparent to caller

6. **Composition**
   - HooksMixin combined with other mixins
   - AIPerfLifecycleMixin uses hooks internally
   - Clean separation of concerns

---

## 16. Summary Table

| Feature | Location | Details |
|---------|----------|---------|
| **Hook Enum** | hooks.py:41-57 | 15 different hook types |
| **Core Mixin** | hooks_mixin.py | Registration, validation, execution |
| **Lifecycle** | aiperf_lifecycle_mixin.py | State machine with 5 lifecycle hooks |
| **Logger** | aiperf_logger_mixin.py | 9 logging levels with lazy eval |
| **Validation** | hooks_mixin.py:78-94, 113 | UnsupportedHookError, type checks |
| **Error Handling** | hooks_mixin.py:181-192 | AIPerfMultiError collection |
| **Parameters** | Hook.resolve_params() | Static or dynamic resolution |
| **Execution** | run_hooks() | Sequential, reverse-capable |
| **Tests** | tests/test_hooks.py | 14+ comprehensive tests |

