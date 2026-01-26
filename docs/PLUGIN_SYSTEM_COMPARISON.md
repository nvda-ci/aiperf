<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Python Plugin System Deep Dive Comparison

A comprehensive analysis of AIPerf's plugin system compared to major Python plugin frameworks.

**Document Version**: 1.0
**Date**: January 2026
**Author**: Auto-generated comparison

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Systems Compared](#systems-compared)
3. [Feature Matrix](#feature-matrix)
4. [Detailed Analysis](#detailed-analysis)
   - [AIPerf Plugin System](#aiperf-plugin-system)
   - [Pluggy (pytest)](#pluggy-pytest)
   - [Stevedore (OpenStack)](#stevedore-openstack)
   - [Raw Entry Points](#raw-entry-points-importlibmetadata)
   - [Yapsy](#yapsy)
5. [Architecture Comparison](#architecture-comparison)
6. [Code Examples](#code-examples)
7. [Performance Considerations](#performance-considerations)
8. [When to Use What](#when-to-use-what)
9. [Sources](#sources)

---

## Executive Summary

| System | Best For | Discovery | Conflict Resolution | Type Safety |
|--------|----------|-----------|---------------------|-------------|
| **AIPerf** | Factory pattern (select ONE impl) | YAML + entry points | Priority-based | Excellent (overloads) |
| **Pluggy** | Hook chains (call ALL impls) | Decorators | Order-based | None |
| **Stevedore** | Managed entry points | Entry points only | First wins / custom | None |
| **Raw Entry Points** | Simple discovery | Entry points | None | None |
| **Yapsy** | File-based plugins | File system scan | None | None |

**AIPerf's Unique Position**: A YAML-first, type-safe factory system with priority-based conflict resolution. Most similar to Stevedore's `DriverManager` but with better developer experience and conflict handling.

---

## Systems Compared

### 1. AIPerf Plugin System
- **Version Analyzed**: As of January 2026 (current codebase)
- **Files**: `_plugin_registry.py` (537 lines), `plugins.py` (575 lines), `types.py` (203 lines)
- **Dependencies**: Pydantic, ruamel.yaml, importlib.metadata

### 2. Pluggy
- **Version**: 1.6.x (latest stable)
- **Repository**: [github.com/pytest-dev/pluggy](https://github.com/pytest-dev/pluggy)
- **Used By**: pytest, tox, devpi, Datasette

### 3. Stevedore
- **Version**: 5.6.x (latest stable)
- **Repository**: [opendev.org/openstack/stevedore](https://opendev.org/openstack/stevedore)
- **Used By**: OpenStack projects, Avocado framework

### 4. Raw Entry Points
- **Source**: Python standard library (`importlib.metadata`)
- **Version**: Python 3.10+ (non-provisional)

### 5. Yapsy
- **Version**: 1.12.x
- **Repository**: [github.com/tibonihoo/yapsy](https://github.com/tibonihoo/yapsy)

---

## Feature Matrix

### Discovery & Registration

| Feature | AIPerf | Pluggy | Stevedore | Entry Points | Yapsy |
|---------|--------|--------|-----------|--------------|-------|
| **Entry point discovery** | Yes | No* | Yes | Yes | No |
| **File-based discovery** | No | No | No | No | Yes |
| **Decorator registration** | No | Yes (`@hookimpl`) | No | No | No |
| **YAML manifest** | Yes | No | No | No | Yes (`.yapsy-plugin`) |
| **Programmatic registration** | Yes (`register()`) | Yes (`pm.register()`) | No | No | No |
| **Auto-discovery on import** | Yes | No | Optional | No | No |

*\*Pluggy uses decorator-based registration (`@hookimpl`) and explicit `pm.register()`. While you can integrate entry points externally, Pluggy itself has no built-in entry point discovery.*

### Loading & Instantiation

| Feature | AIPerf | Pluggy | Stevedore | Entry Points | Yapsy |
|---------|--------|--------|-----------|--------------|-------|
| **Lazy loading** | Yes (on `get_class()`) | No (immediate) | Optional | Manual | No |
| **Cached after load** | Yes (`loaded_class`) | N/A | Yes | Manual | Yes |
| **Invoke on load** | No | N/A | Yes (`invoke_on_load`) | Manual | Yes |
| **Validation without import** | Yes (AST-based) | No | No | No | No |

### Conflict Resolution

| Feature | AIPerf | Pluggy | Stevedore | Entry Points | Yapsy |
|---------|--------|--------|-----------|--------------|-------|
| **Has conflict resolution** | Yes | No (all run) | Partial | No | No |
| **Priority-based** | Yes | No | No | No | No |
| **Built-in vs external** | Yes (external wins) | N/A | No | No | No |
| **Custom resolver** | No | N/A | Yes (`conflict_resolver`) | No | No |
| **First-wins** | Fallback | N/A | Default | Default | Default |

### Type Safety & IDE Support

| Feature | AIPerf | Pluggy | Stevedore | Entry Points | Yapsy |
|---------|--------|--------|-----------|--------------|-------|
| **Return type hints** | Yes (21 overloads) | No | No | No | No |
| **Protocol enforcement** | Yes (categories.yaml) | No | No | No | Yes (IPlugin) |
| **Schema validation** | Yes (Pydantic + JSON Schema) | No | No | No | No |
| **IDE autocomplete** | Excellent | Poor | Poor | Poor | Poor |

### Advanced Features

| Feature | AIPerf | Pluggy | Stevedore | Entry Points | Yapsy |
|---------|--------|--------|-----------|--------------|-------|
| **Hook wrappers** | No | Yes | No | No | No |
| **Hook chains** | No | Yes | Yes (HookManager) | No | No |
| **Execution ordering** | No | Yes (tryfirst/trylast) | No | No | No |
| **Subset calling** | No | Yes | Yes (filter func) | No | No |
| **Dynamic enum generation** | Yes (`create_enum()`) | No | No | No | No |
| **Reverse lookup (class→name)** | Yes | No | No | No | No |
| **URL scheme detection** | Yes | No | No | No | No |
| **Package metadata** | Yes | No | Yes | Yes | No |
| **Testing utilities** | Yes (`reset()`) | Yes | Yes (`make_test_instance`) | No | No |

---

## Detailed Analysis

### AIPerf Plugin System

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      plugins.py (Facade)                     │
│  - 21 type-safe @overload signatures                        │
│  - Module-level functions delegating to singleton           │
├─────────────────────────────────────────────────────────────┤
│                  _plugin_registry.py (Core)                  │
│  - PluginRegistry(Singleton)                                │
│  - YAML loading, entry point discovery                      │
│  - Priority-based conflict resolution                       │
├─────────────────────────────────────────────────────────────┤
│                     types.py (Models)                        │
│  - TypeEntry (frozen Pydantic model)                        │
│  - CategoryMetadata (TypedDict)                             │
│  - PluginError, TypeNotFoundError                           │
├─────────────────────────────────────────────────────────────┤
│                   schema/ (Validation)                       │
│  - PluginsFile, TypeSpec, PackageInfo                       │
│  - JSON Schema generation for IDE support                   │
└─────────────────────────────────────────────────────────────┘
```

#### Key Data Structures

**TypeEntry** (types.py:62-99):
```python
class TypeEntry(BaseModel):
    model_config = ConfigDict(frozen=True)  # Immutable

    category: str      # e.g., "endpoint"
    name: str          # e.g., "chat"
    package: str       # e.g., "aiperf"
    class_path: str    # e.g., "aiperf.endpoints.chat:ChatEndpoint"
    priority: int      # Conflict resolution (higher wins)
    description: str   # Human-readable
    metadata: dict     # Category-specific config
    loaded_class: type | None  # Cached after load()
```

**Conflict Resolution Algorithm** (_plugin_registry.py:495-518):
```
1. If new.priority > existing.priority → new wins
2. If new.priority < existing.priority → existing wins
3. If equal priority:
   a. If new is external AND existing is built-in → new wins
   b. If new is built-in AND existing is external → existing wins
   c. Otherwise → first registered wins (logs warning)
```

#### Unique Features

1. **AST-based Validation** (types.py:145-203)
   - Validates class exists without importing
   - Parses Python source with `ast.parse()`
   - Checks for ClassDef or ImportFrom nodes
   - No side effects from plugin code

2. **Generated Type Overloads** (plugins.py:86-218)
   - 21 `@overload` signatures auto-generated
   - Each category returns its specific Protocol type
   - IDE knows `get_class(ENDPOINT, 'chat')` returns `type[EndpointProtocol]`

3. **WeakKeyDictionary for Reverse Lookup** (_plugin_registry.py:46)
   - Maps loaded classes back to registered names
   - Allows garbage collection of unused classes
   - Enables `find_registered_name(category, cls)`

4. **Dynamic Enum Generation** (plugins.py:489-527)
   - Creates StrEnum from registered types
   - Automatically updates when plugins added
   - Picklable (sets correct `__module__`)

---

### Pluggy (pytest)

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PluginManager                             │
│  - pm.register(plugin) / pm.unregister(plugin)              │
│  - pm.add_hookspecs(module)                                 │
│  - pm.hook.myhook(arg=value)                                │
├─────────────────────────────────────────────────────────────┤
│                     Hook System                              │
│  - @hookspec - defines the contract                         │
│  - @hookimpl - implements the contract                      │
│  - Results collected in list (or first non-None)            │
├─────────────────────────────────────────────────────────────┤
│                    Wrappers                                  │
│  - @hookimpl(wrapper=True) - new style                      │
│  - Yield-based: pre-hook → result = yield → post-hook       │
└─────────────────────────────────────────────────────────────┘
```

#### Key Concepts

**Hook Specification**:
```python
import pluggy
hookspec = pluggy.HookspecMarker("myproject")

class MySpec:
    @hookspec
    def process_item(self, item):
        """Process an item and return result."""

    @hookspec(firstresult=True)
    def get_config(self):
        """Return config (stops at first non-None)."""

    @hookspec(historic=True)
    def on_startup(self):
        """Called for late-registered plugins too."""
```

**Hook Implementation**:
```python
hookimpl = pluggy.HookimplMarker("myproject")

class MyPlugin:
    @hookimpl
    def process_item(self, item):
        return item.upper()

    @hookimpl(tryfirst=True)
    def process_item_priority(self, item):
        """Runs before other implementations."""

    @hookimpl(wrapper=True)
    def process_item_wrapper(self, item):
        print("Before")
        result = yield  # Other impls run here
        print("After")
        return result
```

#### Execution Model

**Default Behavior**: All implementations called, results collected in list.

**Execution Order**:
1. `tryfirst=True` implementations (LIFO within group)
2. Normal implementations (LIFO registration order)
3. `trylast=True` implementations (LIFO within group)
4. Wrappers execute around everything (outermost first)

**Result Handling**:
- Normal: Returns `list[result]` (None values excluded)
- `firstresult=True`: Returns single value, stops at first non-None
- Wrappers can modify/replace results

#### Limitations vs AIPerf

| Aspect | Pluggy | AIPerf |
|--------|--------|--------|
| Use case | Call ALL plugins | Select ONE plugin |
| Return type | `list[Any]` or `Any` | `type[Protocol]` |
| Conflict handling | All run | Priority resolution |
| Validation | None | AST + Schema |

---

### Stevedore (OpenStack)

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Manager Classes                           │
├──────────────────┬──────────────────────────────────────────┤
│ DriverManager    │ Load ONE extension by name               │
│ ExtensionManager │ Load ALL extensions in namespace         │
│ NamedExtManager  │ Load SPECIFIC extensions by name list    │
│ EnabledExtMgr    │ Load extensions passing check_func       │
│ HookManager      │ Multiple impls of same hook name         │
│ DispatchExtMgr   │ Load all, filter at call time            │
└──────────────────┴──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              importlib.metadata.entry_points()               │
│  - Namespace: "myapp.drivers"                               │
│  - Entry: "chat = myapp.drivers.chat:ChatDriver"            │
└─────────────────────────────────────────────────────────────┘
```

#### Manager Comparison

**DriverManager** (closest to AIPerf):
```python
from stevedore import driver

mgr = driver.DriverManager(
    namespace='myapp.drivers',
    name='chat',
    invoke_on_load=True,
    invoke_args=(config,),
    on_load_failure_callback=handle_error,
)
result = mgr.driver.process(data)
```

**ExtensionManager**:
```python
from stevedore import extension

mgr = extension.ExtensionManager(
    namespace='myapp.processors',
    invoke_on_load=True,
    invoke_args=(config,),
)
# Call all extensions
results = mgr.map(lambda ext: ext.obj.process(data))
```

**HookManager**:
```python
from stevedore import hook

mgr = hook.HookManager(
    namespace='myapp.hooks',
    name='on_save',
    invoke_on_load=True,
)
# Returns list of Extension objects
for ext in mgr['on_save']:
    ext.obj.execute()
```

#### Key Parameters (Common to All Managers)

| Parameter | Type | Description |
|-----------|------|-------------|
| `namespace` | str | Entry point group name |
| `invoke_on_load` | bool | Call plugin after loading |
| `invoke_args` | tuple | Positional args for invocation |
| `invoke_kwds` | dict | Keyword args for invocation |
| `on_load_failure_callback` | callable | Error handler |
| `propagate_map_exceptions` | bool | Raise or log exceptions |

#### Comparison with AIPerf

| Aspect | Stevedore DriverManager | AIPerf |
|--------|------------------------|--------|
| Registration | Entry points only | YAML + entry points |
| Loading | `mgr.driver` | `plugins.get_class()` |
| Conflict resolution | First wins / custom resolver | Priority-based |
| Type hints | None | 21 overloads |
| Schema validation | None | Pydantic |
| Invoke on load | Built-in | Manual |

---

### Raw Entry Points (importlib.metadata)

#### Basic Usage

```python
from importlib.metadata import entry_points

# Python 3.10+ (selectable entry points)
eps = entry_points(group='myapp.plugins')

for ep in eps:
    print(f"Name: {ep.name}")
    print(f"Value: {ep.value}")  # "module:attribute"
    print(f"Group: {ep.group}")

    # Load the actual object
    plugin_class = ep.load()
    instance = plugin_class()
```

#### Registration (pyproject.toml)

```toml
[project.entry-points."myapp.plugins"]
chat = "myapp.plugins.chat:ChatPlugin"
completions = "myapp.plugins.completions:CompletionsPlugin"
```

#### Limitations

- No conflict resolution
- No lazy loading (manual)
- No validation
- No type safety
- No metadata beyond name/value

#### When to Use

- Simple plugin discovery
- You implement everything else
- Minimal dependencies required

---

### Yapsy

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PluginManager                             │
│  - setPluginPlaces(['/path/to/plugins'])                    │
│  - collectPlugins()                                         │
│  - getAllPlugins() / getPluginByName()                      │
├─────────────────────────────────────────────────────────────┤
│                 Plugin Detection                             │
│  - Scans directories for .yapsy-plugin files                │
│  - INI format: [Core] Name=X, Module=path                   │
│  - Optional: [Documentation], custom sections               │
├─────────────────────────────────────────────────────────────┤
│                    IPlugin                                   │
│  - Base class plugins must inherit                          │
│  - activate() / deactivate() lifecycle                      │
│  - Optional category filtering via subclasses               │
└─────────────────────────────────────────────────────────────┘
```

#### Plugin Definition

**my_plugin.yapsy-plugin**:
```ini
[Core]
Name = My Plugin
Module = my_plugin

[Documentation]
Author = John Doe
Version = 1.0
Description = Does something useful
```

**my_plugin.py**:
```python
from yapsy.IPlugin import IPlugin

class MyPlugin(IPlugin):
    def activate(self):
        print("Plugin activated")

    def deactivate(self):
        print("Plugin deactivated")

    def process(self, data):
        return data.upper()
```

#### Usage

```python
from yapsy.PluginManager import PluginManager

manager = PluginManager()
manager.setPluginPlaces(["/path/to/plugins"])
manager.collectPlugins()

for plugin in manager.getAllPlugins():
    print(f"Found: {plugin.name}")
    plugin.plugin_object.activate()
    result = plugin.plugin_object.process("hello")
```

#### Comparison with AIPerf

| Aspect | Yapsy | AIPerf |
|--------|-------|--------|
| Discovery | File system scan | Entry points + YAML |
| Registration | INI files | YAML manifest |
| Base class required | Yes (IPlugin) | No (Protocol-based) |
| Lifecycle hooks | Yes (activate/deactivate) | No |
| Distribution | Directory-based | Package-based |

---

## Architecture Comparison

### Registration Flow

**AIPerf**:
```
plugins.yaml → Pydantic validation → TypeEntry → _types dict
     ↓
entry_points → discover_plugins() → load_registry() → merge
```

**Pluggy**:
```
@hookimpl decorator → pm.register(plugin) → hook._hookimpls list
```

**Stevedore**:
```
pyproject.toml → entry_points() → DriverManager → Extension
```

### Retrieval Flow

**AIPerf**:
```
plugins.get_class(category, name)
    → _registry.get_class()
    → _get_class_by_name()
    → TypeEntry.load()  # lazy
    → importlib.import_module()
    → Return cached class
```

**Pluggy**:
```
pm.hook.myhook(arg=value)
    → HookCaller.__call__()
    → Call all hookimpls in order
    → Collect results in list
    → Return list (or first result)
```

**Stevedore**:
```
DriverManager(namespace, name)
    → entry_points(group=namespace)
    → Find matching entry point
    → ep.load()
    → Optional: invoke_on_load
    → mgr.driver
```

### Memory Model

**AIPerf**:
- Singleton registry (one instance)
- TypeEntry objects (frozen, cached)
- WeakKeyDictionary for reverse lookup
- Lazy loading (class loaded on first access)

**Pluggy**:
- PluginManager instance per project
- All plugins loaded at registration
- No caching beyond Python's module cache

**Stevedore**:
- Manager instance per namespace/name
- Optional caching via Extension objects
- Can reload managers for fresh state

---

## Code Examples

### Registering a Plugin

**AIPerf** (plugins.yaml):
```yaml
schema_version: "1.0"
plugin:
  name: my-plugins
  version: 1.0.0

endpoint:
  custom:
    class: my_package.endpoints:CustomEndpoint
    priority: 10
    description: My custom endpoint
    metadata:
      url_schemes: ["http", "https"]
```

**AIPerf** (programmatic):
```python
from aiperf.plugin import plugins

class CustomEndpoint:
    """My custom endpoint implementation."""
    pass

plugins.register('endpoint', 'custom', CustomEndpoint, priority=10)
```

**Pluggy**:
```python
import pluggy

hookimpl = pluggy.HookimplMarker("myapp")

class MyPlugin:
    @hookimpl
    def process_endpoint(self, request):
        return custom_response(request)

pm = pluggy.PluginManager("myapp")
pm.register(MyPlugin())
```

**Stevedore** (pyproject.toml):
```toml
[project.entry-points."myapp.endpoints"]
custom = "my_package.endpoints:CustomEndpoint"
```

### Retrieving a Plugin

**AIPerf**:
```python
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginCategory

# Type-safe: IDE knows this returns type[EndpointProtocol]
EndpointClass = plugins.get_class(PluginCategory.ENDPOINT, 'custom')
endpoint = EndpointClass(config)
```

**Pluggy**:
```python
# Call all plugins, get list of results
results = pm.hook.process_endpoint(request=req)
```

**Stevedore**:
```python
from stevedore import driver

mgr = driver.DriverManager(
    namespace='myapp.endpoints',
    name='custom',
    invoke_on_load=True,
    invoke_args=(config,),
)
endpoint = mgr.driver
```

### Listing Available Plugins

**AIPerf**:
```python
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginCategory

for entry in plugins.list_types(PluginCategory.ENDPOINT):
    print(f"{entry.name}: {entry.description}")
    print(f"  Package: {entry.package}")
    print(f"  Priority: {entry.priority}")
    print(f"  Class: {entry.class_path}")
```

**Pluggy**:
```python
for plugin in pm.get_plugins():
    name = pm.get_canonical_name(plugin)
    print(f"Plugin: {name}")
```

**Stevedore**:
```python
from stevedore import extension

mgr = extension.ExtensionManager(namespace='myapp.endpoints')
for ext in mgr:
    print(f"{ext.name}: {ext.entry_point}")
```

---

## Performance Considerations

### Startup Time

| System | Behavior | Impact |
|--------|----------|--------|
| **AIPerf** | Loads YAML, discovers entry points on import | ~100-500ms |
| **Pluggy** | No auto-loading | Minimal |
| **Stevedore** | Loads on manager creation | ~50-200ms per manager |
| **Entry Points** | Loads on `entry_points()` call | ~10-50ms |

### Runtime Performance

| Operation | AIPerf | Pluggy | Stevedore |
|-----------|--------|--------|-----------|
| Get single plugin | O(1) dict lookup | N/A | O(n) search |
| Call all plugins | N/A | O(n) calls | O(n) calls |
| First load | Import + cache | Already loaded | Import |
| Subsequent loads | O(1) cached | O(1) | O(1) cached |

### Memory Usage

| System | Characteristics |
|--------|-----------------|
| **AIPerf** | TypeEntry objects (~200 bytes each), lazy class loading |
| **Pluggy** | Full plugin objects always in memory |
| **Stevedore** | Extension objects, optional caching |

---

## When to Use What

### Use AIPerf Plugin System When:
- You need to select ONE implementation from a category
- Type safety and IDE support are important
- You want declarative YAML configuration
- Plugin conflicts need intelligent resolution
- You're building a factory-pattern system

### Use Pluggy When:
- All plugins should be called (hook pattern)
- You need execution ordering (tryfirst/trylast)
- Wrappers/middleware pattern is needed
- Building pytest-like extensible tools

### Use Stevedore When:
- You want managed entry points
- Need multiple manager patterns (Driver, Hook, Named, etc.)
- OpenStack ecosystem integration
- Don't need type safety

### Use Raw Entry Points When:
- Simple discovery is sufficient
- You'll implement everything else
- Minimal dependencies required
- Learning/prototyping

### Use Yapsy When:
- File-based plugin distribution
- Plugins as directories (not packages)
- Need lifecycle hooks (activate/deactivate)
- Simple INI configuration preferred

---

## Sources

### Official Documentation
- [Pluggy Documentation](https://pluggy.readthedocs.io/en/latest/)
- [Stevedore Documentation](https://docs.openstack.org/stevedore/latest/)
- [Python Packaging Guide - Plugins](https://packaging.python.org/guides/creating-and-discovering-plugins/)
- [Yapsy Documentation](https://yapsy.readthedocs.io/)
- [importlib.metadata](https://docs.python.org/3/library/importlib.metadata.html)

### Repositories
- [Pluggy GitHub](https://github.com/pytest-dev/pluggy)
- [Stevedore OpenDev](https://opendev.org/openstack/stevedore)
- [Yapsy GitHub](https://github.com/tibonihoo/yapsy)

### Additional Resources
- [Plugin Systems Overview - Sedimental](https://sedimental.org/plugin_systems.html)
- [Abilian Plugin Comparison](https://lab.abilian.com/Tech/Programming%20Techniques/Plugins/)
- [Entry Points Specification](https://packaging.python.org/specifications/entry-points/)

---

## Appendix: AIPerf Plugin System API Reference

### Module: `aiperf.plugin.plugins`

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_class` | `(category, name) -> type` | Get plugin class by name (21 typed overloads) |
| `list_types` | `(category) -> list[TypeEntry]` | List all types in category |
| `list_categories` | `(include_internal=True) -> list[str]` | List all categories |
| `list_packages` | `(builtin_only=False) -> list[str]` | List loaded packages |
| `get_package_metadata` | `(name) -> PackageInfo` | Get package info |
| `get_category_metadata` | `(category) -> CategoryMetadata` | Get category info |
| `is_internal_category` | `(category) -> bool` | Check if category is internal |
| `find_registered_name` | `(category, cls) -> str | None` | Reverse lookup |
| `validate_all` | `(check_class=False) -> dict` | Validate without importing |
| `register` | `(category, name, cls, priority=0)` | Programmatic registration |
| `create_enum` | `(category, name) -> type` | Generate StrEnum from types |
| `detect_type_from_url` | `(category, url) -> str` | Match URL scheme to type |
| `load_registry` | `(path)` | Load additional YAML |
| `reset` | `()` | Reset for testing |

### Class: `TypeEntry`

| Field | Type | Description |
|-------|------|-------------|
| `category` | `str` | Category identifier |
| `name` | `str` | Type name |
| `package` | `str` | Providing package |
| `class_path` | `str` | `module:ClassName` |
| `priority` | `int` | Conflict resolution |
| `description` | `str` | Human-readable |
| `metadata` | `dict` | Category-specific |
| `loaded_class` | `type | None` | Cached class |
| `is_builtin` | `bool` (property) | If package == "aiperf" |

| Method | Description |
|--------|-------------|
| `load() -> type` | Import and cache class |
| `validate(check_class=False) -> (bool, str | None)` | Validate without import |
