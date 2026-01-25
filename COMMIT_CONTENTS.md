<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Commit Contents: Plugin System

This commit adds **33 files** implementing a unified plugin system for AIPerf.

---

## Factory Integration (Included)

The commit modifies `src/aiperf/common/factories.py` to delegate **17 factories** to `UnifiedPluginRegistry`:

| Factory | Registry Protocol |
|---------|-------------------|
| `AIPerfUIFactory` | `ui` |
| `CommunicationClientFactory` | `communication_client` |
| `CommunicationFactory` | `communication` |
| `ComposerFactory` | `dataset_composer` |
| `ConsoleExporterFactory` | `console_exporter` |
| `CreditIssuingStrategyFactory` | `timing_strategy` |
| `CustomDatasetFactory` | `custom_dataset_loader` |
| `DataExporterFactory` | `data_exporter` |
| `DatasetSamplingStrategyFactory` | `dataset_sampler` |
| `EndpointFactory` | `endpoint` |
| `ServiceFactory` | `service` |
| `ServiceManagerFactory` | `service_manager` |
| `RecordProcessorFactory` | `record_processor` |
| `ResultsProcessorFactory` | `results_processor` |
| `RequestRateGeneratorFactory` | `request_rate_generator` |
| `TransportFactory` | `transport` |
| `ZMQProxyFactory` | `zmq_proxy` |

**Example integration:**
```python
class EndpointFactory(...):
    """DEPRECATED: This factory now delegates to UnifiedPluginRegistry.
    Use get_plugin_registry().get('endpoint', name) instead.
    """

    @classmethod
    def create_instance(cls, class_type, **kwargs):
        from aiperf.plugin.plugin_registry import get_plugin_registry
        registry = get_plugin_registry()
        EndpointClass = registry.get("endpoint", impl_name)
        return EndpointClass(**kwargs)
```

All factories emit `DeprecationWarning` to encourage direct registry usage.

---

## What Was Removed (Integration Points)

The original commit (ae990b627) had **186 additional files** for the timing refactor. These were reverted to keep this commit focused on the plugin infrastructure only.

### Phase Hook Integration (Removed)

The timing refactor included a **PhaseHookRegistry** that allowed plugins to receive phase lifecycle events:

```
┌──────────────────────────────────────────────────┐
│ PhaseHookRegistry (Global Singleton)             │
│ ──────────────────────────────────               │
│ Callback Lists:                                  │
│ ├─ phase_start_callbacks[]                       │
│ ├─ phase_complete_callbacks[]                    │
│ ├─ phase_sending_complete_callbacks[]            │
│ └─ phase_timeout_callbacks[]                     │
└──────────────────────────────────────────────────┘
```

**Flow:**
```
Plugin startup
    ↓
hook.register_with_global_registry()
    ↓
PhaseHookRegistry.register_phase_start(callback)
    ↓
Later: PhaseOrchestrator.execute_phase()
    ↓
callbacks = PhaseHookRegistry.get_phase_start_callbacks()
    ↓
for callback in callbacks:
    await callback(phase, tracker)
```

### Removed Files That Used Plugin System

| Category | Files | Description |
|----------|-------|-------------|
| `src/aiperf/timing/phase_hooks.py` | 1 | Global PhaseHookRegistry singleton |
| `src/aiperf/timing/phase_lifecycle_hooks.py` | 1 | BasePhaseLifecycleHook base class |
| `src/aiperf/timing/phase_orchestrator.py` | 1 | Orchestrator that calls hooks |
| `src/aiperf/timing/phase_executor.py` | 1 | Phase execution with hook support |
| `src/aiperf/timing/*.py` | 28 | Other timing refactor files |

### Example Plugin Usage (Still in this commit)

The example plugin in `examples/aiperf-example-plugin/` demonstrates hooks:

```python
class ExampleLoggingHook(BasePhaseLifecycleHook):
    async def on_phase_start(self, phase, tracker):
        # Called when phase starts
        await self._write_log(f"PHASE_START: {phase}")

    async def on_phase_complete(self, phase, tracker):
        # Called when phase completes
        await self._write_log(f"PHASE_COMPLETE: {phase}")
```

**Note:** The example plugin code references `BasePhaseLifecycleHook` which was in the removed timing refactor. This will need to be updated or the timing refactor re-added for the example to work.

---

## Core Plugin System (5 files)

| File | Description |
|------|-------------|
| `src/aiperf/common/plugin_registry.py` | **UnifiedPluginRegistry** - Core registry with lazy loading, entry point discovery, priority-based conflict resolution |
| `src/aiperf/common/plugin_registry.pyi` | Type stubs for the plugin registry |
| `src/aiperf/common/plugin_loader.py` | **PluginLoader** - Plugin initialization, validation, and lifecycle management |
| `src/aiperf/common/factories.py` | **Modified** - 17 factories now delegate to UnifiedPluginRegistry with deprecation warnings |
| `src/aiperf/registry.yaml` | Built-in registry defining all core implementations (endpoints, strategies, services, etc.) |

---

## Plugin CLI (2 files)

| File | Description |
|------|-------------|
| `src/aiperf/cli_commands/__init__.py` | CLI commands module init |
| `src/aiperf/cli_commands/plugins_cli.py` | CLI commands: `aiperf plugins list/show/validate/protocols/info` |

---

## Example Plugin (14 files)

A complete example plugin demonstrating best practices:

### Source Code
| File | Description |
|------|-------------|
| `examples/aiperf-example-plugin/aiperf_example_plugin/__init__.py` | Package init |
| `examples/aiperf-example-plugin/aiperf_example_plugin/hooks.py` | `ExampleLoggingHook`, `ExampleMetricsCollectorHook` |
| `examples/aiperf-example-plugin/aiperf_example_plugin/processors.py` | Example processors |
| `examples/aiperf-example-plugin/aiperf_example_plugin/registry.yaml` | Plugin's registry |
| `examples/aiperf-example-plugin/pyproject.toml` | Package config with entry points |

### Tests
| File | Description |
|------|-------------|
| `examples/aiperf-example-plugin/tests/__init__.py` | Test package init |
| `examples/aiperf-example-plugin/tests/test_hooks.py` | Hook tests |
| `examples/aiperf-example-plugin/tests/test_processors.py` | Processor tests |

### Documentation
| File | Description |
|------|-------------|
| `examples/aiperf-example-plugin/README.md` | Main readme |
| `examples/aiperf-example-plugin/DEVELOPER_GUIDE.md` | Developer guide |
| `examples/aiperf-example-plugin/INDEX.md` | Documentation index |
| `examples/aiperf-example-plugin/PLUGIN_STRUCTURE.md` | Plugin structure guide |
| `examples/aiperf-example-plugin/USAGE_EXAMPLES.md` | Usage examples |
| `examples/aiperf-example-plugin/LICENSE` | License file |

---

## Scripts (1 file)

| File | Description |
|------|-------------|
| `scripts/validate_registry.py` | Registry validation script (YAML syntax, schema, class paths) |

---

## Tests (4 files)

| File | Description |
|------|-------------|
| `tests/unit/common/test_plugin_registry.py` | UnifiedPluginRegistry unit tests |
| `tests/unit/common/test_plugin_loader.py` | PluginLoader unit tests |
| `tests/unit/common/test_bootstrap_plugin_registry.py` | Bootstrap integration tests |
| `tests/unit/cli/test_plugins_cli.py` | Plugin CLI tests |

---

## Design Documentation (7 files)

| File | Description |
|------|-------------|
| `PLUGIN_SYSTEM_DESIGN.md` | Overall plugin system design |
| `LAZY_PLUGIN_REGISTRY_DESIGN.md` | Lazy loading design |
| `YAML_PLUGIN_REGISTRY_DESIGN.md` | YAML registry format design |
| `docs/CLI_PLUGINS.md` | Plugin CLI documentation |
| `docs/CLI_PLUGINS_QUICKSTART.md` | Quick start guide |
| `docs/CLI_PLUGINS_SUMMARY.md` | CLI summary |
| `docs/PLUGIN_BEST_PRACTICES.md` | Best practices guide |

---

## Summary

```
33 files changed, 19318 insertions(+), 47 deletions(-)

Core System:     5 files (includes factories.py)
CLI:             2 files
Example Plugin: 14 files
Scripts:         1 file
Tests:           4 files
Documentation:   7 files
```

---

## Dependencies / Next Steps

The example plugin (`examples/aiperf-example-plugin/`) has dependencies on code that was removed:

1. **`BasePhaseLifecycleHook`** - Base class from `src/aiperf/timing/phase_lifecycle_hooks.py`
2. **`PhaseHookRegistry`** - Global registry from `src/aiperf/timing/phase_hooks.py`
3. **`CreditPhase`** enum - From timing enums

**Options:**
1. Re-add the timing refactor in a separate commit
2. Update the example plugin to not depend on timing hooks
3. Add minimal stub implementations for the hook base classes
