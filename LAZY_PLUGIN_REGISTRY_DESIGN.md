<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Lazy-Loading Plugin Registry Design

## Vision: Integrate Plugins with Factory System

**Goal:** External plugins register protocol implementations. AIPerf discovers them at startup but only loads/instantiates when actually used.

**Like:** Django apps, Stevedore, ImportLib.resources lazy loading

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION STARTUP                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─ 1. Discover plugins via entry points
         │    └─ entry_points(group='aiperf.plugins')
         │       ├─ datadog-plugin
         │       ├─ prometheus-plugin
         │       └─ custom-plugin
         │
         ├─ 2. Register plugin implementations (LAZY!)
         │    └─ For each plugin:
         │        ├─ Load plugin.register() function only
         │        ├─ Plugin declares what it provides:
         │        │  └─ PhaseHookFactory.register_lazy("datadog", EntryPoint)
         │        └─ DON'T load actual implementation yet!
         │
         └─ 3. Continue AIPerf startup
                (plugins registered but NOT loaded)

┌─────────────────────────────────────────────────────────────────┐
│                     DURING EXECUTION                             │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─ AIPerf code needs hook:
         │    hook = PhaseHookFactory.create("datadog", api_key="xxx")
         │           │
         │           ├─ LAZY LOAD: Import actual class now
         │           ├─ Instantiate with config
         │           └─ Return instance
         │
         └─ Hook is now active (loaded on-demand)
```

---

## Implementation

### 1. Lazy Plugin Registry

```python
# ============================================
# src/aiperf/common/plugin_registry.py
# ============================================

import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar('T')


@dataclass
class LazyPlugin:
    """Lazy plugin entry that loads on first access.

    Attributes:
        name: Plugin name (e.g., "datadog")
        entry_point: EntryPoint object for lazy loading
        loaded_class: Cached class after first load (None initially)
    """

    name: str
    entry_point: importlib.metadata.EntryPoint
    loaded_class: type | None = None

    def load(self) -> type:
        """Load plugin class (lazy - only once).

        Returns:
            Plugin class (cached after first load)
        """
        if self.loaded_class is None:
            # Load from entry point (imports module now)
            self.loaded_class = self.entry_point.load()

        return self.loaded_class

    def create_instance(self, **kwargs) -> Any:
        """Load and instantiate plugin.

        Args:
            **kwargs: Arguments passed to plugin __init__

        Returns:
            Plugin instance
        """
        plugin_class = self.load()
        return plugin_class(**kwargs)


class LazyPluginRegistry:
    """Registry for lazy-loading plugins.

    Plugins are discovered at startup but only loaded when accessed.
    Integrates with AIPerf's factory pattern.
    """

    def __init__(self):
        # Lazy plugin storage: protocol_name → {name → LazyPlugin}
        self._registry: dict[str, dict[str, LazyPlugin]] = {}

    def discover_plugins(self, entry_point_group: str = "aiperf.plugins"):
        """Discover all plugins via entry points.

        Called once at application startup.
        Plugins are registered but NOT loaded yet.

        Args:
            entry_point_group: Entry point group to scan
        """
        entry_points = importlib.metadata.entry_points()

        # Python 3.10+
        if hasattr(entry_points, 'select'):
            plugin_eps = entry_points.select(group=entry_point_group)
        else:
            plugin_eps = entry_points.get(entry_point_group, [])

        for ep in plugin_eps:
            # Parse name: "protocol:impl" or just "impl"
            if ':' in ep.name:
                protocol_name, name = ep.name.split(':', 1)
            else:
                # Default protocol (for backward compat)
                protocol_name = 'phase_hook'
                name = ep.name

            # Create lazy plugin entry (NOT loaded yet!)
            lazy_plugin = LazyPlugin(
                name=name,
                entry_point=ep,
                loaded_class=None,  # Will load on first access
            )

            # Register to protocol
            self._registry.setdefault(protocol_name, {})[name] = lazy_plugin

    def register_lazy(
        self,
        protocol_name: str,
        name: str,
        entry_point: importlib.metadata.EntryPoint
    ):
        """Register lazy plugin entry.

        Args:
            protocol_name: Protocol/factory name (e.g., "phase_hook")
            name: Implementation name (e.g., "datadog")
            entry_point: EntryPoint for lazy loading
        """
        lazy_plugin = LazyPlugin(name, entry_point)
        self._registry.setdefault(protocol_name, {})[name] = lazy_plugin

    def get_implementation(self, protocol_name: str, name: str) -> type:
        """Get implementation class (loads lazily).

        Args:
            protocol_name: Protocol/factory name
            name: Implementation name

        Returns:
            Loaded plugin class

        Raises:
            KeyError: If protocol or implementation not found
        """
        if protocol_name not in self._registry:
            raise KeyError(f"Unknown protocol: {protocol_name}")

        if name not in self._registry[protocol_name]:
            raise KeyError(
                f"Unknown implementation '{name}' for protocol '{protocol_name}'"
            )

        lazy_plugin = self._registry[protocol_name][name]
        return lazy_plugin.load()  # Lazy load here!

    def create_instance(
        self,
        protocol_name: str,
        name: str,
        **kwargs
    ) -> Any:
        """Create instance of plugin (loads lazily).

        Args:
            protocol_name: Protocol/factory name
            name: Implementation name
            **kwargs: Arguments for plugin __init__

        Returns:
            Plugin instance
        """
        lazy_plugin = self._registry[protocol_name][name]
        return lazy_plugin.create_instance(**kwargs)

    def list_implementations(self, protocol_name: str) -> list[str]:
        """List available implementations for protocol.

        Does NOT load them - just lists names.

        Args:
            protocol_name: Protocol/factory name

        Returns:
            List of implementation names
        """
        return list(self._registry.get(protocol_name, {}).keys())

    def is_registered(self, protocol_name: str, name: str) -> bool:
        """Check if implementation is registered (doesn't load it).

        Args:
            protocol_name: Protocol name
            name: Implementation name

        Returns:
            True if registered
        """
        return (
            protocol_name in self._registry
            and name in self._registry[protocol_name]
        )


# Global instance
_lazy_registry = LazyPluginRegistry()


def get_plugin_registry() -> LazyPluginRegistry:
    """Get global plugin registry."""
    return _lazy_registry
```

### 2. Integrate with AIPerf Factories

```python
# ============================================
# src/aiperf/common/factories.py (UPDATE EXISTING)
# ============================================

from aiperf.plugin import plugin_registry


class PhaseLifecycleHookFactory:
    """Factory for phase lifecycle hooks.

    Supports both:
    - Built-in implementations (registered via decorator)
    - Plugin implementations (registered via entry points)
    """

    _builtin_registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Register built-in implementation.

        Args:
            name: Implementation name

        Returns:
            Decorator function
        """
        def decorator(hook_class):
            cls._builtin_registry[name] = hook_class
            return hook_class
        return decorator

    @classmethod
    def create_instance(cls, name: str, **kwargs):
        """Create hook instance (checks plugins first, then built-ins).

        Lazy loads from plugin registry if needed.

        Args:
            name: Implementation name (e.g., "datadog", "logging")
            **kwargs: Arguments for hook __init__

        Returns:
            Hook instance

        Raises:
            ValueError: If implementation not found
        """
        # Check plugin registry first (lazy loading)
        if plugin_registry.is_registered('phase_hook', name):
            # Lazy load and instantiate
            return plugin_registry.create_instance('phase_hook', name, **kwargs)

        # Fall back to built-in registry
        if name in cls._builtin_registry:
            return cls._builtin_registry[name](**kwargs)

        # Not found
        available = cls.list_available()
        raise ValueError(
            f"Unknown phase hook implementation: '{name}'. "
            f"Available: {available}"
        )

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available implementations (built-in + plugins).

        Does NOT load plugins - just lists them.

        Returns:
            List of implementation names
        """
        # Built-ins
        builtin = list(cls._builtin_registry.keys())

        # Plugins (lazy - not loaded yet!)
        plugins = plugin_registry.list_implementations('phase_hook')

        return sorted(set(builtin + plugins))
```

### 3. Built-in Hooks Register with Factory

```python
# ============================================
# src/aiperf/timing/phase_lifecycle_hooks.py
# ============================================

from aiperf.common.factories import PhaseLifecycleHookFactory


# Built-in hooks register via decorator
@PhaseLifecycleHookFactory.register("logging")
class LoggingPhaseHook:
    """Built-in logging hook."""

    def __init__(self, logger):
        self._logger = logger

    async def on_phase_start(self, phase, tracker):
        self._logger.info(f"Phase {phase} started")

    async def on_phase_complete(self, phase, tracker):
        stats = tracker.create_stats()
        self._logger.info(f"Phase {phase} completed: {stats}")

    def register_with_registry(self):
        """Register callbacks with PhaseHookRegistry."""
        from aiperf.timing.phase_hooks import PhaseHookRegistry
        PhaseHookRegistry.register_phase_start(self.on_phase_start)
        PhaseHookRegistry.register_phase_complete(self.on_phase_complete)
```

### 4. External Plugin Entry Point

```python
# ============================================
# External: aiperf-datadog-plugin/setup.py
# ============================================

setup(
    name="aiperf-datadog-plugin",
    entry_points={
        # Format: "protocol:name"
        "aiperf.plugins": [
            "phase_hook:datadog = aiperf_datadog_plugin:DatadogPhaseHook"
        ]
    }
)


# ============================================
# External: aiperf_datadog_plugin/__init__.py
# ============================================

class DatadogPhaseHook:
    """Datadog monitoring hook - loaded lazily."""

    def __init__(self, api_key: str, **kwargs):
        from datadog import initialize
        initialize(api_key=api_key)
        self._api_key = api_key

    async def on_phase_complete(self, phase, tracker):
        """Send metrics to Datadog."""
        from datadog import api
        stats = tracker.create_stats()
        api.Metric.send(
            metric="aiperf.phase.duration",
            points=[(stats.end_ns, stats.duration_sec)],
            tags=[f"phase:{phase}"]
        )

    def register_with_registry(self):
        """Called by factory after instantiation to register callbacks."""
        from aiperf.timing.phase_hooks import PhaseHookRegistry
        PhaseHookRegistry.register_phase_complete(self.on_phase_complete)
```

### 5. Plugin Discovery at Startup

```python
# ============================================
# src/aiperf/cli.py or bootstrap
# ============================================

from aiperf.plugin import plugin_registry


def initialize_aiperf():
    """Initialize AIPerf with plugin discovery."""

    # Discover plugins (DOESN'T load them!)
    plugin_registry.discover_plugins(entry_point_group='aiperf.plugins')

    # Log what was found (still not loaded!)
    for protocol in ['phase_hook', 'endpoint', 'sampler']:  # etc
        impls = plugin_registry.list_implementations(protocol)
        if impls:
            print(f"Found {protocol} plugins: {impls}")

    # Continue with normal startup
    # Plugins will be loaded on-demand when factories use them
```

### 6. Config-Driven Hook Loading

```python
# ============================================
# User's config file
# ============================================
# benchmark.yaml

plugins:
  phase_hooks:  # List of hook implementations to activate
    - name: logging
      # Built-in, always available

    - name: datadog
      config:
        api_key: ${DATADOG_API_KEY}
        tags:
          - team:ml-perf

    - name: prometheus
      config:
        endpoint: http://localhost:9091


# ============================================
# AIPerf loads configured hooks
# ============================================
# src/aiperf/timing/phase_orchestrator.py or bootstrap

def load_configured_phase_hooks(config: dict) -> list:
    """Load phase hooks from config (lazy).

    Args:
        config: Config with plugins.phase_hooks list

    Returns:
        List of instantiated hooks
    """
    hooks = []

    phase_hook_configs = config.get("plugins", {}).get("phase_hooks", [])

    for hook_config in phase_hook_configs:
        name = hook_config["name"]
        init_config = hook_config.get("config", {})

        try:
            # LAZY LOAD: Only loads plugin when creating instance!
            hook_instance = PhaseLifecycleHookFactory.create_instance(
                name,
                **init_config
            )

            # Register hook's callbacks with global registry
            if hasattr(hook_instance, 'register_with_registry'):
                hook_instance.register_with_registry()

            hooks.append(hook_instance)

            print(f"✓ Loaded phase hook: {name}")

        except Exception as e:
            print(f"✗ Failed to load phase hook '{name}': {e!r}")

    return hooks


# Called during TimingManager initialization
def setup_phase_hooks(config: dict):
    """Setup phase hooks from config.

    Args:
        config: Application config
    """
    hooks = load_configured_phase_hooks(config)
    print(f"Loaded {len(hooks)} phase hook(s)")
```

---

## Complete Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│ STARTUP (Application Bootstrap)                                          │
└──────────────────────────────────────────────────────────────────────────┘

1. Plugin Discovery (Eager - At Startup)
   ├─ Scan entry_points(group='aiperf.plugins')
   ├─ Find: phase_hook:datadog, phase_hook:prometheus, endpoint:custom
   └─ Register to LazyPluginRegistry:
      └─ _registry['phase_hook'] = {
            'datadog': LazyPlugin(name='datadog', entry_point=EP, loaded_class=None),
            'prometheus': LazyPlugin(name='prometheus', entry_point=EP, loaded_class=None),
         }

   ★ Plugins discovered but NOT loaded yet! (Lazy)
   ★ No imports, no class loading, no initialization


┌──────────────────────────────────────────────────────────────────────────┐
│ CONFIGURATION PARSING                                                     │
└──────────────────────────────────────────────────────────────────────────┘

2. Read config file
   plugins:
     phase_hooks:
       - name: datadog
         config: {api_key: "xxx"}

   ★ Config parsed, but plugins still not loaded!


┌──────────────────────────────────────────────────────────────────────────┐
│ TIMING MANAGER INITIALIZATION                                             │
└──────────────────────────────────────────────────────────────────────────┘

3. TimingManager.__init__() or @on_init
   ├─ setup_phase_hooks(config)
   │  └─ For each hook in config.plugins.phase_hooks:
   │     ├─ PhaseLifecycleHookFactory.create_instance("datadog", api_key="xxx")
   │     │  ├─ Check LazyPluginRegistry first
   │     │  │  └─ Is 'datadog' registered? YES
   │     │  ├─ lazy_plugin.create_instance(api_key="xxx")
   │     │  │  ├─ lazy_plugin.load()  ◄──── LAZY LOAD HAPPENS HERE!
   │     │  │  │  └─ entry_point.load() imports DatadogPhaseHook class
   │     │  │  └─ DatadogPhaseHook(api_key="xxx")  ◄──── INSTANTIATED!
   │     │  └─ Return instance
   │     └─ hook.register_with_registry()
   │        └─ PhaseHookRegistry.register_phase_complete(hook.on_phase_complete)
   │
   └─ Hooks now active in global PhaseHookRegistry

   ★ Only configured plugins are loaded!
   ★ Only loaded when needed!
   ★ Initialized with config!


┌──────────────────────────────────────────────────────────────────────────┐
│ EXECUTION (Phase Execution)                                               │
└──────────────────────────────────────────────────────────────────────────┘

4. PhaseOrchestrator.execute_phase()
   ├─ callbacks = PhaseHookRegistry.get_phase_complete_callbacks()
   │  └─ [datadog_hook.on_phase_complete, ...]  ◄─── From loaded plugins
   │
   └─ for callback in callbacks:
      await callback(phase, tracker)  ◄─── Calls plugin!

   ★ Plugins execute at hook points
   ★ AIPerf doesn't know what plugins they are!
```

---

## Key Differences from Current

### Current (No Lazy Loading)

```python
# All hooks passed at construction
orchestrator = PhaseOrchestrator(..., hooks=[
    DatadogHook(),  # ← Loaded and instantiated already
    PrometheusHook(),
])

# Problems:
# - Who creates these instances?
# - Where does config come from?
# - How do we know what plugins to load?
```

### Proposed (Lazy Loading + Config-Driven)

```python
from aiperf.plugin import plugin_registry

# Startup: Discover (don't load)
plugin_registry.discover_plugins()  # Finds datadog, prometheus
# Plugins registered but NOT loaded!

# Later: Load configured hooks
hooks = load_configured_phase_hooks(config)
# Only loads plugins listed in config!
# Loaded with config parameters!

# PhaseOrchestrator doesn't get hooks parameter
orchestrator = PhaseOrchestrator(executor, publisher)
# Gets hooks from global PhaseHookRegistry (populated by loaded plugins)
```

---

## Integration with Existing Factories

### Pattern: Factories Check Plugin Registry First

```python
# ============================================
# Example: CreditIssuingStrategyFactory
# ============================================

class CreditIssuingStrategyFactory:
    """Factory for credit issuing strategies.

    Checks plugin registry for implementations, then falls back to built-ins.
    """

    _builtin_registry = {
        TimingMode.FIXED_SCHEDULE: FixedScheduleStrategy,
        TimingMode.REQUEST_RATE: RequestRateStrategy,
    }

    @classmethod
    def create_instance(cls, timing_mode: TimingMode, **kwargs):
        """Create strategy instance.

        Checks plugin registry first for custom implementations.

        Args:
            timing_mode: Timing mode enum
            **kwargs: Strategy initialization arguments

        Returns:
            Strategy instance
        """
        from aiperf.plugin import plugin_registry

        mode_name = timing_mode.value  # "fixed_schedule" or "request_rate"

        # Check plugin registry first (lazy load)
        if plugin_registry.is_registered('timing_strategy', mode_name):
            # Plugin provides custom implementation!
            return plugin_registry.create_instance(
                'timing_strategy',
                mode_name,
                **kwargs
            )

        # Fall back to built-in
        if timing_mode in cls._builtin_registry:
            return cls._builtin_registry[timing_mode](**kwargs)

        raise ValueError(f"Unknown timing mode: {timing_mode}")


# ============================================
# External plugin can provide custom strategy!
# ============================================
# setup.py
entry_points={
    "aiperf.plugins": [
        "timing_strategy:custom_replay = my_plugin:CustomReplayStrategy"
    ]
}

# User config
timing_mode: custom_replay  # Plugin implementation!

# AIPerf lazy-loads plugin when creating strategy
```

---

## Entry Point Naming Convention

### Format: `"protocol:name"`

```python
entry_points={
    "aiperf.plugins": [
        # Phase hooks
        "phase_hook:datadog = aiperf_datadog_plugin:DatadogPhaseHook",
        "phase_hook:prometheus = aiperf_prometheus_plugin:PrometheusPhaseHook",

        # Custom strategies
        "timing_strategy:custom_replay = my_plugin:CustomReplayStrategy",
        "timing_strategy:adaptive_rate = my_plugin:AdaptiveRateStrategy",

        # Custom endpoints
        "endpoint:ollama = aiperf_ollama:OllamaEndpoint",

        # Custom samplers
        "dataset_sampler:weighted = my_plugin:WeightedSampler",
    ]
}
```

**Benefits:**
- Clear protocol/implementation separation
- One entry point group for all plugin types
- Easy to discover: "What phase_hooks are available?"

---

## Configuration Schema

### Unified Plugin Config

```yaml
# benchmark.yaml

# Enable/configure plugins
plugins:
  # Phase lifecycle hooks
  phase_hooks:
    - name: logging  # Built-in

    - name: datadog  # From plugin
      config:
        api_key: ${DATADOG_API_KEY}
        site: datadoghq.com

    - name: prometheus  # From plugin
      config:
        endpoint: http://localhost:9091

  # Custom strategy (example)
  timing_strategy:
    name: adaptive_rate  # From plugin
    config:
      initial_rate: 100
      scale_factor: 1.5

# Standard AIPerf config
benchmark:
  timing_mode: adaptive_rate  # Uses plugin!
  concurrency: 10
```

---

## Startup Integration

### Complete Bootstrap Flow

```python
# ============================================
# src/aiperf/main.py or cli.py
# ============================================

from aiperf.plugin import plugin_registry
from aiperf.timing.phase_hooks import setup_phase_hooks


def main():
    """AIPerf CLI entry point with plugin support."""

    # 1. Parse CLI args
    args = parse_args()

    # 2. Load configuration
    config = load_config(args.config_file)

    # 3. ★ DISCOVER PLUGINS ★ (lazy - doesn't load yet!)
    plugin_registry.discover_plugins(entry_point_group='aiperf.plugins')

    # Log what was found (not loaded!)
    print("Discovered plugins:")
    for protocol in ['phase_hook', 'timing_strategy', 'endpoint']:
        impls = plugin_registry.list_implementations(protocol)
        if impls:
            print(f"  {protocol}: {', '.join(impls)}")

    # 4. Load configured phase hooks (lazy - only configured ones loaded!)
    setup_phase_hooks(config)

    # 5. Bootstrap services (factories will lazy-load plugins as needed)
    bootstrap_services(config)

    # 6. Run benchmark
    run_benchmark(config)


def setup_phase_hooks(config: dict):
    """Load and register configured phase hooks.

    Args:
        config: Application config
    """
    hook_configs = config.get("plugins", {}).get("phase_hooks", [])

    if not hook_configs:
        print("No phase hooks configured")
        return

    for hook_config in hook_configs:
        name = hook_config["name"]
        init_config = hook_config.get("config", {})

        try:
            # LAZY LOAD via factory!
            hook = PhaseLifecycleHookFactory.create_instance(name, **init_config)

            # Let hook register its callbacks
            if hasattr(hook, 'register_with_registry'):
                hook.register_with_registry()

            print(f"✓ Loaded phase hook: {name}")

        except Exception as e:
            print(f"✗ Failed to load phase hook '{name}': {e!r}")
            # Continue loading other hooks
```

---

## Benefits of This Design

### 1. Zero Coupling
```
AIPerf ──X→ Plugins  ✅ (No imports)
Plugins ──→ AIPerf   ✅ (One-way dependency)
```

### 2. Lazy Loading
```python
# Startup: Discover 10 plugins (fast!)
plugin_registry.discover_plugins()  # Just scans entry points

# Execution: Only load 2 configured plugins (efficient!)
hooks = load_configured_phase_hooks(config)  # Only loads what's used
```

### 3. Config-Driven
```yaml
plugins:
  phase_hooks:
    - name: datadog  # Enable this
      config: {...}
    # Don't enable prometheus (it won't be loaded)
```

### 4. Factory Integration
```python
# Factories check plugin registry first
strategy = CreditIssuingStrategyFactory.create_instance(
    timing_mode,  # Could be plugin implementation!
    **kwargs
)
# Lazy loads plugin if timing_mode is from plugin
```

### 5. Extensibility
- Phase hooks (monitoring, notifications)
- Custom strategies (new timing modes)
- Custom endpoints (new LLM providers)
- Custom samplers (new dataset sampling)
- Custom anything with factory pattern!

---

## User Experience

```bash
# 1. Install plugin
pip install aiperf-datadog-plugin

# 2. Configure
cat > benchmark.yaml <<EOF
plugins:
  phase_hooks:
    - name: datadog
      config:
        api_key: xxx
EOF

# 3. Run
aiperf --config benchmark.yaml

# Output:
# Discovered plugins:
#   phase_hook: logging, datadog, prometheus
#   endpoint: ollama
# ✓ Loaded phase hook: logging
# ✓ Loaded phase hook: datadog
# ...
# Benchmark running with plugins active!
```

**Zero user code!** ✅
**Just install + configure!** ✅
**Lazy loading!** ✅
**Integrates with factories!** ✅

---

## Implementation Checklist

**To implement:**

1. Create `LazyPluginRegistry` in `src/aiperf/common/plugin_registry.py`
2. Create `PhaseHookRegistry` in `src/aiperf/timing/phase_hooks.py`
3. Update `PhaseLifecycleHookFactory` to check plugin registry
4. Integrate `plugin_registry.discover_plugins()` into startup
5. Add `setup_phase_hooks()` to load configured hooks
6. Update `PhaseOrchestrator` to use `PhaseHookRegistry`
7. Add `plugins:` section to config schema
8. Document plugin development guide

**Want me to implement this complete lazy plugin system?** 🚀

This is the real solution you're envisioning!
