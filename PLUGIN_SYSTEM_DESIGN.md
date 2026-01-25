<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Plugin System Design - Entry Points Pattern

## Application-Level Plugin Discovery

**Goal:** User runs `aiperf`, plugins are automatically loaded and active.

---

## How It Works

### 1. Entry Points Discovery (Standard Python)

```python
# ============================================
# External plugin package
# ============================================
# File: aiperf_datadog_plugin/plugin.py

from aiperf.timing.phase_hooks import PhaseHookRegistry

class DatadogPhasePlugin:
    """Datadog monitoring plugin for AIPerf.

    Automatically loaded via entry points.
    """

    def __init__(self, config: dict):
        """Initialize plugin with config from AIPerf.

        Args:
            config: Plugin configuration from aiperf config file or env vars
        """
        self.api_key = config.get("api_key") or os.getenv("DATADOG_API_KEY")
        self.enabled = self.api_key is not None

        if self.enabled:
            from datadog import initialize
            initialize(api_key=self.api_key)

    async def on_phase_complete(self, phase, tracker):
        """Called when phase completes."""
        if not self.enabled:
            return

        stats = tracker.create_stats()
        await self._send_metrics_to_datadog(stats)

    def register_hooks(self):
        """Register callbacks with AIPerf.

        Called by AIPerf during plugin initialization.
        """
        PhaseHookRegistry.register_phase_complete(self.on_phase_complete)


# File: setup.py
setup(
    name="aiperf-datadog-plugin",
    version="1.0.0",
    packages=["aiperf_datadog_plugin"],
    install_requires=["datadog", "aiperf>=2.0"],
    entry_points={
        "aiperf.plugins": [
            "datadog = aiperf_datadog_plugin.plugin:DatadogPhasePlugin"
        ]
    }
)
```

### 2. AIPerf Plugin Loader (At Startup)

```python
# ============================================
# AIPerf plugin loading system
# ============================================
# File: src/aiperf/common/plugin_loader.py

import importlib.metadata
from typing import Protocol, runtime_checkable

@runtime_checkable
class AIPerfPlugin(Protocol):
    """Protocol for AIPerf plugins.

    All plugins must implement this interface.
    """

    def __init__(self, config: dict): ...

    def register_hooks(self) -> None:
        """Register hooks with AIPerf.

        Called during plugin initialization.
        Plugins should register callbacks with appropriate registries.
        """
        ...


class PluginLoader:
    """Loads and initializes plugins at application startup.

    Discovers plugins via entry points, initializes them with config,
    and calls register_hooks() on each.
    """

    def __init__(self):
        self._loaded_plugins: list = []

    def discover_and_load_plugins(self, plugin_config: dict | None = None):
        """Discover and load all installed plugins.

        Args:
            plugin_config: Configuration for plugins (from config file or env)

        Returns:
            List of loaded plugin instances
        """
        plugin_config = plugin_config or {}

        # Discover via entry points
        entry_points = importlib.metadata.entry_points()

        # Python 3.10+
        if hasattr(entry_points, 'select'):
            plugins_group = entry_points.select(group='aiperf.plugins')
        else:
            plugins_group = entry_points.get('aiperf.plugins', [])

        for entry_point in plugins_group:
            try:
                # Load plugin class
                plugin_class = entry_point.load()

                # Get plugin-specific config
                plugin_name = entry_point.name
                config = plugin_config.get(plugin_name, {})

                # Instantiate plugin
                plugin_instance = plugin_class(config)

                # Let plugin register its hooks
                plugin_instance.register_hooks()

                # Track loaded plugins
                self._loaded_plugins.append(plugin_instance)

                print(f"Loaded plugin: {plugin_name}")

            except Exception as e:
                print(f"Failed to load plugin {entry_point.name}: {e!r}")
                # Don't let one bad plugin break startup

        return self._loaded_plugins


# ============================================
# Global registry for phase hooks
# ============================================
# File: src/aiperf/timing/phase_hooks.py

class PhaseHookRegistry:
    """Global registry for phase lifecycle callbacks.

    Plugins register callbacks here.
    PhaseOrchestrator calls registered callbacks.
    """

    _phase_start_callbacks: list[Callable] = []
    _phase_complete_callbacks: list[Callable] = []
    _phase_timeout_callbacks: list[Callable] = []

    @classmethod
    def register_phase_start(cls, callback: Callable):
        """Register callback for phase start.

        External plugins call this during register_hooks().
        """
        cls._phase_start_callbacks.append(callback)

    @classmethod
    def register_phase_complete(cls, callback: Callable):
        cls._phase_complete_callbacks.append(callback)

    @classmethod
    def register_phase_timeout(cls, callback: Callable):
        cls._phase_timeout_callbacks.append(callback)

    @classmethod
    def get_phase_start_callbacks(cls) -> list[Callable]:
        return cls._phase_start_callbacks.copy()

    @classmethod
    def get_phase_complete_callbacks(cls) -> list[Callable]:
        return cls._phase_complete_callbacks.copy()

    @classmethod
    def clear(cls):
        """Clear all registered callbacks (for testing)."""
        cls._phase_start_callbacks.clear()
        cls._phase_complete_callbacks.clear()
        cls._phase_timeout_callbacks.clear()


# ============================================
# PhaseOrchestrator uses global registry
# ============================================
class PhaseOrchestrator:
    """Orchestrator that calls globally registered hooks.

    Doesn't know what hooks exist - just calls whatever is in registry.
    """

    def __init__(self, executor, publisher):
        self._executor = executor
        self._publisher = publisher
        # No hooks parameter! Gets from global registry

    async def execute_phase(self, tracker, coro, is_last):
        # Get callbacks from global registry (populated by plugins)
        await self._notify_callbacks(
            PhaseHookRegistry.get_phase_start_callbacks(),
            tracker.phase,
            tracker
        )

        await self._executor.execute_phase(tracker, coro, is_last)

        await self._notify_callbacks(
            PhaseHookRegistry.get_phase_complete_callbacks(),
            tracker.phase,
            tracker
        )

    async def _notify_callbacks(self, callbacks, *args):
        for callback in callbacks:
            try:
                await callback(*args)
            except Exception as e:
                self.error(f"Plugin callback failed: {e!r}")


# ============================================
# AIPerf main entry point
# ============================================
# File: src/aiperf/cli.py

def main():
    """AIPerf CLI entry point."""

    # Parse args
    args = parse_args()

    # Load config
    config = load_config(args.config_file)

    # ★ LOAD PLUGINS AT STARTUP ★
    plugin_loader = PluginLoader()
    plugins = plugin_loader.discover_and_load_plugins(
        plugin_config=config.get("plugins", {})
    )

    print(f"Loaded {len(plugins)} plugins")

    # Continue with normal AIPerf startup
    # Plugins are now registered and active!
    run_benchmark(config)
```

---

## Complete Flow

```
┌────────────────────────────────────────────────────────────────┐
│ 1. User runs: aiperf --config benchmark.yaml                   │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. AIPerf main() starts                                        │
│    ├─ Parse args                                               │
│    ├─ Load config                                              │
│    └─ PluginLoader.discover_and_load_plugins()                 │
│       ├─ Scans entry_points(group='aiperf.plugins')            │
│       ├─ Finds: datadog, prometheus, custom_monitor            │
│       └─ For each plugin:                                      │
│           ├─ Load plugin class                                 │
│           ├─ Instantiate: plugin = PluginClass(config)         │
│           └─ plugin.register_hooks()  ◄─── Plugin registers!   │
└────────────────────────────────────────────────────────────────┘
                           │
                           │ Plugins registered in global registry
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ PhaseHookRegistry (global)                                     │
│ ├─ _phase_start_callbacks: [datadog_fn, prometheus_fn, ...]   │
│ └─ _phase_complete_callbacks: [datadog_fn, custom_fn, ...]    │
└────────────────────────────────────────────────────────────────┘
                           │
                           │ Later during execution
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. TimingManager creates PhaseOrchestrator                     │
│    orchestrator = PhaseOrchestrator(executor, publisher)       │
│                                                                │
│    (No plugins mentioned! Gets from global registry)           │
└────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. PhaseOrchestrator.execute_phase()                           │
│    ├─ callbacks = PhaseHookRegistry.get_phase_start_callbacks()│
│    ├─ for callback in callbacks:                               │
│    │   await callback(phase, tracker)  ◄─── Calls all plugins! │
│    │                                                            │
│    ├─ Execute phase...                                         │
│    │                                                            │
│    └─ callbacks = PhaseHookRegistry.get_phase_complete_...     │
│        for callback in callbacks:                              │
│          await callback(phase, tracker)  ◄─── Calls plugins!   │
└────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Config File (benchmark.yaml)

```yaml
# User's benchmark config
benchmark:
  request_rate: 100
  concurrency: 10

# Plugin configuration
plugins:
  datadog:
    api_key: ${DATADOG_API_KEY}  # From env var
    site: datadoghq.com
    tags:
      - benchmark:llm
      - env:production

  prometheus:
    endpoint: http://localhost:9091
    job_name: aiperf_benchmark

  custom_monitor:
    enabled: true
    log_level: INFO
```

### Environment Variables

```bash
export DATADOG_API_KEY="xxx"
export PROMETHEUS_ENDPOINT="http://..."

aiperf --config benchmark.yaml
# Plugins get config from benchmark.yaml + env vars
```

---

## External Plugin Implementation

### Full Example: aiperf-datadog-plugin

```python
# ============================================
# File: aiperf_datadog_plugin/plugin.py
# ============================================

from aiperf.timing.phase_hooks import PhaseHookRegistry
from aiperf.common.enums import CreditPhase
from datadog import initialize, api, statsd
import logging

logger = logging.getLogger(__name__)


class DatadogPhasePlugin:
    """Datadog monitoring plugin for AIPerf phases.

    Automatically loaded via entry points.
    Sends phase metrics and events to Datadog.
    """

    def __init__(self, config: dict):
        """Initialize plugin.

        Args:
            config: Plugin config from aiperf config file
        """
        self.api_key = config.get("api_key")
        self.site = config.get("site", "datadoghq.com")
        self.tags = config.get("tags", [])
        self.enabled = self.api_key is not None

        if self.enabled:
            initialize(api_key=self.api_key, app_key=self.api_key, api_host=self.site)
            logger.info("Datadog plugin initialized")
        else:
            logger.warning("Datadog plugin disabled (no api_key)")

    def register_hooks(self):
        """Register callbacks with AIPerf.

        Called by PluginLoader during application startup.
        """
        if not self.enabled:
            return

        # Register our callbacks
        PhaseHookRegistry.register_phase_start(self.on_phase_start)
        PhaseHookRegistry.register_phase_complete(self.on_phase_complete)

        logger.info("Datadog hooks registered")

    async def on_phase_start(self, phase: CreditPhase, tracker):
        """Send phase start event to Datadog."""
        try:
            api.Event.create(
                title=f"AIPerf Phase Started: {phase}",
                text=f"Phase {phase} started",
                tags=self.tags + [f"phase:{phase}", "event:start"],
                alert_type="info",
            )
        except Exception as e:
            logger.error(f"Failed to send phase start event: {e!r}")

    async def on_phase_complete(self, phase: CreditPhase, tracker):
        """Send phase metrics to Datadog."""
        try:
            stats = tracker.create_stats()

            # Send metrics via statsd
            statsd.gauge(
                "aiperf.phase.duration",
                stats.duration_sec,
                tags=self.tags + [f"phase:{phase}"]
            )

            statsd.gauge(
                "aiperf.phase.requests.sent",
                stats.sent,
                tags=self.tags + [f"phase:{phase}"]
            )

            statsd.gauge(
                "aiperf.phase.requests.completed",
                stats.completed,
                tags=self.tags + [f"phase:{phase}"]
            )

            # Send completion event
            api.Event.create(
                title=f"AIPerf Phase Complete: {phase}",
                text=f"Phase {phase} completed: {stats.sent} requests in {stats.duration_sec:.2f}s",
                tags=self.tags + [f"phase:{phase}", "event:complete"],
                alert_type="success",
            )

        except Exception as e:
            logger.error(f"Failed to send phase metrics: {e!r}")


# ============================================
# File: setup.py
# ============================================
from setuptools import setup, find_packages

setup(
    name="aiperf-datadog-plugin",
    version="1.0.0",
    description="Datadog monitoring plugin for AIPerf",
    packages=find_packages(),
    install_requires=[
        "datadog>=0.44.0",
        "aiperf>=2.0.0",
    ],
    entry_points={
        # AIPerf scans this group at startup
        "aiperf.plugins": [
            "datadog = aiperf_datadog_plugin.plugin:DatadogPhasePlugin"
        ]
    },
    classifiers=[
        "Framework :: AIPerf",
        "Topic :: System :: Monitoring",
    ],
)
```

---

## AIPerf Plugin Loading Infrastructure

### 1. Plugin Loader (At Application Startup)

```python
# ============================================
# src/aiperf/common/plugin_loader.py
# ============================================

import importlib.metadata
import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class AIPerfPlugin(Protocol):
    """Protocol for AIPerf plugins."""

    def __init__(self, config: dict): ...
    def register_hooks(self) -> None: ...


class PluginLoader:
    """Discovers and loads plugins at application startup."""

    def __init__(self):
        self._loaded_plugins: list = []

    def load_all_plugins(self, plugin_config: dict | None = None) -> list:
        """Load all plugins from entry points.

        Called once at application startup.

        Args:
            plugin_config: Plugin configurations from config file

        Returns:
            List of loaded plugin instances
        """
        plugin_config = plugin_config or {}

        logger.info("Discovering AIPerf plugins...")

        # Get entry points
        entry_points = importlib.metadata.entry_points()

        if hasattr(entry_points, 'select'):
            # Python 3.10+
            plugin_eps = entry_points.select(group='aiperf.plugins')
        else:
            # Python 3.9
            plugin_eps = entry_points.get('aiperf.plugins', [])

        # Load each plugin
        for ep in plugin_eps:
            plugin_name = ep.name

            try:
                # Load plugin class from entry point
                logger.debug(f"Loading plugin: {plugin_name}")
                plugin_class = ep.load()

                # Get plugin-specific config
                config = plugin_config.get(plugin_name, {})

                # Instantiate plugin
                plugin = plugin_class(config)

                # Verify implements protocol
                if not isinstance(plugin, AIPerfPlugin):
                    logger.warning(
                        f"Plugin {plugin_name} doesn't implement AIPerfPlugin protocol"
                    )
                    continue

                # Let plugin register its hooks
                plugin.register_hooks()

                # Track loaded plugins
                self._loaded_plugins.append(plugin)

                logger.info(f"✓ Loaded plugin: {plugin_name}")

            except Exception as e:
                logger.error(f"✗ Failed to load plugin {plugin_name}: {e!r}")
                # Continue loading other plugins

        logger.info(f"Loaded {len(self._loaded_plugins)} plugin(s)")
        return self._loaded_plugins

    def get_loaded_plugins(self) -> list:
        """Get list of loaded plugins."""
        return self._loaded_plugins.copy()
```

### 2. Global Hook Registry

```python
# ============================================
# src/aiperf/timing/phase_hooks.py
# ============================================

from collections.abc import Callable

class PhaseHookRegistry:
    """Global registry for phase lifecycle hooks.

    Plugins register callbacks during initialization.
    PhaseOrchestrator calls registered callbacks during execution.

    Thread-safe for plugin loading (happens once at startup).
    """

    _phase_start_callbacks: list[Callable] = []
    _phase_complete_callbacks: list[Callable] = []
    _phase_sending_complete_callbacks: list[Callable] = []
    _phase_timeout_callbacks: list[Callable] = []

    @classmethod
    def register_phase_start(cls, callback: Callable) -> None:
        """Register callback for phase start events.

        Args:
            callback: async def callback(phase: CreditPhase, tracker: CreditPhaseTracker)
        """
        cls._phase_start_callbacks.append(callback)

    @classmethod
    def register_phase_complete(cls, callback: Callable) -> None:
        """Register callback for phase complete events."""
        cls._phase_complete_callbacks.append(callback)

    @classmethod
    def register_phase_sending_complete(cls, callback: Callable) -> None:
        """Register callback for phase sending complete events."""
        cls._phase_sending_complete_callbacks.append(callback)

    @classmethod
    def register_phase_timeout(cls, callback: Callable) -> None:
        """Register callback for phase timeout events."""
        cls._phase_timeout_callbacks.append(callback)

    @classmethod
    def get_phase_start_callbacks(cls) -> list[Callable]:
        return cls._phase_start_callbacks.copy()

    @classmethod
    def get_phase_complete_callbacks(cls) -> list[Callable]:
        return cls._phase_complete_callbacks.copy()

    @classmethod
    def get_phase_sending_complete_callbacks(cls) -> list[Callable]:
        return cls._phase_sending_complete_callbacks.copy()

    @classmethod
    def get_phase_timeout_callbacks(cls) -> list[Callable]:
        return cls._phase_timeout_callbacks.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered callbacks.

        Used for testing to reset state between tests.
        """
        cls._phase_start_callbacks.clear()
        cls._phase_complete_callbacks.clear()
        cls._phase_sending_complete_callbacks.clear()
        cls._phase_timeout_callbacks.clear()
```

### 3. Update PhaseOrchestrator

```python
# ============================================
# src/aiperf/timing/phase_orchestrator.py
# ============================================

from aiperf.timing.phase_hooks import PhaseHookRegistry

class PhaseOrchestrator:
    """Orchestrates phase execution with globally registered hooks.

    Hooks are registered by plugins at application startup.
    Orchestrator doesn't know what hooks exist.
    """

    def __init__(self, executor: PhaseExecutor, publisher: PhasePublisher):
        self._executor = executor
        self._publisher = publisher
        # No hooks parameter! Uses global registry

    async def execute_phase(
        self,
        phase_tracker: CreditPhaseTracker,
        execute_coro: Coroutine,
        is_last_phase: bool,
    ):
        """Execute phase with globally registered hooks."""

        # Notify phase start hooks (from global registry)
        await self._notify_hooks(
            PhaseHookRegistry.get_phase_start_callbacks(),
            phase_tracker.phase,
            phase_tracker,
        )

        # Execute phase via executor
        await self._executor.execute_phase(phase_tracker, execute_coro, is_last_phase)

        # Notify phase complete hooks (from global registry)
        await self._notify_hooks(
            PhaseHookRegistry.get_phase_complete_callbacks(),
            phase_tracker.phase,
            phase_tracker,
        )

    async def _notify_hooks(self, callbacks: list[Callable], *args, **kwargs):
        """Call all callbacks with error handling.

        Plugins can fail without breaking phase execution.
        """
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    # Support sync callbacks too
                    callback(*args, **kwargs)
            except Exception as e:
                # Log but don't propagate
                if hasattr(self, 'error'):
                    self.error(f"Plugin callback {callback.__name__} failed: {e!r}")
```

### 4. Integrate into AIPerf Main

```python
# ============================================
# src/aiperf/cli.py (or wherever main is)
# ============================================

from aiperf.common.plugin_loader import PluginLoader

def main():
    """AIPerf CLI entry point."""

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file path")
    args = parser.parse_args()

    # Load configuration
    config = load_config_file(args.config)

    # ★★★ LOAD PLUGINS AT STARTUP ★★★
    plugin_loader = PluginLoader()
    loaded_plugins = plugin_loader.load_all_plugins(
        plugin_config=config.get("plugins", {})
    )

    # Plugins are now registered in global registries!
    # Continue with normal AIPerf execution

    # Bootstrap services
    # ... existing AIPerf code ...

    # Run benchmark
    # ... existing AIPerf code ...

    # Plugins will automatically be called at hook points!
```

---

## User Experience

### 1. Install Plugin

```bash
pip install aiperf-datadog-plugin
```

### 2. Configure Plugin

```yaml
# benchmark.yaml
plugins:
  datadog:
    api_key: ${DATADOG_API_KEY}
    tags:
      - team:ml-perf
      - benchmark:gpt4
```

### 3. Run AIPerf

```bash
export DATADOG_API_KEY="xxx"
aiperf --config benchmark.yaml
```

**That's it!** Plugin automatically:
- Discovered via entry points
- Loaded at startup
- Initialized with config
- Registered hooks
- Called during execution

**User never writes code to register hooks!** ✅

---

## Comparison to Current

### Current (Requires User Code)

```python
# ❌ User must write code
from aiperf_datadog_plugin import DatadogPhaseHook

# User creates orchestrator somehow?
orchestrator = PhaseOrchestrator(..., hooks=[
    DatadogPhaseHook(api_key="xxx")
])
```

**Problem:** User needs to modify AIPerf code to pass hooks!

### Proposed (Zero User Code)

```bash
# ✅ User just installs and configures
pip install aiperf-datadog-plugin

# Config file
plugins:
  datadog:
    api_key: xxx

# Run
aiperf --config benchmark.yaml
# Plugin automatically active!
```

**No user code needed!** ✅

---

## Implementation Checklist

**To implement this:**

1. ✅ Create `PhaseHookRegistry` (global callback lists)
2. ✅ Create `PluginLoader` (entry point discovery)
3. ✅ Define `AIPerfPlugin` protocol
4. ✅ Update `PhaseOrchestrator` to use global registry
5. ✅ Integrate `PluginLoader` into `main()`
6. ✅ Update config schema to support `plugins:` section
7. ✅ Document plugin development guide
8. ✅ Create example plugin (aiperf-example-plugin)

**Want me to implement this?** This is the TRUE plugin system you're envisioning! 🚀

This achieves:
- ✅ Zero coupling (AIPerf → plugins)
- ✅ Auto-discovery (entry points)
- ✅ Auto-loading (at startup)
- ✅ Config-driven (plugins: section)
- ✅ Zero user code (just install + configure)

**This is how pytest plugins work!** 🎯