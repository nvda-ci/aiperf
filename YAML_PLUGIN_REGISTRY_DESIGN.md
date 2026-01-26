<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# YAML-Based Plugin Registry Design

## Declarative Plugin Manifests

**Concept:** Entry points point to YAML files (manifests) that declare what the plugin provides.

---

## Complete Design

### 1. External Plugin Structure

```
aiperf-datadog-plugin/
├─ aiperf_datadog_plugin/
│  ├─ __init__.py
│  ├─ registry.yaml          ← Plugin manifest
│  ├─ hooks.py               ← Phase hook implementations
│  ├─ endpoints.py           ← Endpoint implementations (if any)
│  └─ samplers.py            ← Sampler implementations (if any)
│
├─ setup.py
└─ README.md
```

### 2. Plugin Manifest (registry.yaml)

```yaml
# ============================================
# aiperf_datadog_plugin/registry.yaml
# ============================================

# Plugin metadata
plugin:
  name: aiperf-datadog-plugin
  version: 1.0.0
  description: Datadog monitoring integration for AIPerf
  author: Datadog Team
  license: Apache-2.0
  homepage: https://github.com/DataDog/aiperf-datadog-plugin

  # Plugin-level dependencies
  requires:
    - datadog>=0.44.0
    - aiperf>=2.0.0

  # Enable/disable entire plugin
  enabled: true

# Phase hook implementations provided by this plugin
phase_hook:
  datadog:
    # Fully qualified class path
    class: aiperf_datadog_plugin.hooks:DatadogPhaseHook

    # Human-readable description
    description: Send phase lifecycle events and metrics to Datadog

    # Config schema (for validation and documentation)
    config_schema:
      api_key:
        type: string
        required: true
        description: Datadog API key
        env_var: DATADOG_API_KEY

      site:
        type: string
        required: false
        default: datadoghq.com
        description: Datadog site (datadoghq.com, datadoghq.eu, etc.)

      tags:
        type: list
        required: false
        default: []
        description: Additional tags for all metrics

    # Optional: Disable specific implementation
    enabled: true

  datadog_legacy:
    class: aiperf_datadog_plugin.hooks:DatadogLegacyHook
    description: Legacy Datadog integration (deprecated)
    enabled: false  # Disabled by default

# Custom endpoint implementations (if plugin provides)
endpoint:
  datadog_llm_gateway:
    class: aiperf_datadog_plugin.endpoints:DatadogLLMGatewayEndpoint
    description: Datadog LLM Observability Gateway endpoint
    config_schema:
      gateway_url:
        type: string
        required: true
      api_key:
        type: string
        required: true

# Custom sampling strategies (if plugin provides)
dataset_sampler:
  datadog_weighted:
    class: aiperf_datadog_plugin.samplers:WeightedSampler
    description: Weighted random sampling with Datadog metrics
```

### 3. Setup.py (Single Entry Point)

```python
# ============================================
# setup.py
# ============================================

from setuptools import setup, find_packages

setup(
    name="aiperf-datadog-plugin",
    version="1.0.0",
    packages=find_packages(),
    package_data={
        "aiperf_datadog_plugin": ["registry.yaml"],  # Include YAML
    },
    install_requires=[
        "datadog>=0.44.0",
        "aiperf>=2.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "aiperf.plugins": [
            # Just one entry point pointing to manifest!
            "datadog = aiperf_datadog_plugin:registry.yaml"
        ]
    },
)
```

### 4. Lazy Plugin Registry (AIPerf)

```python
# ============================================
# src/aiperf/common/lazy_plugin_registry.py
# ============================================

import importlib
import importlib.metadata
import importlib.resources
from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Any


@dataclass
class PluginManifest:
    """Parsed plugin manifest from registry.yaml.

    Attributes:
        plugin_name: Plugin package name
        metadata: Plugin metadata (version, author, etc.)
        implementations: Dict of protocol → {name → impl_info}
    """

    plugin_name: str
    metadata: dict[str, Any]
    implementations: dict[str, dict[str, dict]]  # protocol → name → info

    @classmethod
    def from_yaml(cls, plugin_name: str, yaml_content: str) -> "PluginManifest":
        """Parse manifest from YAML content.

        Args:
            plugin_name: Plugin name from entry point
            yaml_content: YAML file content

        Returns:
            PluginManifest instance
        """
        data = yaml.safe_load(yaml_content)

        # Extract plugin metadata
        metadata = data.get('plugin', {})

        # Extract implementations
        implementations = {}
        for protocol_name, impls in data.items():
            if protocol_name == 'plugin':
                continue  # Skip metadata section

            implementations[protocol_name] = impls

        return cls(
            plugin_name=plugin_name,
            metadata=metadata,
            implementations=implementations,
        )


@dataclass
class LazyImplementation:
    """Lazy-loading implementation entry.

    Attributes:
        protocol_name: Protocol type (e.g., "phase_hook")
        name: Implementation name (e.g., "datadog")
        plugin_name: Plugin providing this implementation
        class_path: Fully qualified class path (e.g., "pkg.module:ClassName")
        metadata: Implementation metadata (description, config_schema, etc.)
        loaded_class: Cached class after first load (None initially)
    """

    protocol_name: str
    name: str
    plugin_name: str
    class_path: str
    metadata: dict[str, Any]
    loaded_class: type | None = None

    def load(self) -> type:
        """Load implementation class (lazy - only once).

        Returns:
            Loaded class (cached)

        Raises:
            ImportError: If class cannot be loaded
        """
        if self.loaded_class is not None:
            return self.loaded_class

        # Parse class path: "module.path:ClassName"
        if ':' not in self.class_path:
            raise ValueError(
                f"Invalid class path '{self.class_path}'. "
                f"Expected format: 'module.path:ClassName'"
            )

        module_path, class_name = self.class_path.split(':', 1)

        # Import module
        module = importlib.import_module(module_path)

        # Get class
        self.loaded_class = getattr(module, class_name)

        return self.loaded_class

    def create_instance(self, **kwargs) -> Any:
        """Load class and create instance.

        Args:
            **kwargs: Arguments for class __init__

        Returns:
            Instance of implementation
        """
        impl_class = self.load()
        return impl_class(**kwargs)

    @property
    def enabled(self) -> bool:
        """Check if implementation is enabled."""
        return self.metadata.get('enabled', True)

    @property
    def description(self) -> str:
        """Get implementation description."""
        return self.metadata.get('description', '')

    @property
    def config_schema(self) -> dict | None:
        """Get config schema if defined."""
        return self.metadata.get('config_schema')


class LazyPluginRegistry:
    """Registry for lazy-loading plugins from YAML manifests.

    Plugins discovered at startup via entry points.
    Manifests parsed to find available implementations.
    Actual classes loaded only when requested.
    """

    def __init__(self):
        # protocol → name → LazyImplementation
        self._implementations: dict[str, dict[str, LazyImplementation]] = {}

        # Track loaded plugins
        self._loaded_plugins: set[str] = set()

    def discover_plugins(self, entry_point_group: str = "aiperf.plugins"):
        """Discover plugins via entry points.

        Loads registry.yaml from each plugin but NOT the implementations.

        Args:
            entry_point_group: Entry point group to scan
        """
        entry_points = importlib.metadata.entry_points()

        if hasattr(entry_points, 'select'):
            plugin_eps = entry_points.select(group=entry_point_group)
        else:
            plugin_eps = entry_points.get(entry_point_group, [])

        for ep in plugin_eps:
            try:
                plugin_name = ep.name

                # Entry point value is path to YAML file
                # Format: "package.module:registry.yaml"
                yaml_ref = ep.value

                # Load YAML manifest
                manifest = self._load_manifest(plugin_name, yaml_ref)

                # Register all implementations from manifest
                self._register_manifest(plugin_name, manifest)

                self._loaded_plugins.add(plugin_name)

            except Exception as e:
                print(f"Failed to load plugin manifest '{ep.name}': {e!r}")

    def _load_manifest(self, plugin_name: str, yaml_ref: str) -> PluginManifest:
        """Load plugin manifest from YAML reference.

        Args:
            plugin_name: Plugin name
            yaml_ref: Reference to YAML (e.g., "pkg.module:registry.yaml")

        Returns:
            Parsed PluginManifest
        """
        # Parse reference: "package.module:file.yaml"
        if ':' in yaml_ref:
            package_path, yaml_filename = yaml_ref.split(':', 1)
        else:
            # Assume registry.yaml in package root
            package_path = yaml_ref
            yaml_filename = "registry.yaml"

        # Load YAML file from package
        try:
            # Python 3.9+
            yaml_content = importlib.resources.files(package_path).joinpath(yaml_filename).read_text()
        except AttributeError:
            # Python 3.8 fallback
            import importlib.resources as resources
            yaml_content = resources.read_text(package_path, yaml_filename)

        # Parse manifest
        return PluginManifest.from_yaml(plugin_name, yaml_content)

    def _register_manifest(self, plugin_name: str, manifest: PluginManifest):
        """Register all implementations from manifest.

        Args:
            plugin_name: Plugin name
            manifest: Parsed manifest
        """
        # Check if plugin is enabled
        if not manifest.metadata.get('enabled', True):
            print(f"Plugin '{plugin_name}' is disabled in manifest")
            return

        # Register each implementation
        for protocol_name, impls in manifest.implementations.items():
            for name, impl_info in impls.items():
                # Create lazy implementation entry
                lazy_impl = LazyImplementation(
                    protocol_name=protocol_name,
                    name=name,
                    plugin_name=plugin_name,
                    class_path=impl_info.get('class') or impl_info,  # Support simple or detailed format
                    metadata=impl_info if isinstance(impl_info, dict) else {},
                )

                # Skip if disabled
                if not lazy_impl.enabled:
                    continue

                # Register (NOT loaded yet!)
                self._implementations.setdefault(protocol_name, {})[name] = lazy_impl

    def get_implementation(self, protocol_name: str, name: str) -> type:
        """Get implementation class (lazy load).

        Args:
            protocol_name: Protocol type
            name: Implementation name

        Returns:
            Loaded class

        Raises:
            KeyError: If not found
        """
        if protocol_name not in self._implementations:
            raise KeyError(f"No implementations registered for protocol: {protocol_name}")

        if name not in self._implementations[protocol_name]:
            available = list(self._implementations[protocol_name].keys())
            raise KeyError(
                f"Implementation '{name}' not found for protocol '{protocol_name}'. "
                f"Available: {available}"
            )

        lazy_impl = self._implementations[protocol_name][name]
        return lazy_impl.load()  # LAZY LOAD HERE!

    def create_instance(self, protocol_name: str, name: str, **kwargs) -> Any:
        """Create instance of implementation (lazy load + instantiate).

        Args:
            protocol_name: Protocol type
            name: Implementation name
            **kwargs: Init arguments

        Returns:
            Instance
        """
        lazy_impl = self._implementations[protocol_name][name]
        return lazy_impl.create_instance(**kwargs)

    def list_implementations(self, protocol_name: str) -> list[str]:
        """List available implementations for protocol (doesn't load them).

        Args:
            protocol_name: Protocol type

        Returns:
            List of implementation names
        """
        return list(self._implementations.get(protocol_name, {}).keys())

    def get_implementation_info(
        self,
        protocol_name: str,
        name: str
    ) -> dict:
        """Get implementation metadata without loading it.

        Args:
            protocol_name: Protocol type
            name: Implementation name

        Returns:
            Metadata dict (description, config_schema, etc.)
        """
        lazy_impl = self._implementations[protocol_name][name]
        return {
            'plugin': lazy_impl.plugin_name,
            'class_path': lazy_impl.class_path,
            'description': lazy_impl.description,
            'config_schema': lazy_impl.config_schema,
            'enabled': lazy_impl.enabled,
        }

    def list_all_plugins(self) -> list[str]:
        """List all discovered plugin names."""
        return sorted(self._loaded_plugins)


# Global singleton
_lazy_plugin_registry = LazyPluginRegistry()


def get_lazy_plugin_registry() -> LazyPluginRegistry:
    """Get global lazy plugin registry."""
    return _lazy_plugin_registry
```

---

## Integration with Factories

### Update Existing Factories to Check Plugin Registry

```python
# ============================================
# src/aiperf/common/factories.py (UPDATE)
# ============================================

from aiperf.common.lazy_plugin_registry import get_lazy_plugin_registry


class PhaseLifecycleHookFactory:
    """Factory with plugin support.

    Checks order:
    1. Plugin registry (lazy-loaded external implementations)
    2. Built-in registry (internal implementations)
    """

    _builtin_registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Register built-in implementation."""
        def decorator(hook_class):
            cls._builtin_registry[name] = hook_class
            return hook_class
        return decorator

    @classmethod
    def create_instance(cls, name: str, **kwargs):
        """Create instance (checks plugins first).

        Args:
            name: Implementation name
            **kwargs: Init arguments

        Returns:
            Hook instance

        Raises:
            ValueError: If not found
        """
        # 1. Check plugin registry first (LAZY LOAD)
        plugin_registry = get_lazy_plugin_registry()
        try:
            return plugin_registry.create_instance('phase_hook', name, **kwargs)
        except KeyError:
            pass  # Not in plugins, try built-ins

        # 2. Check built-in registry
        if name in cls._builtin_registry:
            return cls._builtin_registry[name](**kwargs)

        # 3. Not found
        available = cls.list_available()
        raise ValueError(
            f"Unknown phase hook: '{name}'. Available: {available}"
        )

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available implementations (doesn't load).

        Returns:
            Sorted list of names
        """
        # Built-ins
        builtin = list(cls._builtin_registry.keys())

        # Plugins (not loaded, just listed from manifests)
        plugin_registry = get_lazy_plugin_registry()
        plugins = plugin_registry.list_implementations('phase_hook')

        return sorted(set(builtin + plugins))
```

### Same Pattern for ALL Factories

```python
# CreditIssuingStrategyFactory
# EndpointFactory
# DatasetSamplingStrategyFactory
# RequestRateGeneratorFactory
# ... etc

# All follow same pattern:
# 1. Check plugin registry (lazy load)
# 2. Fall back to built-in registry
# 3. Raise if not found
```

---

## Startup Integration

```python
# ============================================
# src/aiperf/cli.py or main entry point
# ============================================

from aiperf.common.lazy_plugin_registry import get_lazy_plugin_registry


def initialize_plugins():
    """Initialize plugin system at application startup.

    Discovers and registers plugins but doesn't load them.
    Fast even with many plugins installed.
    """
    print("Discovering AIPerf plugins...")

    # Discover all plugins (FAST - just reads YAML files)
    registry = get_lazy_plugin_registry()
    registry.discover_plugins(entry_point_group='aiperf.plugins')

    # Report what was found (still not loaded!)
    plugins = registry.list_all_plugins()
    print(f"Found {len(plugins)} plugin(s): {', '.join(plugins)}")

    # Show available implementations per protocol
    for protocol in ['phase_hook', 'timing_strategy', 'endpoint', 'dataset_sampler']:
        impls = registry.list_implementations(protocol)
        if impls:
            print(f"  {protocol}: {', '.join(impls)}")


def main():
    """AIPerf CLI entry point."""

    # 1. Initialize plugin system
    initialize_plugins()

    # 2. Parse args and load config
    args = parse_args()
    config = load_config(args.config_file)

    # 3. Bootstrap services
    # Factories will lazy-load plugins as needed
    bootstrap_and_run(config)
```

---

## Example: Complete Flow

### Plugin Provides Phase Hook

```yaml
# aiperf_datadog_plugin/registry.yaml
phase_hook:
  datadog:
    class: aiperf_datadog_plugin.hooks:DatadogPhaseHook
    description: Datadog monitoring
    config_schema:
      api_key: {type: string, required: true}
```

```python
# aiperf_datadog_plugin/hooks.py
class DatadogPhaseHook:
    def __init__(self, api_key: str, **kwargs):
        from datadog import initialize
        initialize(api_key=api_key)

    async def on_phase_complete(self, phase, tracker):
        from datadog import api
        stats = tracker.create_stats()
        api.Metric.send(...)

    def register_with_global_registry(self):
        """Register callbacks with PhaseHookRegistry."""
        from aiperf.timing.phase_hooks import PhaseHookRegistry
        PhaseHookRegistry.register_phase_complete(self.on_phase_complete)
```

### User Config

```yaml
# benchmark.yaml
plugins:
  phase_hooks:
    - name: datadog
      config:
        api_key: ${DATADOG_API_KEY}
```

### AIPerf Loads It

```python
# src/aiperf/timing/phase_orchestrator.py or bootstrap

def load_configured_hooks(config: dict) -> list:
    """Load hooks from config."""
    hooks = []

    for hook_config in config.get("plugins", {}).get("phase_hooks", []):
        name = hook_config["name"]
        init_config = hook_config.get("config", {})

        # LAZY LOAD via factory!
        hook = PhaseLifecycleHookFactory.create_instance(name, **init_config)
        #      └─ Checks plugin registry
        #         └─ LazyImplementation.create_instance()
        #            └─ Loads class NOW (first access!)
        #               └─ Instantiates with config

        # Register with global registry
        if hasattr(hook, 'register_with_global_registry'):
            hook.register_with_global_registry()

        hooks.append(hook)

    return hooks
```

---

## Benefits

### 1. Declarative Plugin Definition

```yaml
# Plugin author just writes YAML!
phase_hook:
  my_hook:
    class: my_plugin.hooks:MyHook
    description: Does cool stuff

# No Python code for registration!
```

### 2. Fast Discovery

```python
# Startup scans entry points
for ep in entry_points(group='aiperf.plugins'):
    # Load YAML only (fast!)
    yaml_content = load_yaml(ep.value)
    # Register lazy entries
    register_from_yaml(yaml_content)
    # Don't import Python code yet!

# Fast even with 100 plugins installed!
```

### 3. Lazy Loading

```python
# Only load what's configured
config = """
plugins:
  phase_hooks:
    - name: datadog  # Only load this one!
"""

# Other plugins stay unloaded (save memory, fast startup)
```

### 4. Rich Metadata

```yaml
# Manifest includes everything
phase_hook:
  datadog:
    class: ...
    description: ...
    config_schema: ...  # For validation!
    version: 2.0.0
    requires: [datadog>=0.44.0]
    author: Datadog Team
```

### 5. Plugin Introspection (CLI)

```bash
# List available plugins
aiperf plugins list

# Output:
# Available plugins:
#   datadog (aiperf-datadog-plugin v1.0.0)
#     - phase_hook:datadog
#     - endpoint:datadog_llm_gateway
#
#   prometheus (aiperf-prometheus-plugin v1.2.0)
#     - phase_hook:prometheus

# Show plugin details
aiperf plugins show datadog

# Output:
# Plugin: datadog (v1.0.0)
# Description: Datadog monitoring integration
# Implementations:
#   phase_hook:datadog
#     Description: Send phase metrics to Datadog
#     Config:
#       - api_key (string, required)
#       - site (string, default: datadoghq.com)
```

---

## CLI Commands for Plugin Management

```python
# ============================================
# src/aiperf/cli_plugins.py (NEW)
# ============================================

import click
from aiperf.common.lazy_plugin_registry import get_lazy_plugin_registry


@click.group()
def plugins():
    """Manage AIPerf plugins."""
    pass


@plugins.command()
def list_plugins():
    """List all discovered plugins."""
    registry = get_lazy_plugin_registry()
    registry.discover_plugins()

    plugins = registry.list_all_plugins()

    if not plugins:
        click.echo("No plugins found")
        return

    click.echo(f"Found {len(plugins)} plugin(s):\n")

    for plugin_name in plugins:
        click.echo(f"  • {plugin_name}")

        # Show implementations for this plugin
        for protocol in ['phase_hook', 'endpoint', 'timing_strategy', 'dataset_sampler']:
            impls = registry.list_implementations(protocol)
            # Filter to this plugin
            plugin_impls = [
                impl for impl in impls
                if registry._implementations[protocol][impl].plugin_name == plugin_name
            ]
            if plugin_impls:
                click.echo(f"    {protocol}: {', '.join(plugin_impls)}")


@plugins.command()
@click.argument('protocol')
def list_implementations(protocol):
    """List implementations for protocol."""
    registry = get_lazy_plugin_registry()
    registry.discover_plugins()

    impls = registry.list_implementations(protocol)

    if not impls:
        click.echo(f"No implementations found for: {protocol}")
        return

    click.echo(f"Available {protocol} implementations:\n")

    for name in impls:
        info = registry.get_implementation_info(protocol, name)
        click.echo(f"  • {name}")
        click.echo(f"    Plugin: {info['plugin']}")
        if info['description']:
            click.echo(f"    Description: {info['description']}")


@plugins.command()
@click.argument('plugin_name')
def show(plugin_name):
    """Show plugin details."""
    registry = get_lazy_plugin_registry()
    registry.discover_plugins()

    # Show all implementations from this plugin
    click.echo(f"Plugin: {plugin_name}\n")

    for protocol in ['phase_hook', 'endpoint', 'timing_strategy', 'dataset_sampler']:
        impls = registry.list_implementations(protocol)
        plugin_impls = [
            impl for impl in impls
            if registry._implementations[protocol][impl].plugin_name == plugin_name
        ]

        if plugin_impls:
            click.echo(f"{protocol}:")
            for impl in plugin_impls:
                info = registry.get_implementation_info(protocol, impl)
                click.echo(f"  • {impl}")
                if info['description']:
                    click.echo(f"    {info['description']}")
                if info['config_schema']:
                    click.echo(f"    Config: {list(info['config_schema'].keys())}")


# Add to main CLI
@click.group()
def cli():
    pass

cli.add_command(plugins)
```

---

## Usage Examples

### Install Plugin

```bash
pip install aiperf-datadog-plugin
```

### Discover Available Plugins

```bash
aiperf plugins list

# Output:
# Found 2 plugin(s):
#   • datadog
#     phase_hook: datadog, datadog_legacy
#     endpoint: datadog_llm_gateway
#   • prometheus
#     phase_hook: prometheus
```

### Show Plugin Details

```bash
aiperf plugins show datadog

# Output:
# Plugin: datadog
#
# phase_hook:
#   • datadog
#     Send phase lifecycle events and metrics to Datadog
#     Config: ['api_key', 'site', 'tags']
#   • datadog_legacy
#     Legacy Datadog integration (deprecated)
#
# endpoint:
#   • datadog_llm_gateway
#     Datadog LLM Observability Gateway
#     Config: ['gateway_url', 'api_key']
```

### Configure and Run

```yaml
# benchmark.yaml
plugins:
  phase_hooks:
    - name: datadog
      config:
        api_key: ${DATADOG_API_KEY}
```

```bash
aiperf --config benchmark.yaml

# Output:
# Discovering AIPerf plugins...
# Found 2 plugin(s): datadog, prometheus
#   phase_hook: datadog, datadog_legacy, prometheus, logging
#   endpoint: datadog_llm_gateway, ollama
# ✓ Loaded phase hook: datadog
#
# Running benchmark...
# Phase WARMUP started
# [Datadog] Event sent: Phase started
# ...
```

---

## Advantages Over Entry Point Classes

| Aspect | Entry Point → Class | Entry Point → YAML |
|--------|---------------------|-------------------|
| **Entry points** | Many (one per impl) | Few (one per plugin) |
| **Maintenance** | setup.py for each impl | Just edit YAML |
| **Metadata** | In Python code | In YAML (declarative) |
| **Discovery speed** | Fast | Faster (YAML < Python) |
| **Hot reload** | No | Possible (re-read YAML) |
| **Introspection** | Load class to see | Read YAML (no loading) |
| **Config schema** | In docstring | Structured in YAML |
| **Enable/disable** | Uninstall | Flag in YAML |

---

## Implementation Checklist

**To implement YAML plugin system:**

1. Create `LazyPluginRegistry` in `src/aiperf/common/lazy_plugin_registry.py`
2. Create `PluginManifest`, `LazyImplementation` data classes
3. Update ALL factories to check plugin registry first
4. Add `initialize_plugins()` to startup
5. Create `phase_hooks.py` with global callback registry
6. Update `PhaseOrchestrator` to use global registry
7. Add `aiperf plugins` CLI commands
8. Add `plugins:` section to config schema
9. Create plugin development guide with YAML schema
10. Create example plugin: `aiperf-example-plugin`

**Want me to implement this?** This is the ultimate extensibility system! 🚀

**Benefits:**
- ✅ Lazy loading (fast startup)
- ✅ Declarative (YAML manifests)
- ✅ Factory integration (works everywhere)
- ✅ Config-driven (enable/disable)
- ✅ CLI tools (introspection)
- ✅ Zero coupling (AIPerf → plugins)
- ✅ Rich metadata (schemas, descriptions)

This would make AIPerf world-class extensible! 🌟
