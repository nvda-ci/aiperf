<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Example Plugin

This is a complete working example of an AIPerf plugin demonstrating extensibility patterns for the AIPerf benchmarking framework.

## Overview

The AIPerf plugin system allows developers to extend the framework with custom functionality without modifying core code. This example plugin demonstrates:

- **Phase Lifecycle Hooks**: Monitor and react to phase events (start, sending complete, complete, timeout)
- **Custom Metrics Processing**: Calculate and aggregate custom metrics from results
- **Registry-based Discovery**: Use YAML registry for component metadata and automatic loading
- **Async/Await Patterns**: Proper async handling throughout
- **Configuration Parameters**: Support for runtime configuration

## Directory Structure

```
aiperf-example-plugin/
├── aiperf_example_plugin/
│   ├── __init__.py                 # Package init and public API
│   ├── hooks.py                    # Phase lifecycle hook implementations
│   ├── processors.py               # Result processors and aggregators
│   └── registry.yaml               # Plugin component registry
├── tests/
│   └── test_hooks.py               # Hook unit tests
│   └── test_processors.py          # Processor unit tests
├── setup.py                        # setuptools configuration
├── pyproject.toml                  # Modern Python packaging
└── README.md                       # This file
```

## Installation

### From Source (Development)

Install in editable mode for development:

```bash
cd examples/aiperf-example-plugin
pip install -e .
```

This installs the plugin in your Python environment and registers it with AIPerf's plugin system.

### From PyPI (Production)

Once published to PyPI:

```bash
pip install aiperf-example-plugin
```

### Usage (Auto-Discovery)

Once installed, the plugin is automatically discovered by AIPerf:

```python
# The plugin is auto-discovered via entry points!
from aiperf.plugin import plugin_registry

# List available hooks (includes plugin hooks)
hooks = plugin_registry.list_implementations('phase_hook')
# Returns: ['logging', 'example_logging_hook', 'example_metrics_collector_hook', ...]

# Create instance of plugin hook
hook = plugin_registry.create_instance(
    'phase_hook',
    'example_logging_hook',
    log_file="/tmp/phases.log",
    verbose=True
)
```

## Components

### Phase Lifecycle Hooks

Phase hooks receive notifications at key execution events and can perform logging, metrics collection, or other side effects.

#### ExampleLoggingHook

Simple file-based logging of phase transitions.

```python
from aiperf_example_plugin.hooks import ExampleLoggingHook

hook = ExampleLoggingHook(
    log_file="/var/log/aiperf/phases.log",
    verbose=True
)

# Register with your phase orchestrator
orchestrator.register_hook(hook)
```

**Logged Events:**
- `PHASE_START`: Phase begins execution
- `PHASE_SENDING_COMPLETE`: All credits sent to workers
- `PHASE_COMPLETE`: All credits returned from workers
- `PHASE_TIMEOUT`: Phase exceeded time limit

**Output:**
```
[2025-01-15 10:30:45] PHASE_START: WARMUP
[2025-01-15 10:30:50] PHASE_SENDING_COMPLETE: WARMUP | Sent: 1000, Sessions: 10, Duration: 5.2s
[2025-01-15 10:30:52] PHASE_COMPLETE: WARMUP | Completed: 1000, Cancelled: 0, Total Duration: 7.1s
```

#### ExampleMetricsCollectorHook

Advanced hook that collects phase metrics to JSON for analysis.

```python
from aiperf_example_plugin.hooks import ExampleMetricsCollectorHook

hook = ExampleMetricsCollectorHook(
    metrics_file="/tmp/aiperf_metrics.json",
    aggregate=True
)

orchestrator.register_hook(hook)

# After execution, retrieve collected metrics
metrics = hook.get_aggregated_metrics()
print(f"Phase durations: {metrics['phase_durations']}")
```

**Collected Metrics:**
- Event timestamps and types
- Request counts and session info
- Phase durations (sending and total)
- In-flight request counts

### Post-Processors

Post-processors analyze results after phase execution.

#### ExampleMetricsProcessor

Calculates custom metrics from result sets.

```python
from aiperf_example_plugin.processors import ExampleMetricsProcessor

processor = ExampleMetricsProcessor(
    output_file="/tmp/aiperf_metrics.txt",
    include_percentiles=True
)

# Process results
result = await processor.process(raw_results)

print(f"Processed {result.record_count} records")
print(f"Metrics: {result.metrics}")
```

**Calculated Metrics:**
- Total, successful, and failed request counts
- Success rate
- Latency percentiles (P50, P75, P90, P95, P99)
- Min, max, mean, and standard deviation
- Error rates

#### ExampleResultsAggregator

Combines results from multiple phases or runs.

```python
from aiperf_example_plugin.processors import ExampleResultsAggregator

aggregator = ExampleResultsAggregator()

# Aggregate multiple result sets
summary = await aggregator.aggregate([
    results_warmup_phase,
    results_steady_state_phase,
])

# Generate report
report = await aggregator.generate_report(summary)
print(report)
```

## Usage Examples

### Basic Usage with Module-Level API

```python
from aiperf.plugin import plugin_registry

# Auto-discovery! Plugin hooks are available via registry
hook = plugin_registry.create_instance(
    'phase_hook',
    'example_logging_hook',
    log_file="/tmp/aiperf.log",
    verbose=True
)

# Register with orchestrator
orchestrator.register_hook(hook)

# Execute phases - hook receives notifications automatically
await orchestrator.execute_phase(config)

# Create processor via registry
processor = plugin_registry.create_instance(
    'post_processor',
    'example_metrics_processor',
    output_file="/tmp/metrics.txt"
)
result = await processor.process(phase_results)
```

### Direct Import (Alternative)

```python
from aiperf_example_plugin import (
    ExampleLoggingHook,
    ExampleMetricsProcessor,
)

# Create hook instance directly
hook = ExampleLoggingHook(log_file="/tmp/aiperf.log")

# Register with orchestrator
orchestrator.register_hook(hook)

# Execute phases - hook receives notifications automatically
await orchestrator.execute_phase(config)

# Process results
processor = ExampleMetricsProcessor()
result = await processor.process(phase_results)
```

### Advanced: Custom Hook Extension

Extend the base classes for custom implementations:

```python
from aiperf_example_plugin.hooks import BasePhaseLifecycleHook
from aiperf.common.enums import CreditPhase

class CustomAlertHook(BasePhaseLifecycleHook):
    """Alert on slow phases."""

    async def on_phase_complete(self, phase: CreditPhase, tracker):
        stats = tracker.create_stats()
        duration = (stats.end_ns - stats.start_ns) / 1e9

        if duration > 60:  # More than 60 seconds
            await self._send_alert(
                f"Slow phase detected: {phase} took {duration}s"
            )

    async def _send_alert(self, message: str):
        # Implement your alerting logic
        pass
```

### Integration with Configuration

Use YAML config to enable plugins via the registry:

```yaml
# aiperf_config.yaml
plugins:
  phase_hooks:
    - name: example_logging_hook
      config:
        log_file: /var/log/aiperf/phases.log
        verbose: true

    - name: example_metrics_collector_hook
      config:
        metrics_file: /tmp/metrics.json
        aggregate: true

  post_processors:
    - name: example_metrics_processor
      config:
        output_file: /var/log/aiperf/metrics.txt
        include_percentiles: true
```

The AIPerf framework automatically loads configured plugins via the registry:

```python
from aiperf.plugin import plugin_registry

# Load hooks from config
for hook_config in config.get('plugins', {}).get('phase_hooks', []):
    hook = plugin_registry.create_instance(
        'phase_hook',
        hook_config['name'],
        **hook_config.get('config', {})
    )
    orchestrator.register_hook(hook)
```

## Registry Format

The `registry.yaml` file defines plugin components for discovery and loading:

```yaml
plugin:
  name: aiperf-example-plugin
  version: 1.0.0
  description: Example plugin
  enabled: true

phase_hooks:
  example_logging_hook:
    class: aiperf_example_plugin.hooks:ExampleLoggingHook
    description: Phase event logger
    priority: 50                    # Lower = earlier execution

post_processors:
  example_metrics_processor:
    class: aiperf_example_plugin.processors:ExampleMetricsProcessor
    description: Custom metrics calculator
    priority: 60
```

## Testing

Run the plugin's unit tests:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=aiperf_example_plugin
```

Example test:

```python
import pytest
from aiperf_example_plugin.hooks import ExampleLoggingHook

@pytest.mark.asyncio
async def test_logging_hook():
    hook = ExampleLoggingHook(log_file="/tmp/test_phases.log")

    # Mock phase and tracker
    phase = CreditPhase.WARMUP
    tracker = Mock()
    tracker.create_stats.return_value = Mock(sent=100)

    # Test hook receives events
    await hook.on_phase_start(phase, tracker)

    # Verify log file written
    log_content = Path("/tmp/test_phases.log").read_text()
    assert "PHASE_START: WARMUP" in log_content
```

## Development Guide

### Creating Custom Hooks

1. Inherit from `BasePhaseLifecycleHook`
2. Override event methods as needed
3. Implement your logic (logging, metrics, alerting, etc.)
4. Register with phase orchestrator

```python
from aiperf.timing.phase_lifecycle_hooks import BasePhaseLifecycleHook

class MyCustomHook(BasePhaseLifecycleHook):
    """Your custom phase hook."""

    async def on_phase_start(self, phase, tracker):
        """Called when phase starts."""
        # Your logic here
        pass
```

### Creating Custom Processors

1. Implement `process()` method accepting results
2. Return `ProcessingResult` with metrics
3. Can write to files, send to services, etc.

```python
from aiperf_example_plugin.processors import ProcessingResult

class MyProcessor:
    """Your custom result processor."""

    async def process(self, results):
        """Process results and return metrics."""
        metrics = await self._analyze(results)
        return ProcessingResult(
            success=True,
            record_count=len(results),
            metrics=metrics,
        )
```

### Adding to Registry

Update `registry.yaml` to register your components:

```yaml
phase_hooks:
  my_custom_hook:
    class: aiperf_example_plugin.custom:MyCustomHook
    description: My custom hook
    priority: 50
```

## Architecture Patterns

### Event-Driven Design

Phase lifecycle hooks follow a pub/sub pattern:
- Phase orchestrator publishes events (start, complete, etc.)
- Hooks subscribe to events
- Multiple hooks can observe same events
- Execution order controlled by priority

### Async/Await

All operations use async/await for non-blocking I/O:

```python
# File writes are async to avoid blocking the event loop
async def _write_metrics(self, metrics):
    with open(self.output_file, 'w') as f:
        f.write(json.dumps(metrics))
```

### Immutable Data

Use `Struct` from msgspec for immutable data models:

```python
from msgspec import Struct

class MyData(Struct, frozen=True):
    """Immutable data model."""
    value: int
    timestamp: float
```

### Type Safety

Full type hints throughout:

```python
from typing import Any

async def process(self, results: list[dict[str, Any]]) -> ProcessingResult:
    """Type-safe method signature."""
    pass
```

## Extending AIPerf

### Plugin Discovery

AIPerf discovers plugins via setuptools entry points:

```python
# setup.py
entry_points={
    "aiperf.plugins": [
        "example_plugin = aiperf_example_plugin:__version__",
    ],
}
```

### Configuration Loading

Plugins can provide YAML configs:

```python
# registry.yaml provides component metadata
# AIPerf loader inspects registry to understand plugin capabilities
# Users enable via configuration files
```

### Hook Registration

Hooks are registered when orchestrator initializes:

```python
# Orchestrator introspects registry.yaml
# For each enabled hook, instantiates and registers
await orchestrator.initialize()  # Registers all enabled hooks
```

## Best Practices

1. **Always use type hints** - Makes code maintainable and catches bugs
2. **Use Struct for immutable data** - Prevents accidental mutations
3. **Implement description and priority in registry** - Helps with debugging
4. **Use descriptive field names** - Makes logs and output readable
5. **Implement proper error handling** - Log errors but don't fail silently
6. **Test async code with pytest-asyncio** - Ensures correct async behavior
7. **Document config parameters** - Users need to know what's available
8. **Follow KISS principle** - Keep hooks/processors simple and focused

## Troubleshooting

### Plugin Not Loading

1. Check installation: `pip show aiperf-example-plugin`
2. Verify entry points: `pip show -f aiperf-example-plugin`
3. Check registry.yaml is included: `pip show -f aiperf-example-plugin | grep registry.yaml`
4. Verify AIPerf can import: `python -c "import aiperf_example_plugin"`

### Hook Not Called

1. Check hook is enabled in config
2. Verify priority isn't set too high (lower = earlier)
3. Check hook registered with orchestrator
4. Verify phase is actually running

### Metrics Not Written

1. Check output directory exists and is writable
2. Verify async/await used correctly
3. Check for exceptions in hook/processor

## Publishing to PyPI

To publish your plugin to PyPI:

```bash
# Build distribution
python -m build

# Upload to PyPI (requires PyPI account and credentials)
twine upload dist/*
```

Users can then install with:

```bash
pip install aiperf-example-plugin
```

## License

Apache License 2.0 - See LICENSE file

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure tests pass: `pytest`
5. Submit a pull request

## Support

For issues or questions:

- GitHub Issues: https://github.com/NVIDIA/aiperf/issues
- Documentation: https://github.com/NVIDIA/aiperf/wiki
- Email: sw-dl-dynamo@nvidia.com

## References

- AIPerf Documentation: https://github.com/NVIDIA/aiperf
- Plugin System Design: See `HOOK_REGISTRATION_PATTERNS.md` in main AIPerf repo
- Phase Architecture: See `src/aiperf/timing/phase_lifecycle_hooks.py`
- Message Models: See `src/aiperf/common/models/credit_structs.py`
