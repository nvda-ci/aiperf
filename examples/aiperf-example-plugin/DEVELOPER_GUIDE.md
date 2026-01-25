<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Example Plugin - Developer Guide

This guide explains the architecture and design patterns used in the example plugin, helping you understand and extend it for your own use cases.

## Architecture Overview

### Plugin System Architecture

```
AIPerf Core
    │
    ├─ Plugin Discovery (entry_points)
    │   └─ Scans installed packages for "aiperf.plugins"
    │
    ├─ Registry Loading (registry.yaml)
    │   └─ Reads component metadata
    │
    ├─ Component Instantiation
    │   └─ Creates hook and processor instances
    │
    └─ Event Dispatching
        └─ Notifies registered hooks at phase events
```

### Component Hierarchy

```
BasePhaseLifecycleHook (Protocol)
    │
    ├─ ExampleLoggingHook
    │   └─ Simple file logging
    │
    └─ ExampleMetricsCollectorHook
        └─ JSON metrics collection

BaseProcessor (Your custom interface)
    │
    ├─ ExampleMetricsProcessor
    │   └─ Calculates metrics
    │
    └─ ExampleResultsAggregator
        └─ Aggregates across phases
```

## File Structure and Responsibilities

### `aiperf_example_plugin/__init__.py`

**Purpose**: Package initialization and public API

**Responsibilities**:
- Define `__version__` (used by plugin loader)
- Export public classes
- Provide convenient imports for users

**Key Pattern**: Use `__all__` to explicitly define public API

```python
__all__ = [
    "ExampleLoggingHook",
    "ExampleMetricsCollectorHook",
    "ExampleMetricsProcessor",
    "ExampleResultsAggregator",
]
```

### `aiperf_example_plugin/hooks.py`

**Purpose**: Phase lifecycle hook implementations

**Components**:
1. `ExampleLoggingHook` - Simple logging hook
2. `ExampleMetricsCollectorHook` - Advanced metrics collection

**Key Patterns**:
- Inherit from `BasePhaseLifecycleHook`
- Implement async methods for phase events
- Use instance variables for state (metrics, files)
- Handle file I/O gracefully with error handling

**Phase Event Flow**:
```
on_phase_start()
    ↓
(Phase sends credits)
    ↓
on_phase_sending_complete()
    ↓
(Phase receives credits)
    ↓
on_phase_complete() or on_phase_timeout()
```

### `aiperf_example_plugin/processors.py`

**Purpose**: Post-processing of phase results

**Components**:
1. `ProcessingResult` - Data class for processor output
2. `ExampleMetricsProcessor` - Single-phase metrics
3. `ExampleResultsAggregator` - Multi-phase aggregation

**Key Patterns**:
- `ProcessingResult` provides standard output format
- Processors implement `async def process()`
- Use helper methods for metric calculations
- Support configuration via constructor parameters

**Processing Pipeline**:
```
Phase Complete
    ↓
Extract Results
    ↓
Calculate Metrics (percentiles, stats)
    ↓
Write Output (file, database, etc.)
    ↓
Return ProcessingResult
```

### `aiperf_example_plugin/registry.yaml`

**Purpose**: Component metadata and discovery

**Structure**:
```yaml
plugin:
  # Plugin metadata
  name: string
  version: string
  description: string
  author: string
  enabled: boolean

phase_hooks:
  # Each hook definition
  hook_name:
    class: module.path:ClassName
    description: string
    priority: int          # Lower = earlier execution
    tags: [list, of, tags]
    auto_load: boolean     # false = manual activation
    config_params: [list, of, params]

post_processors:
  # Each processor definition (same structure)
  processor_name:
    ...
```

**Key Patterns**:
- `priority` controls execution order
- `auto_load: false` requires explicit activation
- `config_params` document configuration options
- `tags` help filter/discover components

## Design Patterns Used

### 1. Protocol-Based Extension

**Pattern**: Define interface via Protocol, implement with concrete classes

**Benefits**:
- Loose coupling between AIPerf and plugins
- Type-safe without inheritance requirements
- Works with duck typing

**Example**:
```python
@runtime_checkable
class PhaseLifecycleHook(Protocol):
    async def on_phase_start(self, phase, tracker) -> None: ...

# Any class implementing these methods works
class MyHook:
    async def on_phase_start(self, phase, tracker) -> None:
        pass  # Implementation
```

### 2. Base Class with No-Op Defaults

**Pattern**: Provide base class with empty implementations

**Benefits**:
- Hooks only implement events they care about
- Subclasses can override selectively
- Avoids repetitive empty methods

**Example**:
```python
class BasePhaseLifecycleHook:
    async def on_phase_start(self, phase, tracker) -> None:
        pass  # No-op default

class MyHook(BasePhaseLifecycleHook):
    async def on_phase_complete(self, phase, tracker) -> None:
        # Only implement what we need
        await self._do_something()
```

### 3. Configuration via Constructor

**Pattern**: Accept configuration parameters in `__init__`

**Benefits**:
- Flexible without config files
- Works with YAML configs (instantiate with dict)
- Type-safe (hint types in signature)

**Example**:
```python
class ExampleLoggingHook(BasePhaseLifecycleHook):
    def __init__(
        self,
        log_file: str = "/tmp/phases.log",
        verbose: bool = False
    ) -> None:
        self.log_file = Path(log_file)
        self.verbose = verbose
```

### 4. Immutable Data Models

**Pattern**: Use `Struct` from msgspec for immutable data

**Benefits**:
- Prevents accidental mutations
- More memory efficient than dataclasses
- Type-safe with msgspec

**Example**:
```python
from msgspec import Struct

@dataclass
class ProcessingResult:
    success: bool
    record_count: int = 0
    metrics: dict | None = None
```

### 5. Async/Await for I/O

**Pattern**: Use async for all I/O operations

**Benefits**:
- Non-blocking (doesn't freeze event loop)
- Composable with other async operations
- Required for AIPerf architecture

**Example**:
```python
async def _write_metrics(self, metrics):
    # Async file write (non-blocking)
    with open(self.file, 'w') as f:
        f.write(json.dumps(metrics))
```

### 6. Error Handling with Graceful Degradation

**Pattern**: Log errors but don't crash

**Benefits**:
- Plugin failures don't stop benchmarks
- Issues are visible in logs
- System stays resilient

**Example**:
```python
try:
    await risky_operation()
except Exception as e:
    self.error(f"Operation failed: {e!r}")
    # Continue execution
```

### 7. Registry-Based Discovery

**Pattern**: YAML registry declares components

**Benefits**:
- Components discoverable without code changes
- Metadata available for UI/tools
- Priority-based execution order
- Tagging for filtering

**Example**:
```yaml
phase_hooks:
  my_hook:
    class: module:MyHook
    priority: 50
    tags: [logging]
```

## Key Implementation Details

### Thread Safety

The plugin uses asyncio, so:
- No explicit locking needed
- Atomic operations safe without locks
- Proper use of `await` prevents race conditions

### File I/O Patterns

**Best Practices**:
1. Create directories in `__init__` if possible
2. Use `Path` from pathlib
3. Handle I/O errors gracefully
4. Use async when available

```python
def __init__(self, output_file: str):
    self.output_file = Path(output_file)
    self.output_file.parent.mkdir(parents=True, exist_ok=True)
```

### Metric Calculations

**Percentile Calculation**:
```python
@staticmethod
def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = (percentile / 100.0) * len(sorted_values)
    if index % 1 == 0:
        return sorted_values[int(index) - 1]
    return sorted_values[int(index)]
```

**Success Rate Calculation**:
```python
successful = sum(1 for r in results if not r.get("error"))
success_rate = successful / len(results) if results else 0
```

## Extending the Plugin

### Adding a New Hook

1. **Create Hook Class**:
```python
class MyCustomHook(BasePhaseLifecycleHook):
    async def on_phase_complete(self, phase, tracker):
        # Your logic
        pass
```

2. **Add to Registry**:
```yaml
phase_hooks:
  my_custom_hook:
    class: aiperf_example_plugin.hooks:MyCustomHook
    description: My custom hook
    priority: 50
    tags: [custom]
    auto_load: false
```

3. **Export in `__init__.py`**:
```python
from aiperf_example_plugin.hooks import MyCustomHook
__all__ = [..., "MyCustomHook"]
```

### Adding a New Processor

1. **Create Processor Class**:
```python
class MyProcessor:
    async def process(self, results):
        # Your logic
        metrics = await self._calculate()
        return ProcessingResult(
            success=True,
            record_count=len(results),
            metrics=metrics
        )
```

2. **Add to Registry**:
```yaml
post_processors:
  my_processor:
    class: aiperf_example_plugin.processors:MyProcessor
    description: My processor
    priority: 60
```

### Custom Hook Example: Alerting

```python
class AlertingHook(BasePhaseLifecycleHook):
    def __init__(self, alert_handler):
        self.alert_handler = alert_handler
        self._phase_starts = {}

    async def on_phase_complete(self, phase, tracker):
        stats = tracker.create_stats()

        # Calculate duration
        duration = (stats.end_ns - stats.start_ns) / 1e9

        if duration > 60:
            await self.alert_handler.send_alert(
                f"Slow phase: {phase} took {duration}s"
            )
```

## Testing Patterns

### Testing Hooks

```python
@pytest.mark.asyncio
async def test_hook_receives_events():
    hook = ExampleLoggingHook(log_file="/tmp/test.log")

    tracker = MagicMock()
    tracker.create_stats.return_value = MagicMock(sent=100)

    await hook.on_phase_start(CreditPhase.WARMUP, tracker)

    # Verify hook acted correctly
    assert Path("/tmp/test.log").exists()
```

### Testing Processors

```python
@pytest.mark.asyncio
async def test_processor_calculates_metrics():
    processor = ExampleMetricsProcessor()

    results = [
        {"latency_ms": 50.0, "status": "success"}
        for _ in range(100)
    ]

    result = await processor.process(results)

    assert result.success
    assert result.metrics["request_counts"]["total"] == 100
```

### Test Fixtures

```python
@pytest.fixture
def mock_tracker():
    """Mock phase tracker."""
    tracker = MagicMock()
    tracker.create_stats.return_value = MagicMock(
        sent=100,
        completed=95,
        in_flight=5
    )
    return tracker

@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
```

## Performance Considerations

### Async Best Practices

1. **Yield control frequently**:
```python
for i, result in enumerate(large_list):
    process(result)
    if i % 100 == 0:
        await asyncio.sleep(0)
```

2. **Batch operations**:
```python
# Process in batches instead of one-by-one
for i in range(0, len(results), BATCH_SIZE):
    batch = results[i:i+BATCH_SIZE]
    await process_batch(batch)
```

3. **Use batch file operations**:
```python
# Write once instead of multiple times
with open(file, 'w') as f:
    for item in items:
        f.write(serialize(item) + "\n")
```

### Memory Efficiency

1. **Don't store unnecessary data**:
```python
# Bad: stores all results
self._results = results

# Good: only stores what's needed
self._metrics = calculate_metrics(results)
```

2. **Use generators for large datasets**:
```python
def process_results(results):
    for result in results:
        yield calculate_metric(result)
```

## Debugging Tips

### Enable Verbose Logging

```python
hook = ExampleLoggingHook(
    log_file="/tmp/phases.log",
    verbose=True  # Include detailed stats
)
```

### Add Debug Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebugHook(BasePhaseLifecycleHook):
    async def on_phase_start(self, phase, tracker):
        stats = tracker.create_stats()
        logger.debug(f"Phase {phase} stats: {stats}")
```

### Monitor File I/O

```python
async def _write_log(self, message):
    try:
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        logger.debug(f"Wrote to {self.log_file}")
    except IOError as e:
        logger.error(f"Failed to write: {e}")
```

## Integration Points

### With AIPerf Core

1. **Phase Lifecycle**: Hooks registered with phase orchestrator
2. **Result Pipeline**: Processors called after phase completion
3. **Configuration**: Plugins loaded via registry.yaml
4. **Discovery**: Entry points scan for installed plugins

### With External Systems

1. **Metrics Export**: Send to Prometheus, Datadog, etc.
2. **Alert Services**: Send alerts to PagerDuty, Slack, etc.
3. **Database**: Store results in PostgreSQL, MongoDB, etc.
4. **Log Aggregation**: Send logs to ELK, Splunk, etc.

## Best Practices Checklist

- [ ] Use type hints on all functions
- [ ] Document public methods with docstrings
- [ ] Handle exceptions gracefully
- [ ] Use async/await consistently
- [ ] Test with realistic data
- [ ] Add registry entries for discoverability
- [ ] Export public API in `__init__.py`
- [ ] Use descriptive names for variables/methods
- [ ] Comment "why" not "what"
- [ ] Follow KISS principle (Keep It Simple, Stupid)

## Common Mistakes to Avoid

1. **Blocking the event loop**:
```python
# BAD: blocks
time.sleep(1)
requests.get(url)

# GOOD: async
await asyncio.sleep(1)
await session.get(url)
```

2. **Storing mutable state**:
```python
# BAD: service state persists
class MyHook:
    def __init__(self):
        self.results = []  # Grows unbounded

# GOOD: compute on demand
class MyHook:
    async def _calculate_metrics(self, results):
        return sum(r.latency for r in results)
```

3. **Silent failures**:
```python
# BAD: errors hidden
try:
    dangerous_operation()
except:
    pass

# GOOD: errors logged
try:
    dangerous_operation()
except Exception as e:
    logger.error(f"Operation failed: {e!r}")
```

4. **Not handling missing data**:
```python
# BAD: crashes on missing fields
latency = result['latency_ms']

# GOOD: handle missing data
latency = result.get('latency_ms', 0)
if latency is None:
    latency = 0
```

## Resources

- AIPerf Repository: https://github.com/NVIDIA/aiperf
- Plugin Architecture Docs: See main repo documentation
- AIPerf Design Patterns: Review `CLAUDE.md` in main repo
- Phase Lifecycle: See `phase_lifecycle_hooks.py`

## Next Steps

1. **Run Example**: Install and run the plugin
2. **Review Code**: Study the implementations
3. **Write Tests**: Add tests for new components
4. **Extend Plugin**: Add your own hooks/processors
5. **Integrate**: Use with your benchmarks
6. **Publish**: Share plugin with community
