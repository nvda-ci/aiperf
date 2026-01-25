<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Example Plugin - Usage Examples

Complete working examples demonstrating how to use the plugin components.

## Quick Start

### 1. Basic Hook Registration (Module-Level API)

```python
"""Simple example of registering and using a phase hook via registry."""

import asyncio
from aiperf.plugin import plugin_registry

async def main():
    # Create hook instance via registry (auto-discovered!)
    hook = plugin_registry.create_instance(
        'phase_hook',
        'example_logging_hook',
        log_file="/tmp/aiperf_phases.log",
        verbose=True
    )

    # In real usage, register with your orchestrator
    # orchestrator.register_hook(hook)

    # Hooks are called automatically when phases execute
    # For this example, we'll simulate:
    from unittest.mock import MagicMock
    from aiperf.common.enums import CreditPhase

    phase = CreditPhase.WARMUP
    tracker = MagicMock()
    tracker.create_stats.return_value = MagicMock(
        sent=100,
        completed=95,
        in_flight=5
    )

    # Simulate phase lifecycle
    await hook.on_phase_start(phase, tracker)
    await hook.on_phase_sending_complete(phase, tracker)
    await hook.on_phase_complete(phase, tracker)

    print("Check /tmp/aiperf_phases.log for logged events")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Metrics Collection

```python
"""Example of collecting phase metrics."""

import asyncio
import json
from pathlib import Path
from aiperf.plugin import plugin_registry

async def main():
    # Create metrics collector via registry
    hook = plugin_registry.create_instance(
        'phase_hook',
        'example_metrics_collector_hook',
        metrics_file="/tmp/aiperf_metrics.json",
        aggregate=True
    )

    # Simulate multiple phases
    from unittest.mock import MagicMock
    from aiperf.common.enums import CreditPhase

    phases = [CreditPhase.WARMUP, CreditPhase.STEADY_STATE]

    for phase in phases:
        tracker = MagicMock()
        tracker.create_stats.return_value = MagicMock(
            sent=1000,
            completed=950,
            cancelled=10,
            sent_sessions=50,
            completed_sessions=45
        )

        # Simulate phase execution
        await hook.on_phase_start(phase, tracker)
        await asyncio.sleep(0.1)
        await hook.on_phase_sending_complete(phase, tracker)
        await asyncio.sleep(0.1)
        await hook.on_phase_complete(phase, tracker)

    # Retrieve collected metrics
    metrics = hook.get_aggregated_metrics()
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Processing Results

```python
"""Example of processing results and calculating metrics."""

import asyncio
from aiperf.plugin import plugin_registry

async def main():
    # Create processor via registry
    processor = plugin_registry.create_instance(
        'post_processor',
        'example_metrics_processor',
        output_file="/tmp/aiperf_metrics.txt",
        include_percentiles=True
    )

    # Sample results from a phase
    results = [
        {"id": i, "latency_ms": 50 + i, "status": "success"}
        for i in range(100)
    ]
    # Add some errors
    results.extend([
        {"id": 100 + i, "latency_ms": None, "status": "error", "error": "Timeout"}
        for i in range(5)
    ])

    # Process results
    result = await processor.process(results)

    print(f"Processed {result.record_count} records")
    print(f"Success: {result.success}")
    print(f"Metrics: {result.metrics}")
    print(f"Output written to: {result.output_path}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. Aggregating Results

```python
"""Example of aggregating results from multiple phases."""

import asyncio
from aiperf_example_plugin.processors import ExampleResultsAggregator

async def main():
    aggregator = ExampleResultsAggregator()

    # Results from different phases
    warmup_results = [
        {"id": i, "timestamp": 1000.0 + i, "latency_ms": 50.0, "status": "success"}
        for i in range(50)
    ]

    steady_state_results = [
        {"id": i, "timestamp": 2000.0 + i, "latency_ms": 45.0, "status": "success"}
        for i in range(100)
    ]

    # Aggregate
    summary = await aggregator.aggregate([
        warmup_results,
        steady_state_results
    ])

    # Generate report
    report = await aggregator.generate_report(summary)
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
```

## Integration Examples

### With AIPerf Configuration

```yaml
# aiperf_config.yaml

plugins:
  example_logging_hook:
    enabled: true
    class: aiperf_example_plugin.hooks:ExampleLoggingHook
    config:
      log_file: /var/log/aiperf/phases.log
      verbose: true

  example_metrics_collector:
    enabled: true
    class: aiperf_example_plugin.hooks:ExampleMetricsCollectorHook
    config:
      metrics_file: /tmp/metrics.json
      aggregate: true

  example_metrics_processor:
    enabled: true
    class: aiperf_example_plugin.processors:ExampleMetricsProcessor
    config:
      output_file: /tmp/metrics_report.txt
      include_percentiles: true
```

### With Custom Phase Orchestrator

```python
"""Example of integrating hooks with a custom phase orchestrator."""

import asyncio
from aiperf.plugin import plugin_registry

class MyPhaseOrchestrator:
    """Example orchestrator with plugin support."""

    def __init__(self):
        self._hooks = []
        self._processor = None

    def register_hook(self, hook):
        """Register a phase lifecycle hook."""
        self._hooks.append(hook)

    def set_processor(self, processor):
        """Set post-processor."""
        self._processor = processor

    async def execute_phase(self, phase_config):
        """Execute a phase and notify hooks."""
        from aiperf.common.models import CreditPhaseTracker

        # Create tracker for this phase
        tracker = CreditPhaseTracker(
            config=phase_config,
            dataset_sampler=None
        )
        tracker.mark_started()

        # Notify hooks of phase start
        for hook in self._hooks:
            await hook.on_phase_start(phase_config.phase, tracker)

        # Execute phase (simplified)
        await asyncio.sleep(0.1)  # Simulate phase execution

        tracker.mark_sending_complete()

        # Notify hooks of sending complete
        for hook in self._hooks:
            await hook.on_phase_sending_complete(phase_config.phase, tracker)

        # Collect results
        results = [
            {"id": i, "latency_ms": 50.0, "status": "success"}
            for i in range(100)
        ]

        tracker.mark_completed()

        # Notify hooks of phase complete
        for hook in self._hooks:
            await hook.on_phase_complete(phase_config.phase, tracker)

        # Process results if processor registered
        if self._processor:
            result = await self._processor.process(results)
            print(f"Processor result: {result}")

        return results


async def main():
    # Create orchestrator
    orchestrator = MyPhaseOrchestrator()

    # Register hooks via registry
    logging_hook = plugin_registry.create_instance(
        'phase_hook',
        'example_logging_hook',
        log_file="/tmp/phases.log"
    )
    orchestrator.register_hook(logging_hook)

    metrics_hook = plugin_registry.create_instance(
        'phase_hook',
        'example_metrics_collector_hook',
        metrics_file="/tmp/metrics.json"
    )
    orchestrator.register_hook(metrics_hook)

    # Set processor via registry
    processor = plugin_registry.create_instance(
        'post_processor',
        'example_metrics_processor',
        output_file="/tmp/metrics.txt"
    )
    orchestrator.set_processor(processor)

    # Execute phases
    from aiperf.timing.credit_models import CreditPhaseConfig
    from aiperf.common.enums import CreditPhase

    config = CreditPhaseConfig(
        phase=CreditPhase.WARMUP,
        total_expected_credits=100
    )

    results = await orchestrator.execute_phase(config)
    print(f"Phase completed with {len(results)} results")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Examples

### Custom Hook Implementation

```python
"""Example of extending the plugin with custom hooks."""

from aiperf_example_plugin.hooks import BasePhaseLifecycleHook
from aiperf.common.enums import CreditPhase
import time

class AlertingPhaseHook(BasePhaseLifecycleHook):
    """Custom hook that alerts on slow phases."""

    def __init__(self, slow_threshold_sec=60):
        self.slow_threshold_sec = slow_threshold_sec
        self._phase_start_times = {}

    async def on_phase_start(self, phase, tracker):
        """Record phase start time."""
        self._phase_start_times[str(phase)] = time.time()

    async def on_phase_complete(self, phase, tracker):
        """Alert if phase was slow."""
        phase_key = str(phase)
        start_time = self._phase_start_times.get(phase_key)

        if start_time:
            duration = time.time() - start_time

            if duration > self.slow_threshold_sec:
                await self._send_alert(
                    f"Slow phase detected: {phase} took {duration:.2f}s "
                    f"(threshold: {self.slow_threshold_sec}s)"
                )

    async def _send_alert(self, message: str):
        """Send alert (implement your alerting mechanism)."""
        print(f"ALERT: {message}")
        # Could send email, Slack message, etc.


# Usage
async def main():
    from unittest.mock import MagicMock

    hook = AlertingPhaseHook(slow_threshold_sec=2)

    phase = CreditPhase.WARMUP
    tracker = MagicMock()
    tracker.create_stats.return_value = MagicMock()

    await hook.on_phase_start(phase, tracker)
    await asyncio.sleep(2.5)  # Simulate slow phase
    await hook.on_phase_complete(phase, tracker)

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Processor with Database Export

```python
"""Example of custom processor that exports to database."""

from aiperf_example_plugin.processors import ProcessingResult
from typing import Any

class DatabaseExportProcessor:
    """Processor that exports metrics to a database."""

    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self._results_table = "aiperf_results"

    async def process(self, results: list[dict[str, Any]]) -> ProcessingResult:
        """Process and export results to database."""
        try:
            # Calculate metrics
            metrics = self._calculate_metrics(results)

            # Export to database
            await self._export_to_database(results, metrics)

            return ProcessingResult(
                success=True,
                record_count=len(results),
                metrics=metrics,
                output_path=self.db_connection_string,
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_count=1,
                metrics={"error": str(e)},
            )

    def _calculate_metrics(self, results):
        """Calculate metrics from results."""
        return {
            "total": len(results),
            "successful": sum(1 for r in results if not r.get("error")),
            "failed": sum(1 for r in results if r.get("error")),
        }

    async def _export_to_database(self, results, metrics):
        """Export to database (stub implementation)."""
        # Implement your database export logic
        print(f"Exporting {len(results)} results to {self.db_connection_string}")


# Usage
async def main():
    processor = DatabaseExportProcessor(
        db_connection_string="postgresql://user:pass@localhost/aiperf"
    )

    results = [
        {"id": 1, "latency_ms": 50.0, "status": "success"}
        for _ in range(100)
    ]

    result = await processor.process(results)
    print(f"Export result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Examples

### Testing Custom Hooks

```python
"""Example of testing custom hook implementations."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_custom_hook_collects_metrics():
    """Test that custom hook collects phase metrics."""
    from my_plugin.hooks import MyCustomHook

    hook = MyCustomHook()
    phase = MagicMock()
    tracker = MagicMock()
    tracker.create_stats.return_value = MagicMock(sent=100)

    await hook.on_phase_start(phase, tracker)

    # Verify hook collected metrics
    metrics = hook.get_metrics()
    assert metrics is not None
    assert "start_time" in metrics
```

### Testing Custom Processors

```python
"""Example of testing custom processor implementations."""

import pytest

@pytest.mark.asyncio
async def test_custom_processor_calculates_metrics():
    """Test that custom processor calculates metrics."""
    from my_plugin.processors import MyCustomProcessor

    processor = MyCustomProcessor()

    results = [
        {"id": i, "latency_ms": 50.0, "status": "success"}
        for i in range(100)
    ]

    result = await processor.process(results)

    assert result.success
    assert result.record_count == 100
    assert "custom_metric" in result.metrics
```

## Performance Optimization Examples

### Async Batch Processing

```python
"""Example of async batch processing for high-volume data."""

import asyncio
from typing import Any

class BatchMetricsProcessor:
    """Processor that handles results in batches."""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size

    async def process(self, results: list[dict[str, Any]]):
        """Process results in batches."""
        metrics = {
            "total_batches": 0,
            "total_records": 0,
            "batch_metrics": []
        }

        for i in range(0, len(results), self.batch_size):
            batch = results[i:i + self.batch_size]
            batch_metric = await self._process_batch(batch)
            metrics["batch_metrics"].append(batch_metric)
            metrics["total_batches"] += 1
            metrics["total_records"] += len(batch)

            # Yield to event loop to keep system responsive
            await asyncio.sleep(0)

        return metrics

    async def _process_batch(self, batch):
        """Process a single batch."""
        # Async operation to calculate batch metrics
        await asyncio.sleep(0)
        return {
            "count": len(batch),
            "avg_latency": sum(r.get("latency_ms", 0) for r in batch) / len(batch),
        }
```

## Debugging Examples

### Detailed Logging Hook

```python
"""Example hook with detailed debugging output."""

import json
import logging
from aiperf_example_plugin.hooks import BasePhaseLifecycleHook

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebugPhaseHook(BasePhaseLifecycleHook):
    """Hook that logs detailed debugging information."""

    async def on_phase_start(self, phase, tracker):
        """Log detailed start information."""
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        logger.debug(f"Phase start event: phase={phase}")
        if stats:
            logger.debug(f"Start stats: {json.dumps({
                'sent': stats.sent,
                'completed': stats.completed,
                'in_flight': stats.in_flight,
            }, indent=2)}")

    async def on_phase_complete(self, phase, tracker):
        """Log detailed completion information."""
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        logger.debug(f"Phase complete event: phase={phase}")
        if stats:
            logger.debug(f"Final stats: {json.dumps({
                'final_sent': stats.final_sent_count,
                'final_completed': stats.final_completed_count,
                'final_cancelled': stats.final_cancelled_count,
            }, indent=2)}")
```

## Configuration Examples

### YAML Configuration for Production

```yaml
# production_config.yaml

plugins:
  example_logging_hook:
    enabled: true
    config:
      log_file: /var/log/aiperf/phases.log
      verbose: false

  example_metrics_collector:
    enabled: true
    config:
      metrics_file: /var/log/aiperf/metrics.json
      aggregate: true

  example_metrics_processor:
    enabled: true
    config:
      output_file: /var/log/aiperf/metrics_report.txt
      include_percentiles: true
```

### YAML Configuration for Development

```yaml
# development_config.yaml

plugins:
  example_logging_hook:
    enabled: true
    config:
      log_file: /tmp/aiperf_phases.log
      verbose: true

  example_metrics_collector:
    enabled: true
    config:
      metrics_file: /tmp/aiperf_metrics.json
      aggregate: true
```

## Tips and Best Practices

1. **Use async/await consistently** - Never block the event loop
2. **Log important events** - Makes debugging easier
3. **Handle errors gracefully** - Log but don't crash
4. **Test with realistic data** - Use actual result formats
5. **Profile performance** - Monitor memory and CPU usage
6. **Use fixtures for testing** - Makes tests repeatable
7. **Document your hooks/processors** - Helps other developers
8. **Keep plugins focused** - One concern per component
