<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Plugin System Best Practices Guide 2025

**The ultimate guide for creating high-quality, production-ready AIPerf plugins using modern Python patterns.**

## Table of Contents

1. [Modern Python Patterns](#1-modern-python-patterns)
2. [Plugin Development Checklist](#2-plugin-development-checklist)
3. [Code Quality Standards](#3-code-quality-standards)
4. [Performance Best Practices](#4-performance-best-practices)
5. [Security Considerations](#5-security-considerations)
6. [Testing Patterns](#6-testing-patterns)
7. [Documentation Standards](#7-documentation-standards)
8. [Publishing Checklist](#8-publishing-checklist)
9. [Example Code](#9-example-code)
10. [Anti-Patterns to Avoid](#10-anti-patterns-to-avoid)
11. [Migration from Old Factories](#11-migration-from-old-factories)
12. [Common Issues and Solutions](#12-common-issues-and-solutions)

---

## 1. Modern Python Patterns

### Type Hints with Protocols

**Use runtime-checkable protocols for extensibility:**

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class PhaseLifecycleHook(Protocol):
    """Protocol for phase lifecycle hooks.

    Plugins must implement this protocol to receive phase events.
    """

    async def on_phase_start(
        self,
        phase: CreditPhase,
        tracker: CreditPhaseTracker
    ) -> None:
        """Handle phase start event.

        Args:
            phase: Phase type (WARMUP, PROFILING, etc.)
            tracker: Phase statistics tracker
        """
        ...

    async def on_phase_complete(
        self,
        phase: CreditPhase,
        tracker: CreditPhaseTracker
    ) -> None:
        """Handle phase completion.

        Args:
            phase: Completed phase
            tracker: Final phase statistics
        """
        ...
```

**Why protocols?**
- ✅ Duck typing with type safety
- ✅ No inheritance required
- ✅ Easy to test and mock
- ✅ Clear contract definition

### Pydantic for Validation

**Always use Pydantic for configuration and data models:**

```python
from pydantic import BaseModel, Field, field_validator

class PluginConfig(BaseModel):
    """Configuration for phase logging hook.

    Validates inputs and provides defaults.
    """

    log_file: str = Field(
        default="/tmp/aiperf_phases.log",
        description="Path to write phase logs"
    )
    verbose: bool = Field(
        default=False,
        description="Enable detailed logging with statistics"
    )
    max_file_size_mb: int = Field(
        default=100,
        gt=0,
        le=1000,
        description="Maximum log file size before rotation"
    )

    @field_validator("log_file")
    @classmethod
    def validate_log_path(cls, v: str) -> str:
        """Ensure log directory is writable."""
        from pathlib import Path
        path = Path(v)
        if not path.parent.exists():
            raise ValueError(f"Parent directory does not exist: {path.parent}")
        return v

class PhaseEvent(BaseModel):
    """Phase lifecycle event data."""

    phase: str = Field(description="Phase name")
    timestamp: float = Field(description="Unix timestamp")
    sent: int = Field(default=0, description="Credits sent")
    completed: int = Field(default=0, description="Credits completed")

    class Config:
        frozen = True  # Immutable
```

**Benefits:**
- ✅ Automatic validation
- ✅ Type safety
- ✅ Clear error messages
- ✅ JSON serialization
- ✅ Documentation from descriptions

### pathlib for Paths

**Always use pathlib.Path for file system operations:**

```python
from pathlib import Path

class FileLogger:
    """File logger using pathlib for cross-platform compatibility."""

    def __init__(self, log_file: str | Path) -> None:
        # Convert to Path immediately
        self.log_file = Path(log_file).resolve()

        # Create parent directories
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Check writability
        if not self.log_file.parent.is_dir():
            raise ValueError(f"Parent is not a directory: {self.log_file.parent}")

    async def write_log(self, message: str) -> None:
        """Write log using async I/O."""
        import aiofiles

        async with aiofiles.open(self.log_file, mode="a") as f:
            await f.write(message + "\n")

    def rotate_if_needed(self) -> None:
        """Rotate log file if too large."""
        if not self.log_file.exists():
            return

        size_mb = self.log_file.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            backup = self.log_file.with_suffix(".log.old")
            self.log_file.rename(backup)
```

**Why pathlib?**
- ✅ Cross-platform (Windows/Linux/Mac)
- ✅ Readable and chainable
- ✅ Built-in operations (mkdir, exists, stat)
- ✅ Type-safe path operations

### Async/Await Best Practices

**Use async for all I/O operations:**

```python
import asyncio
import aiofiles
import aiohttp

class DatadogExporter:
    """Export metrics to Datadog using async I/O."""

    def __init__(self, api_key: str, api_url: str) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Create session on enter."""
        self._session = aiohttp.ClientSession(
            headers={"DD-API-KEY": self.api_key}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session on exit."""
        if self._session:
            await self._session.close()

    async def send_metric(
        self,
        metric_name: str,
        value: float,
        tags: list[str]
    ) -> None:
        """Send metric to Datadog asynchronously."""
        if not self._session:
            raise RuntimeError("Session not initialized (use async with)")

        payload = {
            "series": [{
                "metric": metric_name,
                "points": [[int(asyncio.get_event_loop().time()), value]],
                "tags": tags,
            }]
        }

        async with self._session.post(
            f"{self.api_url}/api/v1/series",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status != 202:
                error = await resp.text()
                raise RuntimeError(f"Failed to send metric: {error}")
```

**Async patterns:**
- ✅ Use `async with` for resource management
- ✅ Never block with `time.sleep()` (use `asyncio.sleep()`)
- ✅ Use `aiofiles` for file I/O
- ✅ Use `aiohttp` for HTTP requests
- ✅ Set timeouts on all I/O operations

### Context Managers

**Use context managers for resource cleanup:**

```python
from contextlib import asynccontextmanager

class MetricsCollector:
    """Collect metrics with automatic cleanup."""

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        self._buffer: list[dict] = []
        self._file_handle = None

    @asynccontextmanager
    async def collection_context(self):
        """Context manager for metric collection lifecycle."""
        try:
            # Setup
            self._file_handle = await aiofiles.open(
                self.output_file,
                mode="w"
            )
            await self._file_handle.write("[\n")

            yield self

        finally:
            # Cleanup (always runs)
            await self._flush_buffer()
            if self._file_handle:
                await self._file_handle.write("\n]\n")
                await self._file_handle.close()

    async def add_metric(self, metric: dict) -> None:
        """Add metric to buffer."""
        self._buffer.append(metric)

        # Flush when buffer is large
        if len(self._buffer) >= 1000:
            await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """Flush buffer to file."""
        if not self._buffer or not self._file_handle:
            return

        import json
        for metric in self._buffer:
            await self._file_handle.write(
                json.dumps(metric) + ",\n"
            )

        self._buffer.clear()

# Usage
async def collect_metrics():
    collector = MetricsCollector(Path("/tmp/metrics.json"))

    async with collector.collection_context():
        await collector.add_metric({"name": "latency", "value": 1.2})
        await collector.add_metric({"name": "throughput", "value": 100})
        # Automatically flushed and closed on exit
```

### Match/Case for Parsing

**Use match/case (Python 3.10+) for cleaner parsing:**

```python
from enum import Enum

class EventType(Enum):
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    PHASE_TIMEOUT = "phase_timeout"

class EventHandler:
    """Handle phase events using match/case."""

    async def handle_event(
        self,
        event_type: EventType,
        data: dict
    ) -> None:
        """Route event to appropriate handler."""

        match event_type:
            case EventType.PHASE_START:
                await self._handle_phase_start(data)

            case EventType.PHASE_COMPLETE:
                await self._handle_phase_complete(data)

            case EventType.PHASE_TIMEOUT:
                await self._handle_phase_timeout(data)

            case _:
                self.warning(f"Unknown event type: {event_type}")

    async def parse_phase_stats(self, stats: dict) -> PhaseStats:
        """Parse statistics with match/case."""

        match stats:
            case {"sent": int(sent), "completed": int(completed), **rest}:
                return PhaseStats(
                    sent=sent,
                    completed=completed,
                    extra=rest
                )

            case {"error": str(error)}:
                raise ValueError(f"Stats contain error: {error}")

            case _:
                raise ValueError(f"Invalid stats format: {stats}")
```

---

## 2. Plugin Development Checklist

### Essential Requirements

**Use this checklist for every plugin:**

- [ ] **Use pyproject.toml (not setup.py)**
  ```toml
  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"

  [project]
  name = "aiperf-my-plugin"
  version = "1.0.0"
  description = "My AIPerf plugin"
  requires-python = ">=3.10"
  dependencies = [
      "aiperf>=2.0.0",
  ]

  [project.entry-points."aiperf.plugins"]
  my_plugin = "aiperf_my_plugin:get_registry_path"
  ```

- [ ] **Add complete type hints**
  ```python
  # Good
  async def process(self, data: dict[str, Any]) -> ProcessResult:
      ...

  # Bad
  async def process(self, data):
      ...
  ```

- [ ] **Use Pydantic for config**
  ```python
  class MyPluginConfig(BaseModel):
      api_key: str = Field(description="API key")
      timeout_seconds: int = Field(default=30, gt=0)
  ```

- [ ] **Implement protocol correctly**
  ```python
  @runtime_checkable
  class PhaseHookProtocol(Protocol):
      async def on_phase_start(
          self, phase: CreditPhase, tracker
      ) -> None: ...
  ```

- [ ] **Add comprehensive docstrings**
  ```python
  def my_function(x: int) -> str:
      """Convert integer to string representation.

      Args:
          x: Integer to convert

      Returns:
          String representation of x

      Raises:
          ValueError: If x is negative

      Example:
          >>> my_function(42)
          '42'
      """
  ```

- [ ] **Include tests (>80% coverage)**
  ```python
  # tests/test_my_hook.py
  import pytest

  @pytest.mark.asyncio
  async def test_phase_start():
      hook = MyHook(log_file="/tmp/test.log")
      await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)
      assert Path("/tmp/test.log").exists()
  ```

- [ ] **Add README with examples**
  ```markdown
  # AIPerf My Plugin

  ## Installation
  ```bash
  pip install aiperf-my-plugin
  ```

  ## Usage
  ```yaml
  plugins:
    phase_hooks:
      - name: my_hook
        config:
          verbose: true
  ```
  ```

- [ ] **Use semantic versioning**
  ```
  1.0.0  - Initial release
  1.1.0  - New feature (backward compatible)
  1.1.1  - Bug fix
  2.0.0  - Breaking change
  ```

- [ ] **Include LICENSE file**
  ```
  Apache-2.0, MIT, or other permissive license
  ```

- [ ] **Configure entry point correctly**
  ```python
  # aiperf_my_plugin/__init__.py
  from pathlib import Path

  def get_registry_path() -> Path:
      """Return path to plugin registry.yaml."""
      return Path(__file__).parent / "registry.yaml"
  ```

---

## 3. Code Quality Standards

### Type Checking

**Use pyright or mypy for static type checking:**

```bash
# pyproject.toml
[tool.pyright]
pythonVersion = "3.10"
typeCheckingMode = "strict"
reportMissingTypeStubs = false
```

```python
# Good: Complete type hints
from typing import Any

async def process_data(
    data: dict[str, Any],
    timeout: float = 30.0
) -> tuple[bool, str]:
    """Process data with timeout.

    Args:
        data: Data to process
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    ...

# Bad: Missing types
async def process_data(data, timeout=30.0):
    ...
```

### Linting with Ruff

**Configure ruff for consistent code style:**

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # Line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports
"tests/*" = ["S101"]      # Allow assert in tests
```

```bash
# Format code
ruff format .

# Check and fix issues
ruff check --fix .
```

### Formatting

**Use ruff format (Black-compatible):**

```python
# Good: Properly formatted
def my_function(
    long_parameter_name: str,
    another_parameter: int,
    yet_another_parameter: bool = False,
) -> dict[str, Any]:
    """Function with proper formatting."""
    return {
        "parameter1": long_parameter_name,
        "parameter2": another_parameter,
        "parameter3": yet_another_parameter,
    }

# Bad: Inconsistent formatting
def my_function(long_parameter_name: str, another_parameter: int,
                yet_another_parameter: bool = False) -> dict[str, Any]:
    return {"parameter1": long_parameter_name, "parameter2": another_parameter, "parameter3": yet_another_parameter}
```

### Testing with pytest

**Write comprehensive async tests:**

```python
# tests/test_my_hook.py
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from aiperf_my_plugin.hooks import MyPhaseHook
from aiperf.common.enums import CreditPhase

@pytest.fixture
def temp_log_file(tmp_path):
    """Provide temporary log file."""
    return tmp_path / "test.log"

@pytest.fixture
def mock_tracker():
    """Mock phase tracker."""
    tracker = Mock()
    stats = Mock()
    stats.sent = 100
    stats.completed = 95
    stats.in_flight = 5
    tracker.create_stats.return_value = stats
    return tracker

@pytest.mark.asyncio
async def test_phase_start_creates_log(temp_log_file, mock_tracker):
    """Test that phase start creates log file."""
    hook = MyPhaseHook(log_file=str(temp_log_file), verbose=True)

    await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)

    assert temp_log_file.exists()
    content = temp_log_file.read_text()
    assert "PHASE_START" in content
    assert "WARMUP" in content

@pytest.mark.asyncio
async def test_phase_complete_logs_stats(temp_log_file, mock_tracker):
    """Test that phase complete logs statistics."""
    hook = MyPhaseHook(log_file=str(temp_log_file), verbose=True)

    await hook.on_phase_start(CreditPhase.WARMUP, mock_tracker)
    await hook.on_phase_complete(CreditPhase.WARMUP, mock_tracker)

    content = temp_log_file.read_text()
    assert "Completed: 95" in content

@pytest.mark.asyncio
async def test_concurrent_writes(temp_log_file, mock_tracker):
    """Test thread safety with concurrent writes."""
    hook = MyPhaseHook(log_file=str(temp_log_file))

    # Simulate concurrent phase events
    tasks = [
        hook.on_phase_start(CreditPhase.WARMUP, mock_tracker),
        hook.on_phase_start(CreditPhase.PROFILING, mock_tracker),
        hook.on_phase_complete(CreditPhase.WARMUP, mock_tracker),
    ]

    await asyncio.gather(*tasks)

    # Should have all events logged
    lines = temp_log_file.read_text().splitlines()
    assert len(lines) == 3

@pytest.mark.parametrize("phase,expected", [
    (CreditPhase.WARMUP, "WARMUP"),
    (CreditPhase.PROFILING, "PROFILING"),
    (CreditPhase.STEADY_STATE, "STEADY_STATE"),
])
@pytest.mark.asyncio
async def test_all_phases(temp_log_file, mock_tracker, phase, expected):
    """Test logging for all phase types."""
    hook = MyPhaseHook(log_file=str(temp_log_file))

    await hook.on_phase_start(phase, mock_tracker)

    content = temp_log_file.read_text()
    assert expected in content
```

### Documentation

**Write comprehensive docstrings:**

```python
class DatadogMetricsExporter:
    """Export AIPerf metrics to Datadog.

    This plugin exports phase metrics and benchmark results to Datadog
    for monitoring and alerting. Supports custom tags and aggregation.

    Architecture:
        ┌──────────────────┐
        │ Phase Events     │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Metrics Buffer   │ ← Batching
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Datadog API      │
        └──────────────────┘

    Attributes:
        api_key: Datadog API key for authentication
        api_url: Datadog API URL (default: https://api.datadoghq.com)
        tags: Global tags applied to all metrics
        batch_size: Number of metrics to batch before sending
        flush_interval: Seconds between automatic flushes

    Example:
        ```python
        exporter = DatadogMetricsExporter(
            api_key=os.environ["DD_API_KEY"],
            tags=["env:production", "service:aiperf"]
        )

        async with exporter:
            await exporter.export_metric(
                "aiperf.phase.duration",
                123.45,
                tags=["phase:warmup"]
            )
        ```

    Notes:
        - Metrics are batched for efficiency
        - Automatic retry on transient failures
        - Thread-safe for concurrent use
        - Graceful degradation on API errors

    See Also:
        - Datadog API docs: https://docs.datadoghq.com/api/
        - AIPerf Plugin Guide: docs/PLUGIN_BEST_PRACTICES.md
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.datadoghq.com",
        tags: list[str] | None = None,
        batch_size: int = 100,
        flush_interval: float = 10.0,
    ) -> None:
        """Initialize Datadog exporter.

        Args:
            api_key: Datadog API key (get from DD_API_KEY env var)
            api_url: Datadog API URL (default: https://api.datadoghq.com)
            tags: Global tags for all metrics (e.g., ["env:prod"])
            batch_size: Metrics to batch before sending (default: 100)
            flush_interval: Seconds between flushes (default: 10.0)

        Raises:
            ValueError: If api_key is empty or batch_size <= 0

        Example:
            ```python
            exporter = DatadogMetricsExporter(
                api_key=os.environ["DD_API_KEY"],
                tags=["team:ml", "project:benchmarking"]
            )
            ```
        """
```

### Error Handling

**Use custom exceptions and proper error handling:**

```python
class PluginError(Exception):
    """Base exception for plugin errors."""
    pass

class ConfigurationError(PluginError):
    """Configuration validation failed."""
    pass

class ExportError(PluginError):
    """Failed to export data."""
    pass

class MetricsExporter:
    """Export metrics with robust error handling."""

    async def export_metric(
        self,
        name: str,
        value: float
    ) -> None:
        """Export metric with error handling.

        Args:
            name: Metric name
            value: Metric value

        Raises:
            ExportError: If export fails after retries
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                await self._send_metric(name, value)
                return

            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    raise ExportError(
                        f"Failed to export metric after {max_retries} attempts: {e}"
                    ) from e

                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                # Unexpected error - don't retry
                raise ExportError(f"Unexpected error exporting metric: {e}") from e
```

---

## 4. Performance Best Practices

### Lazy Loading

**Defer expensive operations until needed:**

```python
class HeavyResourcePlugin:
    """Plugin with lazy-loaded heavy resources."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._model = None
        self._embeddings = None

    @property
    def model(self):
        """Lazy-load ML model."""
        if self._model is None:
            import torch
            self._model = torch.load(self.config["model_path"])
        return self._model

    @property
    def embeddings(self):
        """Lazy-load embeddings."""
        if self._embeddings is None:
            import numpy as np
            self._embeddings = np.load(self.config["embeddings_path"])
        return self._embeddings
```

### Async I/O for File Operations

**Use aiofiles for non-blocking file I/O:**

```python
import aiofiles
import aiofiles.os

class AsyncFileWriter:
    """Write files asynchronously."""

    async def write_large_file(self, path: Path, data: list[dict]) -> None:
        """Write large dataset asynchronously."""
        import json

        async with aiofiles.open(path, mode="w") as f:
            await f.write("[\n")

            for i, item in enumerate(data):
                line = json.dumps(item)
                if i < len(data) - 1:
                    line += ","
                await f.write(line + "\n")

            await f.write("]\n")

    async def read_lines(self, path: Path) -> list[str]:
        """Read file lines asynchronously."""
        async with aiofiles.open(path, mode="r") as f:
            lines = await f.readlines()
        return lines
```

### Connection Pooling

**Reuse HTTP connections for efficiency:**

```python
import aiohttp

class APIClient:
    """API client with connection pooling."""

    def __init__(self, base_url: str, max_connections: int = 100) -> None:
        self.base_url = base_url
        self.max_connections = max_connections
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Create session with connection pool."""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
        )

        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=20,
        )

        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )
        return self

    async def __aexit__(self, *args):
        """Close session and connections."""
        if self._session:
            await self._session.close()

    async def post(self, endpoint: str, data: dict) -> dict:
        """Make POST request using connection pool."""
        if not self._session:
            raise RuntimeError("Use async with")

        url = f"{self.base_url}{endpoint}"
        async with self._session.post(url, json=data) as resp:
            resp.raise_for_status()
            return await resp.json()
```

### Caching with functools

**Cache expensive computations:**

```python
from functools import lru_cache, cache
import hashlib

class MetricsProcessor:
    """Process metrics with caching."""

    @staticmethod
    @lru_cache(maxsize=1000)
    def compute_percentile(values_tuple: tuple[float, ...], p: float) -> float:
        """Compute percentile with caching.

        Note: Must use tuple (immutable) for caching.
        """
        import numpy as np
        values = np.array(values_tuple)
        return float(np.percentile(values, p))

    @cache
    def get_metric_schema(self, metric_name: str) -> dict:
        """Get metric schema (cached indefinitely)."""
        # Expensive operation
        return self._load_schema_from_db(metric_name)

    @staticmethod
    def cache_key(data: dict) -> str:
        """Generate cache key from dict."""
        import json
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
```

### Memory Efficiency

**Use slots and frozen dataclasses:**

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class MetricRecord:
    """Memory-efficient metric record.

    slots=True: Reduces memory usage by 40-50%
    frozen=True: Makes immutable (hashable, thread-safe)
    """

    metric_name: str
    value: float
    timestamp: float
    tags: tuple[str, ...]  # Use tuple (immutable)

    def __post_init__(self):
        """Validate after initialization."""
        if self.value < 0:
            raise ValueError("Value must be non-negative")

# Comparison
# Regular class: ~280 bytes per instance
# With slots: ~152 bytes per instance
# 45% memory savings!
```

---

## 5. Security Considerations

### Input Validation

**Always validate and sanitize inputs:**

```python
from pydantic import BaseModel, Field, field_validator

class PluginConfig(BaseModel):
    """Secure plugin configuration."""

    api_key: str = Field(min_length=20, max_length=100)
    endpoint_url: str = Field(pattern=r"^https://.*")
    timeout: int = Field(gt=0, le=300)

    @field_validator("endpoint_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is safe."""
        from urllib.parse import urlparse

        parsed = urlparse(v)
        if parsed.scheme not in ("https", "http"):
            raise ValueError("URL must use http or https")

        # Block internal IPs
        if parsed.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            raise ValueError("Cannot use internal IPs")

        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        if not v.isalnum():
            raise ValueError("API key must be alphanumeric")
        return v
```

### API Key Handling

**Never hardcode secrets - use environment variables:**

```python
import os
from pathlib import Path

class SecureConfig:
    """Secure configuration loading."""

    @staticmethod
    def load_api_key() -> str:
        """Load API key from environment or file.

        Priority:
        1. Environment variable
        2. Secrets file
        3. Fail with clear error
        """
        # Try environment first
        api_key = os.environ.get("DATADOG_API_KEY")
        if api_key:
            return api_key

        # Try secrets file
        secrets_file = Path.home() / ".aiperf" / "secrets.env"
        if secrets_file.exists():
            with open(secrets_file) as f:
                for line in f:
                    if line.startswith("DATADOG_API_KEY="):
                        return line.split("=", 1)[1].strip()

        # Fail explicitly
        raise ValueError(
            "API key not found. Set DATADOG_API_KEY environment variable "
            "or create ~/.aiperf/secrets.env"
        )

    @staticmethod
    def mask_sensitive_data(config: dict) -> dict:
        """Mask sensitive fields for logging."""
        masked = config.copy()

        sensitive_keys = {"api_key", "password", "token", "secret"}

        for key in masked:
            if any(s in key.lower() for s in sensitive_keys):
                value = masked[key]
                if isinstance(value, str) and len(value) > 4:
                    masked[key] = f"{value[:4]}...{value[-4:]}"

        return masked
```

### Error Message Sanitization

**Never leak sensitive information in errors:**

```python
class SafeLogger:
    """Logger that sanitizes sensitive data."""

    @staticmethod
    def sanitize_error_message(error: Exception, context: dict) -> str:
        """Create safe error message."""
        # Remove sensitive data from context
        safe_context = {}
        for key, value in context.items():
            if "password" in key.lower() or "api_key" in key.lower():
                safe_context[key] = "***REDACTED***"
            else:
                safe_context[key] = value

        # Create sanitized message
        return f"Error: {type(error).__name__} | Context: {safe_context}"

    async def log_error(self, error: Exception, **context) -> None:
        """Log error with sanitization."""
        message = self.sanitize_error_message(error, context)
        print(f"ERROR: {message}")
```

### Dependency Pinning

**Pin dependencies for security:**

```toml
# pyproject.toml
[project]
dependencies = [
    "aiperf>=2.0.0,<3.0.0",
    "aiohttp>=3.9.0,<4.0.0",
    "pydantic>=2.5.0,<3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
]
```

### Vulnerability Scanning

**Scan dependencies regularly:**

```bash
# Install safety
pip install safety

# Scan for vulnerabilities
safety check

# Scan requirements file
safety check --file requirements.txt

# Continuous scanning in CI/CD
safety check --continue-on-error
```

---

## 6. Testing Patterns

### Async Test Fixtures

**Create reusable async fixtures:**

```python
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def event_loop():
    """Create event loop for tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def temp_workspace(tmp_path):
    """Create temporary workspace with cleanup."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    yield workspace

    # Cleanup (if needed)
    import shutil
    shutil.rmtree(workspace, ignore_errors=True)

@pytest.fixture
def mock_phase_tracker():
    """Mock phase tracker with realistic data."""
    tracker = Mock()

    stats = Mock()
    stats.sent = 100
    stats.completed = 95
    stats.cancelled = 3
    stats.in_flight = 2
    stats.sent_sessions = 10

    tracker.create_stats.return_value = stats
    return tracker

@pytest.fixture
async def mock_http_session():
    """Mock aiohttp session."""
    session = AsyncMock()

    # Mock successful response
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"success": True})

    session.post = AsyncMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=response)
    session.__aexit__ = AsyncMock()

    return session
```

### Mock External Dependencies

**Isolate tests from external services:**

```python
import pytest
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_datadog_export_success(mock_http_session):
    """Test successful metric export to Datadog."""
    from aiperf_datadog.exporter import DatadogExporter

    # Use mocked session
    with patch("aiohttp.ClientSession", return_value=mock_http_session):
        exporter = DatadogExporter(api_key="test-key")

        async with exporter:
            await exporter.send_metric("test.metric", 123.45, ["tag:value"])

        # Verify API call
        mock_http_session.post.assert_called_once()
        call_args = mock_http_session.post.call_args
        assert "test.metric" in str(call_args)

@pytest.mark.asyncio
async def test_datadog_export_retry_on_failure(mock_http_session):
    """Test retry logic on transient failures."""
    # Mock failure then success
    mock_http_session.post.side_effect = [
        aiohttp.ClientError("Temporary failure"),
        aiohttp.ClientError("Temporary failure"),
        AsyncMock(status=200),
    ]

    exporter = DatadogExporter(api_key="test-key", max_retries=3)

    async with exporter:
        await exporter.send_metric("test.metric", 123.45)

    # Should have retried twice
    assert mock_http_session.post.call_count == 3
```

### Test Isolation

**Reset registries between tests:**

```python
import pytest
from aiperf.timing.phase_hooks import PhaseHookRegistry

@pytest.fixture(autouse=True)
def reset_registries():
    """Reset all global registries before each test."""
    # Clear phase hook registry
    PhaseHookRegistry.clear()

    # Reset plugin registry
    from aiperf.common import plugin_registry
    plugin_registry.reset()

    yield

    # Cleanup after test
    PhaseHookRegistry.clear()
    plugin_registry.reset()

@pytest.mark.asyncio
async def test_hook_registration():
    """Test hook registration (isolated)."""
    from aiperf_my_plugin.hooks import MyHook

    hook = MyHook(config={})
    hook.register_with_global_registry()

    # Verify registration
    callbacks = PhaseHookRegistry.get_phase_start_callbacks()
    assert len(callbacks) == 1
    assert callbacks[0] == hook.on_phase_start
```

### Parametrized Tests

**Test multiple scenarios efficiently:**

```python
import pytest
from aiperf.common.enums import CreditPhase

@pytest.mark.parametrize("phase,expected_msg", [
    (CreditPhase.WARMUP, "WARMUP"),
    (CreditPhase.PROFILING, "PROFILING"),
    (CreditPhase.STEADY_STATE, "STEADY_STATE"),
])
@pytest.mark.asyncio
async def test_phase_logging(tmp_path, mock_phase_tracker, phase, expected_msg):
    """Test logging for different phases."""
    from aiperf_my_plugin.hooks import LoggingHook

    log_file = tmp_path / "test.log"
    hook = LoggingHook(log_file=str(log_file))

    await hook.on_phase_start(phase, mock_phase_tracker)

    content = log_file.read_text()
    assert expected_msg in content

@pytest.mark.parametrize("batch_size,num_metrics,expected_calls", [
    (10, 5, 1),    # Single batch
    (10, 15, 2),   # Two batches
    (10, 20, 2),   # Exact two batches
    (10, 25, 3),   # Three batches
])
@pytest.mark.asyncio
async def test_batching_logic(batch_size, num_metrics, expected_calls):
    """Test metric batching with different sizes."""
    exporter = MetricsExporter(batch_size=batch_size)

    with patch.object(exporter, "_flush_batch") as mock_flush:
        for i in range(num_metrics):
            await exporter.add_metric(f"metric_{i}", float(i))

        await exporter.flush()

    assert mock_flush.call_count == expected_calls
```

### Integration Tests

**Test plugin integration with AIPerf:**

```python
import pytest
from aiperf.common.plugin_loader import PluginLoader
from aiperf.timing.phase_orchestrator import PhaseOrchestrator

@pytest.mark.integration
@pytest.mark.asyncio
async def test_plugin_integration_end_to_end():
    """Test plugin loads and integrates with AIPerf."""
    # Load plugin
    loader = PluginLoader()
    config = {
        "phase_hooks": [
            {"name": "my_logging_hook", "config": {"verbose": True}}
        ]
    }
    loader.initialize_plugin_system(config)

    # Verify plugin loaded
    assert len(loader.get_loaded_plugins()) == 1

    # Verify hook registered
    from aiperf.timing.phase_hooks import PhaseHookRegistry
    callbacks = PhaseHookRegistry.get_phase_start_callbacks()
    assert len(callbacks) > 0

    # Test execution
    orchestrator = PhaseOrchestrator(...)
    await orchestrator.execute_phase(CreditPhase.WARMUP)

    # Verify hook was called
    # (Check log file or metrics)
```

---

## 7. Documentation Standards

### README Structure

**Essential sections for plugin README:**

```markdown
# AIPerf Datadog Plugin

Export AIPerf benchmark metrics to Datadog for monitoring and alerting.

## Features

- Real-time metric export during benchmark execution
- Support for custom tags and aggregation
- Automatic retry on transient failures
- Batching for high-throughput scenarios
- GPU telemetry support

## Installation

```bash
pip install aiperf-datadog-plugin
```

## Quick Start

```yaml
# config.yaml
plugins:
  phase_hooks:
    - name: datadog_exporter
      config:
        api_key: ${DATADOG_API_KEY}
        tags:
          - env:production
          - service:aiperf
```

```bash
export DATADOG_API_KEY="your-api-key"
aiperf --config config.yaml
```

## Configuration

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | Datadog API key |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | string | `https://api.datadoghq.com` | Datadog API URL |
| `tags` | list[string] | `[]` | Global tags for all metrics |
| `batch_size` | int | `100` | Metrics per batch |
| `flush_interval` | float | `10.0` | Seconds between flushes |

## Examples

### Basic Usage

```python
from aiperf_datadog_plugin import DatadogExporter

exporter = DatadogExporter(
    api_key=os.environ["DD_API_KEY"],
    tags=["team:ml", "project:benchmarking"]
)
```

### Advanced Configuration

```yaml
plugins:
  phase_hooks:
    - name: datadog_exporter
      config:
        api_key: ${DATADOG_API_KEY}
        api_url: https://api.datadoghq.eu
        tags:
          - env:production
          - datacenter:us-west-2
          - team:ml-infra
        batch_size: 500
        flush_interval: 5.0
```

## Metrics Exported

| Metric Name | Description | Tags |
|-------------|-------------|------|
| `aiperf.phase.duration` | Phase execution time | `phase`, `status` |
| `aiperf.requests.sent` | Total requests sent | `phase`, `endpoint` |
| `aiperf.requests.completed` | Completed requests | `phase`, `status_code` |
| `aiperf.latency.p50` | 50th percentile latency | `phase`, `endpoint` |
| `aiperf.latency.p95` | 95th percentile latency | `phase`, `endpoint` |
| `aiperf.latency.p99` | 99th percentile latency | `phase`, `endpoint` |

## Development

### Setup

```bash
git clone https://github.com/yourusername/aiperf-datadog-plugin
cd aiperf-datadog-plugin
pip install -e ".[dev]"
```

### Testing

```bash
pytest tests/ -v
pytest tests/ --cov=aiperf_datadog_plugin --cov-report=html
```

### Code Quality

```bash
ruff format .
ruff check --fix .
pyright .
```

## Troubleshooting

### API Authentication Errors

**Problem:** `401 Unauthorized` errors

**Solution:** Verify API key is correct:
```bash
curl -X GET "https://api.datadoghq.com/api/v1/validate" \
  -H "DD-API-KEY: ${DATADOG_API_KEY}"
```

### Metrics Not Appearing

**Problem:** Metrics not visible in Datadog

**Solution:** Check metric names follow Datadog naming conventions:
- Use lowercase
- Use underscores not dots
- Start with letter

## License

Apache-2.0

## Support

- GitHub Issues: https://github.com/yourusername/aiperf-datadog-plugin/issues
- Email: support@example.com
```

### API Reference Documentation

**Document all public interfaces:**

```python
class DatadogExporter:
    """Export AIPerf metrics to Datadog.

    Public API Methods:
        __init__(api_key, **kwargs): Initialize exporter
        send_metric(name, value, tags): Send single metric
        send_metrics(metrics): Send batch of metrics
        flush(): Force flush pending metrics

    Context Manager:
        Use with `async with` for automatic lifecycle management:
        ```python
        async with DatadogExporter(api_key) as exporter:
            await exporter.send_metric("my.metric", 123.45)
        ```

    Lifecycle:
        1. __init__: Create exporter instance
        2. __aenter__: Open HTTP session
        3. send_metric/send_metrics: Export metrics
        4. flush: Flush remaining metrics
        5. __aexit__: Close HTTP session

    Thread Safety:
        - All methods are async-safe
        - Concurrent send_metric calls are batched
        - Flush is thread-safe with lock

    Error Handling:
        - Transient errors: Automatic retry with backoff
        - Permanent errors: Raise ExportError
        - Network errors: Retry up to max_retries

    Performance:
        - Batching: Reduces API calls by ~90%
        - Connection pooling: Reuses connections
        - Async I/O: Non-blocking operations
    """
```

### Usage Examples

**Provide comprehensive examples:**

```python
# examples/basic_usage.py
"""Basic usage example for Datadog plugin."""

import asyncio
import os
from aiperf_datadog_plugin import DatadogExporter

async def main():
    """Basic example."""
    exporter = DatadogExporter(
        api_key=os.environ["DD_API_KEY"]
    )

    async with exporter:
        # Send single metric
        await exporter.send_metric(
            "my.benchmark.duration",
            123.45,
            tags=["env:dev"]
        )

        # Send batch
        await exporter.send_metrics([
            {"name": "my.metric.1", "value": 1.0},
            {"name": "my.metric.2", "value": 2.0},
        ])

if __name__ == "__main__":
    asyncio.run(main())
```

### Troubleshooting Guide

**Document common issues and solutions:**

```markdown
## Common Issues

### Issue: "Failed to import plugin"

**Symptoms:**
```
ImportError: No module named 'aiperf_datadog_plugin'
```

**Causes:**
1. Plugin not installed
2. Wrong Python environment
3. Incorrect package name

**Solutions:**
```bash
# Verify installation
pip list | grep aiperf-datadog-plugin

# Reinstall plugin
pip uninstall aiperf-datadog-plugin
pip install aiperf-datadog-plugin

# Check Python environment
which python
python -c "import aiperf_datadog_plugin"
```

### Issue: "Hook not being called"

**Symptoms:**
- No metrics in Datadog
- No log output from plugin

**Diagnosis:**
```python
from aiperf.timing.phase_hooks import PhaseHookRegistry
callbacks = PhaseHookRegistry.get_phase_start_callbacks()
print(f"Registered callbacks: {len(callbacks)}")
```

**Solutions:**
1. Verify plugin is enabled in config
2. Check hook registration method is called
3. Ensure no exceptions in hook initialization
```

---

## 8. Publishing Checklist

### Pre-Publishing Validation

**Complete this checklist before publishing:**

- [ ] **Version bumped correctly**
  ```bash
  # Check current version
  git tag --list | tail -n 5

  # Update version
  # pyproject.toml: version = "1.1.0"
  # __init__.py: __version__ = "1.1.0"
  ```

- [ ] **Changelog updated**
  ```markdown
  # CHANGELOG.md

  ## [1.1.0] - 2025-01-15

  ### Added
  - GPU telemetry support
  - Automatic retry logic

  ### Fixed
  - Memory leak in metric buffering
  - Thread safety in concurrent writes
  ```

- [ ] **Tests passing**
  ```bash
  pytest tests/ -v --cov=aiperf_my_plugin
  # Ensure >80% coverage
  ```

- [ ] **Type checking passing**
  ```bash
  pyright .
  # 0 errors, 0 warnings
  ```

- [ ] **Linting passing**
  ```bash
  ruff check .
  ruff format --check .
  ```

- [ ] **Documentation complete**
  ```bash
  # Check README
  # Check API docs
  # Check examples
  ```

- [ ] **License file present**
  ```bash
  ls LICENSE
  ```

- [ ] **Security scan passing**
  ```bash
  safety check
  ```

### Build Package

**Build distribution packages:**

```bash
# Install build tools
pip install build twine

# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build package
python -m build

# Verify contents
tar -tzf dist/aiperf_my_plugin-1.0.0.tar.gz
unzip -l dist/aiperf_my_plugin-1.0.0-py3-none-any.whl

# Check distribution
twine check dist/*
```

### Upload to PyPI

**Publish to PyPI:**

```bash
# Test PyPI first (recommended)
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    aiperf-my-plugin

# Verify plugin works
python -c "from aiperf_my_plugin import get_registry_path; print(get_registry_path())"

# Upload to production PyPI
twine upload dist/*
```

### GitHub Releases

**Create GitHub release:**

```bash
# Create and push tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Create release on GitHub with:
# - Release notes from CHANGELOG.md
# - Attached wheel and tar.gz files
# - Migration guide if breaking changes
```

### Post-Publishing

- [ ] **Verify installation**
  ```bash
  pip install aiperf-my-plugin
  python -c "import aiperf_my_plugin"
  ```

- [ ] **Update documentation**
  ```bash
  # Update installation instructions
  # Update version numbers
  ```

- [ ] **Announce release**
  ```markdown
  # GitHub Discussions, Twitter, etc.

  aiperf-my-plugin v1.0.0 released!

  New features:
  - GPU telemetry support
  - Automatic retry logic

  Install: pip install aiperf-my-plugin
  Docs: https://github.com/user/aiperf-my-plugin
  ```

---

## 9. Example Code

### Example 1: Minimal Phase Hook

**Simplest possible phase hook:**

```python
# aiperf_simple_plugin/hooks.py
"""Minimal phase hook example."""

from pathlib import Path
from aiperf.common.enums import CreditPhase
from aiperf.timing.phase_lifecycle_hooks import BasePhaseLifecycleHook

class SimpleLoggingHook(BasePhaseLifecycleHook):
    """Minimal phase logging hook.

    Logs phase transitions to a file.
    """

    def __init__(self, log_file: str = "/tmp/phases.log") -> None:
        """Initialize with log file path."""
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def on_phase_start(self, phase: CreditPhase, tracker) -> None:
        """Log phase start."""
        with open(self.log_file, "a") as f:
            f.write(f"Phase {phase} started\n")

    async def on_phase_complete(self, phase: CreditPhase, tracker) -> None:
        """Log phase completion."""
        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        with open(self.log_file, "a") as f:
            if stats:
                f.write(f"Phase {phase} completed: {stats.completed} requests\n")
            else:
                f.write(f"Phase {phase} completed\n")
```

```yaml
# aiperf_simple_plugin/registry.yaml
schema_version: "1.0"

plugin:
  name: aiperf-simple-plugin
  version: 1.0.0
  description: Simple phase logging hook

phase_hook:
  simple_logging:
    class: aiperf_simple_plugin.hooks:SimpleLoggingHook
    description: Simple file-based phase logging
```

```python
# aiperf_simple_plugin/__init__.py
"""Simple plugin entry point."""

from pathlib import Path

def get_registry_path() -> Path:
    """Return path to registry.yaml."""
    return Path(__file__).parent / "registry.yaml"
```

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "aiperf-simple-plugin"
version = "1.0.0"
description = "Simple AIPerf phase logging plugin"
requires-python = ">=3.10"
dependencies = ["aiperf>=2.0.0"]

[project.entry-points."aiperf.plugins"]
simple_plugin = "aiperf_simple_plugin:get_registry_path"
```

### Example 2: Advanced Phase Hook with Metrics

**Production-ready hook with metrics collection:**

```python
# aiperf_metrics_plugin/hooks.py
"""Advanced phase hook with metrics collection."""

import json
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from aiperf.common.enums import CreditPhase
from aiperf.timing.phase_lifecycle_hooks import BasePhaseLifecycleHook

@dataclass
class PhaseMetrics:
    """Metrics for a single phase."""

    phase: str
    start_time: float
    end_time: float | None = None
    sent: int = 0
    completed: int = 0
    cancelled: int = 0
    in_flight: int = 0
    duration: float = 0.0

    def finalize(self) -> None:
        """Calculate final metrics."""
        if self.end_time:
            self.duration = self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "sent": self.sent,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "in_flight": self.in_flight,
        }

class MetricsCollectorHook(BasePhaseLifecycleHook):
    """Advanced metrics collection hook.

    Features:
    - Collects detailed phase metrics
    - Exports to JSON
    - Calculates aggregates
    - Thread-safe

    Attributes:
        metrics_file: Path to metrics JSON file
        aggregate: Whether to calculate aggregates
    """

    def __init__(
        self,
        metrics_file: str = "/tmp/phase_metrics.json",
        aggregate: bool = True,
    ) -> None:
        """Initialize metrics collector.

        Args:
            metrics_file: Path to output JSON file
            aggregate: Calculate aggregate statistics
        """
        self.metrics_file = Path(metrics_file)
        self.aggregate = aggregate
        self._metrics: dict[str, PhaseMetrics] = {}
        self._phase_order: list[str] = []

        # Ensure parent directory exists
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    async def on_phase_start(self, phase: CreditPhase, tracker) -> None:
        """Record phase start."""
        phase_key = str(phase)

        self._metrics[phase_key] = PhaseMetrics(
            phase=phase_key,
            start_time=time.time(),
        )

        if phase_key not in self._phase_order:
            self._phase_order.append(phase_key)

        await self._write_checkpoint()

    async def on_phase_sending_complete(self, phase: CreditPhase, tracker) -> None:
        """Record sending complete."""
        phase_key = str(phase)

        if phase_key not in self._metrics:
            return

        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        if stats:
            self._metrics[phase_key].sent = stats.sent

    async def on_phase_complete(self, phase: CreditPhase, tracker) -> None:
        """Record phase completion."""
        phase_key = str(phase)

        if phase_key not in self._metrics:
            return

        metrics = self._metrics[phase_key]
        metrics.end_time = time.time()

        stats = tracker.create_stats() if hasattr(tracker, "create_stats") else None
        if stats:
            metrics.completed = stats.completed
            metrics.cancelled = stats.cancelled
            metrics.in_flight = stats.in_flight

        metrics.finalize()
        await self._write_checkpoint()

    async def on_phase_timeout(self, phase: CreditPhase, tracker) -> None:
        """Record phase timeout."""
        phase_key = str(phase)

        if phase_key not in self._metrics:
            return

        metrics = self._metrics[phase_key]
        metrics.end_time = time.time()
        metrics.finalize()

        await self._write_checkpoint()

    async def _write_checkpoint(self) -> None:
        """Write metrics checkpoint to file."""
        output = {
            "metrics": {
                phase: metrics.to_dict()
                for phase, metrics in self._metrics.items()
            },
            "phase_order": self._phase_order,
            "last_updated": time.time(),
        }

        if self.aggregate:
            output["aggregates"] = self._calculate_aggregates()

        try:
            with open(self.metrics_file, "w") as f:
                json.dump(output, f, indent=2)
        except OSError as e:
            print(f"Warning: Failed to write metrics: {e}")

    def _calculate_aggregates(self) -> dict[str, Any]:
        """Calculate aggregate statistics."""
        completed_metrics = [
            m for m in self._metrics.values()
            if m.end_time is not None
        ]

        if not completed_metrics:
            return {}

        total_duration = sum(m.duration for m in completed_metrics)
        total_sent = sum(m.sent for m in completed_metrics)
        total_completed = sum(m.completed for m in completed_metrics)
        total_cancelled = sum(m.cancelled for m in completed_metrics)

        return {
            "total_phases": len(completed_metrics),
            "total_duration": total_duration,
            "total_sent": total_sent,
            "total_completed": total_completed,
            "total_cancelled": total_cancelled,
            "average_phase_duration": total_duration / len(completed_metrics),
            "completion_rate": total_completed / total_sent if total_sent > 0 else 0.0,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics.

        Returns:
            Dictionary with all metrics
        """
        return {
            phase: metrics.to_dict()
            for phase, metrics in self._metrics.items()
        }
```

### Example 3: Custom Endpoint Implementation

**Custom API endpoint with streaming support:**

```python
# aiperf_custom_api_plugin/endpoint.py
"""Custom API endpoint implementation."""

from typing import Any
import aiohttp

from aiperf.common.models import ModelEndpointInfo, ParsedResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint

class CustomAPIEndpoint(BaseEndpoint):
    """Custom API endpoint.

    Implements support for a proprietary API format.

    Features:
    - Custom request format
    - Custom response parsing
    - Streaming support
    - Error handling
    """

    def __init__(self, model_endpoint: ModelEndpointInfo) -> None:
        """Initialize endpoint.

        Args:
            model_endpoint: Endpoint configuration
        """
        super().__init__(model_endpoint)
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        """Initialize HTTP session."""
        await super().initialize()

        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
        )

        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=300),
        )

    async def stop(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
        await super().stop()

    def format_request(self, prompt: str, **params) -> dict[str, Any]:
        """Format request for custom API.

        Args:
            prompt: Input prompt
            **params: Additional parameters

        Returns:
            Formatted request payload
        """
        return {
            "input": {
                "text": prompt,
                "parameters": {
                    "max_length": params.get("max_tokens", 100),
                    "temperature": params.get("temperature", 1.0),
                    "top_p": params.get("top_p", 1.0),
                    "stream": params.get("stream", False),
                }
            },
            "metadata": {
                "request_id": params.get("request_id"),
                "timestamp": params.get("timestamp"),
            }
        }

    async def send_request(
        self,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> ParsedResponse:
        """Send request to API.

        Args:
            request_data: Formatted request
            stream: Whether to stream response

        Returns:
            Parsed response

        Raises:
            RuntimeError: If request fails
        """
        if not self._session:
            raise RuntimeError("Session not initialized")

        url = f"{self.model_endpoint.url}/v1/generate"
        headers = {
            "Authorization": f"Bearer {self.model_endpoint.api_key}",
            "Content-Type": "application/json",
        }

        if stream:
            return await self._stream_request(url, headers, request_data)
        else:
            return await self._non_stream_request(url, headers, request_data)

    async def _non_stream_request(
        self,
        url: str,
        headers: dict[str, str],
        request_data: dict[str, Any],
    ) -> ParsedResponse:
        """Send non-streaming request."""
        async with self._session.post(
            url,
            json=request_data,
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return self.parse_response(data)

    async def _stream_request(
        self,
        url: str,
        headers: dict[str, str],
        request_data: dict[str, Any],
    ) -> ParsedResponse:
        """Send streaming request."""
        full_text = ""

        async with self._session.post(
            url,
            json=request_data,
            headers=headers,
        ) as resp:
            resp.raise_for_status()

            async for line in resp.content:
                if line:
                    chunk = self._parse_stream_chunk(line)
                    if chunk:
                        full_text += chunk

        return ParsedResponse(
            text=full_text,
            finish_reason="stop",
        )

    def parse_response(self, response: dict[str, Any]) -> ParsedResponse:
        """Parse API response.

        Args:
            response: Raw API response

        Returns:
            Parsed response
        """
        return ParsedResponse(
            text=response["output"]["text"],
            finish_reason=response.get("finish_reason", "stop"),
            usage={
                "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
            }
        )

    def _parse_stream_chunk(self, chunk: bytes) -> str | None:
        """Parse streaming chunk.

        Args:
            chunk: Raw chunk bytes

        Returns:
            Extracted text or None
        """
        try:
            import json
            data = json.loads(chunk.decode("utf-8"))
            return data.get("token", {}).get("text", "")
        except Exception:
            return None
```

### Example 4: Custom Post-Processor

**Process and export custom metrics:**

```python
# aiperf_custom_metrics_plugin/processor.py
"""Custom metrics post-processor."""

from pathlib import Path
from typing import Any
import json

class CustomMetricsProcessor:
    """Custom metrics post-processor.

    Processes benchmark results and exports custom metrics.

    Features:
    - Custom metric calculations
    - JSON export
    - CSV export
    - Aggregation
    """

    def __init__(
        self,
        output_dir: str = "/tmp/metrics",
        export_json: bool = True,
        export_csv: bool = True,
    ) -> None:
        """Initialize processor.

        Args:
            output_dir: Directory for output files
            export_json: Export to JSON
            export_csv: Export to CSV
        """
        self.output_dir = Path(output_dir)
        self.export_json = export_json
        self.export_csv = export_csv

        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process_results(self, results: dict[str, Any]) -> None:
        """Process results and export metrics.

        Args:
            results: Benchmark results
        """
        # Calculate custom metrics
        metrics = self._calculate_metrics(results)

        # Export to formats
        if self.export_json:
            await self._export_json(metrics)

        if self.export_csv:
            await self._export_csv(metrics)

    def _calculate_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Calculate custom metrics.

        Args:
            results: Raw results

        Returns:
            Calculated metrics
        """
        # Extract data
        latencies = results.get("latencies", [])
        throughput = results.get("throughput", 0)

        # Custom calculations
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = min_latency = max_latency = 0

        return {
            "average_latency_ms": avg_latency * 1000,
            "min_latency_ms": min_latency * 1000,
            "max_latency_ms": max_latency * 1000,
            "throughput_rps": throughput,
            "total_requests": len(latencies),
        }

    async def _export_json(self, metrics: dict[str, Any]) -> None:
        """Export metrics to JSON."""
        output_file = self.output_dir / "custom_metrics.json"

        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

    async def _export_csv(self, metrics: dict[str, Any]) -> None:
        """Export metrics to CSV."""
        import csv

        output_file = self.output_dir / "custom_metrics.csv"

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
```

### Example 5: Multi-Protocol Plugin

**Plugin with multiple implementations:**

```yaml
# aiperf_multi_plugin/registry.yaml
schema_version: "1.0"

plugin:
  name: aiperf-multi-plugin
  version: 1.0.0
  description: Multi-protocol plugin with hooks and processors

# Phase hooks
phase_hook:
  metrics_collector:
    class: aiperf_multi_plugin.hooks:MetricsCollectorHook
    description: Collect phase metrics

  notification_hook:
    class: aiperf_multi_plugin.hooks:NotificationHook
    description: Send notifications on phase events

# Post-processors
results_processor:
  custom_exporter:
    class: aiperf_multi_plugin.processors:CustomExporter
    description: Export to custom format

  aggregator:
    class: aiperf_multi_plugin.processors:MetricsAggregator
    description: Aggregate metrics across runs

# Custom endpoint
endpoint:
  custom_api:
    class: aiperf_multi_plugin.endpoints:CustomAPIEndpoint
    description: Custom API endpoint implementation
```

---

## 10. Anti-Patterns to Avoid

### Global Mutable State

**DON'T:**
```python
# Bad: Global mutable state
_global_metrics = []  # ANTI-PATTERN!

class BadHook:
    async def on_phase_start(self, phase, tracker):
        _global_metrics.append({"phase": phase})  # NOT THREAD-SAFE!
```

**DO:**
```python
# Good: Instance state
class GoodHook:
    def __init__(self):
        self._metrics: list[dict] = []  # Instance variable

    async def on_phase_start(self, phase, tracker):
        self._metrics.append({"phase": phase})
```

### Blocking I/O in Async

**DON'T:**
```python
# Bad: Blocking operations in async
async def bad_write(self, data: str):
    with open("file.txt", "w") as f:  # BLOCKS EVENT LOOP!
        f.write(data)

    time.sleep(1)  # BLOCKS EVENT LOOP!
```

**DO:**
```python
# Good: Async I/O
async def good_write(self, data: str):
    import aiofiles

    async with aiofiles.open("file.txt", "w") as f:
        await f.write(data)

    await asyncio.sleep(1)  # Non-blocking
```

### Missing Error Handling

**DON'T:**
```python
# Bad: No error handling
async def bad_request(self, url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()  # May fail!
```

**DO:**
```python
# Good: Comprehensive error handling
async def good_request(self, url: str) -> dict | None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.json()

    except aiohttp.ClientError as e:
        self.error(f"Request failed: {e}")
        return None

    except asyncio.TimeoutError:
        self.error(f"Request timeout: {url}")
        return None

    except Exception as e:
        self.error(f"Unexpected error: {e}")
        return None
```

### Hardcoded Paths

**DON'T:**
```python
# Bad: Hardcoded paths
LOG_FILE = "/tmp/my.log"  # Only works on Unix!
CONFIG_FILE = "C:\\Users\\user\\config.yaml"  # Only works on Windows!
```

**DO:**
```python
# Good: Cross-platform paths
from pathlib import Path

LOG_FILE = Path.home() / ".aiperf" / "my.log"
CONFIG_FILE = Path(__file__).parent / "config.yaml"
TEMP_FILE = Path(tempfile.gettempdir()) / "my.log"
```

### Missing Type Hints

**DON'T:**
```python
# Bad: No type hints
def process_data(data, config):  # What types?
    return data["value"] * config["multiplier"]
```

**DO:**
```python
# Good: Complete type hints
def process_data(
    data: dict[str, float],
    config: dict[str, float]
) -> float:
    return data["value"] * config["multiplier"]
```

### Circular Imports

**DON'T:**
```python
# Bad: Circular imports
# module_a.py
from module_b import ClassB

class ClassA:
    def use_b(self):
        return ClassB()

# module_b.py
from module_a import ClassA  # CIRCULAR!

class ClassB:
    def use_a(self):
        return ClassA()
```

**DO:**
```python
# Good: Type hints in TYPE_CHECKING
# module_a.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from module_b import ClassB

class ClassA:
    def use_b(self) -> "ClassB":
        from module_b import ClassB  # Import in function
        return ClassB()
```

### Too Much Magic

**DON'T:**
```python
# Bad: Overly complex metaclasses
class MagicMeta(type):
    def __new__(mcs, name, bases, namespace):
        # 100 lines of magic...
        return super().__new__(mcs, name, bases, namespace)

class MyHook(metaclass=MagicMeta):  # What does this do?
    pass
```

**DO:**
```python
# Good: Explicit is better than implicit
class MyHook:
    def __init__(self, config: dict):
        self.config = config
        self._setup()

    def _setup(self) -> None:
        """Explicit setup logic."""
        # Clear setup steps
```

### Ignoring Lifecycle

**DON'T:**
```python
# Bad: No cleanup
class BadHook:
    def __init__(self):
        self.file = open("log.txt", "w")  # Never closed!

    async def on_phase_start(self, phase, tracker):
        self.file.write(f"Phase {phase}\n")
```

**DO:**
```python
# Good: Proper lifecycle management
class GoodHook:
    def __init__(self):
        self.file = None

    async def __aenter__(self):
        self.file = open("log.txt", "w")
        return self

    async def __aexit__(self, *args):
        if self.file:
            self.file.close()

    async def on_phase_start(self, phase, tracker):
        if self.file:
            self.file.write(f"Phase {phase}\n")
```

---

## 11. Migration from Old Factories

### Step-by-Step Migration Guide

**Old System (Factory-based):**
```python
# OLD: Factory decorator registration
from aiperf.common.factories import EndpointFactory
from aiperf.common.enums import EndpointType

@EndpointFactory.register(EndpointType.MY_ENDPOINT)
class MyEndpoint(EndpointProtocol):
    pass

# OLD: Factory creation
endpoint = EndpointFactory.create_instance(
    EndpointType.MY_ENDPOINT,
    model_endpoint=config
)
```

**New System (Registry-based):**
```yaml
# NEW: registry.yaml declaration
endpoint:
  my_endpoint:
    class: aiperf_my_plugin.endpoints:MyEndpoint
    description: My custom endpoint
```

```python
# NEW: Registry creation
from aiperf.common import plugin_registry

MyEndpoint = plugin_registry.get_class("endpoint", "my_endpoint")
endpoint = MyEndpoint(model_endpoint=config)
```

### Migration Steps

**1. Remove Factory Decorators:**
```python
# Before
@EndpointFactory.register(EndpointType.MY_ENDPOINT)
class MyEndpoint(EndpointProtocol):
    pass

# After (remove decorator)
class MyEndpoint(EndpointProtocol):
    pass
```

**2. Create registry.yaml:**
```yaml
schema_version: "1.0"

plugin:
  name: aiperf-my-plugin
  version: 1.0.0

endpoint:
  my_endpoint:
    class: aiperf_my_plugin.endpoints:MyEndpoint
    description: My custom endpoint implementation
```

**3. Update Imports:**
```python
# Before
from aiperf.common.factories import EndpointFactory
from aiperf.common.enums import EndpointType

# After
from aiperf.common import plugin_registry
```

**4. Update Creation Code:**
```python
# Before
endpoint = EndpointFactory.create_instance(
    EndpointType.MY_ENDPOINT,
    model_endpoint=config
)

# After
endpoint = plugin_registry.create_instance(
    "endpoint",
    "my_endpoint",
    model_endpoint=config
)
```

**5. Update Tests:**
```python
# Before
def test_endpoint():
    endpoint = EndpointFactory.create_instance(...)

# After
import pytest
from aiperf.common import plugin_registry

@pytest.fixture(autouse=True)
def reset_registry():
    plugin_registry.reset()
    yield

def test_endpoint():
    endpoint = plugin_registry.create_instance(...)
```

### Deprecation Handling

**Support both old and new APIs temporarily:**

```python
# Backward compatibility layer
import warnings

def create_endpoint_old_style(endpoint_type, **kwargs):
    """Old factory-style creation (deprecated)."""
    warnings.warn(
        "Factory-style creation is deprecated. "
        "Use plugin_registry.create_instance() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Map old enum to new name
    name_mapping = {
        "MY_ENDPOINT": "my_endpoint",
    }

    from aiperf.common import plugin_registry
    return plugin_registry.create_instance(
        "endpoint",
        name_mapping[endpoint_type],
        **kwargs
    )
```

### Testing Migration

**Test both old and new APIs during transition:**

```python
import pytest

@pytest.mark.parametrize("creation_method", ["old", "new"])
def test_endpoint_creation(creation_method):
    """Test endpoint creation with both APIs."""
    if creation_method == "old":
        endpoint = create_endpoint_old_style("MY_ENDPOINT", ...)
    else:
        from aiperf.common import plugin_registry
        endpoint = plugin_registry.create_instance("endpoint", "my_endpoint", ...)

    # Both should work the same
    assert endpoint is not None
```

---

## 12. Common Issues and Solutions

### Import Errors

**Issue:** `ModuleNotFoundError: No module named 'aiperf_my_plugin'`

**Causes:**
1. Plugin not installed
2. Wrong Python environment
3. Typo in module name

**Solutions:**
```bash
# Check installation
pip list | grep aiperf-my-plugin

# Reinstall in development mode
pip uninstall aiperf-my-plugin
pip install -e .

# Verify import works
python -c "import aiperf_my_plugin"
```

### Priority Conflicts

**Issue:** Plugin implementation not being used

**Diagnosis:**
```python
from aiperf.common import plugin_registry

impls = plugin_registry.list_types("endpoint")
for impl in impls:
    if impl.impl_name == "my_endpoint":
        print(f"Priority: {impl.priority}")
        print(f"Plugin: {impl.plugin_name}")
```

**Solutions:**

1. **Increase priority in registry.yaml:**
```yaml
endpoint:
  my_endpoint:
    priority: 100  # Higher than default (0)
```

2. **Use explicit selection:**
```python
# Always specify explicitly
endpoint = plugin_registry.create_instance(
    "endpoint",
    "my_endpoint",  # Explicit name
    **kwargs
)
```

### Registry Not Initialized

**Issue:** `KeyError: Unknown protocol 'phase_hook'`

**Cause:** Registry not loaded before use

**Solution:**
```python
# Ensure registry is loaded
from aiperf.common import plugin_registry

# Load built-in registry
plugin_registry.load_builtins()

# Discover external plugins
plugin_registry.discover_plugins()

# Now safe to use
hook = plugin_registry.get_class("phase_hook", "my_hook")
```

### Config Validation Failures

**Issue:** `ValidationError: Field required`

**Diagnosis:**
```python
from aiperf_my_plugin.config import PluginConfig

try:
    config = PluginConfig(api_key="test")
except ValidationError as e:
    print(e.json())  # Shows which fields failed
```

**Solutions:**

1. **Provide all required fields:**
```yaml
plugins:
  phase_hooks:
    - name: my_hook
      config:
        api_key: ${API_KEY}  # Required
        timeout: 30          # Required
```

2. **Add defaults in model:**
```python
class PluginConfig(BaseModel):
    api_key: str  # Required
    timeout: int = 30  # Optional with default
```

### Async Issues

**Issue:** `RuntimeError: This event loop is already running`

**Cause:** Mixing sync and async code incorrectly

**Solution:**
```python
# DON'T: Call async from sync
def sync_function():
    asyncio.run(async_function())  # May fail!

# DO: Keep async chain
async def async_function():
    await other_async_function()
```

### Hook Not Called

**Issue:** Phase hook registered but never called

**Diagnosis:**
```python
from aiperf.timing.phase_hooks import PhaseHookRegistry

# Check registration
callbacks = PhaseHookRegistry.get_phase_start_callbacks()
print(f"Registered: {len(callbacks)} callbacks")

for cb in callbacks:
    print(f"  - {cb.__name__}")
```

**Solutions:**

1. **Ensure registration method is called:**
```python
class MyHook:
    def register_with_global_registry(self) -> None:
        """Must be called to register."""
        PhaseHookRegistry.register_phase_start(self.on_phase_start)

# In plugin loader:
hook = MyHook()
hook.register_with_global_registry()  # Required!
```

2. **Check hook is enabled in config:**
```yaml
plugins:
  phase_hooks:
    - name: my_hook  # Must match registry.yaml
      config: {}
```

3. **Verify no exceptions in hook:**
```python
async def on_phase_start(self, phase, tracker):
    try:
        # Hook logic
        ...
    except Exception as e:
        print(f"Hook error: {e}")  # Log errors
        raise
```

---

## Additional Resources

### Related Documentation

- **Migration Guide:** `/home/anthony/nvidia/projects/aiperf/ajc/sticky-credit/docs/MIGRATION_GUIDE.md`
- **Enhanced Registry Design:** `/home/anthony/nvidia/projects/aiperf/ajc/sticky-credit/ENHANCED_REGISTRY_DESIGN.md`
- **Example Plugin:** `/home/anthony/nvidia/projects/aiperf/ajc/sticky-credit/examples/aiperf-example-plugin/`
- **AIPerf Dev Guide:** `/home/anthony/nvidia/projects/aiperf/ajc/sticky-credit/CLAUDE.md`

### Registry Files

- **Built-in Registry:** `/home/anthony/nvidia/projects/aiperf/ajc/sticky-credit/src/aiperf/registry.yaml`
- **Plugin Registry Source:** `/home/anthony/nvidia/projects/aiperf/ajc/sticky-credit/src/aiperf/common/plugin_registry.py`
- **Phase Hooks Registry:** `/home/anthony/nvidia/projects/aiperf/ajc/sticky-credit/src/aiperf/timing/phase_hooks.py`

### External Resources

- **Python Type Hints:** https://docs.python.org/3/library/typing.html
- **Pydantic Documentation:** https://docs.pydantic.dev/
- **Pytest Async:** https://pytest-asyncio.readthedocs.io/
- **Ruff Linter:** https://docs.astral.sh/ruff/
- **PyPI Publishing:** https://packaging.python.org/tutorials/packaging-projects/

---

## Summary

This guide covers modern Python best practices for AIPerf plugin development in 2025:

1. **Modern Patterns:** Type hints with protocols, Pydantic validation, pathlib, async/await, context managers, match/case
2. **Development Checklist:** pyproject.toml, type hints, Pydantic config, protocols, docstrings, tests, README, semver, license, entry points
3. **Code Quality:** Type checking (pyright/mypy), linting (ruff), formatting, testing (pytest), documentation, error handling
4. **Performance:** Lazy loading, async I/O, connection pooling, caching, memory efficiency (slots/frozen)
5. **Security:** Input validation, API key handling, error sanitization, dependency pinning, vulnerability scanning
6. **Testing:** Async fixtures, mock dependencies, test isolation, parametrization, integration tests
7. **Documentation:** README structure, API reference, usage examples, troubleshooting
8. **Publishing:** Pre-publishing validation, package building, PyPI upload, GitHub releases, post-publishing
9. **Examples:** Minimal hook, advanced hook, custom endpoint, post-processor, multi-protocol plugin
10. **Anti-Patterns:** Global mutable state, blocking I/O, missing error handling, hardcoded paths, missing types, circular imports, too much magic, ignoring lifecycle
11. **Migration:** Factory to registry, deprecation handling, testing migration
12. **Common Issues:** Import errors, priority conflicts, registry initialization, config validation, async issues, hooks not called

**Follow these patterns for production-ready, maintainable, secure, and performant AIPerf plugins.**

---

*Generated for AIPerf 2.0 - 2025*
