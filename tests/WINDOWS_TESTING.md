<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Testing AIPerf on Windows

This document explains how to run the AIPerf test suite on Windows and describes the Windows-specific test infrastructure.

## Overview

The AIPerf test suite is now fully compatible with Windows. Tests that require Unix-specific features (like fork, bash commands, or Docker with Unix paths) are automatically skipped on Windows.

## Quick Start

### Running All Tests on Windows

```bash
# Run all tests (Unix-specific tests will be skipped automatically)
pytest

# Run with verbose output
pytest -v

# Run tests in parallel (faster)
pytest -n auto
```

### Running Platform-Specific Tests

```bash
# Run only Windows-specific tests
pytest -m windows

# Run cross-platform tests (skips platform-specific)
pytest -m "not windows and not macos and not linux and not unix"

# Run everything including Unix tests (will skip on Windows)
pytest -m ""
```

## Test Markers

The test suite uses pytest markers to manage platform compatibility:

| Marker | Description | Behavior on Windows |
|--------|-------------|---------------------|
| `@pytest.mark.windows` | Windows-only tests | Run |
| `@pytest.mark.macos` | macOS-only tests | Skip |
| `@pytest.mark.linux` | Linux-only tests | Skip |
| `@pytest.mark.unix` | Unix-only (macOS/Linux) | Skip |
| `@pytest.mark.skip_on_windows` | Skip on Windows | Skip |

## Windows-Specific Test Files

### Core Platform Tests

1. **`tests/common/test_bootstrap_windows.py`**
   - Tests Windows-specific bootstrap behavior
   - Verifies signal handling adaptations
   - Checks that macOS-specific fixes don't run on Windows

2. **`tests/common/test_platform_compatibility.py`**
   - Cross-platform compatibility tests
   - Tests IPC socket path generation
   - Tests virtual environment path detection
   - Tests multiprocessing compatibility
   - Tests signal availability per platform

### What Gets Tested on Windows

✅ **Runs on Windows:**
- All core unit tests
- Configuration tests
- Message handling tests
- Data structure tests
- Cross-platform compatibility tests
- Windows-specific tests

❌ **Skipped on Windows:**
- macOS-specific terminal fixes
- Linux-specific signal handling
- Docker tests with Unix paths
- Bash script tests
- Fork-based multiprocessing tests

## Platform Detection

Tests use automatic platform detection:

```python
import sys

# Check if running on Windows
if sys.platform == "win32":
    # Windows-specific code
    pass

# Check platform.system() for more detail
import platform
if platform.system() == "Windows":
    # Windows detected
    pass
```

## Common Issues and Solutions

### Issue: Tests fail with "IPC not supported"

**Solution:** The code automatically uses `.as_posix()` for IPC paths. If you see this error:
1. Make sure you're running an updated version with the Windows fixes
2. Check that ZMQ is installed: `pip install pyzmq`
3. Consider using TCP backend instead: `--comm-backend zmq_tcp`

### Issue: Tests fail with "SIGKILL not found"

**Solution:** The code uses `signal.SIGTERM` on Windows instead of `SIGKILL`. This is handled automatically.

### Issue: Tests fail with "ForkProcess not found"

**Solution:** The code conditionally imports `ForkProcess`. This is handled automatically in the Windows fixes.

### Issue: Virtual environment not detected

**Solution:** Windows uses `Scripts/python.exe` instead of `bin/python`. The test utilities handle this automatically.

## Writing New Tests

### For Cross-Platform Tests

```python
def test_something_cross_platform():
    """This test will run on all platforms."""
    # Use cross-platform APIs
    import tempfile
    from pathlib import Path

    temp_dir = Path(tempfile.gettempdir())
    # ... rest of test
```

### For Windows-Only Tests

```python
import pytest

@pytest.mark.windows
def test_windows_specific_feature():
    """This test only runs on Windows."""
    import sys
    assert sys.platform == "win32"
    # ... Windows-specific test code
```

### For Unix-Only Tests

```python
import pytest

@pytest.mark.unix
def test_unix_specific_feature():
    """This test only runs on Unix (macOS/Linux)."""
    # ... Unix-specific test code
```

### Skipping on Windows

```python
import pytest

@pytest.mark.skip_on_windows
def test_bash_command():
    """This test requires bash and will skip on Windows."""
    import subprocess
    subprocess.run(["bash", "-c", "echo test"])
```

## Test Fixtures

### Platform Mocking Fixtures

```python
def test_with_windows_mock(mock_platform_windows):
    """Test with Windows platform mocked."""
    # platform.system() will return "Windows"
    pass

def test_with_macos_mock(mock_platform_darwin):
    """Test with macOS platform mocked."""
    # platform.system() will return "Darwin"
    pass

def test_with_linux_mock(mock_platform_linux):
    """Test with Linux platform mocked."""
    # platform.system() will return "Linux"
    pass
```

## CI/CD on Windows

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest -v
```

### Expected Test Results on Windows

- **Total tests:** ~XXX (varies with each version)
- **Passed:** ~XXX (cross-platform + Windows-specific)
- **Skipped:** ~XX (Unix-specific tests)
- **Failed:** 0 (all tests should pass or skip appropriately)

## Troubleshooting

### Import Errors

If you see import errors related to Unix modules:

```python
# ❌ Bad - will fail on Windows
from fcntl import flock

# ✅ Good - conditional import
import sys
if sys.platform != "win32":
    from fcntl import flock
```

### Path Errors

```python
# ❌ Bad - hardcoded Unix path
path = "/tmp/aiperf"

# ✅ Good - use tempfile
import tempfile
from pathlib import Path
path = Path(tempfile.gettempdir()) / "aiperf"
```

### Process Termination Errors

```python
# ❌ Bad - SIGKILL not available on Windows
os.kill(pid, signal.SIGKILL)

# ✅ Good - use appropriate signal per platform
import sys
kill_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
os.kill(pid, kill_signal)
```

## Performance Notes

- **Spawn vs Fork:** Windows only supports spawn multiprocessing (slower than fork)
- **IPC Sockets:** Limited support on Windows; consider using TCP backend
- **File Paths:** Use `pathlib.Path` for cross-platform path handling
- **Line Endings:** Tests handle both CRLF and LF

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [Python Windows FAQ](https://docs.python.org/3/faq/windows.html)
- [Multiprocessing on Windows](https://docs.python.org/3/library/multiprocessing.html#windows)

## Reporting Issues

If you encounter Windows-specific test failures:

1. Check if the test is marked with a platform marker
2. Verify your Python version (3.10+ required)
3. Ensure all dependencies are installed: `pip install -e ".[dev]"`
4. Run with verbose output: `pytest -v tests/path/to/test.py`
5. Report issues with the full traceback and platform information

```bash
# Collect system info for bug reports
python --version
pytest --version
python -c "import sys; print(f'Platform: {sys.platform}')"
python -c "import platform; print(f'System: {platform.system()} {platform.release()}')"
```
