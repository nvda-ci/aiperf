<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plugin CLI - Quick Start Guide

Quick reference for the AIPerf plugin management CLI.

## Installation

The plugin CLI is built-in to AIPerf. No additional installation needed.

```bash
pip install aiperf
```

## Quick Commands

### List all plugins
```bash
aiperf plugins list-plugins
```

### Show what a plugin provides
```bash
aiperf plugins show aiperf
```

### List all protocol types
```bash
aiperf plugins protocols
```

### List implementations for a protocol
```bash
aiperf plugins list-implementations endpoint
```

### Get details about a specific implementation
```bash
aiperf plugins info endpoint chat
```

### Validate a registry file
```bash
aiperf plugins validate
aiperf plugins validate /path/to/registry.yaml
```

### Inspect the entire system
```bash
aiperf plugins inspect
```

## Common Use Cases

### "What endpoints can I use?"
```bash
# List all endpoint implementations
aiperf plugins list-implementations endpoint

# Get details about a specific endpoint
aiperf plugins info endpoint chat
```

### "What timing strategies are available?"
```bash
aiperf plugins list-implementations timing_strategy
```

### "I'm developing a plugin, how do I validate it?"
```bash
# Validate your registry.yaml
aiperf plugins validate /path/to/my-plugin/registry.yaml

# Check if it loads correctly
aiperf plugins show my-plugin

# Verify implementations are registered
aiperf plugins list-implementations my_protocol
```

### "Which plugin provides this implementation?"
```bash
# Show detailed info including plugin name
aiperf plugins info protocol_name impl_name
```

### "What protocols can I extend?"
```bash
# List all extensible protocol types
aiperf plugins protocols
```

## Verbose Mode

Add `--verbose` to `list-implementations` for more details:

```bash
aiperf plugins list-implementations endpoint --verbose
```

This shows:
- Class paths
- Priorities (for conflict resolution)
- Full implementation details

## Getting Help

```bash
# General help
aiperf plugins --help

# Command-specific help
aiperf plugins info --help
aiperf plugins list-implementations --help
```

## Output

The CLI uses rich formatting with:
- **Tables** for listings
- **Panels** for detailed info
- **Color coding** (cyan=names, green=success, red=errors, yellow=warnings)

## Exit Codes

- `0` = Success
- Non-zero = Error (e.g., plugin not found, validation failed)

## Tab Completion

Most shells support tab completion. Install shell completion for best experience.

## See Also

- [Full CLI Documentation](CLI_PLUGINS.md)
- [Plugin Development Guide](PLUGIN_DEVELOPMENT.md)
- [Plugin Registry System](PLUGIN_REGISTRY.md)
