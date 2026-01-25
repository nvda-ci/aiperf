<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plugin Management CLI

AIPerf provides a comprehensive CLI for managing and inspecting plugins. This guide covers all available commands and their usage.

## Table of Contents

- [Overview](#overview)
- [Commands](#commands)
  - [list](#list)
  - [show](#show)
  - [list-implementations](#list-implementations)
  - [protocols](#protocols)
  - [info](#info)
  - [validate](#validate)
  - [inspect](#inspect)
- [Examples](#examples)
- [Common Workflows](#common-workflows)

## Overview

The plugin CLI provides tools for:
- **Discovery**: Find all loaded plugins and their implementations
- **Inspection**: View detailed information about plugins and implementations
- **Validation**: Ensure registry.yaml files are valid
- **Debugging**: Troubleshoot plugin loading and conflicts

All commands are accessed via the `aiperf plugins` subcommand.

## Commands

### list

List all discovered plugins.

```bash
aiperf plugins list [--builtin-only]
```

**Options:**
- `--builtin-only`: Show only built-in plugins (default: false)

**Output:**
- Table showing plugin names and types (Built-in or External)
- Total count of plugins found

**Example:**
```bash
$ aiperf plugins list
                AIPerf Plugins
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Plugin             ┃ Type     ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ aiperf             │ Built-in │
│ aiperf-datadog     │ External │
└────────────────────┴──────────┘

Found 2 plugin(s)
```

### show

Show detailed information about a specific plugin.

```bash
aiperf plugins show <plugin-name>
```

**Arguments:**
- `plugin-name`: Name of the plugin to inspect

**Output:**
- All implementations provided by the plugin, organized by protocol
- Implementation descriptions
- Plugin metadata (version, author, etc.)

**Example:**
```bash
$ aiperf plugins show aiperf
╭──────────────────────────╮
│ Plugin: aiperf           │
╰──────────────────────────╯

endpoint:
  • openai
    OpenAI-compatible API endpoints
  • anthropic
    Anthropic Claude API endpoints

timing_strategy:
  • fixed_schedule
    Fixed schedule strategy for trace replay
  • request_rate
    Request rate strategy with multiple modes

Plugin Metadata:
  Version: 2.0.0
  Description: AIPerf core implementations
  Author: NVIDIA
```

### list-implementations

List all implementations for a specific protocol.

```bash
aiperf plugins list-implementations <protocol> [--verbose]
```

**Arguments:**
- `protocol`: Protocol type (e.g., 'endpoint', 'timing_strategy')

**Options:**
- `--verbose`: Show detailed information including class paths and priorities

**Output:**
- Table of all implementations for the protocol
- Implementation names, plugins, and descriptions
- In verbose mode: class paths and priorities

**Example:**
```bash
$ aiperf plugins list-implementations endpoint
        endpoint Implementations
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name      ┃ Plugin ┃ Description                  ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ anthropic │ aiperf │ Anthropic Claude API         │
│ openai    │ aiperf │ OpenAI-compatible endpoints  │
└───────────┴────────┴──────────────────────────────┘

Found 2 implementation(s)
```

**Verbose Example:**
```bash
$ aiperf plugins list-implementations endpoint --verbose
                        endpoint Implementations
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Name      ┃ Plugin ┃ Class Path                    ┃ Priority ┃ Description       ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ anthropic │ aiperf │ aiperf.endpoints.anthropic... │        0 │ Anthropic Claude  │
│ openai    │ aiperf │ aiperf.endpoints.openai:Op... │        0 │ OpenAI-compatible │
└───────────┴────────┴───────────────────────────────┴──────────┴───────────────────┘
```

### protocols

List all available protocol types.

```bash
aiperf plugins protocols
```

**Output:**
- Table of all protocol types
- Count of implementations per protocol
- Protocol descriptions

**Example:**
```bash
$ aiperf plugins protocols
          Available Protocol Types
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Protocol              ┃ Implementations  ┃ Description                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ dataset               │                3 │ Dataset and data generation  │
│ endpoint              │                2 │ LLM API endpoint impls       │
│ exporter              │                2 │ Result export formats        │
│ timing_strategy       │                2 │ Request timing strategies    │
└───────────────────────┴──────────────────┴──────────────────────────────┘

Found 4 protocol type(s)
```

### info

Show detailed information about a specific implementation.

```bash
aiperf plugins info <protocol> <name>
```

**Arguments:**
- `protocol`: Protocol type (e.g., 'endpoint')
- `name`: Implementation name (e.g., 'openai')

**Output:**
- Comprehensive details about the implementation
- Protocol, plugin, class path, priority
- Built-in status and description
- Plugin metadata if available

**Example:**
```bash
$ aiperf plugins info endpoint openai
╭─────────────── endpoint:openai ───────────────╮
│ Implementation: openai                        │
│ Protocol: endpoint                            │
│ Plugin: aiperf                                │
│ Class Path: aiperf.endpoints.openai:OpenAI... │
│ Priority: 0                                   │
│ Built-in: True                                │
│                                               │
│ Description:                                  │
│ OpenAI-compatible API endpoints supporting    │
│ chat completions and streaming responses      │
│                                               │
│ Plugin Metadata:                              │
│ Version: 2.0.0                                │
│ Author: NVIDIA                                │
│ License: Apache-2.0                           │
╰───────────────────────────────────────────────╯
```

### validate

Validate registry.yaml format and class paths.

```bash
aiperf plugins validate [registry-file]
```

**Arguments:**
- `registry-file`: Optional path to registry.yaml (defaults to built-in registry)

**Validation Checks:**
1. YAML syntax validity
2. Schema version compliance
3. Plugin metadata completeness
4. Alphabetical ordering of implementations
5. No duplicate entries
6. Class path validity (all classes importable)

**Output:**
- Step-by-step validation results
- Detailed error messages if validation fails
- Summary of registry contents on success

**Example:**
```bash
$ aiperf plugins validate
Validating: /path/to/aiperf/registry.yaml
============================================================

1. Validating YAML syntax...
PASSED: Valid YAML syntax

2. Validating schema version...
PASSED: Schema version is valid

3. Validating plugin metadata...
PASSED: Plugin metadata is valid

4. Validating alphabetical ordering...
PASSED: All implementations are alphabetically ordered

5. Validating no duplicates...
PASSED: No duplicate entries

6. Validating class paths (this may take a moment)...
PASSED: All class paths are valid and importable

============================================================
SUCCESS: All validation checks passed!
Registry contains 7 protocol sections
Total implementations: 24
```

**Validate Custom Registry:**
```bash
$ aiperf plugins validate /path/to/custom/registry.yaml
```

### inspect

Inspect the complete plugin system state.

```bash
aiperf plugins inspect [plugin-name]
```

**Arguments:**
- `plugin-name`: Optional plugin name to focus on (defaults to system overview)

**Output (System Overview):**
- All loaded plugins
- All protocols with implementation counts
- Total implementation count
- Plugins contributing to each protocol

**Output (Specific Plugin):**
- Same as `show` command for the specified plugin

**Example:**
```bash
$ aiperf plugins inspect
╭──────────────────────────────────────╮
│ AIPerf Plugin System Overview        │
╰──────────────────────────────────────╯

Loaded Plugins: 2
  • aiperf (built-in)
  • aiperf-datadog (external)

Protocols: 4
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Protocol           ┃ Implementations  ┃ Plugins Contributing  ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ endpoint           │                2 │                     1 │
│ timing_strategy    │                2 │                     1 │
│ exporter           │                3 │                     2 │
└────────────────────┴──────────────────┴───────────────────────┘

Total Implementations: 24
```

## Examples

### Discovering Available Endpoints

```bash
# 1. List all protocols to find endpoint protocol
$ aiperf plugins protocols

# 2. List endpoint implementations
$ aiperf plugins list-implementations endpoint

# 3. Get details about OpenAI endpoint
$ aiperf plugins info endpoint openai
```

### Validating a Custom Plugin

```bash
# Validate your plugin's registry.yaml
$ aiperf plugins validate /path/to/my-plugin/registry.yaml

# If validation passes, check what it provides
$ aiperf plugins show my-plugin
```

### Debugging Plugin Conflicts

```bash
# 1. Inspect full system state
$ aiperf plugins inspect

# 2. Check specific protocol for conflicts
$ aiperf plugins list-implementations endpoint --verbose

# 3. Compare priorities of conflicting implementations
$ aiperf plugins info endpoint openai
$ aiperf plugins info endpoint custom_openai
```

### Finding Implementations for Your Needs

```bash
# 1. See all available protocol types
$ aiperf plugins protocols

# 2. List implementations for timing strategies
$ aiperf plugins list-implementations timing_strategy

# 3. Get detailed info about fixed_schedule
$ aiperf plugins info timing_strategy fixed_schedule
```

## Common Workflows

### Workflow 1: Exploring the Plugin System

```bash
# Start with overview
aiperf plugins inspect

# List all plugins
aiperf plugins list

# Show details for specific plugin
aiperf plugins show aiperf

# Explore specific protocol
aiperf plugins list-implementations endpoint --verbose
```

### Workflow 2: Creating a Custom Plugin

```bash
# 1. Check existing implementations for reference
aiperf plugins list-implementations endpoint

# 2. Get details about similar implementation
aiperf plugins info endpoint openai

# 3. Create your registry.yaml (see plugin development docs)

# 4. Validate your registry
aiperf plugins validate /path/to/your/registry.yaml

# 5. Install and verify
pip install -e /path/to/your/plugin
aiperf plugins list  # Should see your plugin
aiperf plugins show your-plugin
```

### Workflow 3: Troubleshooting

```bash
# Check if plugin is loaded
aiperf plugins list

# Verify protocol is registered
aiperf plugins protocols

# Check implementation exists
aiperf plugins list-implementations your_protocol

# Get detailed info
aiperf plugins info your_protocol your_impl

# Validate registry if issues persist
aiperf plugins validate /path/to/registry.yaml
```

## Tips

1. **Use `--verbose` for debugging**: The verbose flag shows class paths and priorities, useful for understanding plugin loading behavior.

2. **Validate early and often**: Run `validate` on your registry.yaml during development to catch issues early.

3. **Check priorities for conflicts**: If two plugins provide the same implementation, the one with higher priority wins. Use `list-implementations --verbose` to check priorities.

4. **Use `inspect` for overview**: Start with `inspect` to get a quick overview of the entire plugin system.

5. **Tab completion**: Most shells support tab completion for cyclopts commands. Press Tab to auto-complete protocol and implementation names.

## Exit Codes

- `0`: Success
- Non-zero: Error occurred (validation failed, plugin not found, etc.)

## Color Coding

The CLI uses color coding for clarity:
- **Cyan**: Primary information (names, protocols)
- **Green**: Success messages, counts
- **Yellow**: Warnings, empty results
- **Red**: Errors
- **Magenta**: Plugin names, types
- **Dim**: Secondary information, descriptions

## Getting Help

For help with any command:
```bash
aiperf plugins --help
aiperf plugins list --help
aiperf plugins info --help
```

## See Also

- [Plugin Development Guide](PLUGIN_DEVELOPMENT.md)
- [Plugin Registry System](PLUGIN_REGISTRY.md)
- [Plugin Best Practices](PLUGIN_BEST_PRACTICES.md)
